#include <cuda_runtime.h>
#include <math.h>

// AKA_CONTRACT: q, k, v, o, batch, heads, seq, head_dim, scale

__device__ __forceinline__ float warp_max(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_down_sync(0xffffffffu, x, offset));
    }
    return x;
}

__device__ __forceinline__ float warp_sum(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffffu, x, offset);
    }
    return x;
}

__device__ __forceinline__ float score_for(
    const float* __restrict__ qrow,
    const float* __restrict__ krow,
    int head_dim,
    float scale,
    int lane) {
    float acc = 0.0f;
    for (int d = lane; d < head_dim; d += 32) {
        acc = fmaf(qrow[d], krow[d], acc);
    }
    return __shfl_sync(0xffffffffu, warp_sum(acc), 0) * scale;
}

__global__ void dense_attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ o,
    int batch,
    int heads,
    int seq,
    int head_dim,
    float scale) {
    const int row = blockIdx.x;
    const int total_rows = batch * heads * seq;
    if (row >= total_rows) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int bh = row / seq;
    const int qi = row - bh * seq;
    const size_t head_base = (size_t)bh * (size_t)seq * (size_t)head_dim;
    const float* qrow = q + head_base + (size_t)qi * head_dim;
    float* orow = o + head_base + (size_t)qi * head_dim;

    __shared__ float tile_scores[256];
    __shared__ float warp_reduce[8];
    __shared__ float row_max;
    __shared__ float row_sum;

    if (tid == 0) row_max = -CUDART_INF_F;
    __syncthreads();
for (int base = 0; base < seq; base += 256) {
        for (int j = warp * 32 + lane; j < 256; j += 256) {
            const int kj = base + j;
            float s = -CUDART_INF_F;
            if (kj < seq) {
                const float* krow = k + head_base + (size_t)kj * head_dim;
                s = score_for(qrow, krow, head_dim, scale, lane);
            }
            tile_scores[j] = s;
        }
        __syncthreads();
        float x = (tid < 256) ? tile_scores[tid] : -CUDART_INF_F;
        x = warp_max(x);
        if (lane == 0) warp_reduce[warp] = x;
        __syncthreads();
        if (warp == 0) {
            x = (lane < 8) ? warp_reduce[lane] : -CUDART_INF_F;
            x = warp_max(x);
            if (lane == 0) row_max = fmaxf(row_max, x);
        }
        __syncthreads();
    const float m = row_max;

    float sum = 0.0f;
    for (int base = 0; base < seq; base += 256) {
        for (int j = warp * 32 + lane; j < 256; j += 256) {
            const int kj = base + j;
            float s = -CUDART_INF_F;
            if (kj < seq) {
                const float* krow = k + head_base + (size_t)kj * head_dim;
                s = score_for(qrow, krow, head_dim, scale, lane);
            }
            tile_scores[j] = s;
        }
        __syncthreads();
        if (tid < 256) {
            const float s = tile_scores[tid];
            sum += (s == -CUDART_INF_F) ? 0.0f : expf(s - m);
        }
        __syncthreads();
    }
    sum = warp_sum(sum);
    if (lane == 0) warp_reduce[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = (lane < 8) ? warp_reduce[lane] : 0.0f;
        sum = warp_sum(sum);
        if (lane == 0) row_sum = sum;
    }
    __syncthreads();
    const float inv_sum = 1.0f / row_sum;

    for (int d = tid; d < head_dim; d += 256) {
        float acc = 0.0f;
        for (int base = 0; base < seq; base += 256) {
            for (int j = warp * 32 + lane; j < 256; j += 256) {
                const int kj = base + j;
                float s = -CUDART_INF_F;
                if (kj < seq) {
                    const float* krow = k + head_base + (size_t)kj * head_dim;
                    s = score_for(qrow, krow, head_dim, scale, lane);
                }
                tile_scores[j] = s;
            }
            __syncthreads();
            for (int j = 0; j < 256; ++j) {
                const int kj = base + j;
                if (kj < seq) {
                    const float p = expf(tile_scores[j] - m) * inv_sum;
                    acc = fmaf(p, v[head_base + (size_t)kj * head_dim + d], acc);
                }
            }
            __syncthreads();
        }
        orow[d] = acc;
    }
}

extern "C" void launch_kernel(
    float* q,
    float* k,
    float* v,
    float* o,
    int batch,
    int heads,
    int seq,
    int head_dim,
    float scale) {
    const int rows = batch * heads * seq;
    dense_attention_kernel<<<rows, 256>>>(q, k, v, o, batch, heads, seq, head_dim, scale);
}