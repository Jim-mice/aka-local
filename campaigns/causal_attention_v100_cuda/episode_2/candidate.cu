#include <cuda_runtime.h>
#include <math.h>

// AKA_CONTRACT: q, k, v, o, batch, heads, seq, head_dim, scale

__device__ __forceinline__ float warp_sum(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffffu, x, offset);
    }
    return x;
}

__device__ __forceinline__ float block_sum(float x, float* warp_sums) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    x = warp_sum(x);
    if (lane == 0) warp_sums[warp] = x;
    __syncthreads();
    float v = (threadIdx.x < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
    if (warp == 0) v = warp_sum(v);
    if (threadIdx.x == 0) warp_sums[0] = v;
    __syncthreads();
    return warp_sums[0];
}

__global__ void causal_attention_kernel(
    float* q, float* k, float* v, float* o,
    int batch, int heads, int seq, int head_dim, float scale) {
    int row = blockIdx.x;
    int total_rows = batch * heads * seq;
    if (row >= total_rows) return;

    int tid = threadIdx.x;
    int qi = row % seq;
    int bh = row / seq;
    size_t base = (size_t)bh * (size_t)seq * (size_t)head_dim;
    size_t qbase = base + (size_t)qi * (size_t)head_dim;
    __shared__ float warp_sums[8];
    __shared__ float score_shared;
    __shared__ float max_shared;
    __shared__ float denom_shared;

    float row_max = -INFINITY;
    for (int j = 0; j <= qi; ++j) {
        float dot = 0.0f;
        size_t kbase = base + (size_t)j * (size_t)head_dim;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            dot = fmaf(q[qbase + d], k[kbase + d], dot);
        }
        float total = block_sum(dot, warp_sums) * scale;
        if (tid == 0) score_shared = total;
        __syncthreads();
        if (score_shared > row_max) row_max = score_shared;
    }
    if (tid == 0) max_shared = row_max;
    __syncthreads();

    float denom = 0.0f;
    for (int j = 0; j <= qi; ++j) {
        float dot = 0.0f;
        size_t kbase = base + (size_t)j * (size_t)head_dim;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            dot = fmaf(q[qbase + d], k[kbase + d], dot);
        }
        float total = block_sum(dot, warp_sums) * scale;
        if (tid == 0) score_shared = total;
        __syncthreads();
        if (tid == 0) denom += __expf(score_shared - max_shared);
    }
    float denom_total = block_sum(denom, warp_sums);
    if (tid == 0) denom_shared = denom_total;
    __syncthreads();

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j <= qi; ++j) {
            float dot = 0.0f;
            size_t kbase = base + (size_t)j * (size_t)head_dim;
            for (int x = tid; x < head_dim; x += blockDim.x) {
                dot = fmaf(q[qbase + x], k[kbase + x], dot);
            }
            float total = block_sum(dot, warp_sums) * scale;
            if (tid == 0) score_shared = total;
            __syncthreads();
            acc = fmaf(__expf(score_shared - max_shared), v[kbase + d], acc);
        }
        o[qbase + d] = acc / denom_shared;
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
    float scale
) {
    int rows = batch * heads * seq;
    causal_attention_kernel<<<rows, 256>>>(q, k, v, o, batch, heads, seq, head_dim, scale);
}