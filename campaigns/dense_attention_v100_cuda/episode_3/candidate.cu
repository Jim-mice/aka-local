#include <cuda_runtime.h>
#include <math.h>

// AKA_CONTRACT: q, k, v, o, batch, heads, seq, head_dim, scale

__device__ __forceinline__ float warp_sum(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffff, x, offset);
    }
    return x;
}

__device__ __forceinline__ float warp_broadcast(float x) {
    return __shfl_sync(0xffffffff, x, 0);
}

__global__ void dense_attention_kernel(const float* __restrict__ q,
                                       const float* __restrict__ k,
                                       const float* __restrict__ v,
                                       float* __restrict__ o,
                                       int batch, int heads, int seq,
                                       int head_dim, float scale) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int rows_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * rows_per_block + warp_in_block;
    const int total_rows = batch * heads * seq;
    if (row >= total_rows) return;

    const int bh = row / seq;
    const int qi = row - bh * seq;
    const size_t head_base = (size_t)bh * (size_t)seq * (size_t)head_dim;
    const size_t q_row = head_base + (size_t)qi * (size_t)head_dim;

    float row_max = -CUDART_INF_F;
    for (int j = 0; j < seq; ++j) {
        const size_t k_row = head_base + (size_t)j * (size_t)head_dim;
        float dot = 0.0f;
        for (int d = lane; d < head_dim; d += 32) {
            dot = fmaf(q[q_row + d], k[k_row + d], dot);
        }
        dot = warp_sum(dot) * scale;
        row_max = fmaxf(row_max, warp_broadcast(dot));
    }

    float denom = 0.0f;
    for (int j = 0; j < seq; ++j) {
        const size_t k_row = head_base + (size_t)j * (size_t)head_dim;
        float dot = 0.0f;
        for (int d = lane; d < head_dim; d += 32) {
            dot = fmaf(q[q_row + d], k[k_row + d], dot);
        }
        dot = warp_sum(dot) * scale;
        if (lane == 0) denom = denom + expf(dot - row_max);
    }
    denom = warp_broadcast(denom);
    const float inv_denom = 1.0f / denom;

    const size_t o_row = q_row;
    for (int d = lane; d < head_dim; d += 32) {
        float acc = 0.0f;
        for (int j = 0; j < seq; ++j) {
            const size_t k_row = head_base + (size_t)j * (size_t)head_dim;
            float dot = 0.0f;
            for (int x = lane; x < head_dim; x += 32) {
                dot = fmaf(q[q_row + x], k[k_row + x], dot);
            }
            dot = warp_sum(dot) * scale;
            const float p = expf(dot - row_max) * inv_denom;
            const size_t v_index = k_row + (size_t)d;
            acc = fmaf(p, v[v_index], acc);
        }
        o[o_row + d] = acc;
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
    const int rows_per_block = 8;
    const int total_rows = batch * heads * seq;
    const int blocks = (total_rows + rows_per_block - 1) / rows_per_block;
    dense_attention_kernel<<<blocks, 256>>>(q, k, v, o, batch, heads, seq, head_dim, scale);
}