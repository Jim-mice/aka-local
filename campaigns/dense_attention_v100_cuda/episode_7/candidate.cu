#include <cuda_runtime.h>
#include <math.h>

// AKA_CONTRACT: q, k, v, o, batch, heads, seq, head_dim, scale

__device__ __forceinline__ float warp_sum(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffff, x, offset);
    }
    return x;
}

__device__ __forceinline__ float block_sum(float x) {
    __shared__ float warp_totals[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    x = warp_sum(x);
    if (lane == 0) warp_totals[warp] = x;
    __syncthreads();
    x = (threadIdx.x < (blockDim.x >> 5)) ? warp_totals[lane] : 0.0f;
    if (warp == 0) x = warp_sum(x);
    if (threadIdx.x == 0) warp_totals[0] = x;
    __syncthreads();
    return warp_totals[0];
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
    int row = blockIdx.x;
    int total_rows = batch * heads * seq;
    if (row >= total_rows) return;

    extern __shared__ float q_shared[];
    int tid = threadIdx.x;
    int q_base = row * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        q_shared[d] = q[q_base + d];
    }
    __syncthreads();

    int bh = row / seq;
    int qi = row - bh * seq;
    int kv_base = bh * seq * head_dim;
    int out_base = q_base;

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        o[out_base + d] = 0.0f;
    }

    for (int j = 0; j < seq; ++j) {
        int key_base = kv_base + j * head_dim;
        float partial = 0.0f;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            partial = fmaf(q_shared[d], k[key_base + d], partial);
        }
        float score = block_sum(partial) * scale;

        float new_max = fmaxf(row_max, score);
        float old_factor = (row_sum == 0.0f) ? 0.0f : expf(row_max - new_max);
        float weight = expf(score - new_max);
        float new_sum = row_sum * old_factor + weight;

        for (int d = tid; d < head_dim; d += blockDim.x) {
            float prev = o[out_base + d];
            float value = v[key_base + d];
            o[out_base + d] = (prev * row_sum * old_factor + value * weight) / new_sum;
        }
        row_max = new_max;
        row_sum = new_sum;
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
    int rows = batch * heads * seq;
    dim3 block(256, 1, 1);
    dim3 grid(rows, 1, 1);
    size_t shared_bytes = static_cast<size_t>(head_dim) * sizeof(float);
    dense_attention_kernel<<<grid, block, shared_bytes>>>(q, k, v, o, batch, heads, seq, head_dim, scale);
}
