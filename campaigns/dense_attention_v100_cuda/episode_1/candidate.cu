#include <cuda_runtime.h>
#include <math.h>

// AKA_CONTRACT: q, k, v, o, batch, heads, seq, head_dim, scale

namespace {

__device__ __forceinline__ float warp_reduce_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_reduce_max(float value, float* warp_values) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_reduce_max(value);
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();
    value = (threadIdx.x < (blockDim.x >> 5)) ? warp_values[lane] : -CUDART_INF_F;
    if (warp == 0) value = warp_reduce_max(value);
    if (threadIdx.x == 0) warp_values[0] = value;
    __syncthreads();
    return warp_values[0];
}

__device__ __forceinline__ float block_reduce_sum(float value, float* warp_values) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_reduce_sum(value);
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();
    value = (threadIdx.x < (blockDim.x >> 5)) ? warp_values[lane] : 0.0f;
    if (warp == 0) value = warp_reduce_sum(value);
    if (threadIdx.x == 0) warp_values[0] = value;
    __syncthreads();
    return warp_values[0];
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
    const int query_pos = row % seq;
    const int head_row = row / seq;
    const size_t head_base = (size_t)head_row * (size_t)seq * (size_t)head_dim;
    const size_t query_base = head_base + (size_t)query_pos * (size_t)head_dim;

    __shared__ float reduction[8];
    float row_max = -CUDART_INF_F;
    for (int key_pos = tid; key_pos < seq; key_pos += blockDim.x) {
        const float* q_ptr = q + query_base;
        const float* k_ptr = k + head_base + (size_t)key_pos * (size_t)head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot = fmaf(q_ptr[d], k_ptr[d], dot);
        }
        row_max = fmaxf(row_max, dot * scale);
    }
    row_max = block_reduce_max(row_max, reduction);

    float row_sum = 0.0f;
    for (int key_pos = tid; key_pos < seq; key_pos += blockDim.x) {
        const float* q_ptr = q + query_base;
        const float* k_ptr = k + head_base + (size_t)key_pos * (size_t)head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot = fmaf(q_ptr[d], k_ptr[d], dot);
        }
        row_sum += expf(dot * scale - row_max);
    }
    row_sum = block_reduce_sum(row_sum, reduction);
    const float inv_sum = 1.0f / row_sum;

    float* out_ptr = o + query_base;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float result = 0.0f;
        for (int key_pos = 0; key_pos < seq; ++key_pos) {
            const float* q_ptr = q + query_base;
            const float* k_ptr = k + head_base + (size_t)key_pos * (size_t)head_dim;
            float dot = 0.0f;
            for (int kd = 0; kd < head_dim; ++kd) {
                dot = fmaf(q_ptr[kd], k_ptr[kd], dot);
            }
            const float probability = expf(dot * scale - row_max) * inv_sum;
            result = fmaf(probability, v[head_base + (size_t)key_pos * (size_t)head_dim + d], result);
        }
        out_ptr[d] = result;
    }
}

}  // namespace

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
    const int threads = 256;
    const int rows = batch * heads * seq;
    dense_attention_kernel<<<rows, threads>>>(q, k, v, o, batch, heads, seq, head_dim, scale);
}
