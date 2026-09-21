#include <cuda_runtime.h>
#include <math.h>

// AKA_CONTRACT: q, k, v, o, batch, heads, seq, head_dim, scale

__inline__ __device__ float warp_reduce_max(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_down_sync(0xffffffff, x, offset));
    }
    return x;
}

__inline__ __device__ float warp_reduce_sum(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffff, x, offset);
    }
    return x;
}

__inline__ __device__ float block_reduce_max(float x, float* smem) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    x = warp_reduce_max(x);
    if (lane == 0) smem[warp] = x;
    __syncthreads();
    x = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : -CUDART_INF_F;
    if (warp == 0) x = warp_reduce_max(x);
    if (threadIdx.x == 0) smem[0] = x;
    __syncthreads();
    return smem[0];
}

__inline__ __device__ float block_reduce_sum(float x, float* smem) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    x = warp_reduce_sum(x);
    if (lane == 0) smem[warp] = x;
    __syncthreads();
    x = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
    if (warp == 0) x = warp_reduce_sum(x);
    if (threadIdx.x == 0) smem[0] = x;
    __syncthreads();
    return smem[0];
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
    const int qrow = row;
    const int base = (qrow / seq) * seq * head_dim;
    const int qi = qrow % seq;
    const int qoff = base + qi * head_dim;
    const int nwarps = blockDim.x >> 5;
    __shared__ float reduction[16];

    float local_max = -CUDART_INF_F;
    for (int j = tid; j < seq; j += blockDim.x) {
        const int koff = base + j * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot = fmaf(q[qoff + d], k[koff + d], dot);
        }
        local_max = fmaxf(local_max, dot * scale);
    }
    const float row_max = block_reduce_max(local_max, reduction);

    float local_sum = 0.0f;
    for (int j = tid; j < seq; j += blockDim.x) {
        const int koff = base + j * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot = fmaf(q[qoff + d], k[koff + d], dot);
        }
        local_sum += __expf(dot * scale - row_max);
    }
    const float row_sum = block_reduce_sum(local_sum, reduction);
    const float inv_sum = 1.0f / row_sum;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float accum = 0.0f;
        for (int j = 0; j < seq; ++j) {
            const int koff = base + j * head_dim;
            float dot = 0.0f;
            for (int x = 0; x < head_dim; ++x) {
                dot = fmaf(q[qoff + x], k[koff + x], dot);
            }
            accum += (__expf(dot * scale - row_max) * inv_sum) * v[koff + d];
        }
        o[qoff + d] = accum;
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