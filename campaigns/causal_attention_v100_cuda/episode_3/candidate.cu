#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

// AKA_CONTRACT: q, k, v, o, batch, heads, seq, head_dim, scale

static __device__ __forceinline__ float warp_max(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) x = fmaxf(x, __shfl_down_sync(0xffffffff, x, offset));
    return x;
}

static __device__ __forceinline__ float warp_sum(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) x += __shfl_down_sync(0xffffffff, x, offset);
    return x;
}

static __device__ __forceinline__ float block_max(float x, float* warp_buf) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    x = warp_max(x);
    if (lane == 0) warp_buf[warp] = x;
    __syncthreads();
    float y = (threadIdx.x < 8) ? warp_buf[threadIdx.x] : -INFINITY;
    if (warp == 0) y = warp_max(y);
    if (threadIdx.x == 0) warp_buf[0] = y;
    __syncthreads();
    return warp_buf[0];
}

static __device__ __forceinline__ float block_sum(float x, float* warp_buf) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    x = warp_sum(x);
    if (lane == 0) warp_buf[warp] = x;
    __syncthreads();
    float y = (threadIdx.x < 8) ? warp_buf[threadIdx.x] : 0.0f;
    if (warp == 0) y = warp_sum(y);
    if (threadIdx.x == 0) warp_buf[0] = y;
    __syncthreads();
    return warp_buf[0];
}

extern "C" __global__ void causal_attention_kernel(
    float* q, float* k, float* v, float* o,
    int batch, int heads, int seq, int head_dim, float scale) {
    int row = blockIdx.x;
    int rows = batch * heads * seq;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int i = row % seq;
    int bh = row / seq;
    size_t base = (size_t)bh * seq * head_dim;
    const float* qrow = q + base + (size_t)i * head_dim;
    float* outrow = o + base + (size_t)i * head_dim;

    extern __shared__ float smem[];
    float* scores = smem;
    float* warp_buf = smem + seq;

    for (int j = tid; j <= i; j += blockDim.x) {
        const float* krow = k + base + (size_t)j * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot = fmaf(qrow[d], krow[d], dot);
        scores[j] = dot * scale;
    }
    for (int j = tid; j < seq; j += blockDim.x) {
        if (j > i) scores[j] = -INFINITY;
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int j = tid; j <= i; j += blockDim.x) local_max = fmaxf(local_max, scores[j]);
    float m = block_max(local_max, warp_buf);

    float local_sum = 0.0f;
    for (int j = tid; j <= i; j += blockDim.x) {
        float e = expf(scores[j] - m);
        scores[j] = e;
        local_sum += e;
    }
    float denom = block_sum(local_sum, warp_buf);
    float inv_denom = 1.0f / denom;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j <= i; ++j) {
            acc = fmaf(scores[j] * inv_denom, v[base + (size_t)j * head_dim + d], acc);
        }
        outrow[d] = acc;
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
    size_t smem = (size_t)(seq + 8) * sizeof(float);
    dim3 grid(rows);
    dim3 block(256);
    causal_attention_kernel<<<grid, block, smem>>>(q, k, v, o, batch, heads, seq, head_dim, scale);
}