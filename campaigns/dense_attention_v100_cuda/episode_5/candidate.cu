#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

// AKA_CONTRACT: q, k, v, o, batch, heads, seq, head_dim, scale

__device__ __forceinline__ float block_sum(float x, float* smem) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) x += __shfl_down_sync(0xffffffff, x, offset);
    if (lane == 0) smem[warp] = x;
    __syncthreads();
    x = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) x += __shfl_down_sync(0xffffffff, x, offset);
    }
    if (threadIdx.x == 0) smem[0] = x;
    __syncthreads();
    return smem[0];
}

__device__ __forceinline__ float block_max(float x, float* smem) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) x = fmaxf(x, __shfl_down_sync(0xffffffff, x, offset));
    if (lane == 0) smem[warp] = x;
    __syncthreads();
    x = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : -INFINITY;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) x = fmaxf(x, __shfl_down_sync(0xffffffff, x, offset));
    }
    if (threadIdx.x == 0) smem[0] = x;
    __syncthreads();
    return smem[0];
}

extern "C" __global__ void dense_attention_kernel(
    const float* q, const float* k, const float* v, float* o,
    int batch, int heads, int seq, int head_dim, float scale) {
    int row = blockIdx.x;
    int total_rows = batch * heads * seq;
    if (row >= total_rows) return;
    int tid = threadIdx.x;
    int qbase = row * head_dim;
    int kvbase = (row / seq) * seq * head_dim;
    extern __shared__ float shared[];
    float* qtile = shared;
    float* reduce = shared + head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) qtile[d] = q[qbase + d];
    __syncthreads();

    float row_max = -INFINITY;
    for (int j = tid; j < seq; j += blockDim.x) {
        float dot = 0.0f;
        int kbase = kvbase + j * head_dim;
        for (int d = 0; d < head_dim; ++d) dot = fmaf(qtile[d], k[kbase + d], dot);
        dot *= scale;
        row_max = fmaxf(row_max, dot);
    }
    row_max = block_max(row_max, reduce);

    float denom = 0.0f;
    for (int j = tid; j < seq; j += blockDim.x) {
        int kbase = kvbase + j * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot = fmaf(qtile[d], k[kbase + d], dot);
        denom += expf(dot * scale - row_max);
    }
    denom = block_sum(denom, reduce);
    float inv_denom = 1.0f / denom;

    int obase = qbase;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < seq; ++j) {
            int kbase = kvbase + j * head_dim;
            float dot = 0.0f;
            for (int x = 0; x < head_dim; ++x) dot = fmaf(qtile[x], k[kbase + x], dot);
            float p = expf(dot * scale - row_max) * inv_denom;
            acc = fmaf(p, v[kbase + d], acc);
        }
        o[obase + d] = acc;
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
    int threads = 256;
    size_t shared_bytes = (size_t)head_dim * sizeof(float) + (threads / 32) * sizeof(float);
    dense_attention_kernel<<<batch * heads * seq, threads, shared_bytes>>>(q, k, v, o, batch, heads, seq, head_dim, scale);
}