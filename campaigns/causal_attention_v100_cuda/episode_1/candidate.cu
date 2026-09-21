#include <cuda_runtime.h>

// AKA_CONTRACT: q, k, v, o, batch, heads, seq, head_dim, scale

__device__ __forceinline__ float warp_sum(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffff, x, offset);
    }
    return x;
}

__device__ __forceinline__ float block_sum(float x, float* shared) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    x = warp_sum(x);
    if (lane == 0) shared[warp] = x;
    __syncthreads();
    x = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
    if (warp == 0) x = warp_sum(x);
    if (threadIdx.x == 0) shared[0] = x;
    __syncthreads();
    return shared[0];
}

__global__ void causal_attention_kernel(
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
    const int rows = batch * heads * seq;
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const int qi = row % seq;
    const int bh = row / seq;
    const long long base = (long long)bh * seq * head_dim;
    const long long qbase = base + (long long)qi * head_dim;
    const long long obase = qbase;

    __shared__ float red[8];
    __shared__ float row_max;
    __shared__ float row_sum;
    __shared__ float score_weight;
    __shared__ float old_scale;

    if (tid == 0) {
        row_max = -INFINITY;
        row_sum = 0.0f;
    }
    __syncthreads();

    for (int j = 0; j <= qi; ++j) {
        float dot = 0.0f;
        const long long kbase = base + (long long)j * head_dim;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            dot = fmaf(q[qbase + d], k[kbase + d], dot);
        }
        dot = block_sum(dot, red) * scale;

        if (tid == 0) {
            const float new_max = fmaxf(row_max, dot);
            const float rescale = (row_sum == 0.0f) ? 0.0f : __expf(row_max - new_max);
            const float w = __expf(dot - new_max);
            old_scale = rescale;
            score_weight = w;
            row_sum = row_sum * rescale + w;
            row_max = new_max;
        }
        __syncthreads();

        const float rescale = old_scale;
        const float w = score_weight;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            const long long idx = obase + d;
            o[idx] = o[idx] * rescale + w * v[kbase + d];
        }
        __syncthreads();
    }

    const float inv_sum = 1.0f / row_sum;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        o[obase + d] *= inv_sum;
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
    if (rows <= 0 || head_dim <= 0) return;
    causal_attention_kernel<<<rows, 256>>>(q, k, v, o, batch, heads, seq, head_dim, scale);
}
