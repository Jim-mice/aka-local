#include <cuda_runtime.h>
#include <math.h>

// AKA_CONTRACT: q, k, v, p, grad_o, grad_q, grad_k, grad_v, batch, heads, seq, head_dim, scale

__global__ void clear_kernel(float* gq, float* gk, float* gv, long long total) {
    long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (x < total) {
        gq[x] = 0.0f;
        gk[x] = 0.0f;
        gv[x] = 0.0f;
    }
}

__global__ void backward_rows(const float* q, const float* k, const float* v,
                              const float* p, const float* go,
                              float* gq, float* gk, float* gv,
                              int batch, int heads, int seq, int dim, float scale) {
    int row = blockIdx.x;
    int total_rows = batch * heads * seq;
    if (row >= total_rows) return;
    int tid = threadIdx.x;
    int bh = row / seq;
    int qi = row - bh * seq;
    long long qbase = ((long long)bh * seq + qi) * dim;
    long long pbase = ((long long)bh * seq + qi) * seq;
    __shared__ float red[8];
    float local = 0.0f;
    for (int j = tid; j < seq; j += blockDim.x) {
        float dp = 0.0f;
        long long vbase = ((long long)bh * seq + j) * dim;
        for (int d = 0; d < dim; ++d) dp = fmaf(go[qbase + d], v[vbase + d], dp);
        local += dp * p[pbase + j];
    }
    for (int off = 16; off > 0; off >>= 1) local += __shfl_down_sync(0xffffffffu, local, off);
    if ((tid & 31) == 0) red[tid >> 5] = local;
    __syncthreads();
    if (tid < 32) {
        float x = tid < (blockDim.x >> 5) ? red[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) x += __shfl_down_sync(0xffffffffu, x, off);
        if (tid == 0) red[0] = x;
    }
    __syncthreads();
    float dot = red[0];
    for (int d = tid; d < dim; d += blockDim.x) {
        float x = 0.0f;
        for (int j = 0; j < seq; ++j) {
            long long vbase = ((long long)bh * seq + j) * dim;
            float dp = 0.0f;
            for (int t = 0; t < dim; ++t) dp = fmaf(go[qbase + t], v[vbase + t], dp);
            float ds = p[pbase + j] * (dp - dot);
            long long kbase = ((long long)bh * seq + j) * dim;
            x = fmaf(ds, k[kbase + d], x);
            atomicAdd(&gk[kbase + d], ds * q[qbase + d] * scale);
            atomicAdd(&gv[vbase + d], p[pbase + j] * go[qbase + d]);
        }
        gq[qbase + d] = x * scale;
    }
}

extern "C" void launch_kernel(
    float* q,
    float* k,
    float* v,
    float* p,
    float* grad_o,
    float* grad_q,
    float* grad_k,
    float* grad_v,
    int batch,
    int heads,
    int seq,
    int head_dim,
    float scale
) {
    long long elems = (long long)batch * heads * seq * head_dim;
    int clear_blocks = (int)((elems + 255) / 256);
    clear_kernel<<<clear_blocks, 256>>>(grad_q, grad_k, grad_v, elems);
    int rows = batch * heads * seq;
    backward_rows<<<rows, 256>>>(q, k, v, p, grad_o, grad_q, grad_k, grad_v,
                                 batch, heads, seq, head_dim, scale);
}