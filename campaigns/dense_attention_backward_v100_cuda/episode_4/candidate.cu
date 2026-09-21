#include <cuda_runtime.h>
#include <math.h>

// AKA_CONTRACT: q, k, v, p, grad_o, grad_q, grad_k, grad_v, batch, heads, seq, head_dim, scale

static __device__ __forceinline__ long long ah_offset(int bh, int seq, int dim, int i, int d) {
    return ((long long)bh * seq + i) * dim + d;
}

static __device__ __forceinline__ long long p_offset(int bh, int seq, int i, int j) {
    return ((long long)bh * seq + i) * seq + j;
}

__global__ void dense_backward_q_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ p,
    const float* __restrict__ grad_o,
    float* __restrict__ grad_q,
    int batch_heads, int seq, int dim, float scale) {
    long long linear = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)batch_heads * seq * dim;
    if (linear >= total) return;
    int d = (int)(linear % dim);
    long long t = linear / dim;
    int i = (int)(t % seq);
    int bh = (int)(t / seq);

    float dot = 0.0f;
    for (int j = 0; j < seq; ++j) {
        float dp = 0.0f;
        for (int x = 0; x < dim; ++x) {
            dp = fmaf(grad_o[ah_offset(bh, seq, dim, i, x)],
                      v[ah_offset(bh, seq, dim, j, x)], dp);
        }
        dot = fmaf(dp, p[p_offset(bh, seq, i, j)], dot);
    }

    float out = 0.0f;
    float qi = q[ah_offset(bh, seq, dim, i, d)];
    (void)qi;
    for (int j = 0; j < seq; ++j) {
        float dp = 0.0f;
        for (int x = 0; x < dim; ++x) {
            dp = fmaf(grad_o[ah_offset(bh, seq, dim, i, x)],
                      v[ah_offset(bh, seq, dim, j, x)], dp);
        }
        float ds = p[p_offset(bh, seq, i, j)] * (dp - dot);
        out = fmaf(ds, k[ah_offset(bh, seq, dim, j, d)], out);
    }
    grad_q[ah_offset(bh, seq, dim, i, d)] = out * scale;
}

__global__ void dense_backward_k_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ p,
    const float* __restrict__ grad_o,
    float* __restrict__ grad_k,
    int batch_heads, int seq, int dim, float scale) {
    long long linear = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)batch_heads * seq * dim;
    if (linear >= total) return;
    int d = (int)(linear % dim);
    long long t = linear / dim;
    int j = (int)(t % seq);
    int bh = (int)(t / seq);
    float out = 0.0f;
    for (int i = 0; i < seq; ++i) {
        float dot = 0.0f;
        for (int y = 0; y < seq; ++y) {
            float dp = 0.0f;
            for (int x = 0; x < dim; ++x) {
                dp = fmaf(grad_o[ah_offset(bh, seq, dim, i, x)],
                          v[ah_offset(bh, seq, dim, y, x)], dp);
            }
            dot = fmaf(dp, p[p_offset(bh, seq, i, y)], dot);
        }
        float dpj = 0.0f;
        for (int x = 0; x < dim; ++x) {
            dpj = fmaf(grad_o[ah_offset(bh, seq, dim, i, x)],
                       v[ah_offset(bh, seq, dim, j, x)], dpj);
        }
        float ds = p[p_offset(bh, seq, i, j)] * (dpj - dot);
        out = fmaf(ds, q[ah_offset(bh, seq, dim, i, d)], out);
    }
    grad_k[ah_offset(bh, seq, dim, j, d)] = out * scale;
}

__global__ void dense_backward_v_kernel(
    const float* __restrict__ p,
    const float* __restrict__ grad_o,
    float* __restrict__ grad_v,
    int batch_heads, int seq, int dim) {
    long long linear = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)batch_heads * seq * dim;
    if (linear >= total) return;
    int d = (int)(linear % dim);
    long long t = linear / dim;
    int j = (int)(t % seq);
    int bh = (int)(t / seq);
    float out = 0.0f;
    for (int i = 0; i < seq; ++i) {
        out = fmaf(p[p_offset(bh, seq, i, j)],
                   grad_o[ah_offset(bh, seq, dim, i, d)], out);
    }
    grad_v[ah_offset(bh, seq, dim, j, d)] = out;
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
    float scale) {
    int bh = batch * heads;
    long long total = (long long)bh * seq * head_dim;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    dense_backward_q_kernel<<<blocks, threads>>>(q, k, v, p, grad_o, grad_q, bh, seq, head_dim, scale);
    dense_backward_k_kernel<<<blocks, threads>>>(q, k, v, p, grad_o, grad_k, bh, seq, head_dim, scale);
    dense_backward_v_kernel<<<blocks, threads>>>(p, grad_o, grad_v, bh, seq, head_dim);
}