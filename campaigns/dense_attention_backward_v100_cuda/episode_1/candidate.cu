#include <cuda_runtime.h>
#include <math.h>

// AKA_CONTRACT: q, k, v, p, grad_o, grad_q, grad_k, grad_v, batch, heads, seq, head_dim, scale

__device__ __forceinline__ float warp_sum(float x) {
    for (int off = 16; off > 0; off >>= 1) x += __shfl_down_sync(0xffffffff, x, off);
    return x;
}

__device__ __forceinline__ float block_sum(float x) {
    __shared__ float smem[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    x = warp_sum(x);
    if (lane == 0) smem[warp] = x;
    __syncthreads();
    x = (threadIdx.x < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
    if (warp == 0) x = warp_sum(x);
    return x;
}

__global__ void d_q_kernel(const float* __restrict__ q, const float* __restrict__ k,
                           const float* __restrict__ p, const float* __restrict__ go,
                           float* __restrict__ gq, int bh, int s, int d, float scale) {
    int row = blockIdx.y, bhead = blockIdx.x, t = threadIdx.x;
    if (bhead >= bh || row >= s) return;
    size_t base = (size_t)bhead * s * d;
    size_t pbase = (size_t)bhead * s * s + (size_t)row * s;
    float dot = 0.0f;
    for (int j = t; j < s; j += blockDim.x) {
        float dp = 0.0f;
        for (int x = 0; x < d; ++x) dp += go[base + (size_t)row*d + x] * q[base + (size_t)j*d + x];
        dot += dp * p[pbase + j];
    }
    dot = block_sum(dot);
    __shared__ float dot_shared;
    if (t == 0) dot_shared = dot;
    __syncthreads();
    for (int x = t; x < d; x += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < s; ++j) {
            float dp = 0.0f;
            for (int z = 0; z < d; ++z) dp += go[base + (size_t)row*d + z] * q[base + (size_t)j*d + z];
            float ds = p[pbase + j] * (dp - dot_shared);
            acc += ds * k[base + (size_t)j*d + x];
        }
        gq[base + (size_t)row*d + x] = acc * scale;
    }
}

__global__ void d_k_kernel(const float* __restrict__ q, const float* __restrict__ k,
                           const float* __restrict__ p, const float* __restrict__ go,
                           float* __restrict__ gk, int bh, int s, int d, float scale) {
    int key = blockIdx.y, bhead = blockIdx.x, t = threadIdx.x;
    if (bhead >= bh || key >= s) return;
    size_t base = (size_t)bhead * s * d;
    for (int row = 0; row < s; ++row) {
        size_t pbase = (size_t)bhead * s * s + (size_t)row * s;
        float dot = 0.0f;
        for (int j = t; j < s; j += blockDim.x) {
            float dp = 0.0f;
            for (int z = 0; z < d; ++z) dp += go[base + (size_t)row*d + z] * k[base + (size_t)j*d + z];
            dot += dp * p[pbase + j];
        }
        dot = block_sum(dot);
        __shared__ float dot_shared;
        if (t == 0) dot_shared = dot;
        __syncthreads();
        if (t < d) {
            float dp = 0.0f;
            for (int z = 0; z < d; ++z) dp += go[base + (size_t)row*d + z] * k[base + (size_t)key*d + z];
            float ds = p[pbase + key] * (dp - dot_shared);
            gk[base + (size_t)key*d + t] += ds * q[base + (size_t)row*d + t] * scale;
        }
        __syncthreads();
    }
}

__global__ void d_v_kernel(const float* __restrict__ p, const float* __restrict__ go,
                           float* __restrict__ gv, int bh, int s, int d) {
    int key = blockIdx.y, bhead = blockIdx.x, t = threadIdx.x;
    if (bhead >= bh || key >= s || t >= d) return;
    size_t base = (size_t)bhead * s * d;
    float acc = 0.0f;
    for (int row = 0; row < s; ++row)
        acc += p[(size_t)bhead*s*s + (size_t)row*s + key] * go[base + (size_t)row*d + t];
    gv[base + (size_t)key*d + t] = acc;
}

extern "C" void launch_kernel(
    float* q, float* k, float* v, float* p, float* grad_o,
    float* grad_q, float* grad_k, float* grad_v,
    int batch, int heads, int seq, int head_dim, float scale) {
    int bh = batch * heads;
    dim3 grid(bh, seq, 1);
    dim3 block(256, 1, 1);
    cudaMemset(grad_k, 0, (size_t)bh * seq * head_dim * sizeof(float));
    d_q_kernel<<<grid, block>>>(q, k, p, grad_o, grad_q, bh, seq, head_dim, scale);
    d_k_kernel<<<grid, block>>>(q, k, p, grad_o, grad_k, bh, seq, head_dim, scale);
    d_v_kernel<<<grid, block>>>(p, grad_o, grad_v, bh, seq, head_dim);
}