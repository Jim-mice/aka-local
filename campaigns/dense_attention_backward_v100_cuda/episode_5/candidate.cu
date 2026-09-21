#include <cuda_runtime.h>

// AKA_CONTRACT: q, k, v, p, grad_o, grad_q, grad_k, grad_v, batch, heads, seq, head_dim, scale

static __device__ __forceinline__ float warp_sum(float x) {
    for (int off = 16; off; off >>= 1) x += __shfl_down_sync(0xffffffff, x, off);
    return x;
}

__global__ void dense_dq_kernel(const float* __restrict__ k, const float* __restrict__ v,
                                const float* __restrict__ p, const float* __restrict__ go,
                                float* __restrict__ gq, int rows, int seq, int d, float scale) {
    int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (row >= rows) return;
    extern __shared__ float sm[];
    int bh = row / seq;
    const float* gorow = go + (size_t)row * d;
    const float* prow = p + (size_t)row * seq;
    for (int j = 0; j < seq; ++j) {
        const float* vj = v + (size_t)(bh * seq + j) * d;
        float part = 0.0f;
        for (int z = tid; z < d; z += blockDim.x) part += gorow[z] * vj[z];
        part = warp_sum(part);
        if (lane == 0) sm[warp] = part;
        __syncthreads();
        float dot = (tid < 8) ? sm[tid] : 0.0f;
        if (warp == 0) dot = warp_sum(dot);
        if (tid == 0) sm[0] = dot;
        __syncthreads();
        float ds = prow[j] * (dot - sm[0]);
        for (int x = tid; x < d; x += blockDim.x)
            gq[(size_t)row * d + x] += ds * k[(size_t)(bh * seq + j) * d + x] * scale;
        __syncthreads();
    }
}
__global__ void dense_dv_kernel(const float* __restrict__ p, const float* __restrict__ go,
                                float* __restrict__ gv, int total, int seq, int d) {
    int out = blockIdx.x, tid = threadIdx.x;
    int bh = out / seq, j = out - bh * seq;
    if (out >= total || tid >= d) return;
    float acc = 0.0f;
    for (int i = 0; i < seq; ++i)
        acc += p[(size_t)(bh * seq + i) * seq + j] * go[(size_t)(bh * seq + i) * d + tid];
    gv[(size_t)(bh * seq + j) * d + tid] = acc;
}

__global__ void dense_dk_kernel(const float* __restrict__ q, const float* __restrict__ v,
                                const float* __restrict__ p, const float* __restrict__ go,
                                float* __restrict__ gk, int total, int seq, int d, float scale) {
    int out = blockIdx.x, tid = threadIdx.x;
    int bh = out / seq, j = out - bh * seq;
    if (out >= total || tid >= d) return;
    float acc = 0.0f;
    for (int i = 0; i < seq; ++i) {
        const float* gorow = go + (size_t)(bh * seq + i) * d;
        float dpj = 0.0f;
        float mean = 0.0f;
        for (int z = 0; z < seq; ++z) {
            const float* vz = v + (size_t)(bh * seq + z) * d;
            float dp = 0.0f;
            for (int x = 0; x < d; ++x) dp += gorow[x] * vz[x];
            mean += dp * p[(size_t)(bh * seq + i) * seq + z];
            if (z == j) dpj = dp;
        }
        float ds = p[(size_t)(bh * seq + i) * seq + j] * (dpj - mean);
        acc += ds * q[(size_t)(bh * seq + i) * d + tid] * scale;
    }
    gk[(size_t)(bh * seq + j) * d + tid] = acc;
}

extern "C" void launch_kernel(
    float* q, float* k, float* v, float* p, float* grad_o,
    float* grad_q, float* grad_k, float* grad_v,
    int batch, int heads, int seq, int head_dim, float scale) {
    int total = batch * heads * seq;
    dim3 block(256);
    dense_dq_kernel<<<total, block, 8 * sizeof(float)>>>(k, v, p, grad_o, grad_q, total, seq, head_dim, scale);
    dense_dv_kernel<<<total, block>>>(p, grad_o, grad_v, total, seq, head_dim);
    dense_dk_kernel<<<total, block>>>(q, v, p, grad_o, grad_k, total, seq, head_dim, scale);
}


