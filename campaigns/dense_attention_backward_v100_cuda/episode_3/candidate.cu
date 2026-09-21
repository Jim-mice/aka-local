#include <cuda_runtime.h>
#include <math.h>

// AKA_CONTRACT: q, k, v, p, grad_o, grad_q, grad_k, grad_v, batch, heads, seq, head_dim, scale

static __device__ __forceinline__ int tensor_base(int bh, int seq, int dim) {
    return bh * seq * dim;
}

static __device__ __forceinline__ float dot_row_value(const float* go, const float* v, int i, int j, int dim) {
    float x = 0.0f;
    int a = i * dim;
    int b = j * dim;
    for (int d = 0; d < dim; ++d) x = fmaf(go[a + d], v[b + d], x);
    return x;
}

__global__ static void backward_dv_kernel(const float* p, const float* go, float* gv,
                                           int total_bh, int seq, int dim) {
    int bh = blockIdx.x;
    int j = blockIdx.y;
    int d = threadIdx.x;
    if (bh >= total_bh || j >= seq || d >= dim) return;
    int base = tensor_base(bh, seq, dim);
    float sum = 0.0f;
    for (int i = 0; i < seq; ++i) sum = fmaf(p[i * seq + j], go[i * dim + d], sum);
    gv[base + j * dim + d] = sum;
}

__global__ static void backward_dq_kernel(const float* k, const float* v, const float* p,
                                           const float* go, const float* q, float* gq,
                                           int total_bh, int seq, int dim, float scale) {
    int bh = blockIdx.x;
    int i = blockIdx.y;
    if (bh >= total_bh || i >= seq) return;
    int lane = threadIdx.x;
    int base = tensor_base(bh, seq, dim);
    float partial = 0.0f;
    for (int j = lane; j < seq; j += blockDim.x) {
        float dp = dot_row_value(go + base, v + base, i, j, dim);
        partial = fmaf(dp, p[i * seq + j], partial);
    }
    for (int off = 16; off > 0; off >>= 1) partial += __shfl_down_sync(0xffffffff, partial, off);
    __shared__ float warp_sum[8];
    if ((lane & 31) == 0) warp_sum[lane >> 5] = partial;
    __syncthreads();
    float dot = (lane < 8) ? warp_sum[lane] : 0.0f;
    if (lane < 32) for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(0xffffffff, dot, off);
    __shared__ float row_dot;
    if (lane == 0) row_dot = dot;
    __syncthreads();
    for (int d = lane; d < dim; d += blockDim.x) {
        float out = 0.0f;
        for (int j = 0; j < seq; ++j) {
            float dp = dot_row_value(go + base, v + base, i, j, dim);
            float ds = p[i * seq + j] * (dp - row_dot);
            out = fmaf(ds, k[j * dim + d], out);
        }
        gq[base + i * dim + d] = out * scale;
    }
    (void)q;
}

__global__ static void backward_dk_kernel(const float* k, const float* v, const float* p,
                                           const float* go, float* gk,
                                           int total_bh, int seq, int dim, float scale) {
    int bh = blockIdx.x;
    int j = blockIdx.y;
    int d = threadIdx.x;
    if (bh >= total_bh || j >= seq || d >= dim) return;
    int base = tensor_base(bh, seq, dim);
    float sum = 0.0f;
    for (int i = 0; i < seq; ++i) {
        float dot = 0.0f;
        for (int t = 0; t < seq; ++t) {
            float dp = dot_row_value(go + base, v + base, i, t, dim);
            dot = fmaf(p[i * seq + t], dp, dot);
        }
        float ds = p[i * seq + j] * (dot_row_value(go + base, v + base, i, j, dim) - dot);
        sum = fmaf(ds, (go - go) [0] + 0.0f, sum);
        // The query value is loaded through the equivalent input pointer below.
    }
    (void)k;
    (void)sum;
}

__global__ static void backward_dk_real_kernel(const float* q, const float* v, const float* p,
                                                const float* go, float* gk,
                                                int total_bh, int seq, int dim, float scale) {
    int bh = blockIdx.x;
    int j = blockIdx.y;
    int d = threadIdx.x;
    if (bh >= total_bh || j >= seq || d >= dim) return;
    int base = tensor_base(bh, seq, dim);
    float sum = 0.0f;
    for (int i = 0; i < seq; ++i) {
        float dot = 0.0f;
        for (int t = 0; t < seq; ++t) {
            float dp = dot_row_value(go + base, v + base, i, t, dim);
            dot = fmaf(p[i * seq + t], dp, dot);
        }
        float dpj = dot_row_value(go + base, v + base, i, j, dim);
        float ds = p[i * seq + j] * (dpj - dot);
        sum = fmaf(ds, q[i * dim + d], sum);
    }
    gk[base + j * dim + d] = sum * scale;
}

extern "C" void launch_kernel(
    float* q, float* k, float* v, float* p, float* grad_o,
    float* grad_q, float* grad_k, float* grad_v,
    int batch, int heads, int seq, int head_dim, float scale) {
    int total_bh = batch * heads;
    dim3 rows(total_bh, seq, 1);
    dim3 keys(total_bh, seq, 1);
    backward_dv_kernel<<<keys, 256>>>(p, grad_o, grad_v, total_bh, seq, head_dim);
    backward_dq_kernel<<<rows, 256>>>(k, v, p, grad_o, q, grad_q, total_bh, seq, head_dim, scale);
    backward_dk_real_kernel<<<keys, 256>>>(q, v, p, grad_o, grad_k, total_bh, seq, head_dim, scale);
}