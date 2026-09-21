#include <cuda_runtime.h>
#include <math.h>

// AKA_CONTRACT: q, k, v, p, grad_o, grad_q, grad_k, grad_v, batch, heads, seq, head_dim, scale

static __device__ __forceinline__ float warp_sum(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) x += __shfl_down_sync(0xffffffffu, x, offset);
    return x;
}

static __device__ __forceinline__ float block_sum(float x, float* scratch) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    x = warp_sum(x);
    if (lane == 0) scratch[warp] = x;
    __syncthreads();
    x = (threadIdx.x < (blockDim.x >> 5)) ? scratch[lane] : 0.0f;
    if (warp == 0) x = warp_sum(x);
    if (threadIdx.x == 0) scratch[0] = x;
    __syncthreads();
    return scratch[0];
}

__global__ void dense_backward_rows(const float* q, const float* k, const float* v, const float* p,
                                    const float* grad_o, float* grad_q, float* grad_k, float* grad_v,
                                    int batch, int heads, int seq, int head_dim, float scale) {
    int row = blockIdx.x;
    int total_rows = batch * heads * seq;
    if (row >= total_rows) return;
    int bh = row / seq;
    int qi = row - bh * seq;
    long long qrow = ((long long)bh * seq + qi) * head_dim;
    long long prow = ((long long)bh * seq + qi) * seq;
    long long hbase = (long long)bh * seq * head_dim;
    __shared__ float scratch[8];
    float dot = 0.0f;
    for (int j = threadIdx.x; j < seq; j += blockDim.x) {
        long long vrow = hbase + (long long)j * head_dim;
        float dp = 0.0f;
        for (int d = 0; d < head_dim; ++d) dp = fmaf(grad_o[qrow + d], v[vrow + d], dp);
        dot += dp * p[prow + j];
    }
    dot = block_sum(dot, scratch);
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float out = 0.0f;
        for (int j = 0; j < seq; ++j) {
            long long krow = hbase + (long long)j * head_dim;
            float dp = 0.0f;
            for (int x = 0; x < head_dim; ++x) dp = fmaf(grad_o[qrow + x], v[krow + x], dp);
            float ds = p[prow + j] * (dp - dot);
            out = fmaf(ds, k[krow + d], out);
        }
        grad_q[qrow + d] = out * scale;
    }
}

__global__ void dense_dkv(const float* q, const float* k, const float* v, const float* p,
                          const float* grad_o, float* grad_k, float* grad_v,
                          int batch, int heads, int seq, int head_dim, float scale) {
    int row = blockIdx.x;
    int total_rows = batch * heads * seq;
    if (row >= total_rows) return;
    int bh = row / seq;
    int kj = row - bh * seq;
    long long hbase = (long long)bh * seq * head_dim;
    __shared__ float scratch[8];
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float dk = 0.0f, dv = 0.0f;
        for (int i = 0; i < seq; ++i) {
            long long qrow = hbase + (long long)i * head_dim;
            long long prow = ((long long)bh * seq + i) * seq;
            float dp = 0.0f;
            for (int x = 0; x < head_dim; ++x) dp = fmaf(grad_o[qrow + x], v[hbase + (long long)kj * head_dim + x], dp);
            float dot = 0.0f;
            for (int j = 0; j < seq; ++j) {
                float dpl = 0.0f;
                long long vrow = hbase + (long long)j * head_dim;
                for (int x = 0; x < head_dim; ++x) dpl = fmaf(grad_o[qrow + x], v[vrow + x], dpl);
                dot += dpl * p[prow + j];
            }
            float ds = p[prow + kj] * (dp - dot);
            dk = fmaf(ds, q[qrow + d], dk);
            dv = fmaf(p[prow + kj], grad_o[qrow + d], dv);
        }
        grad_k[hbase + (long long)kj * head_dim + d] = dk * scale;
        grad_v[hbase + (long long)kj * head_dim + d] = dv;
    }
    (void)k;
}

extern "C" void launch_kernel(
    float* q, float* k, float* v, float* p, float* grad_o,
    float* grad_q, float* grad_k, float* grad_v,
    int batch, int heads, int seq, int head_dim, float scale) {
    int rows = batch * heads * seq;
    dim3 block(256);
    dim3 grid(rows);
    dense_backward_rows<<<grid, block>>>(q, k, v, p, grad_o, grad_q, grad_k, grad_v,
                                         batch, heads, seq, head_dim, scale);
    dense_dkv<<<grid, block>>>(q, k, v, p, grad_o, grad_k, grad_v,
                               batch, heads, seq, head_dim, scale);
}