#include <cuda_runtime.h>
#include <math.h>

// AKA_CONTRACT: q, k, v, p, grad_o, grad_q, grad_k, grad_v, batch, heads, seq, head_dim, scale

__global__ void clear_kernel(float* __restrict__ x, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) x[i] = 0.0f;
}

__device__ __forceinline__ float dot_row(const float* __restrict__ a,
                                          const float* __restrict__ b,
                                          int d, int head_dim) {
    float s = 0.0f;
    for (int x = d; x < head_dim; x += 32) s = fmaf(a[x], b[x], s);
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffff, s, off);
    return __shfl_sync(0xffffffff, s, 0);
}

__global__ void backward_rows(const float* __restrict__ q,
                              const float* __restrict__ k,
                              const float* __restrict__ v,
                              const float* __restrict__ p,
                              const float* __restrict__ grad_o,
                              float* __restrict__ grad_q,
                              float* __restrict__ grad_k,
                              float* __restrict__ grad_v,
                              int heads, int seq, int head_dim, float scale) {
    int row = (int)blockIdx.x;
    int total_rows = gridDim.x;
    int tid = threadIdx.x;
    if (row >= total_rows) return;

    int d = tid;
    int bh = row / seq;
    int qi = row - bh * seq;
    long long base = (long long)bh * seq * head_dim;
    const float* qrow = q + base + (long long)qi * head_dim;
    const float* gorow = grad_o + base + (long long)qi * head_dim;
    float* gqrow = grad_q + base + (long long)qi * head_dim;
    float* gv = grad_v + base;
    float* gk = grad_k + base;
    const float* prow = p + (long long)bh * seq * seq + (long long)qi * seq;

    if (d < head_dim) {
        float qacc = 0.0f;
        float vacc = 0.0f;
        for (int j = 0; j < seq; ++j) {
            const float* vrow = v + base + (long long)j * head_dim;
            const float* krow = k + base + (long long)j * head_dim;
            float prob = prow[j];
            float dp = 0.0f;
            for (int x = 0; x < head_dim; ++x) dp = fmaf(gorow[x], vrow[x], dp);
            float dot = 0.0f;
            for (int x = 0; x < head_dim; ++x) dot = fmaf(dp * prob, 0.0f, dot);
            // The expression above intentionally does not approximate the softmax dot;
            // compute the exact row statistic below in a compact scalar pass.
            dot = 0.0f;
            for (int t = 0; t < seq; ++t) {
                const float* vt = v + base + (long long)t * head_dim;
                float dpt = 0.0f;
                for (int x = 0; x < head_dim; ++x) dpt = fmaf(gorow[x], vt[x], dpt);
                dot = fmaf(dpt, prow[t], dot);
            }
            float ds = prob * (dp - dot);
            qacc = fmaf(ds, krow[d], qacc);
            vacc = fmaf(prob, gorow[d], vacc);
            atomicAdd(gk + (long long)j * head_dim + d, ds * qrow[d] * scale);
        }
        gqrow[d] = qacc * scale;
        gv[(long long)qi * head_dim + d] = vacc;
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
    int rows = batch * heads * seq;
    int clear_threads = 256;
    long long elems = (long long)batch * heads * seq * head_dim;
    int clear_blocks = (int)((elems + clear_threads - 1) / clear_threads);
    if (clear_blocks > 65535) clear_blocks = 65535;
    clear_kernel<<<clear_blocks, clear_threads>>>(grad_k, elems);
    backward_rows<<<rows, 256>>>(q, k, v, p, grad_o, grad_q, grad_k, grad_v,
                                 heads, seq, head_dim, scale);
}