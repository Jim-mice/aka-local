// Dense attention reference kernel for Tesla V100 (sm_70)
// CUDA 11.8 compatible
// Scaled dot-product attention: O = softmax(Q @ K^T * scale) @ V
// Tensor layout: [batch, heads, seq, head_dim] contiguous row-major
//
// Correctness-first reference: each thread fully computes its own query row.
// No shared memory used — each thread has its own scores[] on the stack.
// Supports seq up to 512 (max stack usage: seq * 4 bytes per thread).

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define MAX_SEQ 512

// AKA_CONTRACT: q, k, v, o, batch, heads, seq, head_dim, scale

extern "C" __global__ void attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ o,
    int batch,
    int heads,
    int seq,
    int head_dim,
    float scale
) {
    int bh = blockIdx.x;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch) return;

    int tid = threadIdx.x;
    size_t stride = (size_t)seq * head_dim;

    const float* q_bh = q + ((size_t)b * heads + h) * stride;
    const float* k_bh = k + ((size_t)b * heads + h) * stride;
    const float* v_bh = v + ((size_t)b * heads + h) * stride;
    float* o_bh = o + ((size_t)b * heads + h) * stride;

    // Per-thread local scores array (on stack)
    float scores[MAX_SEQ];

    // Each thread handles its own query position(s)
    for (int qi = tid; qi < seq; qi += blockDim.x) {
        const float* q_row = q_bh + (size_t)qi * head_dim;

        // Compute all scores, track max
        float my_max = -FLT_MAX;
        for (int kj = 0; kj < seq; kj++) {
            const float* k_row = k_bh + (size_t)kj * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q_row[d] * k_row[d];
            }
            dot *= scale;
            scores[kj] = dot;
            if (dot > my_max) my_max = dot;
        }

        // Stable softmax: compute exp sum
        float exp_sum = 0.0f;
        for (int kj = 0; kj < seq; kj++) {
            exp_sum += expf(scores[kj] - my_max);
        }
        float inv_sum = 1.0f / exp_sum;

        // Weighted V accumulation
        for (int d = 0; d < head_dim; d++) {
            float acc = 0.0f;
            for (int kj = 0; kj < seq; kj++) {
                float weight = expf(scores[kj] - my_max) * inv_sum;
                acc += weight * v_bh[(size_t)kj * head_dim + d];
            }
            o_bh[(size_t)qi * head_dim + d] = acc;
        }
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
    int total_blocks = batch * heads;
    int threads = 128;
    attention_kernel<<<total_blocks, threads, 0>>>(
        q, k, v, o, batch, heads, seq, head_dim, scale
    );
    cudaDeviceSynchronize();
}
