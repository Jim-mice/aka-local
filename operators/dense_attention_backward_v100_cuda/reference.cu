// Dense attention backward reference kernel for Tesla V100 (sm_70)
// CUDA 11.8 compatible
//
// Backward pass for non-causal scaled dot-product attention.
// Given forward Q, K, V, saved softmax P, and upstream grad dO:
//
//   dV = P^T @ dO
//   dP = dO @ V^T
//   For each row i: dot_i = sum_j(dP[i,j] * P[i,j])
//                   dS[i,j] = P[i,j] * (dP[i,j] - dot_i)
//   dQ = dS @ K * scale
//   dK = dS^T @ Q * scale
//
// Tensor layouts:
//   Q, K, V, dO, dQ, dK, dV: [batch, heads, seq, head_dim] contiguous FP32
//   P:                       [batch, heads, seq, seq]      contiguous FP32
//
// Correctness-first reference: each thread handles one query row.
// Uses stack-based arrays. Supports seq up to 128.

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define MAX_SEQ 128

// AKA_CONTRACT: q, k, v, p, grad_o, grad_q, grad_k, grad_v, batch, heads, seq, head_dim, scale

extern "C" __global__ void dense_attention_backward_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ p,
    const float* __restrict__ grad_o,
    float* __restrict__ grad_q,
    float* __restrict__ grad_k,
    float* __restrict__ grad_v,
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
    size_t stride_qkv = (size_t)seq * head_dim;
    size_t stride_p   = (size_t)seq * seq;

    const float* q_bh = q + ((size_t)b * heads + h) * stride_qkv;
    const float* k_bh = k + ((size_t)b * heads + h) * stride_qkv;
    const float* v_bh = v + ((size_t)b * heads + h) * stride_qkv;
    const float* p_bh = p + ((size_t)b * heads + h) * stride_p;
    const float* go_bh = grad_o + ((size_t)b * heads + h) * stride_qkv;
    float* gq_bh = grad_q + ((size_t)b * heads + h) * stride_qkv;
    float* gk_bh = grad_k + ((size_t)b * heads + h) * stride_qkv;
    float* gv_bh = grad_v + ((size_t)b * heads + h) * stride_qkv;

    // ---- Step 1: dV for the rows assigned to this thread ----
    // dV[j,d] = sum_i P[i,j] * dO[i,d]
    for (int qi = tid; qi < seq; qi += blockDim.x) {
        for (int d = 0; d < head_dim; d++) {
            float acc = 0.0f;
            for (int i = 0; i < seq; i++) {
                acc += p_bh[(size_t)i * seq + qi] * go_bh[(size_t)i * head_dim + d];
            }
            gv_bh[(size_t)qi * head_dim + d] = acc;
        }
    }
    __syncthreads();

    // ---- Step 2: dP row for this thread ----
    // dP[i,j] = sum_d dO[i,d] * V[j,d]
    // Then softmax backward: dS[i,j] = P[i,j] * (dP[i,j] - dot_i)
    // where dot_i = sum_j dP[i,j] * P[i,j]
    //
    // Store dS[i,:] locally, then accumulate into dQ and dK

    for (int qi = tid; qi < seq; qi += blockDim.x) {
        // Compute dP row qi
        float dP[MAX_SEQ];
        float dot_i = 0.0f;

        for (int j = 0; j < seq; j++) {
            float acc = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                acc += go_bh[(size_t)qi * head_dim + d] * v_bh[(size_t)j * head_dim + d];
            }
            dP[j] = acc;
        }

        // dot_i = sum_j dP[j] * P[qi,j]
        for (int j = 0; j < seq; j++) {
            dot_i += dP[j] * p_bh[(size_t)qi * seq + j];
        }

        // dS[qi,j] = P[qi,j] * (dP[j] - dot_i)
        float dS[MAX_SEQ];
        for (int j = 0; j < seq; j++) {
            dS[j] = p_bh[(size_t)qi * seq + j] * (dP[j] - dot_i) * scale;
        }

        // ---- Step 3: dQ from dS row ----
        // dQ[qi,d] = sum_j dS[qi,j] * K[j,d]
        for (int d = 0; d < head_dim; d++) {
            float acc = 0.0f;
            for (int j = 0; j < seq; j++) {
                acc += dS[j] * k_bh[(size_t)j * head_dim + d];
            }
            gq_bh[(size_t)qi * head_dim + d] = acc;
        }

        // ---- Step 4: dK from dS column ----
        // dK[j,d] accumulates: dS^T[j,i] * Q[i,d] = dS[i,j] * Q[i,d]
        // Since dS[i,j] is the element at query i, key j,
        // dK[j,d] += dS[qi,j] * Q[qi,d]  (qi is i, j is j)
        for (int j = 0; j < seq; j++) {
            float weight = dS[j];  // dS[qi, j]
            for (int d = 0; d < head_dim; d++) {
                // atomicAdd not used; each row qi handles its own contribution
                // But dK needs accumulation across i! Each thread handles one qi,
                // so we need synchronization for dK accumulation.
                // Alternative: write dK contributions to a scratch buffer,
                // then reduce across threads in a separate kernel.
                //
                // For correctness-first reference, we use atomicAdd:
                atomicAdd(&gk_bh[(size_t)j * head_dim + d], weight * q_bh[(size_t)qi * head_dim + d]);
            }
        }
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
    int total_blocks = batch * heads;
    int threads = 128;

    // Zero grad_k output before kernel (atomicAdd needs zero init)
    cudaMemset(grad_k, 0, (size_t)batch * heads * seq * head_dim * sizeof(float));

    dense_attention_backward_kernel<<<total_blocks, threads, 0>>>(
        q, k, v, p, grad_o, grad_q, grad_k, grad_v,
        batch, heads, seq, head_dim, scale
    );
    cudaDeviceSynchronize();
}
