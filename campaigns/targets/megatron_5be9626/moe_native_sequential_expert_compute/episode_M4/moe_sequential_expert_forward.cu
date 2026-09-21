#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// AKA_TARGET_CONTRACT: a1bb8ee0af6c250663e0935ae3deb07104975fc5904c56bb5b2f0d4e8032d1fe
extern "C" void moe_sequential_expert_forward_fp16_stream(const half* tokens, const int64_t* tokens_per_expert, const half* probs, const half* fc1_weights, const half* fc2_weights, half* output, int64_t token_count, int64_t hidden, int64_t intermediate, int64_t experts, cudaStream_t stream);

namespace {

__device__ __forceinline__ float gated_silu(float x, float gate) {
    return (x / (1.0f + expf(-x))) * gate;
}

__global__ void moe_sequential_expert_kernel(
    const half* __restrict__ tokens,
    const int64_t* __restrict__ tokens_per_expert,
    const half* __restrict__ probs,
    const half* __restrict__ fc1_weights,
    const half* __restrict__ fc2_weights,
    half* __restrict__ output,
    int64_t token_count,
    int64_t hidden,
    int64_t intermediate,
    int64_t experts) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t col = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (row >= token_count || col >= hidden) return;

    int64_t expert = 0;
    int64_t expert_begin = 0;
    for (; expert < experts; ++expert) {
        const int64_t n = tokens_per_expert[expert];
        if (row < expert_begin + n) break;
        expert_begin += n;
    }
    if (expert >= experts) return;

    const half* token_row = tokens + row * hidden;
    const half* fc1 = fc1_weights + expert * (2 * intermediate) * hidden;
    const half* fc2 = fc2_weights + expert * hidden * intermediate;

    float acc = 0.0f;
    for (int64_t k = 0; k < intermediate; ++k) {
        float gate = 0.0f;
        float value = 0.0f;
        for (int64_t h = 0; h < hidden; ++h) {
            const float x = __half2float(token_row[h]);
            gate += x * __half2float(fc1[k * hidden + h]);
            value += x * __half2float(fc1[(intermediate + k) * hidden + h]);
        }
        acc += gated_silu(gate, value) * __half2float(fc2[col * intermediate + k]);
    }
    output[row * hidden + col] = __float2half(acc * __half2float(probs[row]));
}

}  // namespace

extern "C" void moe_sequential_expert_forward_fp16_stream(
    const half* tokens,
    const int64_t* tokens_per_expert,
    const half* probs,
    const half* fc1_weights,
    const half* fc2_weights,
    half* output,
    int64_t token_count,
    int64_t hidden,
    int64_t intermediate,
    int64_t experts,
    cudaStream_t stream) {
    if (token_count <= 0 || hidden <= 0 || intermediate <= 0 || experts <= 0) return;
    const dim3 block(8, 8, 1);
    const dim3 grid(
        static_cast<unsigned>((token_count + block.x - 1) / block.x),
        static_cast<unsigned>((hidden + block.y - 1) / block.y),
        1);
    moe_sequential_expert_kernel<<<grid, block, 0, stream>>>(
        tokens, tokens_per_expert, probs, fc1_weights, fc2_weights, output,
        token_count, hidden, intermediate, experts);
}
