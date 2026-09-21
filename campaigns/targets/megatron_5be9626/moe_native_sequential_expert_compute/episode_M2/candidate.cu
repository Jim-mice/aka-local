#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// MEGATRON_NATIVE_SEQUENTIALMLP_EXPERT_COMPUTE_V1

namespace {

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void sequential_expert_kernel(
    const half* __restrict__ tokens,
    const int64_t* __restrict__ tokens_per_expert,
    const half* __restrict__ probs,
    const half* __restrict__ fc1_weights,
    const half* __restrict__ fc2_weights,
    half* __restrict__ output,
    int64_t total_tokens,
    int64_t hidden,
    int64_t intermediate,
    int64_t experts) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int h = static_cast<int>(threadIdx.x);
    if (row >= total_tokens || h >= hidden) return;

    int64_t remaining = row;
    int64_t expert = 0;
    int64_t expert_row = row;
    int64_t expert_start = 0;
    for (; expert < experts; ++expert) {
        const int64_t count = tokens_per_expert[expert];
        if (remaining < count) {
            expert_row = remaining;
            break;
        }
        remaining -= count;
        expert_start += count;
    }
    if (expert >= experts) return;

    const int64_t token_index = expert_start + expert_row;
    const half* token = tokens + token_index * hidden;
    const half* fc1 = fc1_weights + expert * (2 * intermediate) * hidden;
    const half* fc2 = fc2_weights + expert * hidden * intermediate;

    float result = 0.0f;
    for (int64_t j = 0; j < intermediate; ++j) {
        float gate = 0.0f;
        float value = 0.0f;
        for (int64_t k = 0; k < hidden; ++k) {
            const float x = __half2float(token[k]);
            gate += x * __half2float(fc1[j * hidden + k]);
            value += x * __half2float(fc1[(intermediate + j) * hidden + k]);
        }
        const float activated = silu(gate) * value;
        result += activated * __half2float(fc2[h * intermediate + j]);
    }

    output[token_index * hidden + h] = __float2half_rn(result * __half2float(probs[token_index]));
}

}  // namespace

extern "C" void moe_sequential_expert_forward_fp16_stream(
    const half* tokens,
    const int64_t* tokens_per_expert,
    const half* probs,
    const half* fc1_weights,
    const half* fc2_weights,
    half* output,
    int64_t tokens_count,
    int64_t hidden,
    int64_t intermediate,
    int64_t experts,
    cudaStream_t stream) {
    if (tokens_count <= 0 || hidden <= 0 || intermediate <= 0 || experts <= 0) return;
    const unsigned int threads = static_cast<unsigned int>(hidden);
    sequential_expert_kernel<<<static_cast<unsigned int>(tokens_count), threads, 0, stream>>>(
        tokens, tokens_per_expert, probs, fc1_weights, fc2_weights, output,
        tokens_count, hidden, intermediate, experts);
}
