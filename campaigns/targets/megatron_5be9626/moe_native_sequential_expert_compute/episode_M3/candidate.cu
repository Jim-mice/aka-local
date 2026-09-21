// AKA_TARGET_CONTRACT: a1bb8ee0af6c250663e0935ae3deb07104975fc5904c56bb5b2f0d4e8032d1fe
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + __expf(-x));
}

__global__ void sequential_expert_kernel(
    const half* tokens, const int64_t* tokens_per_expert, const half* probs,
    const half* fc1_weights, const half* fc2_weights, half* output,
    int64_t token_count, int64_t hidden, int64_t intermediate,
    int64_t experts) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = token_count * hidden;
  if (row >= total) return;

  const int64_t local_token = row / hidden;
  const int64_t h = row - local_token * hidden;
  int64_t remaining = local_token;
  int64_t expert = 0;
  for (; expert < experts; ++expert) {
    const int64_t count = tokens_per_expert[expert];
    if (remaining < count) break;
    remaining -= count;
  }
  if (expert >= experts) {
    output[row] = __float2half(0.0f);
    return;
  }

  const int64_t token_base = remaining * hidden;
  const int64_t fc1_base = expert * intermediate * hidden;
  const int64_t fc2_base = expert * hidden * intermediate;
  float value = 0.0f;
  for (int64_t j = 0; j < intermediate; ++j) {
    float a = 0.0f;
    float b = 0.0f;
    for (int64_t k = 0; k < hidden; ++k) {
      const float x = __half2float(tokens[token_base + k]);
      a += x * __half2float(fc1_weights[fc1_base + j * hidden + k]);
      b += x * __half2float(fc1_weights[fc1_base + (intermediate + j) * hidden + k]);
    }
    value += __half2float(fc2_weights[fc2_base + h * intermediate + j]) * silu(a) * b;
  }
  output[row] = __float2half(value * __half2float(probs[local_token]));
}

}  // namespace

extern "C" void moe_sequential_expert_forward_fp16_stream(const half* tokens, const int64_t* tokens_per_expert, const half* probs, const half* fc1_weights, const half* fc2_weights, half* output, int64_t token_count, int64_t hidden, int64_t intermediate, int64_t experts, cudaStream_t stream);

extern "C" void moe_sequential_expert_forward_fp16_stream(const half* tokens, const int64_t* tokens_per_expert, const half* probs, const half* fc1_weights, const half* fc2_weights, half* output, int64_t token_count, int64_t hidden, int64_t intermediate, int64_t experts, cudaStream_t stream) {
  if (token_count <= 0 || hidden <= 0 || intermediate <= 0 || experts <= 0) return;
  const int threads = 128;
  const int64_t elements = token_count * hidden;
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  sequential_expert_kernel<<<blocks, threads, 0, stream>>>(
      tokens, tokens_per_expert, probs, fc1_weights, fc2_weights, output,
      token_count, hidden, intermediate, experts);
}
