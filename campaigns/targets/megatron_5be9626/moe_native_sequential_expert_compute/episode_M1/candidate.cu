#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// SEMANTIC_ABI: Megatron Native SequentialMLP Expert Compute.
// Inputs: tokens [T,H], tokens_per_expert [E], probs [T],
//         fc1 [E,2I,H], fc2 [E,H,I]. Output: [T,H].
// Layouts are row-major and tokens are already contiguous by local expert.
// The public entry point is stream-ordered and does not synchronize.

namespace {
constexpr int kExperts = 4;
constexpr int kHidden = 64;
constexpr int kIntermediate = 128;
constexpr int kTokens = 16;

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}

__global__ void sequential_expert_kernel(const half* __restrict__ tokens,
                                          const int64_t* __restrict__ counts,
                                          const half* __restrict__ probs,
                                          const half* __restrict__ fc1,
                                          const half* __restrict__ fc2,
                                          half* __restrict__ output) {
  const int expert = blockIdx.x;
  const int row_in_expert = blockIdx.y;
  const int h = threadIdx.x;
  if (expert >= kExperts || row_in_expert >= kTokens || h >= kHidden) return;

  int64_t base = 0;
  for (int e = 0; e < expert; ++e) base += counts[e];
  const int64_t n = counts[expert];
  if (row_in_expert >= n) return;
  const int token = static_cast<int>(base) + row_in_expert;

  const half* x = tokens + static_cast<int64_t>(token) * kHidden;
  const half* w1 = fc1 + static_cast<int64_t>(expert) * (2 * kIntermediate) * kHidden;
  const half* w2 = fc2 + static_cast<int64_t>(expert) * kHidden * kIntermediate;

  float acc = 0.0f;
  for (int j = 0; j < kIntermediate; ++j) {
    float up = 0.0f;
    float gate = 0.0f;
    for (int i = 0; i < kHidden; ++i) {
      const float xv = __half2float(x[i]);
      up += xv * __half2float(w1[j * kHidden + i]);
      gate += xv * __half2float(w1[(kIntermediate + j) * kHidden + i]);
    }
    const float activated = silu(gate) * up;
    acc += activated * __half2float(w2[h * kIntermediate + j]);
  }

  output[static_cast<int64_t>(token) * kHidden + h] =
      __float2half_rn(acc * __half2float(probs[token]));
}
}  // namespace

extern "C" void candidate(const half* tokens,
                           const int64_t* tokens_per_expert,
                           const half* probs,
                           const half* expert_fc1,
                           const half* expert_fc2,
                           half* output,
                           cudaStream_t stream) {
  dim3 grid(kExperts, kTokens, 1);
  dim3 block(kHidden, 1, 1);
  sequential_expert_kernel<<<grid, block, 0, stream>>>(
      tokens, tokens_per_expert, probs, expert_fc1, expert_fc2, output);
}
