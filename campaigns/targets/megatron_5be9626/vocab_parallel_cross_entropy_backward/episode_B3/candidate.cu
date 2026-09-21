#include <cuda_runtime.h>
#include <stdint.h>

// AKA_TARGET_CONTRACT: vocab_parallel_cross_entropy_backward:6121f49401f3ef4601549c8732a62870ff17a7a7d7047e9c55d55b79c9ff6e92

namespace {

__global__ void ce_backward_local_fp32_kernel(
    const float* __restrict__ softmax,
    const bool* __restrict__ target_mask,
    const int64_t* __restrict__ masked_target_1d,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int64_t rows,
    int64_t local_vocab) {
  const int64_t total = rows * local_vocab;
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t row = linear / local_vocab;
    const int64_t col = linear - row * local_vocab;
    const float g = grad_output[row];
    float value = softmax[linear] * g;
    if (!target_mask[row] && masked_target_1d[row] == col) {
      value -= g;
    }
    grad_input[linear] = value;
  }
}

}  // namespace

extern "C" void ce_backward_local_fp32_stream(const float* softmax, const bool* target_mask, const int64_t* masked_target_1d, const float* grad_output, float* grad_input, int64_t rows, int64_t local_vocab, cudaStream_t stream) {
  if (rows <= 0 || local_vocab <= 0) {
    return;
  }
  constexpr int threads = 256;
  int64_t blocks64 = (rows * local_vocab + threads - 1) / threads;
  if (blocks64 > 65535) {
    blocks64 = 65535;
  }
  ce_backward_local_fp32_kernel<<<static_cast<unsigned int>(blocks64), threads, 0, stream>>>(
      softmax, target_mask, masked_target_1d, grad_output, grad_input, rows, local_vocab);
}
