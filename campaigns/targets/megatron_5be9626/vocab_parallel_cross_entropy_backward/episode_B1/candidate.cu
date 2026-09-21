#include <cuda_runtime.h>
#include <stdint.h>

// AKA_TARGET_CONTRACT: vocab_parallel_cross_entropy_backward:6121f49401f3ef4601549c8732a62870ff17a7a7d7047e9c55d55b79c9ff6e92

__global__ void ce_backward_local_fp32_kernel(
    const float* __restrict__ softmax,
    const bool* __restrict__ target_mask,
    const int64_t* __restrict__ masked_target_1d,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int64_t rows,
    int64_t local_vocab) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * local_vocab;
  if (linear >= total) return;

  const int64_t row = linear / local_vocab;
  const int64_t col = linear - row * local_vocab;
  float value = softmax[linear] * grad_output[row];
  if (!target_mask[row] && masked_target_1d[row] == col) {
    value -= grad_output[row];
  }
  grad_input[linear] = value;
}

extern "C" void ce_backward_local_fp32_stream(const float* softmax, const bool* target_mask, const int64_t* masked_target_1d, const float* grad_output, float* grad_input, int64_t rows, int64_t local_vocab, cudaStream_t stream) {
  if (rows <= 0 || local_vocab <= 0) return;
  const int threads = 256;
  const int64_t total = rows * local_vocab;
  const int64_t blocks64 = (total + threads - 1) / threads;
  const unsigned int blocks = static_cast<unsigned int>(blocks64);
  ce_backward_local_fp32_kernel<<<blocks, threads, 0, stream>>>(
      softmax, target_mask, masked_target_1d, grad_output, grad_input,
      rows, local_vocab);
}
