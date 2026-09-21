// AKA_TARGET_CONTRACT: vocab_parallel_cross_entropy:112959ca63020e9b
// Rank-local CUDA stages for TP=2 vocab-parallel cross entropy.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace {

__global__ void local_max_kernel(const half* logits, float* row_max,
                                 int rows, int local_vocab) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const half* x = logits + (int64_t)row * local_vocab;
  float m = -CUDART_INF_F;
  for (int col = 0; col < local_vocab; ++col) {
    m = fmaxf(m, __half2float(x[col]));
  }
  row_max[row] = m;
}

__global__ void local_prepare_kernel(
    const half* logits, const int64_t* targets, const float* global_max,
    float* predicted_local, float* denominator_local, float* exp_values,
    unsigned char* target_mask, int64_t* target_local, int rows,
    int local_vocab, int rank) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const half* x = logits + (int64_t)row * local_vocab;
  float* ev = exp_values + (int64_t)row * local_vocab;
  unsigned char* mask = target_mask + (int64_t)row * local_vocab;
  const int64_t target = targets[row];
  const int64_t begin = (int64_t)rank * local_vocab;
  const int64_t end = begin + local_vocab;
  const bool owned = target >= begin && target < end;
  target_local[row] = owned ? target - begin : -1;
  float predicted = 0.0f;
  float denom = 0.0f;
  const float m = global_max[row];
  for (int col = 0; col < local_vocab; ++col) {
    const float shifted = __half2float(x[col]) - m;
    const float e = expf(shifted);
    ev[col] = e;
    denom += e;
    const bool is_target = owned && target_local[row] == col;
    mask[col] = is_target ? 1 : 0;
    if (is_target) predicted = shifted;
  }
  // These are rank-local contributions. The caller must SUM all-reduce both
  // arrays before forming loss; no collective is performed here.
  predicted_local[row] = predicted;
  denominator_local[row] = denom;
}

}  // namespace

extern "C" __attribute__((visibility("default")))
cudaError_t local_max_fp16_stream(const half* logits, float* row_max, int rows,
                                  int local_vocab, cudaStream_t stream) {
  if (!logits || !row_max || rows < 0 || local_vocab <= 0) {
    return cudaErrorInvalidValue;
  }
  const int threads = 128;
  local_max_kernel<<<(rows + threads - 1) / threads, threads, 0, stream>>>(
      logits, row_max, rows, local_vocab);
  return cudaGetLastError();
}

extern "C" __attribute__((visibility("default")))
cudaError_t local_prepare_fp16_stream(
    const half* logits, const int64_t* targets, const float* global_max,
    float* predicted_local, float* denominator_local, float* exp_values,
    unsigned char* target_mask, int64_t* target_local, int rows,
    int local_vocab, int rank, cudaStream_t stream) {
  if (!logits || !targets || !global_max || !predicted_local ||
      !denominator_local || !exp_values || !target_mask || !target_local ||
      rows < 0 || local_vocab <= 0 || rank < 0) {
    return cudaErrorInvalidValue;
  }
  const int threads = 128;
  local_prepare_kernel<<<(rows + threads - 1) / threads, threads, 0, stream>>>(
      logits, targets, global_max, predicted_local, denominator_local,
      exp_values, target_mask, target_local, rows, local_vocab, rank);
  return cudaGetLastError();
}

// ABI-preserving convenience entries.  Stream-aware callers should use the
// *_stream forms; stream 0 is CUDA's caller-selected default stream.
extern "C" __attribute__((visibility("default")))
cudaError_t local_max_fp16(const half* logits, float* row_max, int rows,
                           int local_vocab) {
  return local_max_fp16_stream(logits, row_max, rows, local_vocab, 0);
}

extern "C" __attribute__((visibility("default")))
cudaError_t local_prepare_fp16(
    const half* logits, const int64_t* targets, const float* global_max,
    float* predicted_local, float* denominator_local, float* exp_values,
    unsigned char* target_mask, int64_t* target_local, int rows,
    int local_vocab, int rank) {
  return local_prepare_fp16_stream(
      logits, targets, global_max, predicted_local, denominator_local,
      exp_values, target_mask, target_local, rows, local_vocab, rank, 0);
}
