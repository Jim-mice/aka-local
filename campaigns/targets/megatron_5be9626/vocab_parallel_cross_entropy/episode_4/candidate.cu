#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// AKA_TARGET_CONTRACT: vocab_parallel_cross_entropy:112959ca63020e9b
// Rank-local kernels only.  The caller owns the stream and performs exactly
// one MAX and two SUM all-reduces between these stages and loss/softmax use.

extern "C" __global__ void aka_local_max_kernel(const half* logits,
                                                 float* row_max,
                                                 int rows, int vocab) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const half* x = logits + (size_t)row * vocab;
  float m = __half2float(x[0]);
  for (int j = 1; j < vocab; ++j) {
    float v = __half2float(x[j]);
    m = v > m ? v : m;
  }
  row_max[row] = m;
}

extern "C" __global__ void aka_local_prepare_kernel(
    const half* logits, const int64_t* targets, const float* global_max,
    float* predicted_logit, float* denominator, float* exp_logits,
    unsigned char* target_mask, int64_t* local_target, int rows, int vocab,
    int rank) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  int64_t target = targets[row];
  int64_t begin = (int64_t)rank * vocab;
  int64_t end = begin + vocab;
  bool owns = target >= begin && target < end;
  target_mask[row] = owns ? 0 : 1;
  local_target[row] = owns ? target - begin : 0;
  float selected = owns ? __half2float(logits[(size_t)row * vocab +
                                               (target - begin)]) : 0.0f;
  predicted_logit[row] = selected;
  float sum = 0.0f;
  const half* x = logits + (size_t)row * vocab;
  float shift = global_max[row];
  float* e = exp_logits + (size_t)row * vocab;
  for (int j = 0; j < vocab; ++j) {
    float z = expf(__half2float(x[j]) - shift);
    e[j] = z;
    sum += z;
  }
  denominator[row] = sum;
}

extern "C" void local_max_fp16(const half* logits, float* row_max,
                                int rows, int vocab, cudaStream_t stream) {
  int threads = 128;
  aka_local_max_kernel<<<(rows + threads - 1) / threads, threads, 0, stream>>>(
      logits, row_max, rows, vocab);
}

extern "C" void local_prepare_fp16(
    const half* logits, const int64_t* targets, const float* global_max,
    float* predicted_logit, float* denominator, float* exp_logits,
    unsigned char* target_mask, int64_t* local_target, int rows, int vocab,
    int rank, cudaStream_t stream) {
  int threads = 128;
  aka_local_prepare_kernel<<<(rows + threads - 1) / threads, threads, 0,
                             stream>>>(logits, targets, global_max,
                                       predicted_logit, denominator,
                                       exp_logits, target_mask, local_target,
                                       rows, vocab, rank);
}
