#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

// AKA_TARGET_CONTRACT: vocab_parallel_cross_entropy:112959ca63020e9b
// Rank-local implementation only.  The caller supplies the global MAX and
// performs the SUM all-reduces for predicted_logit and denominator.

namespace {
constexpr int kThreads = 256;

__global__ void max_kernel(const half* __restrict__ logits,
                           float* __restrict__ out, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows) return;
  float v = -CUDART_INF_F;
  for (int c = threadIdx.x; c < cols; c += blockDim.x)
    v = fmaxf(v, __half2float(logits[row * cols + c]));
  __shared__ float s[kThreads];
  s[threadIdx.x] = v;
  __syncthreads();
  for (int d = blockDim.x / 2; d; d >>= 1) {
    if (threadIdx.x < d) s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + d]);
    __syncthreads();
  }
  if (threadIdx.x == 0) out[row] = s[0];
}

__global__ void prepare_kernel(const half* __restrict__ logits,
                               const int64_t* __restrict__ targets,
                               const float* __restrict__ global_max,
                               float* __restrict__ predicted,
                               float* __restrict__ denominator,
                               float* __restrict__ aux,
                               unsigned char* __restrict__ masked,
                               int64_t* __restrict__ local_target,
                               int rows, int cols, int vocab_start) {
  int row = blockIdx.x;
  if (row >= rows) return;
  int64_t target = targets[row];
  bool owns = target >= vocab_start && target < (int64_t)vocab_start + cols;
  masked[row] = owns ? 0 : 1;
  local_target[row] = owns ? target - vocab_start : 0;
  float p = 0.0f, sum = 0.0f;
  float m = global_max[row];
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float x = __half2float(logits[row * cols + c]) - m;
    float e = expf(x);
    sum += e;
    if (owns && c == local_target[row]) p = x;
  }
  __shared__ float ss[kThreads], sp[kThreads];
  ss[threadIdx.x] = sum; sp[threadIdx.x] = p;
  __syncthreads();
  for (int d = blockDim.x / 2; d; d >>= 1) {
    if (threadIdx.x < d) { ss[threadIdx.x] += ss[threadIdx.x+d]; sp[threadIdx.x] += sp[threadIdx.x+d]; }
    __syncthreads();
  }
  if (threadIdx.x == 0) { predicted[row] = sp[0]; denominator[row] = ss[0]; aux[row] = 0.0f; }
}
}

extern "C" void local_max_fp16(const half* logits, float* local_max, int rows, int cols) {
  if (rows > 0 && cols > 0) max_kernel<<<rows, kThreads, 0, 0>>>(logits, local_max, rows, cols);
}

extern "C" void local_prepare_fp16(const half* logits, const int64_t* targets,
    const float* global_max, float* predicted, float* denominator, float* aux,
    unsigned char* masked, int64_t* local_target, int rows, int cols, int vocab_start) {
  if (rows > 0 && cols > 0)
    prepare_kernel<<<rows, kThreads, 0, 0>>>(logits, targets, global_max, predicted,
      denominator, aux, masked, local_target, rows, cols, vocab_start);
}
