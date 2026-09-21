#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// AKA_TARGET_CONTRACT: vocab_parallel_cross_entropy:112959ca63020e9b

extern "C" cudaError_t local_max_fp16_stream(const half* logits, float* row_max, int rows, int local_vocab, cudaStream_t stream);
extern "C" cudaError_t local_prepare_fp16_stream(const half* logits, const int64_t* targets, const float* global_max, float* predicted_local, float* denominator_local, float* exp_values, unsigned char* target_mask, int64_t* target_local, int rows, int local_vocab, int rank, cudaStream_t stream);

namespace {
constexpr int kThreads = 256;
constexpr float kNegMax = -3.402823466e+38F;

__device__ __forceinline__ float warp_max(float x) {
    for (int offset = 16; offset; offset >>= 1) x = fmaxf(x, __shfl_down_sync(0xffffffffu, x, offset));
    return x;
}

__device__ __forceinline__ float warp_sum(float x) {
    for (int offset = 16; offset; offset >>= 1) x += __shfl_down_sync(0xffffffffu, x, offset);
    return x;
}

__device__ __forceinline__ float block_max(float x) {
    __shared__ float partial[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    x = warp_max(x);
    if (!lane) partial[warp] = x;
    __syncthreads();
    x = threadIdx.x < 8 ? partial[lane] : kNegMax;
    if (!warp) x = warp_max(x);
    return __shfl_sync(0xffffffffu, x, 0);
}

__device__ __forceinline__ float block_sum(float x) {
    __shared__ float partial[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    x = warp_sum(x);
    if (!lane) partial[warp] = x;
    __syncthreads();
    x = threadIdx.x < 8 ? partial[lane] : 0.0F;
    if (!warp) x = warp_sum(x);
    return __shfl_sync(0xffffffffu, x, 0);
}

__global__ void local_max_kernel(const half* __restrict__ logits, float* __restrict__ row_max, int rows, int local_vocab) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    float value = kNegMax;
    const int base = row * local_vocab;
    for (int col = threadIdx.x; col < local_vocab; col += blockDim.x)
        value = fmaxf(value, __half2float(logits[base + col]));
    value = block_max(value);
    if (!threadIdx.x) row_max[row] = value;
}

__global__ void local_prepare_kernel(const half* __restrict__ logits, const int64_t* __restrict__ targets,
                                     const float* __restrict__ global_max, float* __restrict__ predicted_local,
                                     float* __restrict__ denominator_local, float* __restrict__ exp_values,
                                     unsigned char* __restrict__ target_mask, int64_t* __restrict__ target_local,
                                     int rows, int local_vocab, int rank) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int64_t target = targets[row];
    const int64_t begin = static_cast<int64_t>(rank) * local_vocab;
    const bool owns = target >= begin && target < begin + local_vocab;
    target_mask[row] = owns ? 1 : 0;
    target_local[row] = owns ? target - begin : 0;
    predicted_local[row] = 0.0F;
    float sum = 0.0F;
    const int base = row * local_vocab;
    const float max_value = global_max[row];
    for (int col = threadIdx.x; col < local_vocab; col += blockDim.x) {
        const float shifted = __half2float(logits[base + col]) - max_value;
        const float e = expf(shifted);
        exp_values[base + col] = e;
        sum += e;
        if (owns && col == target_local[row]) predicted_local[row] = shifted;
    }
    sum = block_sum(sum);
    if (!threadIdx.x) denominator_local[row] = sum;
}
}

extern "C" cudaError_t local_max_fp16_stream(const half* logits, float* row_max, int rows, int local_vocab, cudaStream_t stream) {
    if (rows < 0 || local_vocab < 0) return cudaErrorInvalidValue;
    if (!rows) return cudaSuccess;
    local_max_kernel<<<rows, kThreads, 0, stream>>>(logits, row_max, rows, local_vocab);
    return cudaGetLastError();
}

extern "C" cudaError_t local_prepare_fp16_stream(const half* logits, const int64_t* targets, const float* global_max, float* predicted_local, float* denominator_local, float* exp_values, unsigned char* target_mask, int64_t* target_local, int rows, int local_vocab, int rank, cudaStream_t stream) {
    if (rows < 0 || local_vocab < 0 || rank < 0) return cudaErrorInvalidValue;
    if (!rows) return cudaSuccess;
    local_prepare_kernel<<<rows, kThreads, 0, stream>>>(logits, targets, global_max, predicted_local, denominator_local, exp_values, target_mask, target_local, rows, local_vocab, rank);
    return cudaGetLastError();
}
