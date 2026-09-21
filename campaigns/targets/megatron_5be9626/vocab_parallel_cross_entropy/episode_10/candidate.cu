#include <cuda_fp16.h>
#include <cuda_runtime.h>

// AKA_TARGET_CONTRACT: vocab_parallel_cross_entropy:112959ca63020e9b

__global__ void local_max_fp16_kernel(const half* __restrict__ logits,
                                      float* __restrict__ row_max,
                                      int rows,
                                      int local_vocab) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const half* row_logits = logits + static_cast<size_t>(row) * local_vocab;
    float maximum = -3.402823466e+38F;
    for (int col = 0; col < local_vocab; ++col) {
        maximum = fmaxf(maximum, __half2float(row_logits[col]));
    }
    row_max[row] = maximum;
}

__global__ void local_prepare_fp16_kernel(
    const half* __restrict__ logits,
    const int64_t* __restrict__ targets,
    const float* __restrict__ global_max,
    float* __restrict__ predicted_local,
    float* __restrict__ denominator_local,
    float* __restrict__ exp_values,
    unsigned char* __restrict__ target_mask,
    int64_t* __restrict__ target_local,
    int rows,
    int local_vocab,
    int rank) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const int64_t target = targets[row];
    const int64_t first = static_cast<int64_t>(rank) * local_vocab;
    const int64_t last = first + local_vocab;
    const bool owns_target = target >= first && target < last;
    target_mask[row] = owns_target ? 1 : 0;
    target_local[row] = owns_target ? target - first : 0;

    const half* row_logits = logits + static_cast<size_t>(row) * local_vocab;
    float denominator = 0.0F;
    float predicted = 0.0F;
    const float row_max = global_max[row];
    for (int col = 0; col < local_vocab; ++col) {
        const float shifted = __half2float(row_logits[col]) - row_max;
        const float value = expf(shifted);
        exp_values[static_cast<size_t>(row) * local_vocab + col] = value;
        denominator += value;
        if (owns_target && col == static_cast<int>(target - first)) {
            predicted = shifted;
        }
    }
    predicted_local[row] = predicted;
    denominator_local[row] = denominator;
}

extern "C" cudaError_t local_max_fp16_stream(const half* logits,
                                              float* row_max,
                                              int rows,
                                              int local_vocab,
                                              cudaStream_t stream) {
    if (rows < 0 || local_vocab < 0) return cudaErrorInvalidValue;
    constexpr int threads = 128;
    const int blocks = (rows + threads - 1) / threads;
    local_max_fp16_kernel<<<blocks, threads, 0, stream>>>(
        logits, row_max, rows, local_vocab);
    return cudaGetLastError();
}

extern "C" cudaError_t local_prepare_fp16_stream(const half* logits,
                                                  const int64_t* targets,
                                                  const float* global_max,
                                                  float* predicted_local,
                                                  float* denominator_local,
                                                  float* exp_values,
                                                  unsigned char* target_mask,
                                                  int64_t* target_local,
                                                  int rows,
                                                  int local_vocab,
                                                  int rank,
                                                  cudaStream_t stream) {
    if (rows < 0 || local_vocab < 0 || rank < 0) return cudaErrorInvalidValue;
    constexpr int threads = 128;
    const int blocks = (rows + threads - 1) / threads;
    local_prepare_fp16_kernel<<<blocks, threads, 0, stream>>>(
        logits, targets, global_max, predicted_local, denominator_local,
        exp_values, target_mask, target_local, rows, local_vocab, rank);
    return cudaGetLastError();
}
