#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// AKA_TARGET_CONTRACT: vocab_parallel_cross_entropy:112959ca63020e9b

namespace {

__global__ void local_max_kernel(const half* __restrict__ logits,
                                 float* __restrict__ row_max,
                                 int rows,
                                 int local_vocab) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float value = -3.402823466e+38F;
    const half* row_logits = logits + static_cast<size_t>(row) * local_vocab;
    for (int col = threadIdx.x; col < local_vocab; col += blockDim.x) {
        value = fmaxf(value, __half2float(row_logits[col]));
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) row_max[row] = partial[0];
}

__global__ void local_prepare_kernel(const half* __restrict__ logits,
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
    int row = blockIdx.x;
    if (row >= rows) return;

    const int64_t target = targets[row];
    const int64_t begin = static_cast<int64_t>(rank) * local_vocab;
    const bool owns_target = target >= begin && target < begin + local_vocab;
    target_mask[row] = owns_target ? 1 : 0;
    target_local[row] = owns_target ? target - begin : 0;

    const float shift = global_max[row];
    const half* row_logits = logits + static_cast<size_t>(row) * local_vocab;
    float predicted = 0.0F;
    float denominator = 0.0F;
    for (int col = threadIdx.x; col < local_vocab; col += blockDim.x) {
        const float shifted = __half2float(row_logits[col]) - shift;
        const float exp_value = __expf(shifted);
        exp_values[static_cast<size_t>(row) * local_vocab + col] = exp_value;
        denominator += exp_value;
        if (owns_target && col == static_cast<int>(target - begin)) predicted = shifted;
    }

    __shared__ float sums[512];
    sums[threadIdx.x] = denominator;
    sums[blockDim.x + threadIdx.x] = predicted;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sums[threadIdx.x] += sums[threadIdx.x + stride];
            sums[blockDim.x + threadIdx.x] += sums[blockDim.x + threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        denominator_local[row] = sums[0];
        predicted_local[row] = sums[blockDim.x];
    }
}

}  // namespace

extern "C" cudaError_t local_max_fp16_stream(const half* logits, float* row_max, int rows, int local_vocab, cudaStream_t stream) {
    if (rows <= 0 || local_vocab <= 0) return cudaSuccess;
    local_max_kernel<<<rows, 256, 0, stream>>>(logits, row_max, rows, local_vocab);
    return cudaGetLastError();
}

extern "C" cudaError_t local_prepare_fp16_stream(const half* logits, const int64_t* targets, const float* global_max, float* predicted_local, float* denominator_local, float* exp_values, unsigned char* target_mask, int64_t* target_local, int rows, int local_vocab, int rank, cudaStream_t stream) {
    if (rows <= 0 || local_vocab <= 0) return cudaSuccess;
    local_prepare_kernel<<<rows, 256, 0, stream>>>(logits, targets, global_max, predicted_local, denominator_local, exp_values, target_mask, target_local, rows, local_vocab, rank);
    return cudaGetLastError();
}
