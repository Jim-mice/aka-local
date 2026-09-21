#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// AKA_TARGET_CONTRACT: vocab_parallel_cross_entropy:112959ca63020e9b

__global__ static void local_max_kernel(const half* logits, float* row_max,
                                        int rows, int local_vocab) {
    int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float partial[256];
    float value = -3.402823466e+38F;
    for (int col = threadIdx.x; col < local_vocab; col += blockDim.x) {
        value = fmaxf(value, __half2float(logits[row * local_vocab + col]));
    }
    partial[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) row_max[row] = partial[0];
}

__global__ static void local_prepare_kernel(
    const half* logits, const int64_t* targets, const float* global_max,
    float* predicted_local, float* denominator_local, float* exp_values,
    unsigned char* target_mask, int64_t* target_local, int rows,
    int local_vocab, int rank) {
    int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float partial[256];
    int64_t start = (int64_t)rank * (int64_t)local_vocab;
    int64_t target = targets[row];
    bool in_partition = target >= start && target < start + local_vocab;
    if (threadIdx.x == 0) {
        target_mask[row] = in_partition ? 1 : 0;
        target_local[row] = target - start;
        predicted_local[row] = 0.0F;
    }
    float max_value = global_max[row];
    float sum = 0.0F;
    for (int col = threadIdx.x; col < local_vocab; col += blockDim.x) {
        float shifted = __half2float(logits[row * local_vocab + col]) - max_value;
        float e = expf(shifted);
        exp_values[row * local_vocab + col] = e;
        sum += e;
        if (in_partition && col == (int)(target - start)) {
            predicted_local[row] = shifted;
        }
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denominator_local[row] = partial[0];
}

extern "C" cudaError_t local_max_fp16_stream(const half* logits, float* row_max,
                                               int rows, int local_vocab,
                                               cudaStream_t stream) {
    local_max_kernel<<<rows, 256, 0, stream>>>(logits, row_max, rows, local_vocab);
    return cudaGetLastError();
}

extern "C" cudaError_t local_prepare_fp16_stream(const half* logits, const int64_t* targets,
                                                  const float* global_max, float* predicted_local,
                                                  float* denominator_local, float* exp_values,
                                                  unsigned char* target_mask, int64_t* target_local,
                                                  int rows, int local_vocab, int rank,
                                                  cudaStream_t stream) {
    local_prepare_kernel<<<rows, 256, 0, stream>>>(logits, targets, global_max, predicted_local,
                                                   denominator_local, exp_values, target_mask,
                                                   target_local, rows, local_vocab, rank);
    return cudaGetLastError();
}
