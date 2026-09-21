#include <cuda_fp16.h>
#include <cuda_runtime.h>

// AKA_TARGET_CONTRACT: vocab_parallel_cross_entropy:112959ca63020e9b

namespace {

__global__ void local_max_kernel(const half* __restrict__ logits,
                                 float* __restrict__ row_max,
                                 int rows, int local_vocab) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float smem[];
    float value = -3.402823466e+38F;
    for (int col = threadIdx.x; col < local_vocab; col += blockDim.x) {
        value = fmaxf(value, __half2float(logits[row * local_vocab + col]));
    }
    smem[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) row_max[row] = smem[0];
}

__global__ void local_prepare_kernel(const half* __restrict__ logits,
                                     const int64_t* __restrict__ targets,
                                     const float* __restrict__ global_max,
                                     float* __restrict__ predicted_local,
                                     float* __restrict__ denominator_local,
                                     float* __restrict__ exp_values,
                                     unsigned char* __restrict__ target_mask,
                                     int64_t* __restrict__ target_local,
                                     int rows, int local_vocab, int rank) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float smem[];
    float* sums = smem;
    __shared__ float target_value;
    const int64_t target = targets[row];
    const int64_t begin = static_cast<int64_t>(rank) * local_vocab;
    const int64_t end = begin + local_vocab;
    const bool owns_target = target >= begin && target < end;
    const int64_t local_target = owns_target ? target - begin : -1;

    if (threadIdx.x == 0) {
        target_value = 0.0f;
        target_mask[row] = owns_target ? 1 : 0;
        target_local[row] = local_target;
    }
    __syncthreads();

    float sum = 0.0f;
    const float max_value = global_max[row];
    for (int col = threadIdx.x; col < local_vocab; col += blockDim.x) {
        const float shifted = __half2float(logits[row * local_vocab + col]) - max_value;
        const float e = expf(shifted);
        exp_values[row * local_vocab + col] = e;
        sum += e;
        if (owns_target && static_cast<int64_t>(col) == local_target) target_value = __half2float(logits[row * local_vocab + col]);
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride; stride >>= 1) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        denominator_local[row] = sums[0];
        predicted_local[row] = owns_target ? target_value : 0.0f;
    }
}

}  // namespace

extern "C" cudaError_t local_max_fp16_stream(const half* logits, float* row_max, int rows, int local_vocab, cudaStream_t stream) {
    if (rows < 0 || local_vocab < 0) return cudaErrorInvalidValue;
    constexpr int threads = 256;
    local_max_kernel<<<rows, threads, threads * sizeof(float), stream>>>(logits, row_max, rows, local_vocab);
    return cudaGetLastError();
}

extern "C" cudaError_t local_prepare_fp16_stream(const half* logits, const int64_t* targets, const float* global_max, float* predicted_local, float* denominator_local, float* exp_values, unsigned char* target_mask, int64_t* target_local, int rows, int local_vocab, int rank, cudaStream_t stream) {
    if (rows < 0 || local_vocab < 0) return cudaErrorInvalidValue;
    constexpr int threads = 256;
    local_prepare_kernel<<<rows, threads, threads * sizeof(float), stream>>>(logits, targets, global_max, predicted_local, denominator_local, exp_values, target_mask, target_local, rows, local_vocab, rank);
    return cudaGetLastError();
}
