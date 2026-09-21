#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// AKA_TARGET_CONTRACT: vocab_parallel_cross_entropy:112959ca63020e9b

namespace {

__device__ __forceinline__ float load_fp16(const half* p) {
    return __half2float(*p);
}

__global__ void local_max_kernel(const half* __restrict__ logits,
                                 float* __restrict__ row_max,
                                 int rows, int local_vocab) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    float v = -3.402823466e+38F;
    const int base = row * local_vocab;
    for (int i = threadIdx.x; i < local_vocab; i += blockDim.x) {
        v = fmaxf(v, load_fp16(logits + base + i));
    }

    __shared__ float smem[256];
    smem[threadIdx.x] = v;
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

    const int64_t target = targets[row];
    const int64_t begin = static_cast<int64_t>(rank) * local_vocab;
    const bool owns_target = target >= begin && target < begin + local_vocab;
    target_mask[row] = owns_target ? 1 : 0;
    target_local[row] = owns_target ? target - begin : 0;

    const float max_value = global_max[row];
    const int base = row * local_vocab;
    float sum = 0.0f;
    float predicted = 0.0f;
    for (int i = threadIdx.x; i < local_vocab; i += blockDim.x) {
        const float e = expf(load_fp16(logits + base + i) - max_value);
        exp_values[base + i] = e;
        sum += e;
        if (owns_target && i == static_cast<int>(target - begin)) predicted = e;
    }

    __shared__ float sums[256];
    __shared__ float preds[256];
    sums[threadIdx.x] = sum;
    preds[threadIdx.x] = predicted;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            sums[threadIdx.x] += sums[threadIdx.x + stride];
            preds[threadIdx.x] += preds[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        denominator_local[row] = sums[0];
        predicted_local[row] = preds[0];
    }
}

}  // namespace

extern "C" cudaError_t local_max_fp16_stream(const half* logits, float* row_max, int rows, int local_vocab, cudaStream_t stream) {
    if (!logits || !row_max || rows < 0 || local_vocab < 0) return cudaErrorInvalidValue;
    if (rows == 0) return cudaSuccess;
    local_max_kernel<<<rows, 256, 0, stream>>>(logits, row_max, rows, local_vocab);
    return cudaGetLastError();
}

extern "C" cudaError_t local_prepare_fp16_stream(const half* logits, const int64_t* targets, const float* global_max, float* predicted_local, float* denominator_local, float* exp_values, unsigned char* target_mask, int64_t* target_local, int rows, int local_vocab, int rank, cudaStream_t stream) {
    if (!logits || !targets || !global_max || !predicted_local || !denominator_local || !exp_values || !target_mask || !target_local || rows < 0 || local_vocab < 0) return cudaErrorInvalidValue;
    if (rows == 0) return cudaSuccess;
    local_prepare_kernel<<<rows, 256, 0, stream>>>(logits, targets, global_max, predicted_local, denominator_local, exp_values, target_mask, target_local, rows, local_vocab, rank);
    return cudaGetLastError();
}
