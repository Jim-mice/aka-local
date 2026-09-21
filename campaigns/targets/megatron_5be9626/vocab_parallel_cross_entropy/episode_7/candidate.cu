#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// AKA_TARGET_CONTRACT: vocab_parallel_cross_entropy:112959ca63020e9b

namespace {
constexpr float kNegMax = -3.402823466e+38F;

__global__ void local_max_kernel(const half* __restrict__ logits,
                                 float* __restrict__ row_max,
                                 int rows, int local_vocab) {
    int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float smax[256];
    float v = kNegMax;
    const int base = row * local_vocab;
    for (int col = threadIdx.x; col < local_vocab; col += blockDim.x) {
        v = fmaxf(v, __half2float(logits[base + col]));
    }
    smax[threadIdx.x] = v;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) row_max[row] = smax[0];
}

__global__ void prepare_kernel(const half* __restrict__ logits,
                               const int64_t* __restrict__ targets,
                               const float* __restrict__ global_max,
                               float* __restrict__ predicted_local,
                               float* __restrict__ denominator_local,
                               float* __restrict__ exp_values,
                               unsigned char* __restrict__ target_mask,
                               int64_t* __restrict__ target_local,
                               int rows, int local_vocab, int rank) {
    int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float ssum[256];
    __shared__ float target_shift;
    __shared__ unsigned char owned;
    __shared__ int64_t local_target;
    if (threadIdx.x == 0) {
        const int64_t target = targets[row];
        const int64_t begin = static_cast<int64_t>(rank) * local_vocab;
        owned = (target >= begin && target < begin + local_vocab) ? 1 : 0;
        local_target = owned ? target - begin : -1;
        target_shift = kNegMax;
    }
    __syncthreads();
    const float shift = global_max[row];
    const int base = row * local_vocab;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < local_vocab; col += blockDim.x) {
        float e = expf(__half2float(logits[base + col]) - shift);
        exp_values[base + col] = e;
        target_mask[base + col] = (owned && col == local_target) ? 1 : 0;
        if (owned && col == local_target) target_shift = __half2float(logits[base + col]) - shift;
        sum += e;
    }
    ssum[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) ssum[threadIdx.x] += ssum[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        denominator_local[row] = ssum[0];
        predicted_local[row] = owned ? target_shift : 0.0f;
        target_local[row] = local_target;
    }
}
}  // namespace

extern "C" cudaError_t local_max_fp16_stream(const half* logits, float* row_max, int rows, int local_vocab, cudaStream_t stream) {
    if (!logits || !row_max || rows < 0 || local_vocab < 0) return cudaErrorInvalidValue;
    local_max_kernel<<<rows, 256, 0, stream>>>(logits, row_max, rows, local_vocab);
    return cudaGetLastError();
}

extern "C" cudaError_t local_prepare_fp16_stream(const half* logits, const int64_t* targets, const float* global_max, float* predicted_local, float* denominator_local, float* exp_values, unsigned char* target_mask, int64_t* target_local, int rows, int local_vocab, int rank, cudaStream_t stream) {
    if (!logits || !targets || !global_max || !predicted_local || !denominator_local || !exp_values || !target_mask || !target_local || rows < 0 || local_vocab < 0) return cudaErrorInvalidValue;
    prepare_kernel<<<rows, 256, 0, stream>>>(logits, targets, global_max, predicted_local, denominator_local, exp_values, target_mask, target_local, rows, local_vocab, rank);
    return cudaGetLastError();
}
