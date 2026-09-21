#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// AKA_TARGET_CONTRACT: megatron_torch_rmsnorm_forward:d24ea7a30fb6a91bc1958a955d140992e77e68dad78faa48158f2b907b722731

namespace {

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void rmsnorm_forward_kernel(const __half* __restrict__ x,
                                       const __half* __restrict__ weight,
                                       __half* __restrict__ output,
                                       int64_t rows,
                                       int64_t hidden_size,
                                       float epsilon) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) return;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __shared__ float warp_sums[8];

    const int64_t row_base = row * hidden_size;
    float sum_squares = 0.0f;
    for (int64_t k = threadIdx.x; k < hidden_size; k += blockDim.x) {
        const float value = __half2float(x[row_base + k]);
        sum_squares += value * value;
    }
    sum_squares = warp_sum(sum_squares);
    if (lane == 0) warp_sums[warp] = sum_squares;
    __syncthreads();

    float total = (threadIdx.x < 8) ? warp_sums[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32) total = warp_sum(total);
    __shared__ float inverse_rms;
    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(total / static_cast<float>(hidden_size) + epsilon);
    }
    __syncthreads();

    for (int64_t k = threadIdx.x; k < hidden_size; k += blockDim.x) {
        const float value = __half2float(x[row_base + k]);
        const float scale = __half2float(weight[k]);
        output[row_base + k] = __float2half(value * inverse_rms * scale);
    }
}

}  // namespace

extern "C" void rmsnorm_forward_fp16_stream(const __half* x,
                                              const __half* weight,
                                              __half* output,
                                              int64_t rows,
                                              int64_t hidden_size,
                                              float epsilon,
                                              cudaStream_t stream) {
    if (rows <= 0 || hidden_size <= 0) return;
    const dim3 block(256);
    const dim3 grid(static_cast<unsigned int>(rows));
    rmsnorm_forward_kernel<<<grid, block, 0, stream>>>(
        x, weight, output, rows, hidden_size, epsilon);
}
