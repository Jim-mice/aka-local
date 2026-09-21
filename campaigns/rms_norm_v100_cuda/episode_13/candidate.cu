#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ float warp_reduce_sum(float value) {
    value += __shfl_down_sync(0xffffffffu, value, 16);
    value += __shfl_down_sync(0xffffffffu, value, 8);
    value += __shfl_down_sync(0xffffffffu, value, 4);
    value += __shfl_down_sync(0xffffffffu, value, 2);
    value += __shfl_down_sync(0xffffffffu, value, 1);
    return value;
}

__global__ __launch_bounds__(256, 2)
void rms_norm_kernel(const float* __restrict__ x,
                     const float* __restrict__ weight,
                     float* __restrict__ y,
                     int hidden,
                     float eps) {
    __shared__ float warp_sums[8];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int row_base = row * hidden;

    float sum = 0.0f;
    for (int col = tid; col < hidden; col += blockDim.x) {
        const float value = x[row_base + col];
        sum = fmaf(value, value, sum);
    }

    sum = warp_reduce_sum(sum);
    if ((tid & 31) == 0) {
        warp_sums[tid >> 5] = sum;
    }
    __syncthreads();

    float total = (tid < 8) ? warp_sums[tid] : 0.0f;
    if (tid < 32) {
        total = warp_reduce_sum(total);
    }
    if (tid == 0) {
        warp_sums[0] = rsqrtf(total / static_cast<float>(hidden) + eps);
    }
    __syncthreads();

    const float inv_rms = warp_sums[0];
    for (int col = tid; col < hidden; col += blockDim.x) {
        y[row_base + col] = x[row_base + col] * inv_rms * weight[col];
    }
}

} // namespace

extern "C" void launch_kernel(float* x, float* weight, float* y,
                              int batch, int hidden, float eps) {
    constexpr int threads = 256;
    rms_norm_kernel<<<batch, threads>>>(x, weight, y, hidden, eps);
}