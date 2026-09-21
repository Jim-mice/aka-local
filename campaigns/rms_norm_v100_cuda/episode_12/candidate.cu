#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ __launch_bounds__(256) void rms_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ y,
    int hidden,
    float eps) {
    __shared__ float warp_totals[32];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int base = row * hidden;

    float sum_sq = 0.0f;
    for (int col = tid; col < hidden; col += blockDim.x) {
        const float v = x[base + col];
        sum_sq = fmaf(v, v, sum_sq);
    }

    sum_sq = warp_sum(sum_sq);
    if ((tid & 31) == 0) {
        warp_totals[tid >> 5] = sum_sq;
    }
    __syncthreads();

    float total = (tid < 32) ? warp_totals[tid] : 0.0f;
    if (tid < 32) {
        total = warp_sum(total);
    }
    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrtf(total * (1.0f / static_cast<float>(hidden)) + eps);
    }
    __syncthreads();

    const float scale = inv_rms;
    for (int col = tid; col < hidden; col += blockDim.x) {
        y[base + col] = (x[base + col] * scale) * weight[col];
    }
}

} // namespace

extern "C" void launch_kernel(
    float* x, float* weight, float* y,
    int batch, int hidden, float eps) {
    constexpr int threads = 256;
    rms_norm_kernel<<<batch, threads>>>(x, weight, y, hidden, eps);
}
