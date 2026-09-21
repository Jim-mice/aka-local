#include <cuda_runtime.h>
#include <math.h>

namespace {

__device__ __forceinline__ float warp_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

__global__ void rms_norm_kernel(const float* __restrict__ x,
                                const float* __restrict__ weight,
                                float* __restrict__ y,
                                int hidden,
                                float eps) {
    __shared__ float warp_sums[8];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int base = row * hidden;

    float sum_sq = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float v = x[base + i];
        sum_sq = fmaf(v, v, sum_sq);
    }

    sum_sq = warp_sum(sum_sq);
    if (lane == 0) {
        warp_sums[warp] = sum_sq;
    }
    __syncthreads();

    float total = (tid < 8) ? warp_sums[tid] : 0.0f;
    if (warp == 0) {
        total = warp_sum(total);
        if (tid == 0) {
            warp_sums[0] = rsqrtf(total / static_cast<float>(hidden) + eps);
        }
    }
    __syncthreads();

    const float inv_rms = warp_sums[0];
    for (int i = tid; i < hidden; i += blockDim.x) {
        y[base + i] = x[base + i] * inv_rms * weight[i];
    }
}

} // namespace

extern "C" void launch_kernel(float* x, float* weight, float* y,
                              int batch, int hidden, float eps) {
    constexpr int threads = 256;
    rms_norm_kernel<<<batch, threads>>>(x, weight, y, hidden, eps);
}