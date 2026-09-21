#include <cuda_runtime.h>

namespace {

__inline__ __device__ float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void rms_norm_kernel(const float* __restrict__ x,
                                const float* __restrict__ weight,
                                float* __restrict__ y,
                                int hidden,
                                float eps) {
    __shared__ float warp_sums[32];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row_offset = row * hidden;

    float sum_sq = 0.0f;
    for (int col = tid; col < hidden; col += blockDim.x) {
        const float value = x[row_offset + col];
        sum_sq = fmaf(value, value, sum_sq);
    }

    sum_sq = warp_reduce_sum(sum_sq);
    if (lane == 0) {
        warp_sums[warp] = sum_sq;
    }
    __syncthreads();

    float block_sum = (tid < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        block_sum = warp_reduce_sum(block_sum);
    }

    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrtf(block_sum / static_cast<float>(hidden) + eps);
    }
    __syncthreads();

    for (int col = tid; col < hidden; col += blockDim.x) {
        y[row_offset + col] = x[row_offset + col] * inv_rms * weight[col];
    }
}

}  // namespace

extern "C" void launch_kernel(float* x, float* weight, float* y,
                               int batch, int hidden, float eps) {
    constexpr int threads = 256;
    rms_norm_kernel<<<batch, threads>>>(x, weight, y, hidden, eps);
}