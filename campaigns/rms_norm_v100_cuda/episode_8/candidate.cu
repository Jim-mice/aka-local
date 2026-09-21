#include <cuda_runtime.h>

namespace {

__inline__ __device__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
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

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const float* row_x = x + static_cast<size_t>(row) * hidden;
    float* row_y = y + static_cast<size_t>(row) * hidden;

    float sum_sq = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float v = row_x[i];
        sum_sq = fmaf(v, v, sum_sq);
    }

    sum_sq = warp_sum(sum_sq);
    if (lane == 0) {
        warp_totals[warp] = sum_sq;
    }
    __syncthreads();

    float block_sum = (tid < (blockDim.x >> 5)) ? warp_totals[tid] : 0.0f;
    if (warp == 0) {
        block_sum = warp_sum(block_sum);
        if (tid == 0) {
            warp_totals[0] = rsqrtf(block_sum / static_cast<float>(hidden) + eps);
        }
    }
    __syncthreads();

    const float inv_rms = warp_totals[0];
    for (int i = tid; i < hidden; i += blockDim.x) {
        row_y[i] = row_x[i] * inv_rms * weight[i];
    }
}

}  // namespace

extern "C" void launch_kernel(
    float* x, float* weight, float* y,
    int batch, int hidden, float eps) {
    if (batch <= 0 || hidden <= 0) {
        return;
    }
    rms_norm_kernel<<<batch, 256>>>(x, weight, y, hidden, eps);
}