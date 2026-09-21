#include <cuda_runtime.h>

namespace {
constexpr int BLOCK_THREADS = 256;

__inline__ __device__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void rms_norm_kernel(const float* __restrict__ input,
                                const float* __restrict__ weight,
                                float* __restrict__ output,
                                int rows, int cols, float epsilon) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const int base = row * cols;
    float sum_sq = 0.0f;
    for (int col = tid; col < cols; col += BLOCK_THREADS) {
        const float v = input[base + col];
        sum_sq = fmaf(v, v, sum_sq);
    }

    sum_sq = warp_sum(sum_sq);
    __shared__ float warp_sums[BLOCK_THREADS / 32];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) warp_sums[warp] = sum_sq;
    __syncthreads();

    float total = (tid < BLOCK_THREADS / 32) ? warp_sums[lane] : 0.0f;
    if (warp == 0) total = warp_sum(total);
    __shared__ float inv_rms;
    if (tid == 0) inv_rms = rsqrtf(total / static_cast<float>(cols) + epsilon);
    __syncthreads();

    const float scale = inv_rms;
    for (int col = tid; col < cols; col += BLOCK_THREADS) {
        output[base + col] = input[base + col] * scale * weight[col];
    }
}
}

extern "C" void launch_kernel(float* input, float* weight, float* output,
                              int rows, int cols, float epsilon) {
    if (rows <= 0 || cols <= 0) return;
    rms_norm_kernel<<<rows, BLOCK_THREADS>>>(input, weight, output, rows, cols, epsilon);
}