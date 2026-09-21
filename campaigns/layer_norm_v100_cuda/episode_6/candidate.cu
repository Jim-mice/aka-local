#include <cuda_runtime.h>
#include <math.h>

namespace {

constexpr int BLOCK_THREADS = 256;

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float block_sum(float v, float* warp_sums) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    v = (threadIdx.x < (BLOCK_THREADS / 32)) ? warp_sums[lane] : 0.0f;
    if (warp == 0) v = warp_sum(v);
    if (threadIdx.x == 0) warp_sums[0] = v;
    __syncthreads();
    return warp_sums[0];
}

__global__ void layer_norm_kernel(const float* __restrict__ x,
                                  float* __restrict__ y,
                                  const float* __restrict__ gamma,
                                  const float* __restrict__ beta,
                                  int rows, int cols, float eps) {
    __shared__ float reduction[BLOCK_THREADS / 32];
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const float* row_x = x + static_cast<size_t>(row) * cols;
    float* row_y = y + static_cast<size_t>(row) * cols;

    float sum = 0.0f;
    for (int c = tid; c < cols; c += BLOCK_THREADS) sum += row_x[c];
    const float mean = block_sum(sum, reduction) / static_cast<float>(cols);

    float var_sum = 0.0f;
    for (int c = tid; c < cols; c += BLOCK_THREADS) {
        const float d = row_x[c] - mean;
        var_sum = fmaf(d, d, var_sum);
    }
    const float variance = block_sum(var_sum, reduction) / static_cast<float>(cols);
    const float inv_std = rsqrtf(variance + eps);

    for (int c = tid; c < cols; c += BLOCK_THREADS) {
        const float normalized = (row_x[c] - mean) * inv_std;
        row_y[c] = fmaf(normalized, gamma[c], beta[c]);
    }
}

} // namespace

extern "C" void launch_kernel(float* x, float* y, float* gamma, float* beta,
                               int rows, int cols, float eps) {
    if (rows <= 0 || cols <= 0) return;
    layer_norm_kernel<<<rows, BLOCK_THREADS>>>(x, y, gamma, beta, rows, cols, eps);
}
