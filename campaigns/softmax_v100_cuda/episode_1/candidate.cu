#include <cuda_runtime.h>
#include <math.h>

// AKA_CONTRACT: x, y, rows, cols

namespace {

__device__ __forceinline__ float warp_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__global__ void softmax_rows(const float* __restrict__ x,
                             float* __restrict__ y,
                             int rows,
                             int cols) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    __shared__ float warp_values[8];

    const float* row_x = x + static_cast<size_t>(row) * cols;
    float* row_y = y + static_cast<size_t>(row) * cols;

    float local_max = -CUDART_INF_F;
    for (int j = tid; j < cols; j += blockDim.x) {
        local_max = fmaxf(local_max, row_x[j]);
    }
    local_max = warp_max(local_max);
    if (lane == 0) warp_values[warp] = local_max;
    __syncthreads();

    float row_max = -CUDART_INF_F;
    if (warp == 0) {
        row_max = (lane < (blockDim.x >> 5)) ? warp_values[lane] : -CUDART_INF_F;
        row_max = warp_max(row_max);
        if (lane == 0) warp_values[0] = row_max;
    }
    __syncthreads();
    row_max = warp_values[0];

    float local_sum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        local_sum += __expf(row_x[j] - row_max);
    }
    local_sum = warp_sum(local_sum);
    if (lane == 0) warp_values[warp] = local_sum;
    __syncthreads();

    float row_sum = 0.0f;
    if (warp == 0) {
        row_sum = (lane < (blockDim.x >> 5)) ? warp_values[lane] : 0.0f;
        row_sum = warp_sum(row_sum);
        if (lane == 0) warp_values[0] = row_sum;
    }
    __syncthreads();
    const float inv_sum = __fdividef(1.0f, warp_values[0]);

    for (int j = tid; j < cols; j += blockDim.x) {
        row_y[j] = __expf(row_x[j] - row_max) * inv_sum;
    }
}

} // namespace

extern "C" void launch_kernel(float* x, float* y, int rows, int cols) {
    constexpr int threads = 256;
    softmax_rows<<<rows, threads>>>(x, y, rows, cols);
}
