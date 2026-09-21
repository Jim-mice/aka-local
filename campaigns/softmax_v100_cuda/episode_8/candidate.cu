#include <cuda_runtime.h>
#include <float.h>

// AKA_CONTRACT: x, y, rows, cols

static __device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

static __device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__global__ __launch_bounds__(256, 2)
void softmax_rows(const float* __restrict__ x,
                  float* __restrict__ y,
                  int rows,
                  int cols) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    if (row >= rows) return;

    __shared__ float smem[16];

    const float* row_x = x + (size_t)row * (size_t)cols;
    float* row_y = y + (size_t)row * (size_t)cols;

    // Pass 1: per-warp max reduction
    float local_max = -FLT_MAX;
    #pragma unroll 4
    for (int j = tid; j < cols; j += blockDim.x) {
        float v = row_x[j];
        local_max = v > local_max ? v : local_max;
    }
    local_max = warp_reduce_max(local_max);
    if (lane == 0) smem[warp_id] = local_max;
    __syncthreads();

    float row_max = -FLT_MAX;
    if (warp_id == 0) {
        float wmax = (lane < (blockDim.x >> 5)) ? smem[lane] : -FLT_MAX;
        row_max = warp_reduce_max(wmax);
    }
    if (warp_id == 0 && lane == 0) smem[0] = row_max;
    __syncthreads();
    row_max = smem[0];

    // Pass 2: per-warp exp sum
    float local_sum = 0.0f;
    #pragma unroll 4
    for (int j = tid; j < cols; j += blockDim.x) {
        local_sum += __expf(row_x[j] - row_max);
    }
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) smem[warp_id] = local_sum;
    __syncthreads();

    float row_sum = 0.0f;
    if (warp_id == 0) {
        float wsum = (lane < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
        row_sum = warp_reduce_sum(wsum);
    }
    if (warp_id == 0 && lane == 0) smem[0] = row_sum;
    __syncthreads();
    row_sum = smem[0];
    float inv_sum = __fdividef(1.0f, row_sum);

    // Pass 3: write output, using __expf intrinsic
    #pragma unroll 4
    for (int j = tid; j < cols; j += blockDim.x) {
        row_y[j] = __expf(row_x[j] - row_max) * inv_sum;
    }
}

extern "C" void launch_kernel(
    float* x,
    float* y,
    int rows,
    int cols
) {
    if (rows <= 0 || cols <= 0) return;
    softmax_rows<<<rows, 256>>>(x, y, rows, cols);
}
