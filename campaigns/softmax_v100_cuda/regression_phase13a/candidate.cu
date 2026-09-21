// Softmax reference kernel for Tesla V100 (sm_70)
// CUDA 11.8 compatible
// Stable row-wise softmax: y_j = exp(x_j - max) / sum(exp(x_j - max))

#include <cuda_runtime.h>
#include <math.h>

// AKA_CONTRACT: x, y, rows, cols

extern "C" __global__ void softmax_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x_row = x + (size_t)row * cols;
    float* y_row = y + (size_t)row * cols;
    int tid = threadIdx.x;

    // Step 1: Find row maximum (numerical stability)
    __shared__ float s_max[256];
    float max_val = -INFINITY;
    for (int j = tid; j < cols; j += blockDim.x) {
        float v = x_row[j];
        if (v > max_val) max_val = v;
    }
    s_max[tid] = max_val;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if (s_max[tid + offset] > s_max[tid]) {
                s_max[tid] = s_max[tid + offset];
            }
        }
        __syncthreads();
    }
    float row_max = s_max[0];

    // Step 2: Compute sum of exp(x - max)
    __shared__ float s_sum[256];
    float sum_val = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        sum_val += expf(x_row[j] - row_max);
    }
    s_sum[tid] = sum_val;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sum[tid] += s_sum[tid + offset];
        }
        __syncthreads();
    }
    float inv_sum = 1.0f / s_sum[0];

    // Step 3: Normalize
    for (int j = tid; j < cols; j += blockDim.x) {
        y_row[j] = expf(x_row[j] - row_max) * inv_sum;
    }
}

extern "C" void launch_kernel(
    float* x,
    float* y,
    int rows,
    int cols
) {
    softmax_kernel<<<rows, 256>>>(x, y, rows, cols);
    cudaDeviceSynchronize();
}