// LayerNorm reference kernel for Tesla V100 (sm_70)
// CUDA 11.8 compatible
// y = (x - mean) / sqrt(var + eps) * gamma + beta
// Uses the same reduction pattern as the verified RMSNorm kernel

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void layernorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int batch,
    int hidden,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= batch) return;

    __shared__ float s_buf[512]; // [0..255] = mean partials, [256..511] = var partials
    float* s_mean = s_buf;
    float* s_var = s_buf + 256;

    const float* x_row = x + row * hidden;

    // Step 1: Compute mean
    float sum = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        sum += x_row[i];
    }
    s_mean[tid] = sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_mean[tid] += s_mean[tid + offset];
        }
        __syncthreads();
    }

    float mean = s_mean[0] / (float)hidden;

    // Step 2: Compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float diff = x_row[i] - mean;
        var_sum += diff * diff;
    }
    s_var[tid] = var_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_var[tid] += s_var[tid + offset];
        }
        __syncthreads();
    }

    float inv_std = rsqrtf(s_var[0] / (float)hidden + eps);

    // Step 3: Normalize and scale
    float* y_row = y + row * hidden;
    for (int i = tid; i < hidden; i += blockDim.x) {
        y_row[i] = (x_row[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

extern "C" void launch_kernel(
    float* x, float* gamma, float* beta, float* y,
    int batch, int hidden, float eps
) {
    layernorm_kernel<<<batch, 256>>>(x, gamma, beta, y, batch, hidden, eps);
    cudaDeviceSynchronize();
}