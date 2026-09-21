// LayerNorm reference kernel for Tesla V100 (sm_70)
// CUDA 11.8 compatible
// Computes: y = (x - mean) / sqrt(var + eps) * gamma + beta

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

    __shared__ float s_mean[256];
    __shared__ float s_var[256];

    // Compute mean
    float sum = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        sum += x[row * hidden + i];
    }
    s_mean[tid] = sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) s_mean[tid] += s_mean[tid + offset];
        __syncthreads();
    }

    float mean = s_mean[0] / hidden;

    // Compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float diff = x[row * hidden + i] - mean;
        var_sum += diff * diff;
    }
    s_var[tid] = var_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) s_var[tid] += s_var[tid + offset];
        __syncthreads();
    }

    float inv_std = rsqrtf(s_var[0] / hidden + eps);

    // Normalize and scale
    for (int i = tid; i < hidden; i += blockDim.x) {
        float v = x[row * hidden + i];
        y[row * hidden + i] = (v - mean) * inv_std * gamma[i] + beta[i];
    }
}

extern "C" void launch_kernel(
    float* x, float* gamma, float* beta, float* y,
    int batch, int hidden, float eps
) {
    layernorm_kernel<<<batch, 256>>>(x, gamma, beta, y, batch, hidden, eps);
    cudaDeviceSynchronize();
}
