// RMSNorm reference kernel for Tesla V100 (sm_70)
// CUDA 11.8 compatible

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void rmsnorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ y,
    int batch,
    int hidden,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= batch) return;

    __shared__ float s_sum_sq[256];
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float v = x[row * hidden + i];
        sum_sq += v * v;
    }
    s_sum_sq[tid] = sum_sq;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) s_sum_sq[tid] += s_sum_sq[tid + offset];
        __syncthreads();
    }

    float rms = sqrtf(s_sum_sq[0] / hidden + eps);
    for (int i = tid; i < hidden; i += blockDim.x) {
        float v = x[row * hidden + i];
        y[row * hidden + i] = (v / rms) * weight[i];
    }
}

extern "C" void launch_kernel(
    float* x, float* weight, float* y,
    int batch, int hidden, float eps
) {
    rmsnorm_kernel<<<batch, 256>>>(x, weight, y, batch, hidden, eps);
    cudaDeviceSynchronize();
}
