#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

__inline__ __device__ float warp_reduce_sum(float v) {
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v, 8);
    v += __shfl_down_sync(0xffffffffu, v, 4);
    v += __shfl_down_sync(0xffffffffu, v, 2);
    v += __shfl_down_sync(0xffffffffu, v, 1);
    return v;
}

__global__ __launch_bounds__(256, 2)
void rms_norm_kernel(const float* __restrict__ x,
                     const float* __restrict__ weight,
                     float* __restrict__ y,
                     int hidden, float eps) {
    __shared__ float warp_sums[8];
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const float* row_x = x + (size_t)row * hidden;
    float* row_y = y + (size_t)row * hidden;

    float sum = 0.0f;
    const int vec_count = hidden >> 2;
    const float4* x4 = reinterpret_cast<const float4*>(row_x);
    for (int i = tid; i < vec_count; i += blockDim.x) {
        float4 v = x4[i];
        sum = fmaf(v.x, v.x, sum);
        sum = fmaf(v.y, v.y, sum);
        sum = fmaf(v.z, v.z, sum);
        sum = fmaf(v.w, v.w, sum);
    }
    for (int i = (vec_count << 2) + tid; i < hidden; i += blockDim.x) {
        float v = row_x[i];
        sum = fmaf(v, v, sum);
    }

    sum = warp_reduce_sum(sum);
    if ((tid & 31) == 0) warp_sums[tid >> 5] = sum;
    __syncthreads();

    float total = (tid < 8) ? warp_sums[tid] : 0.0f;
    if (tid < 32) total = warp_reduce_sum(total);
    if (tid == 0) warp_sums[0] = total;
    __syncthreads();

    const float inv_rms = rsqrtf(warp_sums[0] * (1.0f / (float)hidden) + eps);
    const float4* w4 = reinterpret_cast<const float4*>(weight);
    float4* y4 = reinterpret_cast<float4*>(row_y);
    for (int i = tid; i < vec_count; i += blockDim.x) {
        float4 xv = x4[i];
        float4 wv = w4[i];
        float4 out;
        out.x = (xv.x * inv_rms) * wv.x;
        out.y = (xv.y * inv_rms) * wv.y;
        out.z = (xv.z * inv_rms) * wv.z;
        out.w = (xv.w * inv_rms) * wv.w;
        y4[i] = out;
    }
    for (int i = (vec_count << 2) + tid; i < hidden; i += blockDim.x)
        row_y[i] = (row_x[i] * inv_rms) * weight[i];
}

extern "C" void launch_kernel(float* x, float* weight, float* y,
                               int batch, int hidden, float eps) {
    rms_norm_kernel<<<batch, 256>>>(x, weight, y, hidden, eps);
}