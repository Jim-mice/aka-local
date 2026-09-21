#include <cuda_runtime.h>
#include <math.h>

// AKA_CONTRACT: x, weight, y, batch, hidden, eps

namespace {
constexpr int BLOCK_THREADS = 256;

__device__ __forceinline__ float warp_sum(float v) {
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v, 8);
    v += __shfl_down_sync(0xffffffffu, v, 4);
    v += __shfl_down_sync(0xffffffffu, v, 2);
    v += __shfl_down_sync(0xffffffffu, v, 1);
    return v;
}

__global__ void rms_norm_kernel(const float* __restrict__ x,
                                const float* __restrict__ weight,
                                float* __restrict__ y,
                                int batch, int hidden, float eps) {
    const int row = blockIdx.x;
    if (row >= batch) return;

    __shared__ float warp_totals[8];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const float* row_x = x + static_cast<size_t>(row) * hidden;
    float sum = 0.0f;

    if ((((uintptr_t)row_x) & 15u) == 0u && (hidden & 3) == 0) {
        const float4* row_x4 = reinterpret_cast<const float4*>(row_x);
        const int n4 = hidden >> 2;
        for (int k = tid; k < n4; k += BLOCK_THREADS) {
            float4 v = row_x4[k];
            sum = fmaf(v.x, v.x, sum);
            sum = fmaf(v.y, v.y, sum);
            sum = fmaf(v.z, v.z, sum);
            sum = fmaf(v.w, v.w, sum);
        }
    } else {
        for (int k = tid; k < hidden; k += BLOCK_THREADS) {
            const float v = row_x[k];
            sum = fmaf(v, v, sum);
        }
    }

    sum = warp_sum(sum);
    if (lane == 0) warp_totals[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        float total = (lane < 8) ? warp_totals[lane] : 0.0f;
        total = warp_sum(total);
        if (lane == 0) warp_totals[0] = total;
    }
    __syncthreads();

    const float inv_rms = rsqrtf(warp_totals[0] / static_cast<float>(hidden) + eps);
    for (int k = tid; k < hidden; k += BLOCK_THREADS) {
        y[static_cast<size_t>(row) * hidden + k] = row_x[k] * inv_rms * weight[k];
    }
}
}

extern "C" void launch_kernel(float* x, float* weight, float* y,
                               int batch, int hidden, float eps) {
    rms_norm_kernel<<<batch, BLOCK_THREADS>>>(x, weight, y, batch, hidden, eps);
}