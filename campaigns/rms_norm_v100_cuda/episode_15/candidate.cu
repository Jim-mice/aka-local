#include <cuda_runtime.h>
#include <math.h>

namespace {

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__global__ void rms_norm_kernel(const float* __restrict__ x,
                                const float* __restrict__ weight,
                                float* __restrict__ y,
                                int hidden,
                                float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row_x = x + static_cast<size_t>(row) * hidden;
    float* row_y = y + static_cast<size_t>(row) * hidden;

    float sum_sq = 0.0f;
    if ((hidden & 3) == 0) {
        const float4* x4 = reinterpret_cast<const float4*>(row_x);
        for (int i = tid; i < (hidden >> 2); i += blockDim.x) {
            const float4 v = x4[i];
            sum_sq = fmaf(v.x, v.x, sum_sq);
            sum_sq = fmaf(v.y, v.y, sum_sq);
            sum_sq = fmaf(v.z, v.z, sum_sq);
            sum_sq = fmaf(v.w, v.w, sum_sq);
        }
    } else {
        for (int i = tid; i < hidden; i += blockDim.x) {
            const float v = row_x[i];
            sum_sq = fmaf(v, v, sum_sq);
        }
    }

    sum_sq = warp_sum(sum_sq);
    __shared__ float warp_totals[8];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) warp_totals[warp] = sum_sq;
    __syncthreads();

    float total = (tid < 8) ? warp_totals[tid] : 0.0f;
    if (warp == 0) total = warp_sum(total);
    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrtf(total / static_cast<float>(hidden) + eps);
    }
    __syncthreads();
    const float scale = inv_rms;

    if ((hidden & 3) == 0) {
        const float4* x4 = reinterpret_cast<const float4*>(row_x);
        for (int i = tid; i < (hidden >> 2); i += blockDim.x) {
            const float4 xv = x4[i];
            const int base = i << 2;
            row_y[base + 0] = (xv.x * scale) * weight[base + 0];
            row_y[base + 1] = (xv.y * scale) * weight[base + 1];
            row_y[base + 2] = (xv.z * scale) * weight[base + 2];
            row_y[base + 3] = (xv.w * scale) * weight[base + 3];
        }
    } else {
        for (int i = tid; i < hidden; i += blockDim.x) {
            row_y[i] = (row_x[i] * scale) * weight[i];
        }
    }
}

} // namespace

extern "C" void launch_kernel(float* x, float* weight, float* y,
                               int batch, int hidden, float eps) {
    rms_norm_kernel<<<batch, 256>>>(x, weight, y, hidden, eps);
}