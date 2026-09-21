#include <cuda_runtime.h>

namespace {

__inline__ __device__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void rms_norm_kernel(const float* __restrict__ x,
                                const float* __restrict__ weight,
                                float* __restrict__ y,
                                int hidden,
                                float eps) {
    __shared__ float warp_totals[8];
    __shared__ float inv_rms_shared;

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int row_base = row * hidden;
    const float* row_x = x + row_base;
    float* row_y = y + row_base;

    float sum_sq = 0.0f;
    for (int col = tid * 4; col < hidden; col += blockDim.x * 4) {
        const int remaining = hidden - col;
        if (remaining >= 4) {
            const float4 v = *reinterpret_cast<const float4*>(row_x + col);
            sum_sq = fmaf(v.x, v.x, sum_sq);
            sum_sq = fmaf(v.y, v.y, sum_sq);
            sum_sq = fmaf(v.z, v.z, sum_sq);
            sum_sq = fmaf(v.w, v.w, sum_sq);
        } else {
            for (int j = 0; j < remaining; ++j) {
                const float v = row_x[col + j];
                sum_sq = fmaf(v, v, sum_sq);
            }
        }
    }

    sum_sq = warp_sum(sum_sq);
    if ((tid & 31) == 0) {
        warp_totals[tid >> 5] = sum_sq;
    }
    __syncthreads();

    if (tid < 32) {
        float block_sum = (tid < 8) ? warp_totals[tid] : 0.0f;
        block_sum = warp_sum(block_sum);
        if (tid == 0) {
            inv_rms_shared = rsqrtf(block_sum / static_cast<float>(hidden) + eps);
        }
    }
    __syncthreads();

    const float inv_rms = inv_rms_shared;
    for (int col = tid * 4; col < hidden; col += blockDim.x * 4) {
        const int remaining = hidden - col;
        if (remaining >= 4) {
            const float4 xv = *reinterpret_cast<const float4*>(row_x + col);
            const float4 wv = *reinterpret_cast<const float4*>(weight + col);
            float4 out;
            out.x = xv.x * inv_rms * wv.x;
            out.y = xv.y * inv_rms * wv.y;
            out.z = xv.z * inv_rms * wv.z;
            out.w = xv.w * inv_rms * wv.w;
            *reinterpret_cast<float4*>(row_y + col) = out;
        } else {
            for (int j = 0; j < remaining; ++j) {
                row_y[col + j] = row_x[col + j] * inv_rms * weight[col + j];
            }
        }
    }
}

} // namespace

extern "C" void launch_kernel(float* x, float* weight, float* y,
                               int batch, int hidden, float eps) {
    constexpr int threads = 256;
    rms_norm_kernel<<<batch, threads>>>(x, weight, y, hidden, eps);
}