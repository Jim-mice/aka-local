#include <cuda_runtime.h>

namespace {

__global__ void rms_norm_kernel(const float* __restrict__ x,
                                const float* __restrict__ weight,
                                float* __restrict__ y,
                                int hidden,
                                float eps) {
    __shared__ float warp_sums[8];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    float sum = 0.0f;
    const int vec_hidden = hidden & ~3;
    for (int i = tid * 4; i < vec_hidden; i += blockDim.x * 4) {
        float4 v = reinterpret_cast<const float4*>(x)[i >> 2];
        sum = fmaf(v.x, v.x, sum);
        sum = fmaf(v.y, v.y, sum);
        sum = fmaf(v.z, v.z, sum);
        sum = fmaf(v.w, v.w, sum);
    }
    for (int i = vec_hidden + tid; i < hidden; i += blockDim.x) {
        sum = fmaf(x[i], x[i], sum);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        warp_sums[warp] = sum;
    }
    __syncthreads();

    float total = (tid < 8) ? warp_sums[tid] : 0.0f;
    if (warp == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(0xffffffff, total, offset);
        }
        if (tid == 0) {
            warp_sums[0] = rsqrtf(total / (float)hidden + eps);
        }
    }
    __syncthreads();
    const float inv_rms = warp_sums[0];

    for (int i = tid * 4; i < vec_hidden; i += blockDim.x * 4) {
        float4 v = reinterpret_cast<const float4*>(x)[i >> 2];
        float4 w = reinterpret_cast<const float4*>(weight)[i >> 2];
        float4 out;
        out.x = v.x * inv_rms * w.x;
        out.y = v.y * inv_rms * w.y;
        out.z = v.z * inv_rms * w.z;
        out.w = v.w * inv_rms * w.w;
        reinterpret_cast<float4*>(y)[i >> 2] = out;
    }
    for (int i = vec_hidden + tid; i < hidden; i += blockDim.x) {
        y[i] = x[i] * inv_rms * weight[i];
    }
}

} // namespace

extern "C" void launch_kernel(float* x, float* weight, float* y,
                               int batch, int hidden, float eps) {
    constexpr int threads = 256;
    rms_norm_kernel<<<batch, threads>>>(x, weight, y, hidden, eps);
}