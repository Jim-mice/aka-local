#include <cuda_runtime.h>
#include <math.h>

// AKA_CONTRACT: x, gamma, beta, y, batch, hidden, eps

namespace {

__inline__ __device__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__global__ void layer_norm_kernel(const float* __restrict__ x,
                                  const float* __restrict__ gamma,
                                  const float* __restrict__ beta,
                                  float* __restrict__ y,
                                  int batch, int hidden, float eps) {
    const int row = blockIdx.x;
    if (row >= batch) return;

    __shared__ float warp_sums[8];
    __shared__ float warp_sq_sums[8];
    __shared__ float row_mean;
    __shared__ float inv_std;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const float* row_x = x + (size_t)row * (size_t)hidden;

    float sum = 0.0f;
    for (int j = tid; j < hidden; j += blockDim.x) {
        sum += row_x[j];
    }
    sum = warp_sum(sum);
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        float total = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
        total = warp_sum(total);
        if (lane == 0) row_mean = total / (float)hidden;
    }
    __syncthreads();

    const float mean = row_mean;
    float sq_sum = 0.0f;
    for (int j = tid; j < hidden; j += blockDim.x) {
        const float d = row_x[j] - mean;
        sq_sum = fmaf(d, d, sq_sum);
    }
    sq_sum = warp_sum(sq_sum);
    if (lane == 0) warp_sq_sums[warp] = sq_sum;
    __syncthreads();

    if (warp == 0) {
        float total = (lane < (blockDim.x >> 5)) ? warp_sq_sums[lane] : 0.0f;
        total = warp_sum(total);
        if (lane == 0) {
            const float variance = total / (float)hidden;
            inv_std = rsqrtf(variance + eps);
        }
    }
    __syncthreads();

    const float scale = inv_std;
    float4* out4 = reinterpret_cast<float4*>(y + (size_t)row * (size_t)hidden);
    const float4* in4 = reinterpret_cast<const float4*>(row_x);
    const float4* g4 = reinterpret_cast<const float4*>(gamma);
    const float4* b4 = reinterpret_cast<const float4*>(beta);
    const int vec_count = hidden >> 2;
    for (int k = tid; k < vec_count; k += blockDim.x) {
        const float4 xv = in4[k];
        const float4 gv = g4[k];
        const float4 bv = b4[k];
        float4 ov;
        ov.x = fmaf((xv.x - mean) * scale, gv.x, bv.x);
        ov.y = fmaf((xv.y - mean) * scale, gv.y, bv.y);
        ov.z = fmaf((xv.z - mean) * scale, gv.z, bv.z);
        ov.w = fmaf((xv.w - mean) * scale, gv.w, bv.w);
        out4[k] = ov;
    }
    for (int j = (vec_count << 2) + tid; j < hidden; j += blockDim.x) {
        y[(size_t)row * (size_t)hidden + j] = fmaf((row_x[j] - mean) * scale, gamma[j], beta[j]);
    }
}

} // namespace

extern "C" void launch_kernel(
    float* x,
    float* gamma,
    float* beta,
    float* y,
    int batch,
    int hidden,
    float eps) {
    layer_norm_kernel<<<batch, 256>>>(x, gamma, beta, y, batch, hidden, eps);
}