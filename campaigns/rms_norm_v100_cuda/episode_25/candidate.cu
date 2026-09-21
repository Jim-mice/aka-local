#include <cuda_runtime.h>
#include <math.h>

// AKA_CONTRACT: x, weight, y, batch, hidden, eps

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
                                int batch,
                                int hidden,
                                float eps) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= batch) return;

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    __shared__ float warp_sums[8];

    const float* row_x = x + static_cast<size_t>(row) * hidden;
    float* row_y = y + static_cast<size_t>(row) * hidden;

    float sum = 0.0f;
    const int block_stride = blockDim.x;
    int j = tid;
    #pragma unroll 4
    for (; j + 3 * block_stride < hidden; j += 4 * block_stride) {
        const float a = row_x[j];
        const float b = row_x[j + block_stride];
        const float c = row_x[j + 2 * block_stride];
        const float d = row_x[j + 3 * block_stride];
        sum = fmaf(a, a, sum);
        sum = fmaf(b, b, sum);
        sum = fmaf(c, c, sum);
        sum = fmaf(d, d, sum);
    }
    for (; j < hidden; j += block_stride) {
        const float v = row_x[j];
        sum = fmaf(v, v, sum);
    }

    sum = warp_sum(sum);
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        float block_sum = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
        block_sum = warp_sum(block_sum);
        if (lane == 0) warp_sums[0] = block_sum;
    }
    __syncthreads();

    const float inv_rms = rsqrtf(warp_sums[0] / static_cast<float>(hidden) + eps);
    j = tid;
    #pragma unroll 4
    for (; j + 3 * block_stride < hidden; j += 4 * block_stride) {
        row_y[j] = row_x[j] * inv_rms * weight[j];
        row_y[j + block_stride] = row_x[j + block_stride] * inv_rms * weight[j + block_stride];
        row_y[j + 2 * block_stride] = row_x[j + 2 * block_stride] * inv_rms * weight[j + 2 * block_stride];
        row_y[j + 3 * block_stride] = row_x[j + 3 * block_stride] * inv_rms * weight[j + 3 * block_stride];
    }
    for (; j < hidden; j += block_stride) {
        row_y[j] = row_x[j] * inv_rms * weight[j];
    }
}

} // namespace

extern "C" void launch_kernel(float* x, float* weight, float* y,
                              int batch, int hidden, float eps) {
    constexpr int threads = 256;
    rms_norm_kernel<<<batch, threads>>>(x, weight, y, batch, hidden, eps);
}