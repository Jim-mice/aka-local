#include <cuda_fp16.h>

// AKA_TARGET_CONTRACT: swiglu_v1 intermediate,bias,activated,rows,width,glu_linear_offset half_fp16 contiguous activation_only

namespace {

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void swiglu_kernel(const half* __restrict__ intermediate,
                              const half* __restrict__ bias,
                              half* __restrict__ activated,
                              int rows,
                              int width,
                              float glu_linear_offset) {
    const int output_width = width / 2;
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * output_width;
    if (linear >= total) return;

    const int row = linear / output_width;
    const int j = linear - row * output_width;
    const int base = row * width + j;

    float gate = __half2float(intermediate[base]);
    float up = __half2float(intermediate[base + output_width]);
    if (bias != nullptr) {
        gate += __half2float(bias[j]);
        up += __half2float(bias[j + output_width]);
    }
    activated[linear] = __float2half(silu(gate) * (up + glu_linear_offset));
}

}  // namespace

extern "C" void launch_swiglu(const half* intermediate,
                               const half* bias,
                               half* activated,
                               int rows,
                               int width,
                               float glu_linear_offset) {
    constexpr int threads = 256;
    const int total = rows * (width / 2);
    const int blocks = (total + threads - 1) / threads;
    swiglu_kernel<<<blocks, threads>>>(intermediate, bias, activated, rows, width,
                                        glu_linear_offset);
}
