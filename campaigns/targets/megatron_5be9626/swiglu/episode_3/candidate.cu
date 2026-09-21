#include <cuda_fp16.h>

// AKA_TARGET_CONTRACT: swiglu_v1 intermediate,bias,activated,rows,width,glu_linear_offset half_fp16 contiguous activation_only

namespace {

__global__ void swiglu_kernel(const half* __restrict__ intermediate,
                              const half* __restrict__ bias,
                              half* __restrict__ activated,
                              int rows,
                              int width,
                              float glu_linear_offset) {
    const int half_width = width >> 1;
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * half_width;
    if (linear >= total) {
        return;
    }

    const int j = linear % half_width;
    const int row_base = linear - j;
    float gate = __half2float(intermediate[row_base + j]);
    float up = __half2float(intermediate[row_base + half_width + j]);
    if (bias != nullptr) {
        gate += __half2float(bias[j]);
        up += __half2float(bias[half_width + j]);
    }

    const float silu = gate / (1.0f + expf(-gate));
    activated[linear] = __float2half_rn(silu * (up + glu_linear_offset));
}

}  // namespace

extern "C" void launch_swiglu(const half* intermediate,
                               const half* bias,
                               half* activated,
                               int rows,
                               int width,
                               float glu_linear_offset) {
    const int total = rows * (width >> 1);
    constexpr int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    swiglu_kernel<<<blocks, threads>>>(intermediate, bias, activated,
                                       rows, width, glu_linear_offset);
}
