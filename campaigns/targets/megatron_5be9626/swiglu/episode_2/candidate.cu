#include <cuda_fp16.h>

// AKA_TARGET_CONTRACT: swiglu_v1 intermediate,bias,activated,rows,width,glu_linear_offset half_fp16 contiguous activation_only

namespace {

__global__ void swiglu_kernel(const half* __restrict__ intermediate,
                              const half* __restrict__ bias,
                              half* __restrict__ activated,
                              int rows,
                              int width,
                              float offset) {
    const int half_width = width >> 1;
    const int element = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * half_width;
    if (element >= total) {
        return;
    }

    const int row = element / half_width;
    const int j = element - row * half_width;
    const int base = row * width;

    float gate = __half2float(intermediate[base + j]);
    float up = __half2float(intermediate[base + half_width + j]);
    if (bias != nullptr) {
        gate += __half2float(bias[j]);
        up += __half2float(bias[half_width + j]);
    }

    const float silu_gate = gate / (1.0f + __expf(-gate));
    activated[element] = __float2half_rn(silu_gate * (up + offset));
}

}  // namespace

extern "C" void launch_swiglu(const half* intermediate,
                               const half* bias,
                               half* activated,
                               int rows,
                               int width,
                               float glu_linear_offset) {
    const int total = rows * (width >> 1);
    if (total <= 0) {
        return;
    }
    constexpr int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    swiglu_kernel<<<blocks, threads>>>(intermediate, bias, activated,
                                        rows, width, glu_linear_offset);
}
