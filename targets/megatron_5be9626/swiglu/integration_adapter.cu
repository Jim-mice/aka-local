#include <cuda_fp16.h>
#include <cuda_runtime.h>

// AKA_TARGET_INTEGRATION: swiglu_v1 current_stream adapter_from=21e7995911f9af2d

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void swiglu_kernel(const half* intermediate, const half* bias,
                              half* activated, int rows, int width,
                              float offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * (width / 2);
    if (i >= n) return;
    int row = i / (width / 2);
    int col = i % (width / 2);
    int gate_i = row * width + col;
    int up_i = gate_i + width / 2;
    float gate = __half2float(intermediate[gate_i]);
    float up = __half2float(intermediate[up_i]);
    if (bias) {
        gate += __half2float(bias[col]);
        up += __half2float(bias[col + width / 2]);
    }
    activated[i] = __float2half_rn(silu(gate) * (up + offset));
}

extern "C" void launch_swiglu_stream(const half* intermediate,
                                      const half* bias, half* activated,
                                      int rows, int width, float offset,
                                      cudaStream_t stream) {
    int n = rows * (width / 2);
    int blocks = (n + 255) / 256;
    swiglu_kernel<<<blocks, 256, 0, stream>>>(intermediate, bias, activated,
                                               rows, width, offset);
}
