#include <cuda_fp16.h>
#include <cuda_runtime.h>

// AKA_TARGET_CONTRACT: swiglu_backward_v1 intermediate,grad_output,grad_intermediate,rows,width,glu_linear_offset half_fp16 contiguous current_stream

namespace {
__device__ __forceinline__ float sigmoidf_stable(float x) {
    if (x >= 0.0f) { const float z = expf(-x); return 1.0f / (1.0f + z); }
    const float z = expf(x); return z / (1.0f + z);
}

__global__ void swiglu_backward_kernel(const half* __restrict__ intermediate,
                                       const half* __restrict__ grad_output,
                                       half* __restrict__ grad_intermediate,
                                       int rows, int width, float glu_linear_offset) {
    const int cols = width >> 1;
    const int64_t elements = static_cast<int64_t>(rows) * cols;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements; linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int col = static_cast<int>(linear % cols);
        const int64_t row_base = (linear / cols) * width;
        const float gate = __half2float(intermediate[row_base + col]);
        const float up = __half2float(intermediate[row_base + cols + col]);
        const float go = __half2float(grad_output[linear]);
        const float s = sigmoidf_stable(gate);
        const float grad_up = go * (gate * s);
        const float grad_gate = go * (up + glu_linear_offset) * s *
                                (1.0f + gate * (1.0f - s));
        grad_intermediate[row_base + col] = __float2half_rn(grad_gate);
        grad_intermediate[row_base + cols + col] = __float2half_rn(grad_up);
    }
}
}  // namespace

extern "C" void launch_swiglu_backward(const half* intermediate,
                                        const half* grad_output,
                                        half* grad_intermediate, int rows, int width,
                                        float glu_linear_offset, cudaStream_t stream) {
    const int64_t elements = static_cast<int64_t>(rows) * (width >> 1);
    if (elements <= 0) return;
    constexpr int threads = 256;
    const int blocks = static_cast<int>(min<int64_t>((elements + threads - 1) / threads, 4096));
    swiglu_backward_kernel<<<blocks, threads, 0, stream>>>(
        intermediate, grad_output, grad_intermediate, rows, width, glu_linear_offset);
}
