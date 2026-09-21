#include <cuda_fp16.h>
#include <cuda_runtime.h>

// AKA_TARGET_CONTRACT: swiglu_backward_v1 intermediate,grad_output,grad_intermediate,rows,width,glu_linear_offset half_fp16 contiguous current_stream

namespace {

__device__ __forceinline__ half sigmoid_fp16(half x) {
    const float xf = __half2float(x);
    return __float2half(1.0f / (1.0f + expf(-xf)));
}

__global__ void swiglu_backward_kernel(const half* __restrict__ intermediate,
                                       const half* __restrict__ grad_output,
                                       half* __restrict__ grad_intermediate,
                                       int rows,
                                       int width,
                                       half glu_linear_offset) {
    const int half_width = width >> 1;
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * half_width;
    if (linear >= total) {
        return;
    }

    const int col = linear % half_width;
    const int row = linear / half_width;
    const int base = row * width + col;
    const half gate = intermediate[base];
    const half up = intermediate[base + half_width];
    const half go = grad_output[linear];

    const half sigmoid_gate = sigmoid_fp16(gate);
    const half one_minus_sigmoid = __hsub(__float2half(1.0f), sigmoid_gate);
    const half silu_gate = __hmul(gate, sigmoid_gate);
    const half grad_up = __hmul(go, silu_gate);
    const half up_plus_offset = __hadd(up, glu_linear_offset);
    const half gate_derivative = __hmul(
        __hmul(sigmoid_gate, __hadd(__float2half(1.0f),
                                    __hmul(gate, one_minus_sigmoid))),
        up_plus_offset);
    const half grad_gate = __hmul(go, gate_derivative);

    grad_intermediate[base] = grad_gate;
    grad_intermediate[base + half_width] = grad_up;
}

}  // namespace

extern "C" void launch_swiglu_backward(const half* intermediate,
                                        const half* grad_output,
                                        half* grad_intermediate,
                                        int rows,
                                        int width,
                                        float glu_linear_offset,
                                        cudaStream_t stream) {
    const int total = rows * (width >> 1);
    constexpr int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    const half offset = __float2half(glu_linear_offset);
    swiglu_backward_kernel<<<blocks, threads, 0, stream>>>(
        intermediate, grad_output, grad_intermediate, rows, width, offset);
}
