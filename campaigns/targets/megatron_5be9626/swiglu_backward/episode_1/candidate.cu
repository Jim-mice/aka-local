#include <cuda_fp16.h>
#include <cuda_runtime.h>

// AKA_TARGET_CONTRACT: swiglu_backward_v1 intermediate,grad_output,grad_intermediate,rows,width,glu_linear_offset half_fp16 contiguous current_stream

namespace {

__global__ void swiglu_backward_kernel(const half* intermediate,
                                       const half* grad_output,
                                       half* grad_intermediate,
                                       int rows,
                                       int width,
                                       float glu_linear_offset) {
    const int half_width = width / 2;
    const int64_t total = static_cast<int64_t>(rows) * half_width;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int col = static_cast<int>(linear % half_width);
        const int64_t row_base = (linear / half_width) * width;

        const float gate = __half2float(intermediate[row_base + col]);
        const float up = __half2float(intermediate[row_base + half_width + col]);
        const float dy = __half2float(grad_output[linear]);
        const float sigmoid = 1.0f / (1.0f + expf(-gate));
        const float silu = gate * sigmoid;
        const float grad_up = dy * silu;
        const float grad_gate = dy * (up + glu_linear_offset) * sigmoid *
                                (1.0f + gate * (1.0f - sigmoid));

        grad_intermediate[row_base + col] = __float2half_rn(grad_gate);
        grad_intermediate[row_base + half_width + col] = __float2half_rn(grad_up);
    }
}

}  // namespace

extern "C" void launch_swiglu_backward(const half* intermediate,
                                        const half* grad_output,
                                        half* grad_intermediate,
                                        int rows,
                                        int width,
                                        float glu_linear_offset,
                                        cudaStream_t stream) {
    const int64_t elements = static_cast<int64_t>(rows) * (width / 2);
    const int threads = 256;
    const int blocks = static_cast<int>(min<int64_t>((elements + threads - 1) / threads, 4096));
    if (blocks > 0) {
        swiglu_backward_kernel<<<blocks, threads, 0, stream>>>(
            intermediate, grad_output, grad_intermediate, rows, width, glu_linear_offset);
    }
}
