#include <cuda.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void swiglu_forward_kernel(const half* gate, const half* up, half* out, long n) {
    long pair = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)blockDim.x * gridDim.x;
    long pairs = n / 2;
    for (; pair < pairs; pair += stride) {
        const half2 g2 = reinterpret_cast<const half2*>(gate)[pair];
        const half2 u2 = reinterpret_cast<const half2*>(up)[pair];
        float2 g = __half22float2(g2);
        float2 u = __half22float2(u2);
        reinterpret_cast<half2*>(out)[pair] = __floats2half2_rn(silu(g.x) * u.x, silu(g.y) * u.y);
    }
    if ((n & 1) && pair == pairs) {
        float g = __half2float(gate[n - 1]);
        float u = __half2float(up[n - 1]);
        out[n - 1] = __float2half_rn(silu(g) * u);
    }
}

void launch_swiglu(const at::Tensor& gate, const at::Tensor& up, at::Tensor& out) {
    const int threads = 256;
    const long n = gate.numel();
    int blocks = (int)((n + threads - 1) / threads);
    blocks = blocks > 4096 ? 4096 : blocks;
    swiglu_forward_kernel<<<blocks, threads>>>(
        reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(up.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()), n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_swiglu", &launch_swiglu, "Fused SwiGLU forward (CUDA)");
}
