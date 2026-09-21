import torch
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
namespace {
template <typename T> __device__ __forceinline__ float fval(T x) { return static_cast<float>(x); }
template <typename scalar_t>
__global__ void rmsnorm_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ y, int64_t cols, float eps) {
    const int64_t row = static_cast<int64_t>(blockIdx.x); const int tid = threadIdx.x;
    const int lane = tid & 31; const int warp = tid >> 5;
    const scalar_t* xr = x + row * cols; scalar_t* yr = y + row * cols;
    float sum = 0.0f;
    for (int64_t c = tid; c < cols; c += blockDim.x) { float v = fval(xr[c]); sum = fmaf(v, v, sum); }
    for (int off = 16; off; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
    __shared__ float ws[32]; if (lane == 0) ws[warp] = sum; __syncthreads();
    float total = (tid < (blockDim.x >> 5)) ? ws[lane] : 0.0f;
    if (warp == 0) for (int off = 16; off; off >>= 1) total += __shfl_down_sync(0xffffffffu, total, off);
    __shared__ float inv; if (tid == 0) inv = rsqrtf(total / static_cast<float>(cols) + eps); __syncthreads();
    for (int64_t c = tid; c < cols; c += blockDim.x) yr[c] = static_cast<scalar_t>(fval(xr[c]) * inv);
}
}
torch::Tensor rmsnorm_cuda(torch::Tensor x, double eps) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2, "x must be contiguous CUDA 2D");
    TORCH_CHECK(x.size(1) > 0, "normalized dimension must be non-empty");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat, "unsupported dtype");
    auto y = torch::empty_like(x); constexpr int threads = 256;
    auto stream = at::cuda::getDefaultCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::Half, at::BFloat16, x.scalar_type(), "rmsnorm_cuda", [&] {
        rmsnorm_kernel<scalar_t><<<static_cast<unsigned int>(x.size(0)), threads, 0, stream>>>(x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), x.size(1), static_cast<float>(eps));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("rmsnorm_cuda", &rmsnorm_cuda); }
'''

_ext = load_inline(name="rmsnorm_episode9_sm120", cpp_sources="", cuda_sources=_CUDA_SRC, functions=None, with_cuda=True, verbose=False, extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_120,code=sm_120"])

def rmsnorm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    if x.dim() != 2: raise ValueError("rmsnorm expects [rows, cols]")
    return _ext.rmsnorm_cuda(x.contiguous(), eps)

__all__ = ["rmsnorm"]
