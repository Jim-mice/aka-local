import torch
from torch.utils.cpp_extension import load_inline

_CPP = r"""
#include <torch/extension.h>
torch::Tensor rms_norm_cuda(torch::Tensor x, torch::Tensor weight, double eps);
torch::Tensor rms_norm(torch::Tensor x, torch::Tensor weight, double eps) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "x and weight must be CUDA tensors");
    TORCH_CHECK(x.dim() >= 1 && weight.dim() == 1, "invalid tensor dimensions");
    TORCH_CHECK(x.size(-1) == weight.size(0), "weight size must match last dimension");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type(), "x and weight dtypes must match");
    TORCH_CHECK(x.is_floating_point(), "x must be floating point");
    return rms_norm_cuda(x, weight, eps);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm", &rms_norm, "RMSNorm forward (CUDA)", py::arg("x"), py::arg("weight"), py::arg("eps") = 1e-5);
}
"""

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

__inline__ __device__ float warp_sum(float v) {
    for (int delta = 16; delta > 0; delta >>= 1) v += __shfl_down_sync(0xffffffffu, v, delta);
    return v;
}

template <typename scalar_t>
__global__ void rms_norm_kernel(const scalar_t* __restrict__ x, const scalar_t* __restrict__ weight,
                                scalar_t* __restrict__ out, int64_t cols, float eps) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const scalar_t* row_x = x + row * cols;
    scalar_t* row_out = out + row * cols;
    float sum_sq = 0.0f;
    for (int64_t c = tid; c < cols; c += blockDim.x) {
        const float v = static_cast<float>(row_x[c]);
        sum_sq = fmaf(v, v, sum_sq);
    }
    sum_sq = warp_sum(sum_sq);
    __shared__ float warp_sums[32];
    if (lane == 0) warp_sums[warp] = sum_sq;
    __syncthreads();
    if (warp == 0) {
        const int warp_count = (blockDim.x + 31) >> 5;
        float block_sum = lane < warp_count ? warp_sums[lane] : 0.0f;
        block_sum = warp_sum(block_sum);
        if (lane == 0) warp_sums[0] = rsqrtf(block_sum / static_cast<float>(cols) + eps);
    }
    __syncthreads();
    const float inv_rms = warp_sums[0];
    for (int64_t c = tid; c < cols; c += blockDim.x) {
        row_out[c] = static_cast<scalar_t>(static_cast<float>(row_x[c]) * inv_rms * static_cast<float>(weight[c]));
    }
}

torch::Tensor rms_norm_cuda(torch::Tensor x, torch::Tensor weight, double eps) {
    auto xc = x.contiguous(), wc = weight.contiguous(), out = torch::empty_like(xc);
    const int64_t cols = xc.size(-1), rows = xc.numel() / cols;
    constexpr int threads = 256;
    auto stream = at::cuda::getDefaultCUDAStream(xc.device().index());
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                    xc.scalar_type(), "rms_norm_cuda", [&] {
        rms_norm_kernel<scalar_t><<<static_cast<unsigned int>(rows), threads, 0, stream>>>(
            xc.data_ptr<scalar_t>(), wc.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), cols, static_cast<float>(eps));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out.view_as(x);
}
"""

_ext = None
def _extension():
    global _ext
    if _ext is None:
        _ext = load_inline(name="rms_norm_sm120_episode8", cpp_sources=_CPP, cuda_sources=_CUDA,
                           functions=None, with_cuda=True, extra_cflags=["/O2"],
                           extra_cuda_cflags=["-O3", "--use_fast_math", "--generate-code=arch=compute_120,code=sm_120"],
                           verbose=False)
    return _ext

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return _extension().rms_norm(x, weight, eps)

__all__ = ["rms_norm"]
