import torch
from torch.utils.cpp_extension import load_inline

_rmsnorm_ext = None

_CPP_SOURCE = r'''
#include <torch/extension.h>
torch::Tensor rmsnorm_cuda(torch::Tensor x, torch::Tensor weight, double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_cuda", &rmsnorm_cuda, "RMSNorm CUDA");
}
'''

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

template <typename scalar_t>
__global__ void rmsnorm_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ y,
    int64_t cols,
    float eps) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps = blockDim.x >> 5;
    const scalar_t* row_x = x + row * cols;
    scalar_t* row_y = y + row * cols;

    float sum = 0.0f;
    for (int64_t col = tid; col < cols; col += blockDim.x) {
        const float v = static_cast<float>(row_x[col]);
        sum = fmaf(v, v, sum);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float warp_sums[32];
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    float total = (tid < warps) ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(0xffffffff, total, offset);
        }
        if (tid == 0) warp_sums[0] = rsqrtf(total / static_cast<float>(cols) + eps);
    }
    __syncthreads();

    const float inv_rms = warp_sums[0];
    for (int64_t col = tid; col < cols; col += blockDim.x) {
        const float v = static_cast<float>(row_x[col]);
        const float w = static_cast<float>(weight[col]);
        row_y[col] = static_cast<scalar_t>(v * inv_rms * w);
    }
}

} // namespace

torch::Tensor rmsnorm_cuda(torch::Tensor x, torch::Tensor weight, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [rows, cols]");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(1), "weight shape mismatch");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type(), "x and weight dtypes must match");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16 || x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kFloat, "unsupported dtype");

    auto y = torch::empty_like(x);
    const int threads = 256;
    const auto rows = x.size(0);
    const auto cols = x.size(1);
    const auto stream = at::cuda::getDefaultCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, x.scalar_type(), "rmsnorm_cuda", [&] {
        rmsnorm_kernel<scalar_t><<<static_cast<unsigned int>(rows), threads, 0, stream>>>(
            x.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), cols, static_cast<float>(eps));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
'''


def _get_extension():
    global _rmsnorm_ext
    if _rmsnorm_ext is None:
        _rmsnorm_ext = load_inline(
            name="rmsnorm_cuda_ext",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=None,
            extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-Xcompiler=/Zc:preprocessor"],
            with_cuda=True,
            verbose=False,
        )
    return _rmsnorm_ext


class Model(torch.nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return _get_extension().rmsnorm_cuda(x, weight, self.eps)
