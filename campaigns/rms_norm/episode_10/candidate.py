import torch
from torch.utils.cpp_extension import load_inline

_rmsnorm_ext = None

_CPP_SOURCE = r'''
#include <torch/extension.h>

torch::Tensor rmsnorm_cuda_binding(torch::Tensor x, torch::Tensor weight, double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_cuda", &rmsnorm_cuda_binding, "RMSNorm CUDA");
}
'''

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

// One block processes one logical row. Each thread accumulates strided elements
// in registers, then a two-level warp reduction computes the row mean square.
__global__ void rmsnorm_cuda(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t rows,
    int64_t cols,
    float eps) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const float* row_x = x + row * cols;
    float* row_out = output + row * cols;

    float sum_sq = 0.0f;
    for (int64_t col = tid; col < cols; col += blockDim.x) {
        const float v = row_x[col];
        sum_sq = fmaf(v, v, sum_sq);
    }

    sum_sq = warp_sum(sum_sq);
    __shared__ float warp_sums[32];
    if (lane == 0) {
        warp_sums[warp] = sum_sq;
    }
    __syncthreads();

    float total = (tid < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        total = warp_sum(total);
    }
    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrtf(total / static_cast<float>(cols) + eps);
    }
    __syncthreads();

    for (int64_t col = tid; col < cols; col += blockDim.x) {
        row_out[col] = row_x[col] * inv_rms * weight[col];
    }
}

} // namespace

torch::Tensor rmsnorm_cuda_binding(torch::Tensor x, torch::Tensor weight, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");
    TORCH_CHECK(weight.dim() == 1, "weight must be one-dimensional");
    TORCH_CHECK(x.size(-1) == weight.size(0), "last dimension must match weight");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "inputs must be contiguous");

    const auto cols = x.size(-1);
    const auto rows = x.numel() / cols;
    auto output = torch::empty_like(x);

    int threads = 32;
    while (threads < cols && threads < 256) {
        threads <<= 1;
    }
    const dim3 grid(static_cast<unsigned int>(rows));
    const dim3 block(threads);
    const auto stream = at::cuda::getDefaultCUDAStream(x.device().index());
    rmsnorm_cuda<<<grid, block, 0, stream.stream()>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
        rows, cols, static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
'''


def _get_extension():
    global _rmsnorm_ext
    if _rmsnorm_ext is None:
        _rmsnorm_ext = load_inline(
            name="rmsnorm_sm120a_ext",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=None,
            extra_cflags=["/O2"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_120a,code=sm_120a"],
            with_cuda=True,
            verbose=False,
        )
    return _rmsnorm_ext


class Model(torch.nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return _get_extension().rmsnorm_cuda(x.contiguous(), weight.contiguous(), self.eps)
