from __future__ import annotations

import torch
from torch.utils.cpp_extension import load_inline

_rmsnorm_ext = None

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

__inline__ __device__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}

__global__ void rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ weight,
                               float* __restrict__ out, int64_t rows, int64_t cols, float eps) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_count = blockDim.x >> 5;
    __shared__ float warp_sums[32];
    const float* row_x = x + static_cast<int64_t>(row) * cols;
    float* row_out = out + static_cast<int64_t>(row) * cols;
    float sum_sq = 0.0f;
    for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float value = row_x[col];
        sum_sq = fmaf(value, value, sum_sq);
    }
    sum_sq = warp_sum(sum_sq);
    if (lane == 0) warp_sums[warp] = sum_sq;
    __syncthreads();
    float total = (threadIdx.x < warp_count) ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        total = warp_sum(total);
        if (lane == 0) warp_sums[0] = total;
    }
    __syncthreads();
    const float inv_rms = rsqrtf(warp_sums[0] / static_cast<float>(cols) + eps);
    for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) row_out[col] = row_x[col] * inv_rms * weight[col];
}

void rmsnorm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor out, double eps) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda() && out.is_cuda(), "all tensors must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && weight.scalar_type() == torch::kFloat32 && out.scalar_type() == torch::kFloat32, "float32 tensors required");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && out.is_contiguous(), "contiguous tensors required");
    TORCH_CHECK(x.dim() >= 1 && weight.dim() == 1 && x.size(-1) == weight.size(0), "invalid shapes");
    TORCH_CHECK(out.sizes() == x.sizes(), "out must match x");
    const int64_t cols = x.size(-1);
    const int64_t rows = x.numel() / cols;
    rmsnorm_kernel<<<static_cast<unsigned int>(rows), 256, 0, at::cuda::getDefaultCUDAStream()>>>(x.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(), rows, cols, static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''

def _get_extension():
    global _rmsnorm_ext
    if _rmsnorm_ext is None:
        _rmsnorm_ext = load_inline(name="rmsnorm_sm120a_ext", cpp_sources="", cuda_sources=_CUDA_SRC,
            functions=["rmsnorm_cuda"], with_cuda=True, verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_120,code=sm_120a"])
    return _rmsnorm_ext

class Model(torch.nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x_contig = x.contiguous()
        weight_contig = weight.contiguous()
        out = torch.empty_like(x_contig)
        _get_extension().rmsnorm_cuda(x_contig, weight_contig, out, self.eps)
        return out
