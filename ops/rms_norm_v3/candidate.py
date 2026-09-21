import torch
from torch.utils.cpp_extension import load_inline

_CUDA = r'''
#include <torch/extension.h>
#include <cuda_bf16.h>

namespace {

__device__ __forceinline__ float warp_reduce_sum(float value) {
    for (int delta = 16; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, delta);
    }
    return value;
}

__global__ void rmsnorm_forward_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int64_t rows,
    int64_t hidden,
    float eps) {
    __shared__ float warp_totals[8];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) return;

    const int64_t row_start = row * hidden;
    float sum_sq = 0.0f;
    for (int64_t col = tid; col < hidden; col += blockDim.x) {
        const float value = __bfloat162float(x[row_start + col]);
        sum_sq = fmaf(value, value, sum_sq);
    }

    sum_sq = warp_reduce_sum(sum_sq);
    if (lane == 0) warp_totals[warp] = sum_sq;
    __syncthreads();

    if (warp == 0) {
        float block_sum = (tid < 8) ? warp_totals[tid] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (tid == 0) warp_totals[0] = block_sum;
    }
    __syncthreads();

    const float inv_rms = rsqrtf(warp_totals[0] / static_cast<float>(hidden) + eps);
    for (int64_t col = tid; col < hidden; col += blockDim.x) {
        const float value = __bfloat162float(x[row_start + col]);
        const float scale = __bfloat162float(weight[col]);
        out[row_start + col] = __float2bfloat16_rn(value * inv_rms * scale);
    }
}

}  // namespace

void rmsnorm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor out, double eps) {
    const int64_t rows = x.size(0);
    const int64_t hidden = x.size(1);
    rmsnorm_forward_kernel<<<static_cast<unsigned int>(rows), 256>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        rows,
        hidden,
        static_cast<float>(eps));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_cuda", &rmsnorm_cuda, "RMSNorm forward CUDA");
}
'''

_rmsnorm_ext = load_inline(
    name="rmsnorm_episode_3_warp_reduce",
    cpp_sources="",
    cuda_sources=_CUDA,
    functions=None,
    extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-Xcompiler=/Zc:preprocessor"],
    with_cuda=True,
    verbose=False,
)


class Model(torch.nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        _rmsnorm_ext.rmsnorm_cuda(x, weight, out, self.eps)
        return out
