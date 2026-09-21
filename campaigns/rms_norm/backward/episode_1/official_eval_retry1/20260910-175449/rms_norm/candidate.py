import torch
import torch.utils.cpp_extension as cpp_extension
from torch.utils.cpp_extension import load_inline


# Windows MSVC version probing can emit OEM-encoded bytes; preserve the
# candidate kernel while matching the already-validated local loader behavior.
cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")


_CUDA = r'''
#include <torch/extension.h>
#include <cuda_bf16.h>

__global__ void rmsnorm_row_kernel(const __nv_bfloat16* __restrict__ x,
                                   const __nv_bfloat16* __restrict__ weight,
                                   __nv_bfloat16* __restrict__ out,
                                   int64_t rows, int64_t hidden, float eps) {
    __shared__ float partial[256];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;
    const int64_t base = static_cast<int64_t>(row) * hidden;

    float sum = 0.0f;
    for (int64_t col = tid; col < hidden; col += blockDim.x) {
        const float v = __bfloat162float(x[base + col]);
        sum = fmaf(v, v, sum);
    }
    partial[tid] = sum;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(hidden) + eps);
    for (int64_t col = tid; col < hidden; col += blockDim.x) {
        const float v = __bfloat162float(x[base + col]);
        const float w = __bfloat162float(weight[col]);
        out[base + col] = __float2bfloat16_rn(v * inv_rms * w);
    }
}

void rmsnorm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor out,
                  double eps) {
    const auto rows = x.size(0);
    const auto hidden = x.size(1);
    rmsnorm_row_kernel<<<static_cast<unsigned>(rows), 256>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        rows, hidden, static_cast<float>(eps));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_cuda", &rmsnorm_cuda, "fused RMSNorm CUDA");
}
'''


_rmsnorm_ext = load_inline(
    name="rmsnorm_episode_1_fused",
    cpp_sources="",
    cuda_sources=_CUDA,
    functions=None,
    extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
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
