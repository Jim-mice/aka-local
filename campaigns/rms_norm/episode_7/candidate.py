import torch
from torch.utils.cpp_extension import load_inline

# Windows-native CUDA extension for contiguous 2-D RMSNorm: [rows, hidden].
# Compilation is intentionally deferred until the function is first called.
_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

template <typename T>
__device__ __forceinline__ float to_float(T x) {
    return static_cast<float>(x);
}

template <typename T>
__device__ __forceinline__ T from_float(float x) {
    return static_cast<T>(x);
}
__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

template <typename scalar_t>
__global__ void rmsnorm_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int64_t hidden,
    float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const scalar_t* row_in = input + static_cast<int64_t>(row) * hidden;
    scalar_t* row_out = output + static_cast<int64_t>(row) * hidden;

    float sum = 0.0f;
    // Register accumulation plus warp shuffles avoids a block-wide reduction
    // in the hot path while retaining coalesced, grid-stride loads.
    for (int64_t col = tid; col < hidden; col += blockDim.x) {
        const float v = to_float(row_in[col]);
        sum = fmaf(v, v, sum);
    }
    sum = warp_sum(sum);

    __shared__ float warp_sums[8];
    if (lane == 0) {
        warp_sums[warp] = sum;
    }
    __syncthreads();

    float mean_square = 0.0f;
    if (warp == 0) {
        mean_square = (lane < (blockDim.x + 31) / 32) ? warp_sums[lane] : 0.0f;
        mean_square = warp_sum(mean_square);
        if (lane == 0) {
            warp_sums[0] = mean_square / static_cast<float>(hidden);
        }
    }
    __syncthreads();

    const float inv_rms = rsqrtf(warp_sums[0] + eps);
    for (int64_t col = tid; col < hidden; col += blockDim.x) {
        const float x = to_float(row_in[col]);
        const float g = to_float(weight[col]);
        row_out[col] = from_float<scalar_t>(x * inv_rms * g);
    }
}

void rmsnorm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor output, double eps) {
    const auto rows = input.size(0);
    const auto hidden = input.size(1);
    constexpr int threads = 256;
    const dim3 blocks(static_cast<unsigned int>(rows));
    const auto stream = at::cuda::getDefaultCUDAStream(input.device().index());
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
        "rmsnorm_cuda", [&] {
            rmsnorm_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(), hidden, static_cast<float>(eps));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_cuda", &rmsnorm_cuda, "RMSNorm CUDA kernel");
}
'''

_ext = None

def _extension():
    global _ext
    if _ext is None:
        _ext = load_inline(
            name="rmsnorm_sm120_episode7",
            cpp_sources="",
            cuda_sources=_CUDA_SRC,
            functions=None,
            with_cuda=True,
            extra_cuda_cflags=[
                "-gencode=arch=compute_120,code=sm_120",
                "--expt-relaxed-constexpr",
            ],
            verbose=False,
        )
    return _ext


def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Contiguous 2-D CUDA RMSNorm with one 256-thread block per row."""
    if not input.is_cuda or not weight.is_cuda:
        raise ValueError("input and weight must be CUDA tensors")
    if input.dim() != 2 or weight.dim() != 1:
        raise ValueError("expected input [rows, hidden] and weight [hidden]")
    if input.size(1) != weight.numel():
        raise ValueError("weight length must equal input hidden size")
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("supported dtypes are float16, bfloat16, and float32")
    if weight.dtype != input.dtype:
        raise TypeError("input and weight must have the same dtype")
    x = input.contiguous()
    g = weight.contiguous()
    out = torch.empty_like(x)
    _extension().rmsnorm_cuda(x, g, out, float(eps))
    return out


__all__ = ["rms_norm"]

