import os
import torch
from torch.utils.cpp_extension import load_inline


# This is the fixed target GPU for the frozen Human Challenge.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")


_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> rms_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    double eps);

std::vector<torch::Tensor> rms_backward_cuda(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor inv_rms);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rms_forward_cuda, "RMSNorm forward (CUDA)");
    m.def("backward", &rms_backward_cuda, "RMSNorm backward (CUDA)");
}
"""


_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <vector>

namespace {

constexpr int HIDDEN = 1024;
constexpr int BLOCK = 256;
constexpr int ITEMS_PER_THREAD = HIDDEN / BLOCK;  // 4


__device__ __forceinline__ float warp_reduce_sum(float value) {
    value += __shfl_down_sync(0xffffffff, value, 16);
    value += __shfl_down_sync(0xffffffff, value, 8);
    value += __shfl_down_sync(0xffffffff, value, 4);
    value += __shfl_down_sync(0xffffffff, value, 2);
    value += __shfl_down_sync(0xffffffff, value, 1);
    return value;
}


__device__ __forceinline__ float block_reduce_sum(float value) {
    __shared__ float warp_sums[32];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    constexpr int NUM_WARPS = BLOCK / 32;

    value = warp_reduce_sum(value);

    if (lane == 0) {
        warp_sums[warp] = value;
    }

    __syncthreads();

    if (warp == 0) {
        float warp_value =
            (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;

        warp_value = warp_reduce_sum(warp_value);

        if (lane == 0) {
            warp_sums[0] = warp_value;
        }
    }

    __syncthreads();

    return warp_sums[0];
}


__global__ void rms_forward_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ y,
    float* __restrict__ inv_rms,
    int rows,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    const int row_base = row * HIDDEN;

    // Keep x alive across the reduction so that x is read from
    // global memory only once.
    float xv[ITEMS_PER_THREAD];

    float local_sum_sq = 0.0f;

    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
        const int col = tid + k * BLOCK;
        const float value = __half2float(x[row_base + col]);

        xv[k] = value;
        local_sum_sq += value * value;
    }

    const float sum_sq = block_reduce_sum(local_sum_sq);

    __shared__ float q_shared;

    if (tid == 0) {
        const float mean_sq = sum_sq * (1.0f / float(HIDDEN));
        q_shared = rsqrtf(mean_sq + eps);
        inv_rms[row] = q_shared;
    }

    __syncthreads();

    const float q = q_shared;

    // Reuse the x values already resident in registers.
    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
        const int col = tid + k * BLOCK;

        const float w = __half2float(weight[col]);
        const float out = xv[k] * w * q;

        y[row_base + col] = __float2half_rn(out);
    }
}


__global__ void rms_backward_kernel(
    const half* __restrict__ grad_y,
    const half* __restrict__ x,
    const half* __restrict__ weight,
    const float* __restrict__ inv_rms,
    half* __restrict__ grad_x,
    float* __restrict__ grad_weight_fp32,
    int rows)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    const int row_base = row * HIDDEN;
    const float q = inv_rms[row];

    // After the initial loads, only x and (g*w) need to remain
    // live across the row reduction.
    float xv[ITEMS_PER_THREAD];
    float gwv[ITEMS_PER_THREAD];

    float local_dot = 0.0f;

    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
        const int col = tid + k * BLOCK;
        const int idx = row_base + col;

        const float xx = __half2float(x[idx]);
        const float gg = __half2float(grad_y[idx]);
        const float ww = __half2float(weight[col]);

        const float gw = gg * ww;

        xv[k] = xx;
        gwv[k] = gw;

        // Contribution to:
        //
        // s = sum_i(g_i * w_i * x_i)
        local_dot += gw * xx;

        // Contribution to:
        //
        // dw_i = sum_rows(g_i * x_i * inv_rms_row)
        //
        // FP32 accumulation is intentional.  The final result is
        // cast back to the weight dtype after the kernel.
        atomicAdd(
            grad_weight_fp32 + col,
            gg * xx * q
        );
    }

    const float s = block_reduce_sum(local_dot);

    // dx_i =
    // q * (g_i*w_i)
    // - x_i * (q^3 / H) * sum_j(g_j*w_j*x_j)
    const float coeff =
        (s * (1.0f / float(HIDDEN))) * q * q * q;

    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
        const int col = tid + k * BLOCK;
        const int idx = row_base + col;

        const float dx =
            q * gwv[k]
            - xv[k] * coeff;

        grad_x[idx] = __float2half_rn(dx);
    }
}

} // namespace


std::vector<torch::Tensor> rms_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    double eps)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16,
                "x must be float16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16,
                "weight must be float16");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(x.size(-1) == HIDDEN,
                "hidden size must be 1024");
    TORCH_CHECK(weight.numel() == HIDDEN,
                "weight size must be 1024");

    const int64_t rows64 = x.numel() / HIDDEN;
    TORCH_CHECK(rows64 > 0 && rows64 <= INT_MAX,
                "invalid row count");

    const int rows = static_cast<int>(rows64);

    auto y = torch::empty_like(x);

    auto inv_rms = torch::empty(
        {rows},
        x.options().dtype(torch::kFloat32)
    );

    rms_forward_kernel<<<
        rows,
        BLOCK,
        0,
        at::cuda::getCurrentCUDAStream()
    >>>(
        reinterpret_cast<const half*>(
            x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(
            weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(
            y.data_ptr<at::Half>()),
        inv_rms.data_ptr<float>(),
        rows,
        static_cast<float>(eps)
    );

    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "rms_forward_kernel launch failed: ",
        cudaGetErrorString(err)
    );

    return {y, inv_rms};
}


std::vector<torch::Tensor> rms_backward_cuda(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor inv_rms)
{
    TORCH_CHECK(grad_y.is_cuda(), "grad_y must be CUDA");
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(inv_rms.is_cuda(), "inv_rms must be CUDA");

    TORCH_CHECK(
        grad_y.scalar_type() == torch::kFloat16,
        "grad_y must be float16"
    );
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat16,
        "x must be float16"
    );
    TORCH_CHECK(
        weight.scalar_type() == torch::kFloat16,
        "weight must be float16"
    );
    TORCH_CHECK(
        inv_rms.scalar_type() == torch::kFloat32,
        "inv_rms must be float32"
    );

    TORCH_CHECK(grad_y.is_contiguous(),
                "grad_y must be contiguous");
    TORCH_CHECK(x.is_contiguous(),
                "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(),
                "weight must be contiguous");

    TORCH_CHECK(x.size(-1) == HIDDEN,
                "hidden size must be 1024");
    TORCH_CHECK(grad_y.sizes() == x.sizes(),
                "grad_y shape mismatch");

    const int64_t rows64 = x.numel() / HIDDEN;
    TORCH_CHECK(rows64 > 0 && rows64 <= INT_MAX,
                "invalid row count");

    const int rows = static_cast<int>(rows64);

    TORCH_CHECK(inv_rms.numel() == rows,
                "inv_rms row count mismatch");

    auto grad_x = torch::empty_like(x);

    // Accumulate dweight in FP32 to avoid FP16 atomic accumulation.
    auto grad_weight_fp32 = torch::zeros(
        {HIDDEN},
        x.options().dtype(torch::kFloat32)
    );

    rms_backward_kernel<<<
        rows,
        BLOCK,
        0,
        at::cuda::getCurrentCUDAStream()
    >>>(
        reinterpret_cast<const half*>(
            grad_y.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(
            x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(
            weight.data_ptr<at::Half>()),
        inv_rms.data_ptr<float>(),
        reinterpret_cast<half*>(
            grad_x.data_ptr<at::Half>()),
        grad_weight_fp32.data_ptr<float>(),
        rows
    );

    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "rms_backward_kernel launch failed: ",
        cudaGetErrorString(err)
    );

    // Parameter is FP16 in the frozen contract, so return an FP16
    // gradient after FP32 accumulation.
    auto grad_weight =
        grad_weight_fp32.to(weight.scalar_type());

    return {grad_x, grad_weight};
}
"""


_ext = load_inline(
    name="human_rmsnorm_v1_ext",
    cpp_sources=[_CPP_SRC],
    cuda_sources=[_CUDA_SRC],
    functions=None,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class _HumanRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        # Frozen challenge inputs are expected to be contiguous.
        # Keep a safe path for framework integration without using
        # the reference RMSNorm implementation.
        x_c = x if x.is_contiguous() else x.contiguous()
        w_c = weight if weight.is_contiguous() else weight.contiguous()

        y, inv_rms = _ext.forward(
            x_c,
            w_c,
            float(eps)
        )

        ctx.save_for_backward(
            x_c,
            w_c,
            inv_rms
        )

        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, weight, inv_rms = ctx.saved_tensors

        g_c = (
            grad_y
            if grad_y.is_contiguous()
            else grad_y.contiguous()
        )

        grad_x, grad_weight = _ext.backward(
            g_c,
            x,
            weight,
            inv_rms
        )

        return grad_x, grad_weight, None


class TritonRMSNorm(torch.nn.Module):
    """
    Frozen Human Challenge ABI.

    The historical class name is TritonRMSNorm, but this v1 candidate
    intentionally uses a native CUDA extension internally.
    """

    def __init__(
        self,
        hidden_size,
        eps=1.0e-5,
        num_warps=4,
    ):
        super().__init__()

        if int(hidden_size) != 1024:
            raise ValueError(
                f"Human v1 is specialized for hidden_size=1024, "
                f"got {hidden_size}"
            )

        self.hidden_size = int(hidden_size)
        self.eps = float(eps)
        self.num_warps = int(num_warps)

        self.weight = torch.nn.Parameter(
            torch.ones(self.hidden_size)
        )

        self.invocation_count = 0

    def forward(self, x):
        self.invocation_count += 1

        return _HumanRMSNormFunction.apply(
            x,
            self.weight,
            self.eps,
        )
