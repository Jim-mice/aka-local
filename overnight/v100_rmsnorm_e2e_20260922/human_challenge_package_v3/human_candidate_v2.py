import os
import torch
from torch.utils.cpp_extension import load_inline


# Frozen challenge target: Tesla V100, sm_70.
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
    m.def("forward", &rms_forward_cuda, "RMSNorm forward CUDA");
    m.def("backward", &rms_backward_cuda, "RMSNorm backward CUDA");
}
"""


_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <climits>
#include <vector>

namespace {

constexpr int HIDDEN = 1024;
constexpr int BLOCK = 256;
constexpr int ITEMS_PER_THREAD = HIDDEN / BLOCK; // 4

// Human v2:
// each dweight partial block covers 8 rows x 256 columns.
constexpr int ROW_TILE = 8;


__device__ __forceinline__
float warp_reduce_sum(float value) {
    value += __shfl_down_sync(0xffffffff, value, 16);
    value += __shfl_down_sync(0xffffffff, value, 8);
    value += __shfl_down_sync(0xffffffff, value, 4);
    value += __shfl_down_sync(0xffffffff, value, 2);
    value += __shfl_down_sync(0xffffffff, value, 1);
    return value;
}


__device__ __forceinline__
float block_reduce_sum(float value) {
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
        float v =
            (lane < NUM_WARPS)
            ? warp_sums[lane]
            : 0.0f;

        v = warp_reduce_sum(v);

        if (lane == 0) {
            warp_sums[0] = v;
        }
    }

    __syncthreads();

    return warp_sums[0];
}


// ============================================================
// Forward
//
// Intentionally kept structurally the same as Human v1:
// one block per row, four elements per thread,
// keep x alive across the reduction,
// save inv_rms for backward.
// ============================================================

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

    float xv[ITEMS_PER_THREAD];

    float local_sum_sq = 0.0f;

    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
        const int col = tid + k * BLOCK;

        const float value =
            __half2float(x[row_base + col]);

        xv[k] = value;
        local_sum_sq += value * value;
    }

    const float sum_sq =
        block_reduce_sum(local_sum_sq);

    __shared__ float q_shared;

    if (tid == 0) {
        const float mean_sq =
            sum_sq * (1.0f / float(HIDDEN));

        q_shared =
            rsqrtf(mean_sq + eps);

        inv_rms[row] = q_shared;
    }

    __syncthreads();

    const float q = q_shared;

    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
        const int col = tid + k * BLOCK;

        const float w =
            __half2float(weight[col]);

        y[row_base + col] =
            __float2half_rn(
                xv[k] * w * q
            );
    }
}


// ============================================================
// Backward kernel A: grad_x only
//
// One block per row.
// No dweight atomic operations.
//
// s = sum_i(g_i * w_i * x_i)
//
// dx_i =
// q * (g_i * w_i)
// - x_i * (q^3 / H) * s
//
// Only x and g*w remain live across the row reduction.
// ============================================================

__global__ void rms_dx_kernel(
    const half* __restrict__ grad_y,
    const half* __restrict__ x,
    const half* __restrict__ weight,
    const float* __restrict__ inv_rms,
    half* __restrict__ grad_x,
    int rows)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    const int row_base = row * HIDDEN;

    const float q = inv_rms[row];

    float xv[ITEMS_PER_THREAD];
    float gwv[ITEMS_PER_THREAD];

    float local_dot = 0.0f;

    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
        const int col = tid + k * BLOCK;
        const int idx = row_base + col;

        const float xx =
            __half2float(x[idx]);

        const float gg =
            __half2float(grad_y[idx]);

        const float ww =
            __half2float(weight[col]);

        const float gw = gg * ww;

        xv[k] = xx;
        gwv[k] = gw;

        local_dot += gw * xx;
    }

    const float s =
        block_reduce_sum(local_dot);

    const float coeff =
        (s * (1.0f / float(HIDDEN)))
        * q * q * q;

    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
        const int col = tid + k * BLOCK;
        const int idx = row_base + col;

        const float dx =
            q * gwv[k]
            - xv[k] * coeff;

        grad_x[idx] =
            __float2half_rn(dx);
    }
}


// ============================================================
// Backward kernel B: tiled partial dweight
//
// dw[col] = sum_row(
//     grad_y[row,col]
//     * x[row,col]
//     * inv_rms[row]
// )
//
// Grid:
//   x dimension -> 256-column tiles
//   y dimension -> ROW_TILE-row chunks
//
// Each thread owns exactly one column inside its 256-column tile
// and accumulates ROW_TILE rows in one FP32 register.
//
// No atomics.
// No row-wise reduction.
// Coalesced x/g loads for every row iteration.
// ============================================================

__global__ void rms_dw_partial_kernel(
    const half* __restrict__ grad_y,
    const half* __restrict__ x,
    const float* __restrict__ inv_rms,
    float* __restrict__ partial_dw,
    int rows,
    int num_row_tiles)
{
    const int tid = threadIdx.x;

    const int col =
        blockIdx.x * BLOCK + tid;

    const int row_tile =
        blockIdx.y;

    const int row_begin =
        row_tile * ROW_TILE;

    __shared__ float q_shared[ROW_TILE];

    // Only the first ROW_TILE threads load q values.
    if (tid < ROW_TILE) {
        const int row =
            row_begin + tid;

        q_shared[tid] =
            (row < rows)
            ? inv_rms[row]
            : 0.0f;
    }

    __syncthreads();

    if (col >= HIDDEN) {
        return;
    }

    float acc = 0.0f;

    #pragma unroll
    for (int rr = 0; rr < ROW_TILE; ++rr) {
        const int row =
            row_begin + rr;

        if (row < rows) {
            const int idx =
                row * HIDDEN + col;

            const float xx =
                __half2float(x[idx]);

            const float gg =
                __half2float(grad_y[idx]);

            acc +=
                gg * xx * q_shared[rr];
        }
    }

    // Every partial element has a unique writer.
    // Therefore torch::empty is sufficient; no zeroing is needed.
    partial_dw[
        row_tile * HIDDEN + col
    ] = acc;
}


// ============================================================
// Backward kernel C: final dweight reduction
//
// One thread owns one weight column.
//
// Reads all row-tile partials into one FP32 accumulator,
// then directly writes FP16 grad_weight.
//
// Thus:
// - no global atomics
// - no zero/fill kernel
// - no separate FP32 -> FP16 cast kernel
// ============================================================

__global__ void rms_dw_finalize_kernel(
    const float* __restrict__ partial_dw,
    half* __restrict__ grad_weight,
    int num_row_tiles)
{
    const int col =
        blockIdx.x * BLOCK + threadIdx.x;

    if (col >= HIDDEN) {
        return;
    }

    float acc = 0.0f;

    for (int tile = 0;
         tile < num_row_tiles;
         ++tile)
    {
        acc +=
            partial_dw[
                tile * HIDDEN + col
            ];
    }

    grad_weight[col] =
        __float2half_rn(acc);
}

} // namespace


std::vector<torch::Tensor>
rms_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    double eps)
{
    TORCH_CHECK(
        x.is_cuda(),
        "x must be CUDA"
    );

    TORCH_CHECK(
        weight.is_cuda(),
        "weight must be CUDA"
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
        x.is_contiguous(),
        "x must be contiguous"
    );

    TORCH_CHECK(
        weight.is_contiguous(),
        "weight must be contiguous"
    );

    TORCH_CHECK(
        x.size(-1) == HIDDEN,
        "hidden size must be 1024"
    );

    TORCH_CHECK(
        weight.numel() == HIDDEN,
        "weight size must be 1024"
    );

    const int64_t rows64 =
        x.numel() / HIDDEN;

    TORCH_CHECK(
        rows64 > 0 &&
        rows64 <= INT_MAX,
        "invalid row count"
    );

    const int rows =
        static_cast<int>(rows64);

    auto y =
        torch::empty_like(x);

    auto inv_rms =
        torch::empty(
            {rows},
            x.options().dtype(
                torch::kFloat32
            )
        );

    rms_forward_kernel<<<
        rows,
        BLOCK,
        0,
        at::cuda::getCurrentCUDAStream()
    >>>(
        reinterpret_cast<const half*>(
            x.data_ptr<at::Half>()
        ),
        reinterpret_cast<const half*>(
            weight.data_ptr<at::Half>()
        ),
        reinterpret_cast<half*>(
            y.data_ptr<at::Half>()
        ),
        inv_rms.data_ptr<float>(),
        rows,
        static_cast<float>(eps)
    );

    cudaError_t err =
        cudaGetLastError();

    TORCH_CHECK(
        err == cudaSuccess,
        "rms_forward_kernel launch failed: ",
        cudaGetErrorString(err)
    );

    return {
        y,
        inv_rms
    };
}


std::vector<torch::Tensor>
rms_backward_cuda(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor inv_rms)
{
    TORCH_CHECK(
        grad_y.is_cuda(),
        "grad_y must be CUDA"
    );

    TORCH_CHECK(
        x.is_cuda(),
        "x must be CUDA"
    );

    TORCH_CHECK(
        weight.is_cuda(),
        "weight must be CUDA"
    );

    TORCH_CHECK(
        inv_rms.is_cuda(),
        "inv_rms must be CUDA"
    );

    TORCH_CHECK(
        grad_y.scalar_type()
            == torch::kFloat16,
        "grad_y must be float16"
    );

    TORCH_CHECK(
        x.scalar_type()
            == torch::kFloat16,
        "x must be float16"
    );

    TORCH_CHECK(
        weight.scalar_type()
            == torch::kFloat16,
        "weight must be float16"
    );

    TORCH_CHECK(
        inv_rms.scalar_type()
            == torch::kFloat32,
        "inv_rms must be float32"
    );

    TORCH_CHECK(
        grad_y.is_contiguous(),
        "grad_y must be contiguous"
    );

    TORCH_CHECK(
        x.is_contiguous(),
        "x must be contiguous"
    );

    TORCH_CHECK(
        weight.is_contiguous(),
        "weight must be contiguous"
    );

    TORCH_CHECK(
        grad_y.sizes() == x.sizes(),
        "grad_y shape mismatch"
    );

    TORCH_CHECK(
        x.size(-1) == HIDDEN,
        "hidden size must be 1024"
    );

    const int64_t rows64 =
        x.numel() / HIDDEN;

    TORCH_CHECK(
        rows64 > 0 &&
        rows64 <= INT_MAX,
        "invalid row count"
    );

    const int rows =
        static_cast<int>(rows64);

    TORCH_CHECK(
        inv_rms.numel() == rows,
        "inv_rms row count mismatch"
    );

    const int num_row_tiles =
        (rows + ROW_TILE - 1)
        / ROW_TILE;

    auto grad_x =
        torch::empty_like(x);

    // No memset:
    // every partial element is overwritten by exactly one thread.
    auto partial_dw =
        torch::empty(
            {num_row_tiles, HIDDEN},
            x.options().dtype(
                torch::kFloat32
            )
        );

    // Direct final FP16 destination.
    // No separate FP32->FP16 .to() kernel.
    auto grad_weight =
        torch::empty_like(weight);

    // --------------------------------------------------------
    // Kernel A: dx
    // --------------------------------------------------------

    rms_dx_kernel<<<
        rows,
        BLOCK,
        0,
        at::cuda::getCurrentCUDAStream()
    >>>(
        reinterpret_cast<const half*>(
            grad_y.data_ptr<at::Half>()
        ),
        reinterpret_cast<const half*>(
            x.data_ptr<at::Half>()
        ),
        reinterpret_cast<const half*>(
            weight.data_ptr<at::Half>()
        ),
        inv_rms.data_ptr<float>(),
        reinterpret_cast<half*>(
            grad_x.data_ptr<at::Half>()
        ),
        rows
    );

    cudaError_t err =
        cudaGetLastError();

    TORCH_CHECK(
        err == cudaSuccess,
        "rms_dx_kernel launch failed: ",
        cudaGetErrorString(err)
    );

    // --------------------------------------------------------
    // Kernel B: partial dweight
    // --------------------------------------------------------

    const dim3 dw_grid(
        (HIDDEN + BLOCK - 1) / BLOCK,
        num_row_tiles,
        1
    );

    rms_dw_partial_kernel<<<
        dw_grid,
        BLOCK,
        0,
        at::cuda::getCurrentCUDAStream()
    >>>(
        reinterpret_cast<const half*>(
            grad_y.data_ptr<at::Half>()
        ),
        reinterpret_cast<const half*>(
            x.data_ptr<at::Half>()
        ),
        inv_rms.data_ptr<float>(),
        partial_dw.data_ptr<float>(),
        rows,
        num_row_tiles
    );

    err = cudaGetLastError();

    TORCH_CHECK(
        err == cudaSuccess,
        "rms_dw_partial_kernel launch failed: ",
        cudaGetErrorString(err)
    );

    // --------------------------------------------------------
    // Kernel C: final dweight reduction
    // --------------------------------------------------------

    const int weight_blocks =
        (HIDDEN + BLOCK - 1)
        / BLOCK;

    rms_dw_finalize_kernel<<<
        weight_blocks,
        BLOCK,
        0,
        at::cuda::getCurrentCUDAStream()
    >>>(
        partial_dw.data_ptr<float>(),
        reinterpret_cast<half*>(
            grad_weight.data_ptr<at::Half>()
        ),
        num_row_tiles
    );

    err = cudaGetLastError();

    TORCH_CHECK(
        err == cudaSuccess,
        "rms_dw_finalize_kernel launch failed: ",
        cudaGetErrorString(err)
    );

    return {
        grad_x,
        grad_weight
    };
}
"""


_ext = load_inline(
    name="human_rmsnorm_v2_ext",
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

        x_c = (
            x
            if x.is_contiguous()
            else x.contiguous()
        )

        w_c = (
            weight
            if weight.is_contiguous()
            else weight.contiguous()
        )

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

        x, weight, inv_rms =
            ctx.saved_tensors

        g_c = (
            grad_y
            if grad_y.is_contiguous()
            else grad_y.contiguous()
        )

        grad_x, grad_weight =
            _ext.backward(
                g_c,
                x,
                weight,
                inv_rms
            )

        return (
            grad_x,
            grad_weight,
            None
        )


class TritonRMSNorm(torch.nn.Module):
    """
    Snapshot-C Human Challenge ABI.

    Historical ABI class name is TritonRMSNorm.

    Human v2 internally uses native CUDA.

    Relative to Human v1:
      - forward strategy intentionally unchanged
      - dx separated from dweight
      - no dweight global atomics
      - no dweight memset kernel
      - no separate FP32->FP16 cast kernel
    """

    def __init__(
        self,
        hidden_size,
        eps=1.0e-5,
        num_warps=4
    ):
        super().__init__()

        if int(hidden_size) != HIDDEN_PY:
            raise ValueError(
                "Human v2 is specialized "
                "for hidden_size=1024, got "
                + str(hidden_size)
            )

        self.hidden_size =
            int(hidden_size)

        self.eps =
            float(eps)

        self.num_warps =
            int(num_warps)

        self.weight =
            torch.nn.Parameter(
                torch.ones(
                    self.hidden_size
                )
            )

        self.invocation_count = 0


    def forward(self, x):

        self.invocation_count += 1

        return _HumanRMSNormFunction.apply(
            x,
            self.weight,
            self.eps
        )


HIDDEN_PY = 1024