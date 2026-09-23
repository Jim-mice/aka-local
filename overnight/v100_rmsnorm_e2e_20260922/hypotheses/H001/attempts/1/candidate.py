"""H001 attempt 1: fused Triton RMSNorm with analytic autograd."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_fwd(x_ptr, w_ptr, y_ptr, r_ptr, rows: tl.constexpr, hidden: tl.constexpr,
             eps: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, block)
    mask = cols < hidden
    x = tl.load(x_ptr + row * hidden + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) / hidden
    inv_rms = tl.rsqrt(mean_square + eps)
    tl.store(y_ptr + row * hidden + cols, x * inv_rms * w, mask=mask)
    tl.store(r_ptr + row, inv_rms)


@triton.jit
def _rms_dx(x_ptr, w_ptr, g_ptr, r_ptr, dx_ptr, hidden: tl.constexpr,
            block: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, block)
    mask = cols < hidden
    x = tl.load(x_ptr + row * hidden + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + row * hidden + cols, mask=mask, other=0.0).to(tl.float32)
    inv_rms = tl.load(r_ptr + row).to(tl.float32)
    projection = tl.sum(g * w * x, axis=0)
    dx = g * w * inv_rms - x * (inv_rms * inv_rms * inv_rms) * projection / hidden
    tl.store(dx_ptr + row * hidden + cols, dx, mask=mask)


@triton.jit
def _rms_dw(x_ptr, g_ptr, r_ptr, dw_ptr, rows: tl.constexpr, hidden: tl.constexpr,
            row_block: tl.constexpr):
    col = tl.program_id(0)
    row_offsets = tl.arange(0, row_block)
    mask = row_offsets < rows
    x = tl.load(x_ptr + row_offsets * hidden + col, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + row_offsets * hidden + col, mask=mask, other=0.0).to(tl.float32)
    inv_rms = tl.load(r_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(dw_ptr + col, tl.sum(g * x * inv_rms, axis=0))


class _RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, num_warps):
        if not x.is_cuda or x.dtype != torch.float16 or not x.is_contiguous():
            raise ValueError("candidate requires contiguous CUDA float16 input")
        if weight.dtype != torch.float16 or not weight.is_contiguous():
            raise ValueError("candidate requires contiguous float16 weight")
        hidden = x.shape[-1]
        if hidden != 1024:
            raise ValueError("H001 is specialized to frozen H=1024")
        rows = x.numel() // hidden
        y = torch.empty_like(x)
        inv_rms = torch.empty((rows,), device=x.device, dtype=torch.float32)
        _rms_fwd[(rows,)](x, weight, y, inv_rms, rows=rows, hidden=hidden,
                          eps=eps, block=triton.next_power_of_2(hidden), num_warps=num_warps)
        ctx.save_for_backward(x, weight, inv_rms)
        ctx.num_warps = num_warps
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, inv_rms = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        hidden = x.shape[-1]
        rows = x.numel() // hidden
        dx = torch.empty_like(x)
        dw = torch.empty_like(weight)
        _rms_dx[(rows,)](x, weight, grad_output, inv_rms, dx, hidden=hidden,
                         block=triton.next_power_of_2(hidden), num_warps=ctx.num_warps)
        _rms_dw[(hidden,)](x, grad_output, inv_rms, dw, rows=rows, hidden=hidden,
                           row_block=triton.next_power_of_2(rows), num_warps=4)
        return dx, dw, None, None


class TritonRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size=1024, eps=1.0e-5, num_warps=4):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)
        self.num_warps = int(num_warps)
        self.invocation_count = 0

    def forward(self, x):
        self.invocation_count += 1
        return _RMSNormFunction.apply(x, self.weight, self.eps, self.num_warps)
