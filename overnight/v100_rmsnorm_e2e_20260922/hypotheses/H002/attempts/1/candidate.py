"""H002 attempt 1: fuse dx production with atomic dweight accumulation."""

import torch
import triton
import triton.language as tl

from candidate_base import _rms_fwd


@triton.jit
def _rms_dx_dw_atomic(x_ptr, w_ptr, g_ptr, r_ptr, dx_ptr, dw_accum_ptr,
                      hidden: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, block)
    mask = cols < hidden
    x = tl.load(x_ptr + row * hidden + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + row * hidden + cols, mask=mask, other=0.0).to(tl.float32)
    inv_rms = tl.load(r_ptr + row).to(tl.float32)
    contribution = g * x * inv_rms
    projection = tl.sum(g * w * x, axis=0)
    dx = g * w * inv_rms - x * (inv_rms * inv_rms * inv_rms) * projection / hidden
    tl.store(dx_ptr + row * hidden + cols, dx, mask=mask)
    tl.atomic_add(dw_accum_ptr + cols, contribution, mask=mask)


class _AtomicRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        hidden = x.shape[-1]
        if hidden != 1024 or x.dtype != torch.float16 or not x.is_contiguous():
            raise ValueError("H002 requires contiguous fp16 CUDA input with H=1024")
        rows = x.numel() // hidden
        y = torch.empty_like(x)
        inv_rms = torch.empty(rows, device=x.device, dtype=torch.float32)
        _rms_fwd[(rows,)](x, weight, y, inv_rms, rows=rows, hidden=hidden, eps=eps,
                          block=1024, num_warps=8)
        ctx.save_for_backward(x, weight, inv_rms)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, inv_rms = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        hidden = x.shape[-1]
        rows = x.numel() // hidden
        dx = torch.empty_like(x)
        dw_accum = torch.zeros(hidden, device=x.device, dtype=torch.float32)
        _rms_dx_dw_atomic[(rows,)](x, weight, grad_output, inv_rms, dx, dw_accum,
                                   hidden=hidden, block=1024, num_warps=8)
        return dx, dw_accum.to(weight.dtype), None


class TritonRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size=1024, eps=1.0e-5, num_warps=8):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)
        self.invocation_count = 0

    def forward(self, x):
        self.invocation_count += 1
        return _AtomicRMSNormFunction.apply(x, self.weight, self.eps)
