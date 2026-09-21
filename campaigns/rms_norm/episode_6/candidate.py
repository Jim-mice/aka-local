"""Candidate RMSNorm implementation.

This file intentionally contains no benchmark or CUDA execution on import.  The
public ``rmsnorm`` function accepts an input tensor and optional weight, and
uses PyTorch's fused pointwise machinery when available.  It is written so the
benchmark harness can call it directly or use ``RMSNorm`` as a module.
"""

from __future__ import annotations

from typing import Optional

import torch


def rmsnorm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    dim: int = -1,
) -> torch.Tensor:
    """Apply RMSNorm over one dimension without materializing centered data."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim == 0:
        raise ValueError("x must have at least one dimension")
    if dim < 0:
        dim += x.ndim
    if dim < 0 or dim >= x.ndim:
        raise ValueError("dim is out of range")
    if eps <= 0:
        raise ValueError("eps must be positive")

    # Accumulate in fp32 for low-precision inputs, then cast back once.  The
    # reduction stays on-device and avoids an intermediate normalized tensor.
    acc_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x_acc = x.to(acc_dtype) if x.dtype != acc_dtype else x
    inv_rms = torch.rsqrt(torch.mean(x_acc * x_acc, dim=dim, keepdim=True) + eps)
    y = (x_acc * inv_rms).to(dtype=x.dtype) if x.dtype != acc_dtype else x_acc * inv_rms

    if weight is not None:
        if weight.ndim != 1 or weight.numel() != x.shape[dim]:
            raise ValueError("weight must be a vector matching the normalized dimension")
        shape = [1] * x.ndim
        shape[dim] = weight.numel()
        y = y * weight.reshape(shape).to(dtype=y.dtype)
    return y


class RMSNorm(torch.nn.Module):
    """Drop-in module wrapper for the candidate implementation."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6, *, device=None, dtype=None):
        super().__init__()
        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        self.weight = torch.nn.Parameter(torch.ones(self.normalized_shape, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.normalized_shape:
            raise ValueError("last dimension does not match normalized_shape")
        return rmsnorm(x, self.weight, self.eps, -1)


# Common harness aliases.
forward = rmsnorm
rms_norm = rmsnorm

__all__ = ["RMSNorm", "forward", "rms_norm", "rmsnorm"]
