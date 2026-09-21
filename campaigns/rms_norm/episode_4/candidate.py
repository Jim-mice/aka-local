"""CUDA RMSNorm candidate.

The implementation intentionally delegates the reduction and normalization to
PyTorch's fused RMSNorm operator when available.  This keeps the hot path in a
single CUDA kernel and preserves fp32 accumulation semantics for fp16/bf16
inputs.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    normalized_shape: Optional[Sequence[int] | int] = None,
) -> torch.Tensor:
    """Apply RMSNorm over the trailing normalized dimensions.

    Args:
        x: Input tensor, normally CUDA fp16/bf16 or fp32.
        weight: Affine scale with shape ``normalized_shape``.
        eps: Numerical-stability epsilon.
        normalized_shape: Optional shape override.  Defaults to weight.shape.
    """
    if normalized_shape is None:
        normalized_shape = tuple(weight.shape)
    elif isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    else:
        normalized_shape = tuple(normalized_shape)

    if x.shape[-len(normalized_shape) :] != normalized_shape:
        raise ValueError(
            f"input trailing shape {tuple(x.shape[-len(normalized_shape):])} "
            f"does not match normalized_shape {normalized_shape}"
        )
    if tuple(weight.shape) != normalized_shape:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} does not match "
            f"normalized_shape {normalized_shape}"
        )

    # torch.nn.functional.rms_norm selects the optimized backend on CUDA.
    # It also handles dtype promotion/accumulation consistently with PyTorch.
    return F.rms_norm(x, normalized_shape, weight=weight, eps=eps)


class RMSNorm(torch.nn.Module):
    """Drop-in module wrapper for the candidate kernel path."""

    def __init__(
        self,
        normalized_shape: int | Sequence[int],
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape: Tuple[int, ...] = tuple(normalized_shape)
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.ones(self.normalized_shape, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps, self.normalized_shape)


# Common harness aliases.
forward = rms_norm
apply = rms_norm

__all__ = ["RMSNorm", "rms_norm", "forward", "apply"]
