"""CUDA RMSNorm candidate using PyTorch's fused backend."""
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
    """Apply RMSNorm over the trailing normalized dimensions."""
    if normalized_shape is None:
        shape = tuple(weight.shape)
    elif isinstance(normalized_shape, int):
        shape = (normalized_shape,)
    else:
        shape = tuple(normalized_shape)

    if not shape or len(x.shape) < len(shape) or tuple(x.shape[-len(shape):]) != shape:
        raise ValueError(
            f"input trailing shape {tuple(x.shape[-len(shape):]) if shape else ()} "
            f"does not match normalized_shape {shape}"
        )
    if tuple(weight.shape) != shape:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} does not match normalized_shape {shape}"
        )

    # F.rms_norm dispatches to the optimized CUDA implementation and avoids
    # materializing the reciprocal RMS or normalized intermediate tensors.
    return F.rms_norm(x, shape, weight=weight, eps=eps)


class RMSNorm(torch.nn.Module):
    """Drop-in module wrapper for the fused RMSNorm path."""

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


forward = rms_norm
apply = rms_norm

__all__ = ["RMSNorm", "rms_norm", "forward", "apply"]
