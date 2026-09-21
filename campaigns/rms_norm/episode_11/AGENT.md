# AGENT.md

## Change summary

Created the required `candidate.py`, `hypothesis.json`, and `AGENT.md` files for the Windows RTX 5060 Laptop RMSNorm candidate.

## Implementation

- Added `Model(torch.nn.Module)` with `__init__(eps: float = 1e-6)` and the required `forward(x, weight)` signature.
- Added module-level lazy extension state via `_rmsnorm_ext = None`.
- Compiled the CUDA extension with `torch.utils.cpp_extension.load_inline` and the required `rmsnorm_cuda` entry point.
- Implemented one CUDA block per logical row, register-based sum-of-squares accumulation, warp-shuffle reduction, and shared-memory staging for warp totals.
- Added an in-place-to-output kernel path using a preallocated contiguous output tensor, while preserving the input shape for tensors with arbitrary leading dimensions.
- Selected Windows-compatible CUDA flags and explicitly targeted `sm_120a`.

## Rationale

The design follows the successful prior direction of warp reduction, shared memory, and register accumulation. It avoids introducing Linux-only compiler flags and avoids the previously unsuccessful techniques listed in the campaign context. No benchmarks or CUDA execution were run, as required.
