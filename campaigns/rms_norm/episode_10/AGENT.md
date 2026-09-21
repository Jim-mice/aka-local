# AGENT.md

## RMSNorm candidate summary

Created the required `candidate.py`, `hypothesis.json`, and this summary for the Windows RTX 5060 Laptop / CUDA 13.4 `sm_120a` target.

### What changed

- Added `Model(torch.nn.Module)` with `__init__(eps: float = 1e-6)` and the required `forward(x, weight)` signature.
- Added module-level lazy extension state via `_rmsnorm_ext = None`.
- Used `torch.utils.cpp_extension.load_inline` to compile the CUDA extension on the first forward call.
- Implemented the required CUDA kernel function `rmsnorm_cuda`.
- Used one CUDA block per logical row, register-resident strided accumulation, warp-shuffle reduction, and a small shared array containing one sum per warp.
- Fused inverse-RMS calculation and the weighted output pass in the same kernel.
- Selected Windows-compatible compiler options and explicitly targeted `sm_120a`; no Linux-only flags were added.

### Why

Episode 3 showed a positive result from warp reduction and register accumulation. This candidate keeps that direction while reducing shared-memory use to one value per warp and avoiding a separate reduction/output kernel launch. It does not introduce route-specific or unrelated tuning parameters.

No benchmarks or CUDA code were executed, as requested.
