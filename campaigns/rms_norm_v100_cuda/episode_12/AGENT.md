# AGENT.md

Implemented a standalone CUDA RMSNorm kernel for Tesla V100 (sm_70).

- Added `candidate.cu` with the required `extern "C" void launch_kernel(...)` entry point.
- Uses one 256-thread block per batch row.
- Computes the sum of squares with coalesced loads, warp shuffle reduction, and 32-entry shared memory for warp totals.
- Uses `rsqrtf(sum_sq / hidden + eps)` and a second coalesced pass to write normalized values multiplied by the per-feature weight.
- Keeps the implementation self-contained and independent of Python, PyTorch, and architecture-specific features newer than sm_70.
- Added `hypothesis.json` describing the optimization and expected Volta effect.

No CUDA compilation or benchmark was run, per instructions.
