# AGENT.md

## Change summary

Implemented a standalone CUDA softmax kernel for Tesla V100 (sm_70) in `candidate.cu`.

- Launches one 256-thread block per input row.
- Uses coalesced, strided row access so all tested row widths are covered.
- Performs a numerically stable two-pass softmax: row maximum, then exponent sum after subtracting that maximum.
- Uses warp shuffle reductions for both maximum and sum, with only one small shared-memory array for the eight warp results.
- Uses `__expf` and reciprocal division for a throughput-oriented standalone CUDA implementation.
- Exposes the required exact `extern "C" void launch_kernel(float* x, float* y, int rows, int cols)` entry point.

No CUDA compilation or benchmark was run, as requested. Only the three requested files were created or updated.
