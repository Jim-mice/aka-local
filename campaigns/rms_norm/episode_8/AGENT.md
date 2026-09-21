# Episode 8 Agent Summary

## What changed

- Added candidate.py, a lazy-built PyTorch CUDA extension implementing forward RMSNorm.
- Added hypothesis.json documenting the optimization hypothesis and constraints.

## Why

The candidate follows the previously successful warp_reduce_register_accumulation direction. Each row uses register accumulation for sum of squares, warp shuffle reduction, and one shared-memory value per warp before the final block reduction. The same kernel performs normalization and affine scaling without an intermediate tensor.

## Compatibility and boundaries

- Uses CUDA 13.4 code generation for compute_120 / sm_120.
- Uses Windows-compatible host flag /O2 and no -fopenmp or -march=native.
- Does not run benchmarks or execute CUDA work at import time; compilation is lazy and only occurs when rms_norm is called.
- No benchmark or CUDA execution was run for this episode, as requested.
