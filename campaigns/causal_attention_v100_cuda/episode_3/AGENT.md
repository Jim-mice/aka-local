# Agent Notes

Implemented a standalone CUDA causal scaled dot-product attention kernel for Tesla V100 (`sm_70`).

## What changed

- Added `candidate.cu` with the exact `extern "C" void launch_kernel(...)` interface and required contract marker.
- Assigned one 256-thread CUDA block to each `(batch, head, query_position)` row.
- Computed only visible causal scores (`j <= i`) and explicitly initialized masked positions to `-INFINITY`.
- Staged row scores in shared memory so the max, exponentiation, normalization, and value accumulation are fused into one kernel launch.
- Used warp shuffle reductions plus a small shared warp-result buffer for row maximum and softmax denominator reductions.
- Used fused multiply-add for QK dot products and value accumulation.
- Added bounds checks for row and causal-key accesses.

## Why

The design targets the supplied V100 latency diagnosis by avoiding auxiliary kernels and reducing block-wide synchronization during reductions. Shared score staging keeps the softmax statistics and normalized weights available to all output lanes while preserving numerical stability through max subtraction.

No CUDA compilation or benchmark was run, per task instructions.