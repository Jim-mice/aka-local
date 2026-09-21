# AGENT.md

Implemented a standalone CUDA causal scaled dot-product attention kernel for Tesla V100 (sm_70).

## What changed
- Added `candidate.cu` with the exact required `extern "C" void launch_kernel(...)` entry point.
- Added the required contract marker.
- Uses one 256-thread block per `(batch, head, query_position)` row.
- Uses warp-shuffle reductions plus a small shared-memory array to reduce dot products and normalization sums.
- Computes a numerically stable causal softmax by subtracting the row maximum.
- Masks all future keys by restricting every key loop to `j <= i`.
- Uses fused multiply-add for QK dot products and coalesced strided accesses for Q, K, V, and O.
- Added `hypothesis.json` with the required operator metadata and strategy tags.

## Rationale
The row-wise block mapping gives all threads a common causal row and lets them cooperate on each QK dot product. Warp shuffles avoid unnecessary shared-memory traffic for the reduction, while shared memory broadcasts each score and the row statistics to the block. The implementation deliberately recomputes scores during the softmax and value-accumulation phases to avoid storing an entire score row, keeping memory use bounded for sequence length 4096.

No CUDA compilation or benchmark was run, per the task requirements.