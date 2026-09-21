# Agent Summary

Implemented a standalone CUDA dense-attention candidate for Tesla V100 (sm_70).

## What changed

- Added `candidate.cu` with the exact `launch_kernel` ABI and required contract marker.
- Used 256-thread blocks and 256-element sequence tiles.
- Used warp shuffle reductions for dot products, row maxima, and normalization sums.
- Used shared memory for score-tile reuse during the softmax and value accumulation phases.
- Used fused multiply-add operations in dot products and output accumulation.
- Added `hypothesis.json` documenting the optimization hypothesis, risks, and contract metadata.

## Why

The design keeps score work organized by warps, avoids materializing the full attention matrix, and improves reuse of the current sequence tile while preserving numerically stable row-wise softmax via subtraction of the row maximum.

Per instruction, no CUDA compilation or benchmark was run.