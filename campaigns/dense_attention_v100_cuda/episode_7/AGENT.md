# Agent Notes

Implemented a standalone CUDA dense-attention forward kernel for NVIDIA Volta sm_70.

## What changed

- Added `candidate.cu` with the exact required `extern "C" void launch_kernel(...)` entry point and contract marker.
- Uses one 256-thread block per `(batch, head, query_position)` row.
- Stages the query row in dynamic shared memory so it is loaded once and reused for every key.
- Computes each QK dot product with fused multiply-add instructions.
- Uses warp shuffle reduction followed by a small shared-memory warp reduction for the score.
- Fuses score calculation, numerically stable online softmax, and V accumulation into one kernel, avoiding an intermediate score/probability matrix.
- Writes output through coalesced strided accesses across the head dimension.

## Why

The operator has sequence length 4096, so materializing an attention matrix would be expensive in both memory traffic and synchronization. The online-softmax recurrence preserves numerical stability while allowing each key row to be processed once. Warp-level reduction minimizes synchronization for the QK dot product, while the shared-memory query tile avoids rereading Q for every key.

No CUDA compilation or benchmark was run, per the task instructions. The implementation is intended for the specified CUDA 11.8 / Tesla V100 sm_70 environment.
