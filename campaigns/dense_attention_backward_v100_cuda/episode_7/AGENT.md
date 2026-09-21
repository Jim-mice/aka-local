# Agent summary

Implemented a standalone CUDA dense-attention backward candidate for Tesla V100 (sm_70).

- Added the required `launch_kernel` C ABI and exact contract marker.
- Uses 256-thread row-wise CUDA blocks.
- Uses warp shuffle plus shared-memory block reduction for the softmax backward row statistic.
- Computes dQ, dK, and dV without a separate clear kernel or global atomic accumulation.
- Uses fused multiply-add operations in accumulation loops and coalesced per-head/per-dimension accesses where possible.
- Added the required hypothesis metadata and contract hash.

No CUDA compilation or benchmark was run, per task instructions.