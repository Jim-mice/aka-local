# AGENT.md

Implemented a standalone CUDA causal-attention kernel for Tesla V100 (sm_70).

- Maps one 256-thread block to each batch/head/query row.
- Computes only causal keys (j <= i), so future positions never contribute.
- Uses warp shuffle plus a small shared-memory warp reduction for QK dot products.
- Uses an online, numerically stable softmax update: each score updates the running row maximum, normalization sum, and output accumulator without materializing an attention matrix.
- Uses fused multiply-add for dot products and __expf for fast exponentiation.
- Keeps loads and output updates strided/coalesced across the block and supports arbitrary positive head dimensions.
- The launch entry point and contract marker match the required interface exactly.

No CUDA compilation or benchmarks were run, as requested.
