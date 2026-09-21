# Episode 2

## 研究问题

Can a warp-level reduction replace V1's full shared-memory tree for the official RMSNorm workload?

## V1 原来怎么做

V1 used one 256-thread block per row, float accumulation, a 256-entry shared-memory tree, a second row traversal, and fused normalization/output.

## Agent hypothesis

The Agent selected `warp_reduce_shared_finalize`: eight warp partial sums are produced with `__shfl_down_sync`, then warp 0 finalizes them. The row mapping, block size, two-pass access, float accumulation, epsilon, and BF16 output were kept unchanged.

## CUDA 原理

Warp shuffle reduction exchanges values through warp lanes without the full shared-memory tree. Only eight warp totals need shared storage, reducing shared-memory footprint from 2048 B to 1056 B in the compiled candidate.

## 实测结果

Official compile and correctness passed 56/56. Five independent same-process A/B/B/A batches with 20 warmups and 100 repetitions gave arithmetic mean speedup 1.074215x, geometric mean 1.071904x, and total-time ratio 1.096522x. Shape 17 (2048x128) won all 5 batches: mean 1.075416x, std 0.014928.

## 为什么这个结果重要

The initial 0.812x observation for shape 17 was not reproduced. The controlled reduction-only change showed a repeatable improvement for that shape and for the aggregate in this exact RTX 5060 workload set.

## 已经证明什么

This V2 implementation passed the official 56-shape contract and was faster than V1 under the repeated local ABBA protocol.

## 没有证明什么

It does not prove that warp shuffle is universally faster than shared memory, nor that the result transfers to V100/A100 or another hidden-size distribution. NCU was not needed and occupancy was not measured here.

## 用户接下来需要学习的知识点

Warp shuffle intrinsics, warp-synchronous programming, partial reductions, synchronization cost, shared-memory footprint, and architecture-specific performance transfer.
