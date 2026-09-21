# Dense Attention Backward V100 candidate

## Summary

This candidate implements the required standalone CUDA `launch_kernel` entry point for FP32 non-causal dense attention backward on NVIDIA Volta (`sm_70`). It preserves the supplied forward probabilities `p` and does not recompute softmax.

A 256-thread block is assigned to each `(batch, head, query-row)`. Threads first compute the row statistic `dot_i = sum_j(dP[i,j] * P[i,j])`; warp shuffle reduction performs the intra-warp reduction and a small shared-memory array combines warp results. The same row pass then computes `dQ` and atomically accumulates `dK` and `dV`, followed by a separate zeroing kernel for all output gradients.

## Why

The row mapping gives regular parallelism across all four evaluation batch sizes, while warp shuffles reduce synchronization and shared-memory traffic for the normalization statistic. FMA operations are used in the innermost reductions. The implementation is intentionally correctness-first and uses FP32 arithmetic throughout.

## Tradeoffs and risks

`dK` and `dV` use atomic accumulation because query-row blocks independently contribute to the same key/value rows. This can create contention, especially at larger batch/head counts. The current implementation also recomputes `dP` while producing each query-gradient dimension, increasing arithmetic work and register pressure. No CUDA compilation or benchmark was run, as required.