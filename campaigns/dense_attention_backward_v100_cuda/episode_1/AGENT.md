# AGENT.md

## Summary

This candidate implements the required standalone CUDA dense-attention backward contract for Volta sm_70.

## Changes

- Added the exact `launch_kernel` C ABI and contract marker.
- Computes all three outputs: `grad_q`, `grad_k`, and `grad_v`.
- Uses 256-thread blocks mapped across `(batch * heads, sequence-row)` work.
- Uses warp shuffle reduction followed by a small shared-memory warp reduction for the softmax Jacobian row statistic.
- Keeps `grad_v` as a direct per-output-element reduction over query rows, avoiding atomics.
- Clears `grad_k` before the key-gradient accumulation kernel; key gradients are accumulated over all query rows.
- Uses FP32 arithmetic and the supplied forward probabilities `p`; it does not recompute softmax.

## Rationale and risks

The row-oriented layout gives coalesced accesses for contiguous head-dimension elements and exposes parallelism over the long sequence dimension. Warp-level reduction minimizes synchronization for the softmax backward dot product, while the shared array only stores one value per warp. The implementation deliberately remains standalone and uses no Python or PyTorch.

The main risks are register pressure and repeated dot-product work needed to form row statistics. No CUDA compilation or benchmark was run, as required.