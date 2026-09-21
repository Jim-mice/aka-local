# AGENT.md

## Change summary

This candidate implements RMSNorm as a standalone CUDA kernel for NVIDIA Volta (sm_70).
Each batch row is assigned one 256-thread block. Threads read and write the hidden
vector in coalesced `float4` chunks, with guarded scalar handling for a final partial
chunk when `hidden` is not divisible by four.

The RMS sum of squares is accumulated with `fmaf`, reduced within each warp using
Volta warp shuffles, and then reduced across the eight warp totals in shared memory.
The resulting inverse RMS is computed with `rsqrtf` and broadcast through shared memory.
A second coalesced pass applies the normalization and per-element weight.

The implementation is intentionally self-contained and exposes only the required
`extern "C" void launch_kernel(...)` entry point. It uses no host-side allocation,
no PyTorch, and no synchronization beyond the two block barriers needed to publish
the reduction result and the inverse RMS.

## Why this change

The workload has four 4096-wide shapes, so one block per row keeps launch behavior
simple while providing enough parallelism for every row. `float4` accesses reduce
instruction and transaction overhead, warp shuffles avoid a full shared-memory tree
reduction, and the eight-element shared-memory reduction keeps cross-warp traffic
small. The guarded tail preserves correctness for other hidden sizes without changing
the fast path.