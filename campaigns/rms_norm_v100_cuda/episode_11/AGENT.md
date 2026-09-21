# RMSNorm CUDA candidate

Implemented a standalone CUDA RMSNorm kernel for Tesla V100 (sm_70).

- Launches one 256-thread block per batch row.
- Computes the sum of squares with coalesced strided loads and `fmaf` accumulation.
- Uses warp shuffle reduction followed by an 8-entry shared-memory reduction.
- Reuses the reduced inverse RMS for a second coalesced pass that applies the per-feature weight.
- Keeps shared memory minimal and avoids dynamic allocation and bank-conflict-prone layouts.
- Preserves the required C ABI entry point and standalone CUDA interface.

No benchmark or CUDA compilation was run, as requested.