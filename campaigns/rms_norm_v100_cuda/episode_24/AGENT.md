# Episode 24: RMSNorm CUDA kernel

Implemented a standalone CUDA RMSNorm kernel for Tesla V100 (sm_70) with one 256-thread block per input row. Each thread accumulates a strided portion of the row, performs a warp-level shuffle reduction, and uses a small shared-memory array for the eight warp totals. The normalization pass uses `rsqrtf` and coalesced writes.

For aligned rows whose hidden size is divisible by four, the sum-of-squares pass uses `float4` loads while retaining a scalar fallback for general inputs. The sum uses fused multiply-add operations. The kernel keeps the required `launch_kernel` ABI and contract marker, and uses no framework dependencies.

No CUDA compilation or benchmark was run, as required. The implementation is intended to remain robust across the requested 1, 4, 8, and 32 row shapes rather than specializing for a single batch size.