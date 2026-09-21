# RMSNorm CUDA optimization summary

This candidate targets Tesla V100 (sm_70) with one 256-thread block per input row. Each thread accumulates a strided sum of squares using coalesced `float4` loads when the hidden size is divisible by four, with a scalar tail for general hidden sizes. The reduction uses warp shuffle instructions followed by eight warp totals in shared memory, minimizing synchronization and avoiding a large shared-memory reduction buffer.

After the row RMS is computed, the kernel performs normalization and weighting in the same launch. The second pass reuses coalesced vectorized loads for input, weight, and output, and uses `rsqrtf` plus FMA-based accumulation for the reduction. The launch wrapper preserves the required standalone `extern "C"` interface and launches one block for each batch row.

The implementation is intentionally limited to `candidate.cu`, `hypothesis.json`, and this file. No benchmark or CUDA compilation was run.