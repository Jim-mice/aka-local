# AGENT.md

## Episode 13 kernel

This candidate implements standalone CUDA RMSNorm for Volta sm_70. It launches one 256-thread block per batch row. Each thread processes hidden elements with a coalesced strided access pattern, accumulating the sum of squares with `fmaf`.

The reduction first occurs within each warp using `__shfl_down_sync`, then combines the eight warp totals through a small shared-memory array. Only the warp totals, rather than per-thread partials, are written to shared memory. The resulting inverse RMS is computed once with `rsqrtf`, synchronized, and reused by the fused output pass:

`y = x * inv_rms * weight`.

The implementation is self-contained and uses only CUDA runtime declarations. It preserves the required C ABI entry point:

`extern "C" void launch_kernel(float* x, float* weight, float* y, int batch, int hidden, float eps)`.

The design targets the evaluated 4096-wide rows while remaining valid for arbitrary positive `batch` and `hidden` values. No benchmark or compilation was run for this episode.