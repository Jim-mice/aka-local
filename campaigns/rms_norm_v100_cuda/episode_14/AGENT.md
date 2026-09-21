# RMSNorm CUDA kernel update

Implemented a standalone CUDA RMSNorm kernel for V100 `sm_70`.

## What changed

- Uses one 256-thread block per batch row, which keeps the launch mapping simple and robust for all requested batch sizes.
- Loads the main contiguous region through `float4` for coalesced global-memory traffic, with a scalar tail for non-multiple-of-four hidden sizes.
- Computes the sum of squares with per-thread FMA accumulation, warp shuffle reduction, and an 8-entry shared-memory warp reduction.
- Uses `rsqrtf` after adding `eps` to obtain the inverse RMS.
- Performs the weighted output in a second coalesced pass, reusing the shared inverse RMS and avoiding an intermediate buffer.
- Keeps shared memory limited to eight warp partials to minimize synchronization and bank-conflict exposure.

The file is self-contained and exports the required `extern "C" void launch_kernel(...)` entry point. No benchmark or CUDA compilation was run.