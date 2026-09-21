# RMSNorm CUDA kernel

This candidate targets Tesla V100 (`sm_70`) with a fixed 256-thread block per row. Each thread accumulates a strided subset of the row, then warp shuffle instructions reduce the sum of squares. One value per warp is placed in shared memory, and the first warp performs the final reduction. The inverse RMS is computed once with `rsqrtf`, synchronized through shared memory, and reused by the coalesced output pass.

The implementation is standalone CUDA and exposes the required C-linkage `launch_kernel(float*, float*, float*, int, int, float)` entry point. It uses no PyTorch or Python components and does not modify the input or scale vectors.