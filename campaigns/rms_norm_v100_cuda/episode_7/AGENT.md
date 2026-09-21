# AGENT.md

## Change summary

Created `candidate.cu` as a self-contained standalone CUDA RMSNorm implementation with the required C ABI:

```cpp
extern "C" void launch_kernel(
    float* x, float* weight, float* y,
    int batch, int hidden, float eps
);
```

Each CUDA block processes one batch row with 256 threads. Threads load hidden-dimension elements in a coalesced, block-strided pattern, accumulate the sum of squares with `fmaf`, and reduce it first within warps using Volta-supported warp shuffle instructions. Warp leaders write one value each to a 32-element shared-memory array; the first warp performs the final reduction. One thread computes the inverse RMS with `rsqrtf`, and all threads reuse it for the normalized, weighted output pass.

## Why this is appropriate for sm_70

- Uses only CUDA features supported by compute capability 7.0.
- Uses eight warps per 256-thread block, matching the requested Volta-oriented reduction strategy.
- Keeps global loads and stores coalesced for contiguous rows.
- Uses only a small shared-memory reduction buffer, avoiding shared-memory bank conflicts because each warp leader writes a distinct entry.
- Computes the normalization factor once per row and reuses it across all output elements.

No benchmark or CUDA compilation was run, as requested. No Python files or PyTorch components were created or used.