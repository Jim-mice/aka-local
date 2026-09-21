# AGENT.md

## Episode 9 change summary

Created a Windows-native PyTorch CUDA extension in `candidate.py` implementing unweighted, row-wise RMSNorm for contiguous 2D CUDA tensors. The kernel supports float16, bfloat16, and float32 inputs and accumulates the sum of squares in float registers.

## Why this design

The prior successful direction was warp reduction with register accumulation. This candidate extends that direction without introducing Linux-only flags or unrelated control paths: each block handles one row, performs a coalesced register accumulation, reduces inside each warp with `__shfl_down_sync`, stores only one value per warp in shared memory, and completes the block reduction in warp 0. A second coalesced pass applies the inverse RMS value.

The hypothesis is recorded in `hypothesis.json`. The extension uses `torch.utils.cpp_extension.load_inline` and explicit CUDA 13.4-compatible `nvcc` flags for `sm_120`.

## Scope and validation status

- No benchmark was run.
- No CUDA code was executed.
- No physical-device or runtime result is claimed.
- The three requested files were written only: `candidate.py`, `hypothesis.json`, and `AGENT.md`.
