# Episode 7 candidate summary

Created three files for a Windows-native CUDA RMSNorm candidate targeting RTX 5060 Laptop / sm_120:

- `candidate.py`: a complete `torch.utils.cpp_extension.load_inline` extension exposing `rms_norm(input, weight, eps)`. It uses one 256-thread block per row, register accumulation, warp-shuffle reduction, a small shared-memory array for warp totals, and coalesced grid-stride loads. It supports float16, bfloat16, and float32 contiguous 2-D inputs and emits the same dtype.
- `hypothesis.json`: records the optimization hypothesis, constraints, and expected effect.
- No benchmark or CUDA execution was run, per the task requirement.

The implementation uses the prior successful warp/register reduction direction and does not introduce Linux-only flags. The nvcc configuration explicitly requests `compute_120` / `sm_120` code generation.
