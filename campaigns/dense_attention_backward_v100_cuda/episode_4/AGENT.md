# Change summary

Implemented `candidate.cu` as a standalone CUDA 11.8-compatible FP32 dense-attention backward kernel for Volta `sm_70`.

## What changed

- Preserved the exact `extern "C" void launch_kernel(...)` contract and required contract marker.
- Added separate CUDA kernels for `dQ`, `dK`, and `dV`.
- Used a 1D grid with 256-thread blocks so all batch/head/sequence/feature elements are independently parallelized.
- Used coalesced contiguous feature-dimension accesses for gradient outputs and `fmaf` accumulation for FP32 arithmetic.
- Consumed the supplied softmax probability tensor `p` directly; softmax is not recomputed.
- Kept all indexing in 64-bit intermediate arithmetic so the `[B,H,S,D]` and `[B,H,S,S]` layouts are addressed safely for the required long-sequence shapes.

## Correctness structure

For each query row, the dQ and dK kernels compute the softmax Jacobian term using:

`dot_i = sum_j(dP[i,j] * P[i,j])`

and then accumulate `dS[i,j] = P[i,j] * (dP[i,j] - dot_i)`. The dV kernel computes `P^T @ grad_o`. The resulting dQ and dK values are multiplied by `scale` as required.

## Constraints honored

- No Python files or PyTorch dependencies.
- No benchmark or CUDA compilation was run.
- No architecture newer than `sm_70` was introduced.
- Only `candidate.cu`, `hypothesis.json`, and `AGENT.md` were written for this task.