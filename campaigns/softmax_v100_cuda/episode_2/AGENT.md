# AGENT.md

Implemented a standalone CUDA softmax kernel for Tesla V100 (sm_70).

## What changed

- Added `candidate.cu` with the exact required `launch_kernel(float*, float*, int, int)` entry point and contract marker.
- Uses one 256-thread block per matrix row.
- Performs row maximum and row sum reductions with warp shuffle instructions.
- Stores only warp-level partial results in shared memory, minimizing shared-memory traffic.
- Computes the stable softmax in one kernel: subtract the row maximum, exponentiate with `__expf`, normalize by the reciprocal row sum, and write coalesced output.
- Added `hypothesis.json` describing the optimization strategy and risks.

## Why

The evaluated shapes all have 4096 columns, so a 256-thread row block gives each thread a regular strided workload while keeping global memory accesses coalesced within each iteration. Warp shuffles avoid repeated block-wide reduction traffic and synchronization; shared memory is used only for the small cross-warp reduction. The implementation is limited to CUDA features compatible with Volta sm_70 and does not use PyTorch or Python.

No CUDA compilation or benchmark was run, per the task requirements.