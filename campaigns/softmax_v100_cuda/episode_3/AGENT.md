# Change summary

- Implemented a standalone stable row-wise softmax in `candidate.cu` for NVIDIA Volta (sm_70).
- Assigned one 256-thread block to each row, so all four target shapes use the same predictable execution strategy.
- Used warp shuffle reductions for both the row maximum and the normalization sum, with only a small shared-memory exchange between warps.
- Kept row loads and stores coalesced and used `__restrict__` pointers for clearer aliasing information.
- Subtracted the row maximum before every exponential evaluation to preserve numerical stability.
- Kept the required C ABI entry point and contract marker exactly as specified.

No CUDA compilation or benchmark was run, per the task constraints.