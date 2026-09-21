# LayerNorm V100 candidate

Implemented `candidate.cu` as a standalone CUDA LayerNorm kernel for Tesla V100 (`sm_70`). The launch uses one 256-thread block per input row, which gives independent parallelism across the tested batch sizes while keeping each row's statistics local to one block.

The mean and variance reductions use warp shuffle operations, with only one value per warp stored in shared memory. The first warp completes the block-level reductions, and shared scalars broadcast the row mean and inverse standard deviation to all threads. The output pass uses coalesced `float4` accesses for the 4096-element hidden dimension and applies normalization, gamma, and beta in the same pass with fused multiply-add operations.

The implementation preserves the required `launch_kernel` ABI and includes the exact contract marker. `hypothesis.json` records the optimization strategy and associated risks. No benchmark or CUDA compilation was run, per the task instructions.