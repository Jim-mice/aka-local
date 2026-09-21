# LayerNorm CUDA kernel summary

`candidate.cu` implements the required standalone `extern "C" void launch_kernel(float*, float*, float*, float*, int, int, float)` entry point for Tesla V100 (`sm_70`). It launches one 256-thread block per row and performs the mean and variance reductions with warp shuffle instructions, using only eight shared-memory values to combine warp results. Each thread then writes a contiguous strided subset of the row, applying the affine transform with `fmaf` and computing the inverse standard deviation with `rsqrtf`.

The implementation is self-contained and uses no host framework or Python. The small shared reduction buffer limits shared-memory traffic and avoids a full-block shared reduction, while coalesced row accesses preserve throughput across the requested row counts.
