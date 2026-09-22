# Episode 28

## Hypothesis

Use one CUDA block per RMSNorm row and keep the complete operation fused in one kernel.

For hidden size 4096, launch 256 threads per block. Each thread owns four `float4`
chunks, or sixteen scalar elements.

The input values are loaded once and retained in registers. While loading them,
each thread accumulates its local sum of squares.

The reduction is hierarchical:

1. each thread computes a local partial sum;
2. each warp reduces using `__shfl_down_sync`;
3. the lane-0 thread of each of the eight warps writes one value to shared memory;
4. warp 0 performs the final reduction;
5. one reciprocal RMS value is produced with `rsqrtf`.

Only eight floats of shared memory are used for cross-warp communication.

After the normalization factor is available, each thread reuses its cached input
values, loads the corresponding `float4` weight vectors, computes the scaled
normalized outputs, and performs vectorized stores.

## Expected advantage

The main intended gain is eliminating the second global-memory load of `x` while
also reducing synchronization and shared-memory traffic compared with a
traditional block-wide shared-memory tree reduction.

## Main risk

Keeping sixteen input floats live per thread increases register pressure and may
reduce occupancy. The implementation is also intentionally specialized for the
frozen hidden size of 4096.
