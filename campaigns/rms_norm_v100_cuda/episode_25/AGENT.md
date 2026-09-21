# RMSNorm CUDA candidate

This candidate assigns one 256-thread block to each input row. Each thread performs a four-way unrolled, contiguous-stride pass over the row and accumulates the sum of squares with `fmaf`. Warp-level `__shfl_down_sync` reduces partial sums, while eight shared-memory entries combine the warp results with only two block barriers. A second coalesced pass applies the reciprocal RMS and per-feature weight.

The design targets Tesla V100 sm_70 and the required 4096-feature shapes. It keeps the reduction state small, avoids a full shared-memory reduction array, and fuses the arithmetic needed for the RMSNorm output. No benchmarks or CUDA compilation were run as requested.