# Softmax Episode 4: 128-thread Optimization

## Profile Feedback
NSYS profiling of Episode 2 revealed KERNEL_LATENCY as the bottleneck:
- softmax_rows dominates 99.6% of GPU time
- average kernel call ~21.3us (0.982x vs PyTorch reference)

## Changes from Episode 2/3
1. **128 threads per block** (was 256): Better occupancy on V100 (80 SMs).
   More blocks per SM when row count is moderate.
2. **__launch_bounds__(128, 4)**: Hint to compiler for register allocation.
3. **__fdividef(1.0f, row_sum)**: Intrinsic reciprocal for faster division.
4. **Reduced shared memory**: Only 8 floats (32 bytes) from 32 floats.
5. **Aggressive #pragma unroll 4**: Force unrolling on element loops for ILP.
6. **__expf intrinsic**: Kept from Episode 2.

## Expected Impact
- Better occupancy with 128 threads (4 warps instead of 8 per block)
- Less shared memory contention
- More instruction-level parallelism via unrolling
- __fdividef may reduce division latency
