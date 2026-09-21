# Softmax Episode 8: 256-thread + Intrinsic Optimization

## Strategy
Based on Episode 2 (0.982x) and Episode 3 (0.978x) results, 256 threads
performs significantly better than 128 threads (0.717x in Episode 7).

## Changes from Episode 2
1. __fdividef(1.0f, row_sum) instead of 1.0f / row_sum for faster reciprocal
2. #pragma unroll 4 on all element loops for better ILP
3. __launch_bounds__(256, 2) for register allocation hint
4. Reduced shared memory to 16 floats (from 32)
5. Consistent __expf intrinsic usage

## Profile Feedback
Episode 2 NSYS showed KERNEL_LATENCY with softmax_rows at 99.6% GPU time.
Focus on instruction-level throughput within the single dominant kernel.
