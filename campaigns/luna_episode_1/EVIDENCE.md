GPU: NVIDIA GeForce RTX 5060 Laptop GPU, sm_120
Operator: FP16 SwiGLU, out = silu(gate) * up
Shapes: M=256/1024/4096, D=4096
Incumbent: fused half2 CUDA kernel, 256 threads/block, grid-stride loop
Measured incumbent speedups vs PyTorch reference: 1.582x / 1.506x / 1.673x
ptxas: 30 registers/thread, 0 spills, 0-byte stack, 0 shared memory
NCU: theoretical occupancy 100%, achieved occupancy about 70.74%, DRAM throughput 45.82%, compute throughput 50.92%
Constraint: choose exactly one optimization direction; do not rediscover fusion or half2; do not change semantics; no PyTorch fallback; no external dependencies; target sm_120; stop after editing kernel.py.
