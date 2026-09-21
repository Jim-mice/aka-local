# Dense Attention CUDA Candidate

## What changed

This candidate implements the required standalone `launch_kernel` entry point for Tesla V100 (`sm_70`). It assigns one 256-thread CUDA block to each `(batch, head, query position)` row.

The kernel uses warp shuffle reductions for the row maximum and softmax denominator, with one eight-float shared-memory slot per warp. Dot products use fused multiply-add operations. Query/key accesses are contiguous across the inner head dimension, and output dimensions are distributed across block threads for coalesced writes.

The softmax is numerically stable: the row maximum is computed first, every exponent uses `score - row_max`, and the normalized reciprocal sum is then applied to the value accumulation.

## Why

A block-per-row mapping exposes parallelism across all query rows, while warp-level reductions avoid a full shared-memory reduction and reduce synchronization overhead. The implementation keeps shared memory small and avoids storing the full attention matrix, which is important for sequence length 4096.

The score is recomputed for the value pass rather than materializing scores or probabilities. This trades extra arithmetic and register pressure for much lower global-memory traffic and bounded memory use.

No benchmark or CUDA compilation was run, as required. Only `candidate.cu`, `hypothesis.json`, and this `AGENT.md` were created.
