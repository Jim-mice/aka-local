# Isolated CUDA candidate

- Hypothesis: one fused CUDA kernel with one block per token row and a shared-memory float32 tree reduction will reduce launch/intermediate-memory traffic for the real RMSNorm workload.
- Evidence: the official contract performs one row-wise sum-of-squares reduction followed by normalization; all 56 supplied shapes have contiguous row vectors, with hidden sizes from 128 to 8192.
- Changed mechanism: `candidate.py` uses exactly one custom CUDA kernel. Each 256-thread block reduces one row into shared memory, then rereads that row and writes bfloat16 output using float32 arithmetic. No PyTorch fallback or unrelated tuning is included.
- Correctness risk: block reduction order differs from eager PyTorch reduction, and the kernel assumes CUDA bfloat16 tensors with a two-dimensional contiguous layout. The evaluator's official inputs satisfy the dtype/device contract; numerical tolerance must account for float32 reduction-order differences.
- GPU Wiki record IDs used: none.
