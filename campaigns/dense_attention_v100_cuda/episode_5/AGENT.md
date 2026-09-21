# AGENT.md

## Summary

Implemented `candidate.cu` as a standalone CUDA scaled dot-product attention kernel for Tesla V100 (`sm_70`). The implementation preserves the required C ABI and contract marker, performs row-wise numerically stable softmax, and uses a 256-thread block for each `(batch, head, query position)` row.

## What changed and why

- Stages the query row in dynamic shared memory so each key dot product reuses query values from on-chip storage.
- Uses warp shuffle reductions plus a small shared reduction buffer for row maximum and softmax denominator.
- Uses fused multiply-add operations for QK dot products and value accumulation.
- Keeps memory indexing contiguous for Q, K, V, and O tensors.
- Guards blocks whose row index is outside the logical workload.
- Exposes exactly the requested `extern "C" void launch_kernel(...)` entry point.

## Constraints followed

- Only `candidate.cu`, `hypothesis.json`, and `AGENT.md` were created.
- No Python, PyTorch, benchmark, or CUDA compilation was used.
- The implementation targets the supplied generic `sm_70` compilation command and does not reference newer architectures.
- The required `// AKA_CONTRACT: q, k, v, o, batch, heads, seq, head_dim, scale` marker is present.

## Risk notes

This is a correctness-first baseline rather than a benchmark-validated final optimization. It recomputes QK scores during the normalization and value phases, which increases arithmetic work but avoids storing the full attention matrix. Dynamic shared-memory usage grows with `head_dim`; launch-time resource limits should be checked by the evaluator for unusually large head dimensions.