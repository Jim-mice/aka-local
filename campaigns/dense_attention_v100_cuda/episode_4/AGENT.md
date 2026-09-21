# Agent Notes

Implemented `candidate.cu` as a standalone CUDA scaled dot-product attention kernel for Tesla V100 (`sm_70`). The implementation uses one 256-thread block per query row, warp-shuffle reductions combined with a small shared-memory staging area for row maximum and softmax-sum reductions, and `fmaf` for the QK dot products.

The kernel performs numerically stable softmax evaluation by subtracting the row maximum before exponentiation. It preserves the required contiguous `[batch, heads, seq, head_dim]` layout and exposes the exact `extern "C" void launch_kernel(...)` interface. The launch maps the flattened `(batch, head, query-position)` space to one CUDA block per row.

The implementation intentionally uses only CUDA/C++ source and does not depend on PyTorch or Python. No compilation or benchmark was run, as required. The main performance tradeoff is that attention scores are recomputed for the value accumulation pass, which keeps temporary memory use low but increases arithmetic work and may increase register pressure.