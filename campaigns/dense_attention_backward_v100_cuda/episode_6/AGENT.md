# Dense Attention Backward V100 Candidate

Implemented a standalone CUDA candidate for Tesla V100 (`sm_70`) with the required file-scope `extern "C" void launch_kernel` ABI.

- `clear_kernel` zeroes `grad_k` before accumulation.
- `backward_rows` assigns one 256-thread block to each query row.
- The row kernel computes `grad_q` and `grad_v` together while traversing the dense attention row.
- `grad_k` is accumulated with atomic additions because multiple query rows contribute to each key row.
- Arithmetic is FP32 and uses fused multiply-add operations where applicable.
- Tensor indexing follows the specified `[batch, heads, seq, head_dim]` and `[batch, heads, seq, seq]` layouts.

No CUDA compilation or benchmark was run, as required.