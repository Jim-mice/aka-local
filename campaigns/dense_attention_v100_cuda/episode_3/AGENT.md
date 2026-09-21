# Change summary

Implemented a standalone CUDA dense-attention kernel for Tesla V100 (sm_70).

- Assigns one warp to each query row and uses warp shuffle reductions for QK dot products.
- Uses fused multiply-add operations for dot products and value accumulation.
- Performs a numerically stable two-pass softmax by subtracting the row maximum.
- Avoids allocating or storing the full sequence-by-sequence attention matrix.
- Uses contiguous row-major indexing for Q, K, V, and O and coalesced lane accesses.
- Launches 256 threads per block, providing eight independent query-row warps per block.

The kernel recomputes scores in the normalization and value passes to keep the implementation standalone and avoid large shared-memory or global-memory score storage. No benchmark or CUDA compilation was run, as requested.