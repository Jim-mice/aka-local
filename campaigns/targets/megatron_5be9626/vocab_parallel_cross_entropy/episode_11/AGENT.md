# Candidate agent notes

This candidate targets only the local forward preparation stage of
`vocab_parallel_cross_entropy`.

- The exported ABI prototypes and contract marker are preserved verbatim.
- `local_max_fp16_stream` computes the rank-local row maximum.
- `local_prepare_fp16_stream` uses the globally reduced maximum for every
  shifted logit, local target contribution, and exponent.
- `target_mask` and `target_local` are written once per row.
- The caller must retain exactly one real MAX all-reduce and two real SUM
  all-reduces. No backward path or collective is implemented here.
- The implementation is intended for CUDA 11.8 / V100 sm_70 and does not use
  `CUDART_INF_F`.
