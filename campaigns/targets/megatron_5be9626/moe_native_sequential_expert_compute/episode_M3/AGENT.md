# M3 candidate

Fresh implementation for the Megatron Native SequentialMLP Expert Compute target.

- Exports only `moe_sequential_expert_forward_fp16_stream`.
- Contains the required source-delivery marker and exact prototype.
- Uses the caller-provided CUDA stream.
- Processes contiguous expert-local tokens in expert order; zero-count experts emit no rows.
- Implements FC1 gated SiLU followed by FC2 and probability scaling.
- No routing, permutation, communication, TE, backward, or `CUDART_INF_F` code is used.
