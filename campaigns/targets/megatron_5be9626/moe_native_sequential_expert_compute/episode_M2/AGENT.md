# M2 candidate

This standalone candidate exports the required `extern "C"` entry point:

`moe_sequential_expert_forward_fp16_stream`

It implements contiguous permuted-token expert order, per-expert counts (including empty experts), FP16 FC1/FC2 weights, gated SiLU, and per-token probability scaling on the caller-provided CUDA stream. It performs no device allocation and does not use NCCL or `CUDART_INF_F`.

No Megatron upstream files were modified. This candidate was not benchmarked or flashed.
