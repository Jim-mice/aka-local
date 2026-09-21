# Candidate agent notes

This candidate changes only the local CUDA compute stage. It does not modify Megatron, add collectives, remove collectives, implement backward, or reconstruct the full vocabulary.

`local_max_fp16_stream` performs a per-row FP32 local maximum. `local_prepare_fp16_stream` consumes the caller-provided global maximum and computes `(logit - global_max)` before `expf`. It writes row-shaped target metadata and the local predicted contribution and denominator required by the existing MAX and SUM collectives.

The implementation uses CUDA 11.8-compatible headers and the verified finite initializer `-3.402823466e+38F`; it does not use `CUDART_INF_F`.
