# Candidate Agent Notes

This directory contains a single CUDA candidate for the specified Megatron-LM
non-TE `torch.nn.RMSNorm` forward target.

The implementation is intentionally limited to FP16 contiguous input, weight,
and output tensors. It uses FP32 scalar conversion and accumulation, the
supplied CUDA stream, and no allocation or external runtime work.

No build, execution, benchmark, flash, or physical-device validation was
performed. The ABI marker and exported prototype in `candidate.cu` are part of
the active contract and must remain unchanged.
