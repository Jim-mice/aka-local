# SwiGLU backward candidate

This directory contains one standalone CUDA candidate for the requested
optional-bias-free activation backward boundary.

The kernel reads contiguous FP16 `intermediate` as `[gate, up]` along the
width dimension, computes `grad_gate` and `grad_up`, and writes them back in
the same packed layout. It launches on the caller-provided CUDA stream and
does not synchronize.

No forward path, GEMM, collective, bias-gradient logic, or Megatron source is
included or modified.
