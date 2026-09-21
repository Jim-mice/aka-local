# SwiGLU sidecar candidate

This directory contains one standalone CUDA candidate for the `swiglu_v1`
activation-only replacement boundary.

The kernel fuses optional bias addition, gate/up splitting, SiLU on the gate,
and multiplication by the offset-adjusted up value. FC1, FC2, tensor-parallel
communication, Transformer Engine, FP8, dropout, and residual BDA remain out
of scope.

No build, benchmark, Megatron modification, or runtime validation was
performed for this candidate.
