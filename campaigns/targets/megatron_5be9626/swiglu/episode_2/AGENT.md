# SwiGLU sidecar candidate

This directory contains one standalone CUDA candidate for the `swiglu_v1`
activation-only replacement boundary.

The kernel fuses optional bias addition, gate/up splitting, SiLU on the gate,
and multiplication by the offset-adjusted up value. It uses the caller's
current CUDA stream, allocates no workspace, and performs no synchronization.

No Megatron source was modified. Compilation, correctness replay, and
benchmarking were intentionally not run for this candidate.
