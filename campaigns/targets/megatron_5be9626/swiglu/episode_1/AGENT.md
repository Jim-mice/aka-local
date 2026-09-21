# SwiGLU CUDA Sidecar

This directory contains one standalone CUDA candidate for the specified
Megatron SwiGLU replacement boundary.

`candidate.cu` exports `launch_swiglu` and implements optional bias addition,
gate/up splitting, SiLU on the gate, and multiplication by the offset-adjusted
up projection. It assumes contiguous FP16 row-major inputs and outputs and
launches on the caller's current CUDA stream without synchronization or
workspace allocation.

No Megatron files were modified. No commands or benchmarks were run.
