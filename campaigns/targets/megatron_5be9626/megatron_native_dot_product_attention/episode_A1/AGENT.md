# Standalone candidate

`candidate.cu` implements the requested native dense attention forward boundary:
FP16 contiguous Q/K/V, FP32 softmax semantics, p=0 identity dropout, and FP16
output. It launches one CUDA kernel on the supplied stream, uses dynamic shared
memory for its per-query tile, and performs no allocation or framework edit.

The candidate is scoped to the official `head_dim=64` configurations. No
benchmark, Megatron integration, physical execution, or runtime accuracy claim
was made.
