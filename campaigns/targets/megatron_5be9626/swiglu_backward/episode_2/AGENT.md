# SwiGLU backward candidate

This episode contains one standalone candidate for the optional-bias-free
activation backward boundary at commit `5be9626709af2722333bf54797c954c09edeada3`.

`candidate.cu` launches exactly one kernel on the stream supplied by the ABI.
It reads contiguous FP16 `intermediate` and `grad_output`, computes the fused
sigmoid/SiLU and gate derivative, and writes the inverse-packed
`[grad_gate, grad_up]` result to `grad_intermediate`.

The candidate does not implement or alter FC1, FC2, forward, collectives, or
bias gradients. It performs no synchronization and contains no GEMM.
