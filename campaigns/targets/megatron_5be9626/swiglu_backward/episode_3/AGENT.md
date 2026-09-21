# Candidate Agent Notes

Standalone optional-bias-free SwiGLU backward candidate.

- Reads contiguous FP16 `intermediate` and `grad_output`.
- Computes sigmoid, SiLU, derivative, both products, and `[grad_gate, grad_up]` packing in one kernel.
- Gate and up occupy the first and second halves of the final `width` dimension.
- Uses the supplied stream without synchronization.
- Excludes FC1, FC2, forward, collectives, bias gradients, and GEMMs.
