# Candidate notes

This candidate is a standalone CUDA 11.8 implementation of the required
rank-local, no-label-smoothing backward boundary. It uses one fused FP32
grid-stride kernel and the caller-provided CUDA stream.

The input softmax is read-only. Target subtraction is performed only when the
row owns the target; non-owning rows retain the plain softmax-times-gradient
result. No collective, synchronization, allocation, cast-back, or forward
operation is present.
