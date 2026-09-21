# Candidate notes

This candidate is intentionally limited to the active RMSNorm ABI. It uses one
CUDA block per row, performs the sum of squares in FP32, computes the inverse
RMS once per row, and fuses the final scale and FP16 store.

The wrapper launches only on the caller-provided stream and performs no memory
allocation or host/device synchronization. It does not claim compilation,
benchmark, or device-validation results.
