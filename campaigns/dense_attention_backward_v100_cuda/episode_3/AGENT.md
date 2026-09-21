# Agent summary

Implemented `candidate.cu` as a standalone CUDA 11.8 FP32 dense-attention backward kernel for Volta sm_70. The entry point matches the required C ABI and includes the exact contract marker.

The implementation launches separate kernels for dV, dQ, and dK so each output is written without atomics or cross-row write races. The dQ kernel computes the per-query softmax-backward dot term using warp shuffle reduction plus a small shared-memory staging array, then applies the scaled dS-to-dQ product. dV and dK use coalesced per-head/per-key/per-dimension writes and fused multiply-add accumulation.

No CUDA compilation or benchmark was run, as required. Only the three requested files were created.