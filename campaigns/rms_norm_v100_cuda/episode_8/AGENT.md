# AGENT.md

## Change summary

Created `candidate.cu` as a self-contained standalone CUDA RMSNorm implementation with the required `extern "C" void launch_kernel(float* x, float* weight, float* y, int batch, int hidden, float eps)` entry point.

The kernel assigns one 256-thread block to each batch row. Each thread performs a coalesced strided load over the hidden dimension, accumulates the sum of squares with fused multiply-add, and participates in a warp-shuffle reduction. One 32-element shared-memory array stores warp totals; the first warp completes the block reduction and publishes the inverse RMS. A second coalesced pass applies normalization and the per-feature weight.

This design targets V100 sm_70 by using warp-level primitives for the reduction, only a small conflict-free shared-memory staging area, and independent blocks across rows. It is intended for the requested 4x4096, 1x4096, and 8x4096 shapes without relying on newer architectures or external frameworks.

`hypothesis.json` records the optimization claim and expected effect. No benchmark or CUDA compilation was run.