# Phase 20-A vendor-GEMM-preserving evidence

The real native path already folds the 0.125 scale into `torch.baddbmm` alpha.
With `masked_softmax_fusion=false`, dense no-mask p=0 uses the Torch softmax
fallback: FP16 score to FP32, softmax, then FP16 probabilities. Diagnostic
NSYS confirms that an explicit evaluator-only hybrid preserving vendor QK/PV
reproduces the same six-kernel S=64 graph. Phase 20-B research should target
only fusion/elimination within this local softmax/conversion graph, never a
full-core row-serial substitute. The current readiness snapshot is noisy due
to asymmetric GPU state and an unrelated CUDA process; it is not scoring data.
