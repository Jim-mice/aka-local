# Candidate agent

This candidate targets only the local preparation stage of
`_VocabParallelCrossEntropy`.

It preserves the required exported ABI and contract marker. The preparation
kernel fuses shifted target extraction, row-shaped target ownership metadata,
exponent output, and local denominator accumulation. All local arithmetic is
FP32 and uses `logit - global_max`.

The caller must retain the real one-MAX and two-SUM all-reduces. No backward
path, partitioning change, local-only denominator, or full-vocabulary
reconstruction is introduced. No benchmark or Megatron source modification
was performed for this candidate.
