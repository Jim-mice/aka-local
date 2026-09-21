# Candidate agent notes

This candidate implements only rank-local CUDA work for the TP=2 forward path.

- `local_max_fp16` computes one FP32 maximum per local row.
- `local_prepare_fp16` uses the caller-provided globally reduced maximum,
  extracts the rank-local target, computes shifted exponentials, and returns
  rank-local predicted-logit and denominator contributions.
- `predicted_local` and `denominator_local` must each participate in the
  caller's required SUM all-reduce. No collective or process-group setup is
  present here.
- `target_mask`, `target_local`, and `exp_values` are intermediate outputs;
  storage is caller-owned and laid out row-major.
- All arithmetic after half loads is FP32. The `_stream` entry points accept
  the caller's CUDA stream and do not synchronize; the preferred ABI wrappers
  use CUDA stream 0.

The implementation deliberately contains no backward path, full-vocabulary
reconstruction, local-only final denominator, or local-only final loss.
