# Candidate agent

This candidate targets the rank-local forward computation for TP=2 vocab-parallel cross entropy.

`local_max_fp16` computes each rank's local FP32 maximum. The caller must perform the single real MAX all-reduce before invoking `local_prepare_fp16`.

`local_prepare_fp16` fuses shifted exponentiation, local target extraction, and per-rank partial denominator accumulation. Its outputs are partial values: the caller must perform the two required real SUM all-reduces (predicted target contribution and denominator) and then apply the global loss/softmax logic.

The candidate launches work on the current CUDA stream, performs no synchronization, initializes no process group, and contains no backward implementation. `target_logit_partial` is the local target exponential contribution, not a finalized global target probability or loss input.
