# Candidate notes

This standalone CUDA candidate implements only rank-local work for TP=2.

- `local_max_fp16` computes one FP32 maximum per local row.
- `local_prepare_fp16` extracts the local target, computes the shifted FP32 exponentials, and accumulates the local denominator.
- The caller must perform one real MAX all-reduce after `local_max_fp16`, then real SUM all-reduces for predicted target logits and denominators. No process group, collective, synchronization, loss, softmax normalization, or backward path is implemented here.
- `exp_logits` is row-major `[rows, vocab]`; `target_mask` is 0 when this rank owns the target and 1 otherwise.

No benchmark was run, and no Megatron files were modified.
