# Candidate

This candidate targets `vocab_parallel_cross_entropy` at commit `5be9626709af2722333bf54797c954c09edeada3`.

`local_max_fp16` computes one FP32 maximum per local row. After the caller's real MAX all-reduce, `local_prepare_fp16` computes the masked local target index, partial shifted target logit, and partial FP32 exponential denominator in one kernel. The caller must perform exactly one MAX and two SUM all-reduces and finish loss/softmax formation.

The entry points use the default stream (`stream 0`) because this standalone ABI does not carry a stream argument. No synchronization or process-group code is present; the caller owns stream selection and collective ordering.
