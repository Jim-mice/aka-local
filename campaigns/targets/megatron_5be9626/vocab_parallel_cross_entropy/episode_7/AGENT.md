# Candidate

This candidate targets only the local prepare stage. It fuses shifted FP32 exponentiation, target ownership/extraction, and the local denominator reduction in one kernel per row. The exported ABI and contract marker are preserved verbatim, and the shifted target contribution is `logit[target] - global_max`.

No Megatron source was modified and no benchmark or runtime validation was run.
