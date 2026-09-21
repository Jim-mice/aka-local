# Candidate 6

This candidate targets only the local preparation stage. It uses one row-wise
local-max kernel and one fused preparation kernel for shifted exponentials,
local denominator, target ownership, and local target-logit extraction.

The mandatory C ABI and contract marker are preserved. The caller must retain
exactly one real MAX all-reduce and two real SUM all-reduces; this candidate
does not implement or remove collectives, backward, partitioning, or the
full-vocabulary reconstruction.

No benchmark, Megatron modification, build, or hardware validation was run.
