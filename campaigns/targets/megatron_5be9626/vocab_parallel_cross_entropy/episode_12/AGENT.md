# Episode 12

This candidate is a bounded rank-local refinement for the frozen TP=2
`vocab_parallel_cross_entropy` contract. It retains both required C ABI entry
points and the exact stream ABI. Each entry launches one row-per-block kernel;
warp shuffles and a small shared warp-partial array perform FP32 reductions.

The intended profile-guided change is lower local reduction and temporary
synchronization overhead. The implementation preserves shifted target-logit
semantics, owning-rank target metadata, FP32 exponentiation/reduction, and the
caller-owned one-MAX/two-SUM collective sequence. It does not implement
backward or reconstruct the global vocabulary.

No benchmark, evaluator, correctness result, or hardware result is claimed for
this episode.
