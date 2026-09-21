# Phase 15-B.3 — Distributed Performance Recovery

## Verdict

**PARTIAL / PERFORMANCE BLOCKED.** Corrected semantic validation is now valid and Episodes 7, 8, and 10 pass real TP=2 correctness. The Phase 15-A baseline remains scientifically valid. A candidate benchmark harness attempt terminated before producing timing rows, so no candidate speedup, NSYS candidate trace, or incumbent is claimed.

## 1. Evaluator provenance

Evaluator metadata: `targets/megatron_5be9626/vocab_parallel_cross_entropy/evaluator_metadata.json`.

- version: `tp2_semantic_v2`
- evaluator hash: `7cc0fac8d80b2d27`
- fixture version: `coherent_global_fixture_v2`
- semantic stage hash: `81982bc4aa2021cc`
- ABI mapping hash: `06fcc34413149d1a`

Fixed defects were rank-dependent TP fixture generation, unsafe masked-target diagnostic indexing, and explicit correctness handoff between candidate CUDA stream and collectives.

## 2. Baseline provenance

The frozen baseline `f11fd1fc6846d6f2` came from the independent Phase 15-A `replay.py` real Megatron path and `tp2_bench.txt`. It did not invoke Agent candidates or the defective candidate evaluator. Its fixtures were generated coherently for the distributed replay. Therefore the baseline remains valid and was not versioned again.

## 3. Historical verdict policy

Original Episodes 6–10 result files are preserved. Their old TP=2 correctness verdicts are superseded where the evaluator used rank-dependent fixtures. New `semantic_rerun.json` and `corrected_validation_v2.json` files record corrected evidence; no old result was silently replaced.

## 4. Corrected candidate validation

| Episode | TP=1 | TP=2 | Result |
|---:|---|---|---|
| 7 | PASS | PASS on both ranks | eligible |
| 8 | not rerun independently in final pass | PASS on both ranks | eligible |
| 10 | not rerun independently in final pass | PASS on both ranks | eligible |

All valid runs reported loss max absolute error `0.00190449`, saved-softmax error at most `1.49e-8`, and local gradient error at most `2.98e-8`. Each retained one MAX and two SUM all-reduces. Golden validation independently passed TP=1 and TP=2 and remains excluded from scoring.

Episode 6 and 9 remain correctness failures under coherent fixtures because of `TARGET_LOGIT_WRONG`.

## 5. Benchmark status

The candidate benchmark harness was added to measure real TP=2 CUDA-event intervals including local candidate stages, real collectives, saved-softmax construction, and local gradient arithmetic. The first run for Episode 7 terminated without timing rows after process startup; Episodes 8/10 were not promoted to timing based on that incomplete harness result. No score was computed and the frozen baseline was not overwritten.

This is deliberately not reported as a performance regression or speedup. A subsequent phase must repair the benchmark harness and rerun only corrected-valid candidates against baseline `f11fd1fc6846d6f2`.

## 6. NSYS, diagnosis, and incumbent

No candidate reached a valid benchmark artifact, so candidate NSYS/NCCL trace, local-vs-collective timing attribution, distributed diagnosis, and incumbent decision are pending. The incumbent remains `NO_INCUMBENT`. The baseline NSYS evidence from Phase 15-A remains authoritative.

## 7. New Agent context

Episode 11 was generated after the corrected evaluator work. Its prompt contains the corrected ABI, semantic stage rules, prior `TARGET_LOGIT_WRONG` knowledge, and the real TP=2 baseline profile context. It passed preflight; correctness has not yet been run. No candidate was manually repaired. No valid candidate-specific profile existed to feed, so no false profile feedback is claimed.

## 8. Regression safeguards

The corrected evaluator now enforces coherent rank fixtures, safe non-owning target indices, and independent saved-softmax checks. The golden TP=2 run confirms these safeguards while preserving the collective invariant. Failure cleanup remains torchrun-bounded; no persistent NCCL hang occurred.

## 9. Knowledge and upstream integrity

Only the real-target CE namespace was updated with evaluator provenance and corrected validation. Standalone/SwiGLU knowledge and historical score eras were not changed.

Megatron final verification:

```text
HEAD = 5be9626709af2722333bf54797c954c09edeada3
working tree = clean
```

## 10. Remaining limitation

Semantic correctness recovery is complete. Scientific performance recovery is not complete because the candidate timing harness did not yet produce auditable TP=2 timing rows. The next safe step is benchmark-harness repair, followed by the existing baseline comparison and NSYS pipeline; no collective or contract change is needed.
