# Phase 15-B.2 — Semantic Differential Audit

## Verdict

**PASS: current ABI and evaluator are semantically sufficient after evaluator fixes.** The original Agent candidates were not the only source of failure: the TP=2 evaluator generated different full logits on each rank and its gradient diagnostic indexed masked targets unsafely. After correcting those diagnostic defects, a correctness-only golden ABI implementation passed TP=1 and real TP=2 loss, saved-softmax, and rank-local gradient checks. Episodes 7, 8, and 10 also pass the corrected TP=2 correctness harness. No performance scoring, incumbent, or candidate NSYS was started.

## 1. Exact Megatron stage graph

The exact source is `megatron/core/tensor_parallel/cross_entropy.py` at commit `5be9626709af2722333bf54797c954c09edeada3`.

```text
rank-local FP16 logits
  -> source FP32 logits buffer
  -> local max [N] FP32
  -> REAL MAX all-reduce -> global max [N] FP32
  -> shifted logits = logits - global max, in-place in FP32 buffer
  -> target mask [N] bool and masked target [N] int64
  -> shifted local target logit or zero
  -> exp(shifted logits) [N,V/TP] FP32, in-place
  -> local denominator [N] FP32
  -> REAL SUM all-reduce(predicted target)
  -> REAL SUM all-reduce(denominator)
  -> loss = log(global denominator) - global predicted
  -> softmax = exp logits / global denominator, in-place
  -> save softmax, target mask, masked target
  -> backward: subtract owned target probability and multiply by grad output
```

The full machine-readable stage specification is `targets/megatron_5be9626/vocab_parallel_cross_entropy/semantic_stage_spec.json`.

## 2. ABI-to-semantics mapping

The mapping is in `abi_semantic_mapping.json`. In particular:

- `row_max` is local max before it is overwritten by the real MAX all-reduce.
- `global_max` is the completed global MAX result, never a local max.
- `predicted_local` is the shifted target logit if this rank owns the target, otherwise zero.
- `denominator_local` is the local sum of `exp(logit-global_max)`.
- `exp_values` is the FP32 exponent buffer later divided in-place by the global denominator and saved as softmax.
- `target_mask` is a row-shaped ownership mask and `target_local` is the local target index consumed by backward.

No argument remains semantically described only as a generic workspace.

## 3. ABI sufficiency audit

Classification: **SUFFICIENT**.

The current two-entry ABI supplies the global max after the required MAX barrier, gives the local rank and global targets needed for ownership/masked indexing, exposes independent predicted and denominator contributions, exposes the exponent buffer that becomes saved softmax, and exposes both backward metadata outputs. No hidden state is required.

The original prompt was semantically too weak, and the original evaluator failed to test all roles, but the ABI itself can express the exact Megatron stages.

## 4. Evaluator call-order audit

Corrected diagnostic order:

```text
candidate local_max_fp16_stream
  -> CUDA synchronization for diagnostic handoff
  -> REAL MAX all-reduce(row_max)
  -> candidate local_prepare_fp16_stream(global_max)
  -> CUDA synchronization for diagnostic handoff
  -> REAL SUM all-reduce(predicted_local)
  -> REAL SUM all-reduce(denominator_local)
  -> loss and saved softmax
  -> local analytical backward check
```

The original evaluator had two semantic defects:

1. It used `torch.manual_seed(700 + rows + rank)`, so each TP rank constructed a different full-vocabulary fixture before all-reduce. TP=1 could pass while TP=2 necessarily combined unrelated logits.
2. Its gradient diagnostic used an unbounded local target index for targets owned by another rank, causing an out-of-range device assertion.

Both were fixed only in the correctness diagnostic harness. No scoring baseline or Megatron source was changed.

## 5. Golden implementation

`targets/megatron_5be9626/vocab_parallel_cross_entropy/golden_reference_candidate.cu` is deliberately simple and correctness-first. It is marked:

```text
PURPOSE: EVALUATOR_VALIDATION_ONLY
eligible_for_scoring: false
eligible_for_incumbent: false
```

It exports the exact frozen ABI, uses FP32 arithmetic for max/exp/sums, applies global max once, produces shifted predicted logits, writes row target metadata, and launches on the supplied stream. It is not an optimization candidate.

## 6. Golden correctness

The golden implementation passed through the same compile and distributed evaluator:

| Test | Loss max abs | Saved softmax max abs | Gradient max abs | Result |
|---|---:|---:|---:|---|
| TP=1 | 0.00190449 | 5.96e-8 | 5.96e-8 | PASS |
| TP=2 rank 0 | 0.00190449 | 1.49e-8 | 2.98e-8 | PASS |
| TP=2 rank 1 | 0.00190449 | 1.49e-8 | 2.98e-8 | PASS |

The TP=2 process used real NCCL and retained one MAX plus two SUM all-reduces. Evidence is frozen in `golden_validation.json`.

This proves the current ABI/evaluator model is sufficient. No contract version change is needed.

## 7. Target partition and edge semantics

The diagnostic fixture specification is `semantic_edge_fixtures.json`. It covers first/last local indices, targets owned by the other rank, large positive/negative logits, equal logits, and global maxima on either rank. The golden mapping uses a row-shaped ownership mask and safe masked index, while only the owning rank contributes the shifted target logit.

## 8. Numerical audit

The reference path casts logits to FP32 before max and exponentiation. `expf` operates on shifted FP32 values. The saved softmax is FP32. Target IDs remain int64. The corrected golden implementation follows these rules. Episodes 6–10 variously used incorrect predicted-logit or target metadata semantics; the evaluator defect initially obscured that distinction.

## 9. Episodes 6–10 differential classification

Semantic rerun files are stored in each episode as `semantic_rerun.json`.

| Episode | Corrected TP=2 | First divergence |
|---:|---|---|
| 6 | FAIL | TARGET_LOGIT_WRONG: emitted unshifted target logit |
| 7 | PASS | none |
| 8 | PASS | none |
| 9 | FAIL | TARGET_LOGIT_WRONG: emitted exponentiated target contribution |
| 10 | PASS | none |

Episodes 7, 8, and 10 also pass the corrected TP=1/TP=2 correctness harness. Their original Phase 15-B failure records are preserved; the semantic rerun explicitly records the evaluator revision.

## 10. Differential harness

`semantic_differential.py` provides stage summaries containing shape, dtype, min, max, mean, selected values, and checksum, plus a first-divergence comparator. The evaluator now independently observes loss, saved softmax, and analytical rank-local gradient. The required diagnostic vocabulary is `FIRST_DIVERGENCE_STAGE`, max absolute error, and max relative error.

## 11. Collective invariant and failure containment

All golden and corrected candidate correctness runs retain exactly:

```text
1 MAX all-reduce
2 SUM all-reduces
```

No local simulation replaced NCCL. Compile failures and contract failures remain pre-collective. Rank failures are propagated by torchrun. The earlier device assertion exposed a diagnostic bug and was corrected by safe index clamping; no persistent NCCL hang occurred.

## 12. Performance and incumbent policy

This phase performed no candidate benchmark, no candidate NSYS, no performance diagnosis, and no promotion. The Phase 15-B baseline hash `f11fd1fc6846d6f2` and optimization contract hash `112959ca63020e9b` remain unchanged. The incumbent remains `NO_INCUMBENT`.

The corrected candidates are correctness evidence only at this stage; their local kernel behavior must not be treated as performance evidence.

## 13. Knowledge changes

Only `knowledge/targets/megatron_5be9626/vocab_parallel_cross_entropy/` was changed. New semantic facts are:

- TP2 fixtures must be rank-independent before all-reduce.
- masked target diagnostic indices must be safe on non-owning ranks.
- predicted target contribution is shifted, not raw and not exponentiated.
- saved softmax must be checked independently from loss.

The golden implementation is explicitly excluded from scoring and incumbent knowledge.

## 14. Upstream integrity

Final verification:

```text
HEAD = 5be9626709af2722333bf54797c954c09edeada3
working tree = clean
```

No Megatron source was modified.

## 15. Clearance status

The semantic blocker is cleared: current ABI + evaluator are sufficient, and new Agent candidates 7, 8, and 10 reach real TP=2 correctness. Performance work remains intentionally stopped. A later phase may resume the original TP=2 benchmark/NSYS pipeline using the unchanged scientific contract and baseline.
