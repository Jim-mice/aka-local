# Phase 15-B.6 Comparable Benchmark Era

## Verdict

The new shared benchmark era is implemented and semantically aligned, but Phase 15-B.6 remains **PARTIAL / PERFORMANCE STABILITY BLOCKED**. The old baseline is not reused. The shared reference and candidate paths pass correctness and use the same timing orchestration, but candidate repeatability is not yet sufficient for incumbent promotion: Episode 7 showed intermittent multi-millisecond rank-completion outliers.

## 1. Phase 15-B.5 scope mismatch

The old Phase 15-A baseline `f11fd1fc6846d6f2` timed a differentiable clone and real autograd backward inside the event. The former candidate harness preallocated buffers outside timing and used manual local backward. Phase 15-B.5 correctly classified that pair `SCOPE_NOT_EQUIVALENT`. The old baseline remains historical only and is marked in `legacy_baseline_metadata.json`.

## 2. Shared benchmark architecture

`remote_comparable_bench.py` is the single TP=2 orchestration. It selects either `reference` or `candidate` only for the two rank-local implementation functions:

`local_max → MAX all-reduce → local_prepare → SUM(target) → SUM(denominator) → common loss/saved-softmax/common analytical backward`.

The same fixtures, tensors, output buffers, barriers, CUDA events, process group, and rank aggregation are used for both modes. Candidate mode loads only the immutable episode CUDA source; reference mode uses a trusted PyTorch local implementation.

## 3. Active performance contract

Artifact: `targets/megatron_5be9626/vocab_parallel_cross_entropy/performance_contract.json`

- version: `tp2_scope_v1`
- era: `tp2_comparable_v1`
- semantic contract: `112959ca63020e9b`
- evaluator: `7cc0fac8d80b2d27`
- fixture: `coherent_global_fixture_v2`
- TP: 2, FP16
- collectives: exactly 1 MAX + 2 SUM
- warmups: 5
- measurements: 5
- aggregate: maximum rank completion per iteration
- contract hash: `6632730d9957fbb1a0da4268f2b3c168b2836862db167935d8908758a00b5bea` (canonical placeholder-field hash)

The timed boundary includes local max, all three real NCCL operations, global-max-dependent preparation, common loss finalization, saved softmax, and common local gradient generation. Allocations of the declared buffers and fixture generation are outside timing for both paths.

## 4. Correctness and A/A validation

The reference and Episodes 7/8/10 all pass the shared harness oracle for loss, saved softmax, and rank-local gradient on all three configurations. The reference A/A runs are complete and numerically consistent:

| Reference run | rows 8 | rows 64 | rows 256 |
|---|---:|---:|---:|
| A0 | 13369.8 us | 13382.7 us | 13433.2 us |
| A1 | 13365.7 us | 13376.9 us | 13429.4 us |

The A/A result shows no systematic semantic or structural change, although distributed timing still has normal scheduling variability.

## 5. New baseline

The new reference baseline is stored in `baseline_manifest_tp2_comparable_v1.json`. Its distributed means are approximately `13367.7, 13379.8, 13431.3 us` for the three official configurations. It uses the common manual analytical backward, common loss finalization, common saved state, and common preallocated buffer policy. The legacy baseline hash is not overwritten.

## 6. Candidate results

All three candidates pass shared correctness and complete-row validation:

| Episode | candidate means (us) | provisional ratios vs new reference |
|---:|---|---|
| 7 | 5428.8, 6009.0, 5451.2 | 2.462x, 2.227x, 2.464x |
| 8 | 5416.1, 5415.3, 5451.0 | 2.467x, 2.472x, 2.464x |
| 10 | 5428.0, 5436.2, 5552.3 | 2.462x, 2.462x, 2.419x |

These are comparable shared-harness observations, not yet a promoted campaign score because repeatability is a required gate.

## 7. Repeatability blocker

Episode 7 independent runs included one stable run and two runs with an intermittent outlier:

- stable reference-aligned run: approximately `5429, 5417, 5449 us`
- repeat 1: approximately `5430, 5431, 6586 us`, CV `0.383` on the largest configuration
- latest run: approximately `5429, 6009, 5451 us`, CV `0.216` on the middle configuration

The outliers are rank-completion timing events, not correctness failures. They must not be discarded or replaced by medians without a contract change. Therefore no candidate has a stable score suitable for incumbent promotion.

## 8. Allocation, saved state, and synchronization policy

Both paths preallocate `lm`, `pred`, `den`, `exp`, `target_mask`, `target_local`, `loss`, `softmax`, and `grad` before measurement. Both canonicalize ownership state and use the same `loss = log(den) - pred`, `softmax = exp/den`, and rank-local gradient operation. Both use a barrier and completed CUDA events around the same graph and preserve current-stream ordering.

## 9. Incumbent and replay

The incumbent remains `NO_INCUMBENT`. No deterministic promotion replay is claimed because the repeatability gate is unresolved. The immutable candidate correctness artifacts and shared timing artifacts remain auditable.

## 10. NSYS and Agent resume

No promotion NSYS/NCCL profile or profile-guided new Agent episode was started. This obeys the rule that profile-guided optimization begins only after a stable comparable score. No new Agent candidate was generated.

## 11. Regression artifacts

The shared runner retains both rank records, requires matching configurations and sample counts, and fails closed on incomplete rows. Phase 15-B.2 coherent-fixture/ownership validation remains in force. JSON timing artifacts validate successfully and Python syntax checks pass.

## 12. Historical provenance and integrity

Phase 15-A through 15-B.5 artifacts were preserved. The old baseline is explicitly legacy and scope-mismatched. The semantic optimization contract remains unchanged. Megatron was verified at `5be9626709af2722333bf54797c954c09edeada3` with a clean working tree.

## Remaining limitation

The next task is to explain and control the intermittent rank-completion outliers—likely distributed scheduling/measurement synchronization, but not asserted without trace evidence—then rerun the same shared era. Until that is resolved, the observed candidate ratios must not become an incumbent or trigger profile-guided Agent work.
