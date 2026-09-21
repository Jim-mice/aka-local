# Phase 15-B.4 Distributed Benchmark Recovery

## Verdict

The missing-row failure was repaired. Episode 7, Episode 8, and Episode 10 now produce complete TP=2 raw timing records from both ranks, and the collector aggregates only complete rank/configuration sets. The frozen scientific contract `112959ca63020e9b` and baseline `f11fd1fc6846d6f2` were not changed.

No incumbent is promoted in this repair report. Candidate timing is retained as recovery evidence; a promotion requires an explicitly comparable run against the frozen baseline methodology and the existing campaign policy.

## 1. Phase 15-B.3 blocker and root cause

The first candidate benchmark produced no usable rows because the remote harness allocated `target_mask` as `[rows]`, while the immutable Episode 7 ABI implementation writes local element positions over `[rows, local_vocab]`. The result was an asynchronous CUDA/device failure before the rank summaries were emitted. The old controller also did not persist rank-local raw records or child exit status in a structured result.

Two additional measurement defects were found during recovery: per-stage `torch.cuda.synchronize()` calls polluted the timed interval, and CUDA event completion was queried before an explicit post-event synchronization. The former produced the earlier approximately <REMOTE_HOST> ms readings; the latter produced `CUDA error: device not ready` after the synchronization removal.

## 2. Frozen provenance

| Item | Value |
|---|---|
| Megatron commit | `5be9626709af2722333bf54797c954c09edeada3` |
| optimization contract | `112959ca63020e9b` |
| frozen baseline | `f11fd1fc6846d6f2` |
| evaluator | `tp2_semantic_v2` / `7cc0fac8d80b2d27` |
| TP | 2 real NCCL ranks |
| collective invariant | 1 MAX all-reduce + 2 SUM all-reduces |
| candidate set | Episodes 7, 8, 10; corrected TP=2 valid |

The Phase 15-A baseline is independent of the defective candidate evaluator and remains valid. Episodes 6/9 remain `TARGET_LOGIT_WRONG` and were not benchmarked. The golden implementation remains excluded from score and incumbent decisions.

## 3. Timing schema and aggregation

`targets/megatron_5be9626/vocab_parallel_cross_entropy/timing_row_schema.json` defines the required row provenance. Each rank emits config-local samples and the controller retains the raw rank records. For each iteration, the distributed latency is `max(rank0_latency, rank1_latency)`, representing the slower rank's completion. A score is invalid if either rank, any configuration, or any iteration is missing or mismatched; the result is then `INCOMPLETE_DISTRIBUTED_TIMING`.

The repaired runner writes `tp2_benchmark_v2.txt` with stdout, stderr, and exit code, then writes `tp2_benchmark_v2.json` only with `valid_complete_rows=true` when both ranks and all three configurations are present.

## 4. Barrier and CUDA-event audit

Each warmup and measurement enters the same distributed barrier before the operation. The measured stream sequence is:

`local_max` → real MAX all-reduce → `local_prepare` → real SUM(target) → real SUM(denominator) → saved-softmax/local-gradient arithmetic.

The start event is recorded after the pre-measurement barrier and a completion synchronization; the end event is recorded after the final stream operation and synchronized before elapsed time is queried. Allocations, fixture generation, compilation/import, and warmups are outside the interval. No rank exits the loop early on a successful run.

## 5. Recovery results

All three eligible candidates produced complete rows:

| Episode | Config means (us, rows 8/64/256) | Complete |
|---:|---:|:---:|
| 7 | 2459.03, 2544.03, 2630.04 | PASS |
| 8 | 2616.93, 2627.58, 2641.92 | PASS |
| 10 | 2615.71, 2625.53, 2695.99 | PASS |

Episode 7 independent recovery runs were complete in both cases. The first repaired run was approximately `2459/2544/2630 us`; the second was `2593/2612/2627 us`. Both retain five samples/configuration and rank-0/rank-1 raw evidence.

The corresponding candidate-vs-frozen-baseline ratios, using the existing baseline means, are:

| Episode | Per-config ratio | Geometric mean |
|---:|---|---:|
| 7 | 1.0697, 1.2015, 1.0707 | 1.1123x |
| 8 | 1.0600, 1.1945, 1.0647 | 1.1047x |
| 10 | 1.0605, 1.1954, 1.0434 | 1.0977x |

These ratios are recorded for recovery comparison, but are not promoted as a new scientific score until the exact frozen baseline timing implementation and candidate timing scope are jointly re-audited. In particular, the candidate harness executes the frozen local stage/collective graph and local gradient arithmetic, while the baseline manifest describes the Phase 15-A real Megatron forward-plus-backward measurement. No claim is made that the two paths are identical merely because rows now exist.

## 6. Episode 11

Episode 11 was not regenerated. Its source and hypothesis are preserved. It was generated after the corrected evaluator but no corrected TP=2 correctness artifact was available during this recovery, so it was not benchmarked or included in the eligible set.

## 7. NSYS, diagnostic, incumbent, and replay status

The benchmark harness is now structurally ready for a valid profile: it retains both rank records and fails closed. A candidate NSYS/NCCL trace and a deterministic performance replay were not claimed in this repair artifact because the recovered timing scope still requires the explicit baseline-equivalence audit above. Therefore the incumbent remains `NO_INCUMBENT`; no candidate is promoted from incomplete or incomparable evidence.

## 8. Regression and failure containment

The repaired collector rejects absent ranks, unequal row counts, mismatched configuration IDs, and invalid child exit status. The remote launcher reports rank exit codes and PyTorch elastic cleanup. The Phase 15-B.2 coherent-fixture and target-ownership fixes remain in the evaluator; Episode 7/8/10 corrected validation artifacts still reference evaluator `7cc0fac8d80b2d27`. Episodes 6/9 were not admitted to timing.

## 9. Knowledge and integrity

The new timing schema and recovery records belong to the vocab-parallel CE target namespace only. Historical Phase 15-B, 15-B.1, 15-B.2, and 15-B.3 artifacts were not overwritten. The authoritative Megatron checkout was verified at `5be9626709af2722333bf54797c954c09edeada3` with a clean working tree.

## Remaining limitations

The missing-row harness defect is fixed and complete two-rank rows are reproducible. The remaining gate before a defensible incumbent is an apples-to-apples scope audit between the Phase 15-A baseline driver and the repaired candidate driver, followed by comparable candidate/baseline NSYS traces. No new Agent episode was generated before that gate.
