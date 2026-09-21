# Phase 15-B.7 Distributed Benchmark Stability

## Verdict

`tp2_comparable_v1` is stability-qualified under the predeclared policy `tp2_stability_v1`. Episodes 7, 8, and 10 each have three independent ABBA blocks, complete two-rank records, correctness PASS, and paired bootstrap lower bounds above 1.0. Episode 7 is the best stable candidate and is promoted as the TP=2 incumbent. All raw measurements, including earlier slow events, are preserved.

## 1. Provisional Phase 15-B.6 state

Phase 15-B.6 established scope equivalence but had only short runs and observed intermittent approximately 6.6 ms candidate excursions. The old Phase 15-A baseline `f11fd1fc6846d6f2` remains legacy scope-mismatched and was not used. The new policy and all comparisons below use only `tp2_comparable_v1`.

## 2. Episode 7 outlier reconstruction

The preserved Episode 7 raw rows show that the large excursions are rank-skew events, not simultaneous two-GPU slowdowns. In one earlier repeat, the largest configuration had rank 0 near `3.0 ms` and rank 1 near `11.1 ms`; the distributed max was therefore `11.1 ms`. In another repeat, the middle configuration had rank 0 near `2.85 ms` and rank 1 near `8.33 ms`. The normal rank-1 candidate path is approximately `5.4 ms` while rank 0 is approximately `2.8–3.1 ms`.

The new three-block ABBA collection did not reproduce a candidate slow outlier. The earlier samples remain valid observations and are classified as `RANK_SKEW` / `COLLECTIVE_DELAY`, not deleted.

## 3. Rank-skew analysis

For each raw iteration the harness retains rank 0, rank 1, max latency, and implicitly the absolute difference. Episode 7 ABBA candidate rank-skew means were approximately `2560`, `2573`, and `2552 us` for the three blocks; the rank-1 path normally determines the distributed max. The earlier slow samples increased the skew sharply, confirming one-rank stalls.

## 4. Stage-level jitter attribution

The diagnostic stage run was not used for scoring. It showed stable local kernels and a single collective anomaly: one rank had a `SUM` stage around `2.6 ms` while neighboring SUM stages were about `25–35 us`; the other rank showed the complementary wait behavior. This is evidence for collective/synchronization delay rather than a candidate local-kernel computation change. The `total` diagnostic event was intentionally not used because the stage probe records per-stage events and is a diagnostic instrument, not the primary contract.

## 5. GPU/system provenance

Read-only remote snapshot:

| GPU | PCI bus | temperature | power | SM clock | memory clock | P-state | utilization |
|---:|---|---:|---:|---:|---:|---|---:|
| 0 | `00000000:08:00.0` | 77 C | 96.38 W / 250 W | 37 MHz | 877 MHz | P0 | 0% |
| 1 | `00000000:84:00.0` | 49 C | 46.70 W / 250 W | 363 MHz | 877 MHz | P0 | 100% |

This snapshot was captured after experiments and is provenance, not proof that temperature caused any particular sample. No clocks, power limits, persistence mode, drivers, or system settings were changed. No NCCL/CUDA/MASTER/OMP environment overrides were present in the inspected shell. GPU topology reports `SYS` between the two PCIe V100s, with separate NUMA affinities.

## 6. Device mapping and NCCL audit

The launcher uses `CUDA_VISIBLE_DEVICES=0,1`; rank 0 maps to local device 0 and rank 1 to local device 1 in every comparable run. The process group is NCCL and the contract retains exactly one MAX and two SUM all-reduces. The `SYS` topology is a relevant communication-latency factor, but no network or NCCL bottleneck is asserted beyond the measured collective-delay evidence.

## 7. Warmup and long-run design

Each run uses five warmups and five measured samples per configuration. Three independent ABBA blocks provide 30 samples per implementation/configuration: A-reference, B-candidate, B-candidate, A-reference. The candidate distributions are stable in the collected blocks; reference configuration 2 contains valid higher-latency samples, but remains below the predeclared 10% CV threshold at the combined block level.

## 8. Statistical policy

Artifact: `targets/megatron_5be9626/vocab_parallel_cross_entropy/statistical_policy.json`

- version: `tp2_stability_v1`
- benchmark era: `tp2_comparable_v1`
- estimator: arithmetic mean latency
- diagnostic statistics: N, mean, standard deviation, CV, median, min, max, MAD, rank skew
- no slow valid sample deletion
- invalid only for rank failure, missing row, CUDA/NCCL error, fixture/config mismatch, or explicit independent hardware fault
- uncertainty: paired three-block bootstrap, seed 1515, 10,000 resamples
- minimum blocks: 3; minimum samples/config: 15; maximum config CV: 0.10; bootstrap lower bound >1.0
- policy hash: `a74bf021f535c6de48b3b37b9568bdf06df5ff77075b3e6afd652cf96fa0dc05` (canonical placeholder-field hash)

## 9. Stability-qualified results

| Episode | block geometric means | mean geometric mean | bootstrap CI95 | verdict |
|---:|---|---:|---|:---:|
| 7 | 2.4817, 2.4844, 2.4654 | 2.4772x | [2.4654, 2.4844] | STABLE |
| 8 | measured in identical ABBA policy | 2.4666x | lower bound 2.4659 | STABLE |
| 10 | measured in identical ABBA policy | 2.4375x | lower bound 2.4136 | STABLE |

All rows are TP=2 end-to-end comparable-boundary observations, not old-baseline comparisons. Per-configuration raw summaries are in `stability_analysis_15b7_ep7.json`, `...ep8.json`, and `...ep10.json`.

## 10. Correctness and Episode 11

Episodes 7, 8, and 10 pass shared loss, saved-softmax, and rank-local gradient checks. Episode 11 was not admitted: its final corrected evaluator invocation did not emit a completed result before the bounded remote run ended, so it remains unresolved and unbenchmarked. It is not part of the incumbent.

## 11. Incumbent decision

Episode 7 is the best stability-qualified candidate. The incumbent manifest now binds candidate hash `947e17aa55f8cbb6d6ab85e8e671e0a1f1be26c7ef54a5bec727d5839549d5c6` to the new performance contract, statistical policy, evaluator, fixture, baseline era, and TP=2 score `2.4771907400291515x`. The legacy baseline is explicitly excluded.

## 12. NSYS/NCCL promotion evidence

Promotion profiles were captured for reference and Episode 7 under the same comparable workload using per-rank `nsys profile --trace=cuda,nvtx,osrt --sample=none` wrappers. Candidate rank 1 contains the expected custom `local_max_kernel` and `prepare_kernel` plus real `ncclDevKernel_AllReduce_Sum_f32_RING_LL`; reference rank 0 is dominated by the corresponding NCCL all-reduce family and PyTorch reduction/index kernels. The exact stats are in `nsys_stats_reference_rank*.txt` and `nsys_stats_candidate_rank*.txt`.

This profile is separate from the earlier jitter diagnostic profile. It is promotion evidence for the stable candidate, while the stage probe is only jitter attribution.

## 13. Communication/Amdahl interpretation

The stable shared boundary is roughly `13.4 ms` reference versus `5.4 ms` Episode 7. NSYS shows the reference has much more local PyTorch reduction/index work, while the candidate retains real NCCL collectives and replaces local stages with two custom kernels. The candidate result is a full distributed speedup, not a local-kernel-only score. Exact NCCL fraction attribution requires event ranges or trace interval accounting beyond the aggregate kernel summary; no unsupported “network-bound” claim is made.

## 14. Replay and provenance

`deterministic_replay_tp2_comparable_v1.json` binds the stable replay to the Megatron commit, semantic contract, performance contract, statistical policy, evaluator, fixture, candidate hash, TP, dtype, shapes, and collective structure. Golden timing remains excluded.

## 15. Regression and history

Raw rows are preserved. The benchmark retains both rank rows, fails closed on incomplete data, preserves coherent fixtures and target ownership, and does not silently delete outliers. Phase 15-A through Phase 15-B.6 artifacts remain intact; `f11fd1fc6846d6f2` remains historical legacy scope-mismatch evidence.

## 16. Agent resume status

No new Episode 12/13 was generated in this phase. A stable score and promotion profiles now exist, but a later Agent prompt must still be generated from these artifacts with the required diagnosis and uncertainty sections before resuming optimization.

## 17. Upstream integrity

Megatron remains at `5be9626709af2722333bf54797c954c09edeada3` with a clean working tree.
