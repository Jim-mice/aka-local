# Phase 15-B.8 CE Forward Campaign Closure

## Verdict

The bounded profile-guided closure is complete. Episode 7 remains the
stability-qualified TP=2 incumbent at `2.4771907400291515x`. Episode 11 and
Episode 12 both passed the corrected TP=2 correctness gates and the frozen
statistical policy, but neither exceeded Episode 7. Episode 13 was not run:
Episode 12 produced no actionable profile delta and did not improve the
incumbent.

The forward CE campaign is now explicitly frozen. No Episode 14+ should be
generated without a new phase and contract decision.

## 1. Stable Phase 15-B.7 starting point

The starting point was `tp2_comparable_v1` with performance contract
`tp2_scope_v1` and statistical policy `tp2_stability_v1`. The semantic
optimization contract remained `112959ca63020e9b`. The corrected evaluator was
`tp2_semantic_v2`, hash `7cc0fac8d80b2d27`, with fixture
`coherent_global_fixture_v2`.

Episode 7 had three independent ABBA blocks, complete two-rank rows, and a
paired bootstrap CI95 of `[2.465431157986491x, 2.484391708222999x]`. Earlier
slow samples were retained and classified as valid rank-skew/collective-delay
observations.

## 2. Episode 7 incumbent provenance

The incumbent manifest binds:

- Megatron commit `5be9626709af2722333bf54797c954c09edeada3`
- semantic contract `112959ca63020e9b`
- performance contract hash `6632730d9957fbb1a0da4268f2b3c168b2836862db167935d8908758a00b5bea`
- statistical policy hash `a74bf021f535c6de48b3b37b9568bdf06df5ff77075b3e6afd652cf96fa0dc05`
- active baseline hash `587a9dbd4f2815b6e9e8608ff1b6e622623bb73fc67d678a816cd04ac84ba777`
- evaluator hash `7cc0fac8d80b2d27`
- fixture `coherent_global_fixture_v2`
- candidate hash `947e17aa55f8cbb6d6ab85e8e671e0a1f1be26c7ef54a5bec727d5839549d5c6`
- TP=2, FP16, official configurations `[[8,1,32],[32,2,64],[128,2,128]]`
- collective structure `1 MAX + 2 SUM`
- stable score and bootstrap evidence.

The legacy Phase 15-A baseline `f11fd1fc6846d6f2` remains excluded because
its timing scope was not equivalent.

## 3. Episode 11 resolution

Episode 11 was not regenerated or manually repaired. Its immutable source
hash is `3792b0200c828d70a1e45ef00c8a7cebe0a6222aeb7b76ecac14182f96ab4f68`.
The final corrected evaluator run passed on both ranks:

| check | result |
|---|---:|
| preflight / compile | PASS |
| TP=1 sanity | PASS |
| TP=2 loss max absolute error | 0.0019044876098632812 |
| TP=2 saved-softmax max absolute error | 1.4901161193847656e-8 |
| TP=2 rank-local gradient max absolute error | 1.4901161193847656e-8 |
| collective structure | 1 MAX + 2 SUM |

Episode 11 then completed the same three-block ABBA policy. Its stable
geometric-mean speedup was `2.473246856646392x`, CI95
`[2.4691712380220494x, 2.4809308996501547x]`. It is valid but not promoted.

## 4. Episode 7 profile decomposition

The promotion NSYS artifacts are the four rank-specific files
`nsys_stats_reference_rank*.txt` and `nsys_stats_candidate_rank*.txt`.
The structured summary is `promotion_nsys_summary_15b8.json`.

The candidate trace contains the two custom local kernels and real NCCL
all-reduce kernels. In the profiled aggregate, candidate rank 0 reported:

- custom local kernels: 60 instances, 624,209 ns total;
- NCCL SUM kernel family: 121 instances, 330,125,573 ns total;
- total GPU kernel time in the report: 337,608,794 ns.

The reference rank 0 reported no candidate custom kernels and 529,254,488 ns
across 121 NCCL SUM-kernel instances, with 542,112,115 ns total GPU kernel
time in the report. Rank 1 reports a complementary distribution of the
collective/wait activity; it must not be averaged away.

The stage diagnostic independently observed normal candidate local stages of
approximately 78--96 us for local max and 12--15 us for prepare, with normal
SUM stages around 24--34 us. A representative slow event showed one SUM
stage around 2.6 ms and complementary waiting on the other rank. This is
evidence for collective delay/synchronization, not a measured local-kernel
failure.

## 5. Communication Amdahl analysis

The stable comparable boundary is approximately 13.4 ms for the reference
and 5.4 ms for Episode 7, yielding the measured 2.47719x full TP=2 speedup.
Episode 7 has already removed most of the replaceable PyTorch local
reduction/index/elementwise work. The remaining critical path includes the
three mandatory NCCL collectives and rank synchronization.

The normal two custom local stages are only about 0.09--0.11 ms in the stage
diagnostic. Therefore, even an ideal elimination of those two local launches
would provide only roughly a 1.02x incremental ceiling relative to the
already-optimized 5.4 ms boundary. This is a profile-bounded estimate, not a
claim that all remaining time is network bandwidth. The direct NSYS evidence
supports the narrower conclusion that collective and synchronization activity
is now a major remaining component.

## 6. Structured diagnosis

The final diagnostic is in `diagnostic_15b8.json`:

- `COLLECTIVE_DOMINATED`: supported by the NCCL kernel share in the promotion
  profile and the local-vs-collective stage timings;
- `SYNCHRONIZATION_OVERHEAD`: supported by complementary rank waits and
  rank-skewed slow samples;
- `RANK_SKEW`: supported by preserved samples where rank 1 reached roughly
  8.3--11.1 ms while rank 0 remained around 2.8--3.0 ms.

No unsupported network bottleneck or NCU hardware diagnosis is claimed.

## 7. Episode 12 prompt evidence and hypothesis

Episode 12 was the only new Agent episode in this closure. Its prompt was
generated after the stable Episode 7 profile and included the active era,
statistical policy, incumbent score/CI, real NCCL/local-kernel evidence,
Amdahl interpretation, and the exact Episode 6/9 semantic failure reasons.
The prompt dry-run checked all mandatory ABI symbols, both collective-count
invariants, contract/evaluator/toolchain markers, and profile sections.

The Agent hypothesis was a bounded refinement: retain the two-entry ABI and
caller-owned collectives while using one row-per-block kernel with
warp-shuffle/shared-memory reductions. No candidate source was manually
edited.

## 8. Episode 12 correctness

Episode 12 source hash is
`a62f35782ec13afd19fc0bf973c535e1d92bf606cd3c2f33bcddf9ac9cdf762`.
Preflight, CUDA 11.8 compilation, TP=1, and corrected real TP=2 all passed.
Both ranks reported loss max absolute error `0.0019044876098632812`, saved
softmax max absolute error `1.4901161193847656e-8`, and gradient max absolute
error `1.4901161193847656e-8`. The runtime retained exactly one MAX and two
SUM collectives.

## 9. Episode 12 stable performance

Episode 12 used three ABBA blocks under the existing policy, with 30 samples
per configuration per implementation and no deletion of slow valid samples.
The designated raw runs are recorded in `stability_analysis_15b8_ep12.json`.
The extra diagnostic runs `A12_4` and `B12_4` are preserved but excluded from
the predeclared three-block score because they were outside the designated
ABBA schedule.

| block | geometric-mean speedup |
|---|---:|
| 1 | 2.46572198235247x |
| 2 | 2.428665425370318x |
| 3 | 2.4966954984984393x |
| aggregate | 2.4636943020737423x |

The paired bootstrap CI95 was `[2.428665425370318x, 2.4966954984984393x]`.
The result is `STABLE` under the frozen policy: all blocks were complete,
all ranks were present, candidate CV stayed below 10%, and the lower bound
was above 1.0. It nevertheless remains below Episode 7 and was not promoted.

## 10. Episode 13 decision

Episode 13 was not run. Episode 12 was correct and stable but did not improve
the incumbent, and no Episode 12 profile showed a concrete new local
opportunity beyond the already-profiled two-kernel path. Running a second
new Agent merely to consume the episode budget would violate the bounded
evidence-driven rule.

## 11. Final incumbent

Episode 7 remains the final incumbent. Episode 8 and Episode 10 remain
historically valid stable candidates at approximately `2.46657x` and
`2.43751x`, respectively. Episode 11 and Episode 12 are now fully resolved
and valid but lower-scoring. Episodes 6 and 9 remain excluded because their
corrected first divergence is `TARGET_LOGIT_WRONG`.

## 12. Final deterministic replay

`final_replay_15b8.json` and `final_replay_15b8_tp2.txt` bind the final
Episode 7 replay to the frozen Megatron commit, semantic/performance/
statistical contracts, active baseline, evaluator, fixture, candidate hash,
TP=2, FP16, official shape set, and `1 MAX + 2 SUM` structure. Both ranks
passed loss, saved-softmax, and local-gradient checks.

## 13. Final promotion NSYS/NCCL

The existing Episode 7 promotion NSYS is reusable: its workload and
provenance match the active performance contract and its candidate hash is
the final incumbent hash. It proves the custom `local_max_kernel` and
`prepare_kernel` execute alongside real NCCL all-reduce kernels. The earlier
stage probe remains diagnostic-only and was not substituted for promotion
NSYS.

## 14. Backward-local opportunity analysis

Backward remains outside the forward Agent contract and was not optimized.
At this commit it has no collective and consumes saved softmax, target mask,
and masked/local target metadata. The common reference implementation uses
softmax normalization, target-metadata canonicalization, a gradient copy, and
owned-target subtraction.

A correctness-only stage diagnostic at the representative `[64,64]` local
configuration measured the backward-labelled interval around 0.87 ms on one
rank and 5.32 ms on the other rank; those values include rank waiting because
the stage events were deliberately inserted around distributed execution.
They are evidence that backward is a meaningful future boundary, not an
isolated kernel benchmark. The current promotion NSYS aggregate reports
total `cudaLaunchKernel` API activity (not a backward-specific count), so an
exact backward kernel count is not claimed. A future Phase 15-C should add
explicit backward NVTX/range provenance before optimizing.

## 15. Diminishing-return analysis

Across the valid candidates, Episode 7 is best at 2.47719x, Episode 11 is
2.47325x, Episode 8 is approximately 2.46657x, Episode 12 is 2.46369x, and
Episode 10 is approximately 2.43751x. The spread among the top two is about
0.16 percentage points of speedup, well within the scale where the frozen
stability/CI evidence matters. Episode 12's bounded local refinement did not
convert the remaining collective/synchronization floor into a better full
operator score.

The forward-only local CE search therefore has diminishing returns under the
current three-collective TP=2 boundary. The useful next optimization target,
if desired, is a separately contracted backward-local boundary or a proven
collective/overlap design; neither is started here.

## 16. Forward campaign freeze metadata

`campaign_state_15b8.json` records `FORWARD_CAMPAIGN_FROZEN`, final Episode 7
incumbency, all active hashes, fixture, TP/dtype, and collective structure.
It explicitly forbids Episode 14+ without a new phase and contract.

## 17. Framework lessons and knowledge changes

The CE knowledge namespace now records the complete chain: exact ABI,
toolchain preflight, golden semantic validation, coherent distributed
fixtures, fail-closed rank timing, explicit performance-scope contracts,
comparable eras, statistical qualification, and profile-before-Agent
generation. Candidate performance lessons remain separate from evaluator and
benchmark-infrastructure failures.

No standalone-operator or SwiGLU score history was modified.

## 18. Upstream integrity

At closure, `<LOCAL_USER_HOME>\projects\megatron-lm` was checked for the
required HEAD `5be9626709af2722333bf54797c954c09edeada3` and a clean working
tree. No authoritative Megatron source was modified.

## 19. Remaining limitations

The NSYS files are aggregate per-rank summaries, so they identify the
collective/local structure and delay pattern but do not provide a complete
event-range accounting of every microsecond in the timed boundary. The
backward diagnostic includes synchronization waits and does not claim a pure
backward kernel latency or count. The final result is a TP=2 V100 FP16
forward/local-backward study; it does not cover other TP sizes, FP8,
Transformer Engine, or a backward optimization campaign.
