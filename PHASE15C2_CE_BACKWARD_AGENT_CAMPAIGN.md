# Phase 15-C.2 — CE Backward Agent Campaign

Status: PARTIAL / STABILITY-BLOCKED. No backward incumbent was promoted and no forward artifact was modified.

## Starting point and active contract

Phase 15-C.1 qualified the real Megatron rank-local, no-label-smoothing backward boundary. The active backward optimization contract is `6121f49401f3ef4601549c8732a62870ff17a7a7d7047e9c55d55b79c9ff6e92`; it binds the Phase 15-C.1 replay contract `4841f90840fee98eeff354e999fd25a1626c99e561cd5402b0dc68ec8f0d94da`, performance contract `afbfcff56ee6891b888ca7c6fbf632b74ccfab71f849d7934c56dc1d4bd9e205`, and statistical policy `7a750bc7f4f3c7b768bd5b35d695dab6762131d4c6681d6925100bf5557cf18f`.

The active ABI is out-of-place: real saved FP32 softmax is read-only; a separate preallocated FP32 output is supplied. The reference copies the template to that output outside timing and uses the exact Megatron helper path. A candidate may eliminate reference temporaries but must not mutate the saved input, allocate output inside timing, or launch NCCL.

## Episodes and correctness

| Episode | Preflight | CUDA build | REAL TP=2 correctness | Result |
|---|---|---|---|---|
| B1 | PASS | PASS after harness command correction | PASS | Benchmarked; stability blocked |
| B2 | PASS | PASS | PASS | Correct but not benchmarked after B1 stability failure |
| B3 | PASS | PASS | PASS | Correct but not benchmarked after B1 stability failure |

B1's first two build records remain preserved: the framework's `-std=c++17` and `--std=c++17` invocations were rejected by the audited remote nvcc. The candidate source was not edited. A revalidation using the accepted no-standard-flag invocation compiled and passed all official TP=2 gradients. The contract marker, `extern "C"` symbol, FP32/bool/int64 ABI, and forbidden `CUDART_INF_F` guard were checked before build.

All TP=2 official configurations passed against the exact Megatron helper oracle with direct FP32 tolerance `1e-5`; saved softmax remained byte-identical. Candidate boundary contains zero collectives by construction and evaluator policy.

## B1 paired benchmark

The frozen policy was used: three independent invocations, five warmups, three blocks of ten samples per rank, and A-B-B-A ordering. All raw rows are preserved in `campaigns/targets/megatron_5be9626/vocab_parallel_cross_entropy_backward/episode_B1/paired_benchmark_raw.json`.

| Config SxBxV | Reference proxy us | Candidate proxy us | Raw proxy speedup |
|---|---:|---:|---:|
| 8x1x32 | 312.076 | 115.115 | 2.7110x |
| 32x2x64 | 312.063 | 120.883 | 2.5815x |
| 128x2x128 | 283.990 | 118.858 | 2.3893x |

Raw geometric mean: 2.55717x. This is not a publishable or promotable score.

The frozen stability gate requires maximum per-rank CV <= 0.20. B1 candidate CVs passed (0.107, 0.167, 0.185), but the paired reference rank-0 CV was 0.209 for 8x1x32 and 0.223 for 32x2x64. Valid slow samples were retained, including reference maxima 717.856 us and 1010.688 us. Consequently the entire paired comparison is `STABILITY_BLOCKED`; no CI, incumbent, deterministic replay, promotion NSYS, sidecar integration, or combined training score is claimed.

## Kernel graph and diagnosis

The Phase 15-C.1 reference NSYS graph has about eight local kernel families (arange, indexed access/update, copy/fill, multiply, and tensor-iterator arithmetic) and zero NCCL kernels. B1 source is one direct CUDA launch for the semantic update. This is a source/build observation, not a completed candidate NSYS proof; because stability qualification failed, promotion profiling was deliberately not run.

Evidence-backed diagnosis remains `MULTI_KERNEL_OVERHEAD` and `LOCAL_KERNEL_LATENCY`. The data does not justify memory-bound, compute-bound, occupancy, or register conclusions.

## Provenance, isolation, and next condition

The forward Episode 7 campaign state was re-read as `FORWARD_CAMPAIGN_FROZEN`; no forward score, candidate, contracts, or collective structure changed. Megatron remained at `5be9626709af2722333bf54797c954c09edeada3` with a clean worktree.

Backward knowledge is isolated under `knowledge/targets/megatron_5be9626/vocab_parallel_cross_entropy/backward/`. The current blocker is environmental/reference stability under the already-frozen statistical policy, not candidate correctness. A subsequent explicitly authorized continuation would need to collect another complete, equally paired block under recorded physical GPU state before any candidate can be promoted; it must not alter the statistical policy or delete B1 samples.
