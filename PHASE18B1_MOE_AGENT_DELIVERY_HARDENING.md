# Phase 18-B.1 MoE Agent Delivery Hardening

## Verdict

`STABILITY_BLOCKED`. M4 is the final authorized new episode. It passed the
hardened delivery gate, CUDA 11.8/sm70 compilation, and official forward
correctness, but the frozen reference/candidate campaign failed the reference
CV gate for the two-empty distribution. No incumbent or performance score is
claimed.

## Starting state and frozen contracts

Phase 18-A qualified `Megatron Native SequentialMLP Expert Compute`; TE grouped
GEMM remains dependency-blocked. The real boundary is local fixed-routing
expert compute, excluding router, permutation, communication, combine, and
backward. Replay hash: `e73232a4e11649d6519785c79ab4a823eddbd5588e2e318edbdae3abb99ecaec`.
Performance hash: `3086f89790979dac0e1a1918b9fb51953ac292820db6f14bbf414312858bdaef`.
Optimization contract SHA256: `a1bb8ee0af6c250663e0935ae3deb07104975fc5904c56bb5b2f0d4e8032d1fe`.
Statistical policy SHA256: `046d79bdd115a94346898fe3a146d7676b6f8c9edb23871120f170637e666479`.
The policy remained 3 invocations × 3 blocks × 10 samples, CV ≤ 0.20,
preserving all valid raw samples.

## M1/M2 immutable failures

- M1 source SHA256 `8f0552530bb831b28809912562452e2e86b9a235c3572c78acd993bd938e7080` exported `candidate`, producing the precise first divergence `MISSING_REQUIRED_SYMBOL` / `WRONG_EXPORTED_SYMBOL`.
- M2 source SHA256 `113a30309375ec776356434c103dc816daf3a36416e509540077a83f124af82` exported the required symbol but omitted the semantic marker, producing `MISSING_CONTRACT_MARKER`.

Neither was compiled, benchmarked, or manually repaired.

## Delivery contract and preflight

The frozen symbol is `moe_sequential_expert_forward_fp16_stream` with the exact
11-argument FP16/int64/current-stream prototype in
`required_abi.json` and `abi_skeleton.cuh`. The exact marker is
`// AKA_TARGET_CONTRACT: a1bb8ee0af6c250663e0935ae3deb07104975fc5904c56bb5b2f0d4e8032d1fe`.
`marker_spec.json` defines the machine-readable format. Generic
`lab/core/delivery.py` now distinguishes missing symbol, missing marker, hash
mismatch, ABI mismatch, and forbidden CUDA constructs before compilation.
Regression preflight classifies M1 and M2 correctly. The evaluator-only source
is explicitly marked non-candidate/non-benchmark/non-incumbent and compiled
successfully with CUDA 11.8 `nvcc -arch=sm_70 -O3`.

The prompt audit showed that M1 had the semantic entry point insufficiently
enforced and M2 treated the marker as documentation rather than a hard source
requirement. The hardened template puts the literal marker and full prototype
in a mandatory delivery header. Dry-run validation passed for marker,
prototype, all contract hashes, distributions, CUDA 11.8/sm70, zero-token
semantics, and routing/communication exclusions.

## M3

M3 was freshly generated after Tasks 0–10; it passed delivery preflight and
CUDA compilation. Its candidate SHA256 is
`dacde96cc107ba11fcc4093bf7d8aeb76e73e9d74c3c84d2715e2e9360d9febb`.
Independent FP32 oracle correctness failed: max absolute errors were
0.0141607, 0.0169661, and 0.00964587 for balanced, moderate, and two-empty
distributions. The failure was actionable semantic/layout evidence, not a
delivery failure, so M4 was authorized. No M3 benchmark or NSYS was run.

## M4

M4 was a new Agent output, not a source edit of M3. Source file:
`episode_M4/moe_sequential_expert_forward.cu`; SHA256
`1fb2921a0e0706d2e71bc42073036ca3b048661edaa7c1e76ae22f2e61e9e549`.
Delivery preflight and CUDA 11.8/sm70 build passed. Independent oracle results:

| distribution | max abs | max rel | finite |
|---|---:|---:|---|
| [4,4,4,4] | 1.8803e-6 | 4.7544e-4 | yes |
| [1,3,5,7] | 2.8396e-6 | 4.4785e-4 | yes |
| [0,0,8,8] | 2.7297e-6 | 4.5252e-4 | yes |

The candidate therefore passed the official forward correctness gate. The
first oversampled diagnostic benchmark was preserved separately and was not
used for scoring. A fresh complete frozen-policy run then collected 90 raw
samples per side and distribution. Results:

| distribution | reference mean/CV (us) | candidate mean/CV (us) | ratio |
|---|---:|---:|---:|
| [4,4,4,4] | 2042.12 / 0.0292 | 1596.30 / 0.0468 | 1.2793x |
| [1,3,5,7] | 2058.55 / 0.0424 | 1389.45 / 0.0495 | 1.4816x |
| [0,0,8,8] | 1124.45 / 0.3733 | 1174.64 / 0.0037 | 0.9573x |

The two-empty reference CV exceeds the immutable 0.20 gate. Consequently the
campaign is `STABILITY_BLOCKED`; no aggregate speedup, CI, incumbent, NSYS
promotion profile, or deterministic promotion replay is valid. The raw JSON
preserves every sample, including the valid slow reference excursions.

## Gates not reached

Because the reference stability gate failed, edge/negative/backward promotion
gates, promotion NSYS, and sidecar integration are not promotion evidence and
were not used to manufacture a score. No M5 was generated. M4 is not an
incumbent. The next legal step is a separately authorized clean-environment
requalification using the same frozen policy, not selective reruns.

## Report corrections and frozen campaigns

Earlier long-run text claiming that attention A2 NSYS or MoE replay/oracle was
missing is superseded: `STALE_SUMMARY_SUPERSEDED`. Attention A2 NSYS is
complete; Phase 18-A MoE replay, oracle, and NSYS are complete. CE forward
remains `FORWARD_CAMPAIGN_FROZEN`; CE backward remains
`ENVIRONMENT_STABILITY_BLOCKED`; RMSNorm remains `STABILITY_BLOCKED_R1_R2`;
attention remains `AGGREGATE_WIN_PER_CONFIG_MIXED` with no incumbent.

## Integrity

No authoritative Megatron source was modified. Required HEAD is
`5be9626709af2722333bf54797c954c09edeada3` and its working tree is clean.
M1–M4 sources and raw artifacts remain preserved in their episode directories.
