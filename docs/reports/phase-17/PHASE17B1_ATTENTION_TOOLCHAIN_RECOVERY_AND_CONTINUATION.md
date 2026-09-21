# Phase 17-B.1 — Attention Toolchain Recovery and Continuation

Status: `STABILITY_QUALIFIED_BUT_PER_CONFIG_MIXED` (A2); no unconditional attention incumbent.

Phase 17-B remains historically `BUILD_BLOCKED`: A1 is immutable and remains `REJECT_TOOLCHAIN`. Its source contained `CUDART_INF_F`, which CUDA 11.8 rejected as an undefined identifier. A generic preflight rule now rejects that construct before remote compilation. A CUDA 11.8 / sm70 probe verified a finite negative FP32 sentinel and `-INFINITY` compile and run successfully.

## A2 provenance

A2 was a newly generated toolchain-feedback-guided Agent episode; A1 was not edited or copied. A2 passed marker/ABI preflight, CUDA 11.8/sm70 build, official forward correctness, edge correctness, negative adapter checks, trusted reference-backward dQ/dK/dV compatibility, and zero-collective execution. Candidate source hash and all episode artifacts remain under `campaigns/targets/megatron_5be9626/megatron_native_dot_product_attention/episode_A2/`.

## Frozen contracts

Replay: `65058b45af7fb867a135d143b06e6600f75fbe7a13c523f351eef20d48741736`.
Performance: `5fc6c0031cd077ccc801ed1ae1397a7308e63c27760d7fc4ba3ddb6adf0d4f24`.
Optimization: `90839e70b4edbfdc9cdfd903fe9c51230311a8fc7c13fb1b2a0673acd261916c`.
Statistics: `692f9718d1e12d922561e6d8c69cef835a37cd9c0f6e087c2ab0b342e231ae41`.
Baseline: `764cc0e3145635091024b80ae07ce48f5e618f0fe8335d1cfc93645032e335db`.

## Complete paired result

The official corrected run contains 3 invocations × 3 blocks × 10 samples per implementation/configuration. No old or overcount diagnostic rows were mixed into the result. Both reference and candidate satisfy CV ≤ 0.20 for every configuration.

| configuration | reference mean (us) | candidate mean (us) | speedup |
|---|---:|---:|---:|
| 16×1×1024 | 777.830 | 131.210 | 5.928x |
| 64×2×1024 | 780.754 | 668.934 | 1.167x |
| 128×2×1024 | 802.111 | 2312.363 | 0.347x |

Geometric mean: `1.338883x`; paired block bootstrap CI95 (seed 1717, 10,000 resamples): `[1.307531x, 1.368066x]`. This is a policy-qualified aggregate improvement, but the candidate is materially slower at S=128. It must not be described as a uniform win.

## Runtime interpretation

A2 is a `FULL_CORE_CUSTOM_FP32_ROW_SOFTMAX` implementation. Phase 17-A reference NSYS remains the authoritative reference graph: vendor QK GEMM, Megatron softmax, conversion/local work, and vendor PV GEMM, with zero NCCL. A2's correctness and timing evidence prove behavior but do not by themselves prove a kernel-count reduction; candidate NSYS/profile provenance is therefore a remaining promotion-profile artifact. The S=128 regression is consistent with the candidate's serial/shared-memory strategy, but this is an implementation diagnosis, not an unsupported hardware bottleneck claim.

## Decision

A2 passes the frozen aggregate stability and uncertainty gates, subject to the project promotion rule requiring candidate profiling/provenance. No A3 is generated until a candidate NSYS exists and demonstrates an actionable next hypothesis. No attention source is patched upstream. The next authorized work is bounded candidate profiling, then Phase 18-A MoE qualification.

Prior CE and RMSNorm campaign states remain unchanged.
