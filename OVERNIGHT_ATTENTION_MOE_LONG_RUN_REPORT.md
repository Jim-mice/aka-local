# Overnight Attention / MoE Long-Run Report

## Current checkpoint

The run began with Phase 17-B `BUILD_BLOCKED`. A1 remains immutable and remains `REJECT_TOOLCHAIN` because CUDA 11.8/sm70 rejected `CUDART_INF_F`. Generic preflight was hardened to reject that construct before remote compilation. A standalone CUDA 11.8 probe passed with a finite negative FP32 sentinel and `-INFINITY`.

## Attention continuation

A2 was generated as a new toolchain-feedback-guided real Agent episode. It passed preflight, CUDA build, official forward correctness, edge correctness, negative adapter tests, trusted reference-backward dQ/dK/dV compatibility, and zero-collective execution. The corrected complete paired ABBA run used the frozen 3 invocation × 3 block × 10 sample policy and was not mixed with the preserved overcount diagnostic.

A2 statistics: 16×1×1024 reference/candidate 777.830/131.210 us, 5.928x; 64×2×1024 780.754/668.934 us, 1.167x; 128×2×1024 802.111/2312.363 us, 0.347x. All CVs passed ≤0.20. Aggregate geometric mean was 1.338883x with bootstrap CI95 [1.307531x, 1.368066x]. The result is scientifically qualified as a mixed, configuration-dependent aggregate, not a uniform win. A2 profile provenance remains the next attention artifact before any incumbent claim; A3 is not authorized without profile/failure evidence.

## MoE Phase 18-A / 18-B

Exact source mapping confirms that “MoE Grouped GEMM” is not one unconditional operator. The source path is router → dispatcher/permutation/optional communication → `TEGroupedMLP` or `SequentialMLP` → combine. TE grouped linear is dependency-blocked in the established environment. The source-valid fallback is `Megatron Native SequentialMLP Expert Compute`, whose `forward` splits by `tokens_per_expert`, invokes each local expert sequentially, then concatenates. FC1, activation/SwiGLU, and FC2 are distinct semantic stages.

Phase 18-A completed as `QUALIFIED_FOR_AGENT_CAMPAIGN`. The real replay used E=4, H=64, intermediate=128, FP16, TP=1/EP=1 and distributions [4,4,4,4], [1,3,5,7], [0,0,8,8]. Independent FP32 oracle max absolute errors were 2.84e-6, 3.05e-6, and 3.91e-6; input gradients were finite. Qualification means were 4144.2, 4068.4, and 3689.2 us. The complete 90-sample-per-distribution reference policy rerun passed with CV 0.0518, 0.0501, and 0.0636.

REAL NSYS qdrep evidence showed 13 kernel instances balanced, 13 under moderate imbalance, and 7 with two empty experts, including expert GEMM families, fused SiLU pointwise work, and concatenation; no NCCL. Contracts are recorded in the canonical target namespace, with replay hash `e73232a4e11649d6519785c79ab4a823eddbd5588e2e318edbdae3abb99ecaec` and performance hash `3086f89790979dac0e1a1918b9fb51953ac292820db6f14bbf414312858bdaef`.

Phase 18-B began under the authorized bounded policy. Real M1 failed contract preflight because it exported `candidate` instead of the required semantic symbol. Failure-guided M2 then exported the symbol but lacked the required semantic target marker, so it also failed `REJECT_CONTRACT`. Neither candidate was manually repaired, compiled, benchmarked, or promoted. The bounded M1/M2 budget is exhausted; no MoE incumbent exists.

## Frozen campaigns and integrity

CE forward remains `FORWARD_CAMPAIGN_FROZEN`; CE backward remains `ENVIRONMENT_STABILITY_BLOCKED`; RMSNorm remains `STABILITY_BLOCKED_R1_R2`. No prior score was rewritten. The authoritative Megatron checkout is at `5be9626709af2722333bf54797c954c09edeada3` with clean working tree.

## Phase 18-B.1 delivery hardening correction

`STALE_SUMMARY_SUPERSEDED`: the earlier statements that Attention A2 NSYS and
MoE replay/oracle were missing are no longer current. A2 NSYS is complete;
Phase 18-A MoE replay, independent oracle, and NSYS are complete.

M3 passed delivery and CUDA build but failed correctness with max absolute
errors 0.01416, 0.01697, and 0.00965. M4 was then generated as the final
authorized episode. M4 passed exact ABI/marker delivery, CUDA 11.8/sm70 build,
and official oracle correctness (max abs 1.88e-6, 2.84e-6, 2.73e-6). Its new
complete frozen-policy benchmark is `STABILITY_BLOCKED`: the two-empty
reference distribution had CV 0.3733, while the immutable gate is CV <= 0.20.
No M4 score, incumbent, promotion NSYS, or integration claim exists. No M5 is
authorized.

## Remaining blockers

1. Attention has no unconditional incumbent because its qualified aggregate is configuration-mixed.
2. MoE Agent promotion is blocked by frozen reference stability for the two-empty distribution.
3. Transformer Engine grouped GEMM remains unavailable and must not be faked.
