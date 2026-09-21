# Phase 19-A — Five-Target Campaign Closure

## Purpose and canonical states

Phase 19-A closes the original five real Megatron targets without new Agent
episodes. On-disk artifacts are authoritative. The registry is
`real_target_campaign_index.json` and machine entry point is
`five_target_campaign_state.json`.

| Original target | Canonical target | State |
|---|---|---|
| SwiGLU | Megatron MLP SwiGLU activation boundary | `COMPLETED_INTEGRATION_LIMITED` |
| Vocab-Parallel CE | Megatron Vocab-Parallel Cross Entropy | forward `FORWARD_CAMPAIGN_FROZEN`; backward `ENVIRONMENT_STABILITY_BLOCKED` |
| Residual Add RMSNorm | non-TE WrappedTorchNorm → torch.nn.RMSNorm | `STABILITY_BLOCKED_R1_R2` |
| Dense Fused Attention | Native DotProductAttention dense/no-mask/p=0 core | `AGGREGATE_WIN_PER_CONFIG_MIXED` |
| MoE Grouped GEMM | Native SequentialMLP Expert Compute | `STABILITY_BLOCKED` |

## Provenance and terminology

CE forward Episode 7 is `PROMOTED_SCORE`. CE backward B1, RMSNorm R1/R2, and
MoE M4 remain raw/non-promotable. Attention A2 is stability-qualified but
per-configuration mixed and not promoted. Earlier claims that attention A2
NSYS or MoE replay/oracle were missing are `STALE_SUMMARY_SUPERSEDED`; no raw
historical evidence was rewritten. Contract/path audit is recorded in
`PHASE19A_CONTRACT_HASH_AUDIT.json`.

## M4 diagnostic-only NSYS and scaling

Diagnostic-only NSYS captured real SequentialMLP reference and M4 for all
official distributions with zero NCCL. Reference has 13 balanced, 13 moderate,
and 7 two-empty instances: small GEMMs, SiLU/gated activation, and concat. M4
has one `moe_sequential_expert_kernel` for each distribution, at approximately
1.645 ms, 1.650 ms, and 1.596 ms. This is `DIAGNOSTIC_ONLY_NOT_PROMOTION`.

M4 raw ratios are 1.2793x balanced, 1.4816x moderate, and 0.9573x two-empty.
Source and trace support `LAUNCH_REDUCTION`, `ACTIVATION_FUSION`,
`MANY_SMALL_GEMMS`, `EMPTY_EXPERT_OVERHEAD`, and `IMBALANCE_SENSITIVITY`.
M4 remains near fixed-cost when experts are empty while reference skips empty
expert work. No memory/occupancy claim is made.

Two-empty reference CV is 0.3733. The preserved largest sample is 4.9899 ms at
invocation 2/block 2/sample 0; its preceding reference was 1.0963 ms and its
ABBA candidate neighbors were 1.1769/1.1711 ms. It is reference-only in that
neighborhood but causation is `UNKNOWN`. No samples were deleted.

## Stability and promotion infrastructure

`lab/core/environment_qualification.py` and its policy implement read-only
hardware/process provenance plus a reference-only readiness probe. `READY`,
`NOISY`, and `UNKNOWN` never replace frozen CV qualification. Cross-target
evidence is in `CROSS_TARGET_STABILITY_AUDIT.md`; common physical-GPU causation
is not established.

`SCIENTIFIC_PROMOTION_GATE_REGISTRY.md` freezes source identity → contracts →
delivery → toolchain → correctness → scope → environment → reference/candidate
stability → promotion profile → replay → integration. Delivery regressions
cover M1-like wrong symbol, M2-like missing marker, hash and ABI mismatch,
CUDA11.8 forbidden construct, and valid evaluator-only input. They pass
deterministically, as do environment probe and registry JSON validation. The
CLI audit found target-specific replay/benchmark/profile scripts rather than a
single uniform non-destructive CLI for all five targets; this is documented as
a reproducibility infrastructure gap, not silently claimed as complete.

## Next legal actions

- SwiGLU: only a newly authorized FC1/custom-GEMM research phase.
- CE forward: remain frozen.
- CE backward and RMSNorm: one independent clean-environment requalification under unchanged policy only.
- Attention: only a newly authorized vendor-GEMM-preserving/softmax-surrounding strategy phase; no A4.
- MoE: only one independent clean-environment M4 requalification under unchanged policy; no M5.

`CLEAN_ENVIRONMENT_REQUALIFICATION_PLAN.md` specifies the one-shot rule; it is
not authorization to rerun until a pass occurs.

## Integrity and open questions

Megatron HEAD is `5be9626709af2722333bf54797c954c09edeada3` and the working
tree is clean. No frozen score, threshold, candidate source, or authoritative
Megatron source changed. Open questions are noise attribution, a
vendor-GEMM-preserving attention strategy, and availability of a real grouped
GEMM backend without changing dependency truth.
