# Phase 19-A：五个 target 的 campaign 收尾

## 目的与规范状态

Phase 19-A 在不创建新的 Agent episode 的前提下，结束最初的五个真实 Megatron target。磁盘中的 artifact 是权威依据；规范注册表为 `real_target_campaign_index.json`，机器入口为 `five_target_campaign_state.json`。

| 原始 target | 真实研究目标 | 状态 |
|---|---|---|
| SwiGLU | Megatron MLP SwiGLU activation boundary | `COMPLETED_INTEGRATION_LIMITED` |
| Vocab-Parallel CE | Megatron Vocab-Parallel Cross Entropy | forward `FORWARD_CAMPAIGN_FROZEN`；backward `ENVIRONMENT_STABILITY_BLOCKED` |
| Residual Add RMSNorm | non-TE WrappedTorchNorm → torch.nn.RMSNorm | `STABILITY_BLOCKED_R1_R2` |
| Dense Fused Attention | Native DotProductAttention dense/no-mask/p=0 core | `AGGREGATE_WIN_PER_CONFIG_MIXED` |
| MoE Grouped GEMM | Native SequentialMLP Expert Compute | `STABILITY_BLOCKED` |

## 溯源与术语

CE forward Episode 7 是 `PROMOTED_SCORE`。CE backward B1、RMSNorm R1/R2 和 MoE M4 仍是 raw/non-promotable。Attention A2 已通过稳定性资格确认，但各 configuration 表现不一致，不能晋升。此前关于 attention A2 NSYS 或 MoE replay/oracle 缺失的说法均为 `STALE_SUMMARY_SUPERSEDED`；没有重写任何原始历史证据。contract/path 审计记录在 `docs/audits/PHASE19A_CONTRACT_HASH_AUDIT.json`。

## M4 diagnostic-only NSYS 与 scaling

diagnostic-only NSYS 在所有 official distribution 上记录了真实 SequentialMLP reference 和 M4，且 NCCL 为零。reference 有 13 个 balanced、13 个 moderate 和 7 个 two-empty instance：包含小 GEMM、SiLU/gated activation 与 concat。M4 在每种 distribution 中各有一个 `moe_sequential_expert_kernel`，约为 1.645 ms、1.650 ms 和 1.596 ms。这是 `DIAGNOSTIC_ONLY_NOT_PROMOTION`。

M4 raw ratio 为 balanced 1.2793x、moderate 1.4816x、two-empty 0.9573x。源码和 trace 支持 `LAUNCH_REDUCTION`、`ACTIVATION_FUSION`、`MANY_SMALL_GEMMS`、`EMPTY_EXPERT_OVERHEAD` 与 `IMBALANCE_SENSITIVITY`。当 expert 为空时，M4 仍接近固定开销，而 reference 会跳过空 expert 工作；不对 memory/occupancy 作结论。

two-empty reference 的 CV 为 0.3733。保留的最大样本是 invocation 2/block 2/sample 0 的 4.9899 ms；其前一个 reference 为 1.0963 ms，ABBA candidate 邻居为 1.1769/1.1711 ms。该异常只在该邻域的 reference 中出现，因果关系为 `UNKNOWN`。没有删除样本。

## 稳定性与晋升基础设施

`lab/core/environment_qualification.py` 及其 policy 提供只读硬件/process provenance 和 reference-only readiness probe。`READY`、`NOISY` 和 `UNKNOWN` 不会取代冻结的 CV qualification。跨 target 证据在 `docs/audits/CROSS_TARGET_STABILITY_AUDIT.md`；尚未建立共同 physical-GPU cause。

`docs/audits/SCIENTIFIC_PROMOTION_GATE_REGISTRY.md` 固化了 source identity → contracts → delivery → toolchain → correctness → scope → environment → reference/candidate stability → promotion profile → replay → integration。delivery regression 覆盖 M1-like wrong symbol、M2-like missing marker、hash/ABI mismatch、CUDA11.8 forbidden construct 和 valid evaluator-only input；它们均确定性通过，environment probe 和 registry JSON validation 也通过。CLI 审计发现五个 target 使用各自的 replay/benchmark/profile 脚本，而不是统一的非破坏性 CLI；这被记录为复现基础设施缺口，不宣称已完成。

## 下一步允许动作

- SwiGLU：仅允许新授权的 FC1/custom-GEMM research phase。
- CE forward：保持冻结。
- CE backward 和 RMSNorm：只允许在不改变 policy 下各做一次独立 clean-environment requalification。
- Attention：仅允许新授权的 vendor-GEMM-preserving/softmax-surrounding strategy phase；不创建 A4。
- MoE：只允许一次独立 M4 clean-environment requalification；不创建 M5。

`docs/maintenance/CLEAN_ENVIRONMENT_REQUALIFICATION_PLAN.md` 规定一次性规则；它不授权反复重跑直至通过。

## 完整性与未决问题

Megatron HEAD 为 `5be9626709af2722333bf54797c954c09edeada3`，working tree clean。没有修改冻结 score、threshold、candidate source 或权威 Megatron source。未决问题是 noise attribution、vendor-GEMM-preserving attention strategy，以及不改变 dependency truth 的前提下是否存在真实 grouped GEMM backend。
