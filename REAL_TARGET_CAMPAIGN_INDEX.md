# 真实 Megatron target campaign 索引

这是项目级的规范索引；各条目的原始报告仍是权威实验依据。

| 历史名称 | 真实研究目标 | 终态 | 下一步允许动作 |
|---|---|---|---|
| SwiGLU | Megatron MLP SwiGLU activation boundary | `COMPLETED_INTEGRATION_LIMITED` | 仅在获得新授权后开展 FC1/custom-GEMM 研究。 |
| Vocab-Parallel CE | Megatron Vocab-Parallel Cross Entropy | forward `FORWARD_CAMPAIGN_FROZEN`；backward `ENVIRONMENT_STABILITY_BLOCKED` | 只允许在不改变既有 policy 的条件下做一次 clean-environment backward requalification。 |
| Residual Add RMSNorm | non-TE WrappedTorchNorm → torch.nn.RMSNorm | `STABILITY_BLOCKED_R1_R2` | 只允许做一次 clean-environment requalification。 |
| Dense Fused Attention | Native DotProductAttention dense/no-mask/p=0 forward core | `AGGREGATE_WIN_PER_CONFIG_MIXED` | 仅允许明确授权的 vendor-GEMM-preserving/softmax research phase。 |
| MoE Grouped GEMM | Native SequentialMLP Expert Compute | `STABILITY_BLOCKED` | 只允许一次独立的 clean-environment M4 requalification；不创建 M5。 |

机器可读的详细信息和 contract 引用位于 `real_target_campaign_index.json`。
