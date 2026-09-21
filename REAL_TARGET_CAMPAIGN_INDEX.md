# Real Megatron Target Campaign Index

This index is the canonical project-level registry; raw reports remain the
authoritative evidence for each entry.

| Historical name | Canonical real target | Terminal state | Next legal action |
|---|---|---|---|
| SwiGLU | Megatron MLP SwiGLU activation boundary | `COMPLETED_INTEGRATION_LIMITED` | New FC1/custom-GEMM research authorization only. |
| Vocab-Parallel CE | Megatron Vocab-Parallel Cross Entropy | forward `FORWARD_CAMPAIGN_FROZEN`; backward `ENVIRONMENT_STABILITY_BLOCKED` | One clean-environment backward requalification under unchanged policy only. |
| Residual Add RMSNorm | non-TE WrappedTorchNorm → torch.nn.RMSNorm | `STABILITY_BLOCKED_R1_R2` | One clean-environment requalification only. |
| Dense Fused Attention | Native DotProductAttention dense/no-mask/p=0 forward core | `AGGREGATE_WIN_PER_CONFIG_MIXED` | Explicit vendor-GEMM-preserving/softmax research phase only. |
| MoE Grouped GEMM | Native SequentialMLP Expert Compute | `STABILITY_BLOCKED` | One clean-environment M4 requalification only; no M5. |

Machine-readable details and contract references are in
`real_target_campaign_index.json`.
