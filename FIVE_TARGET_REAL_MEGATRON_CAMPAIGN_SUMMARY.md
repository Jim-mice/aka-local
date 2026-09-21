# 五个真实 Megatron target 的 campaign 汇总

| 真实研究目标 | 后端 | 当前最可靠结果 | 晋升状态 | 当前阻塞项 |
|---|---|---|---|---|
| MLP SwiGLU activation | Megatron/PyTorch activation | 独立 forward 约 2.8x，backward 约 3.46x；已完成 current-stream integration | completed, integration-limited | 周边 GEMM/autograd 开销 |
| Vocab-Parallel CE forward | native TP CE | Episode 7 2.4771907400291515x，CI95 [2.46543,2.48439] | promoted/frozen | 无；已冻结 |
| Vocab-Parallel CE backward | native rank-local backward | correctness-valid raw B1 约 2.56x | not promoted | 环境稳定性 |
| WrappedTorchNorm RMSNorm | torch.nn.RMSNorm non-TE | correctness-valid raw R1/R2 约 1.64x | not promoted | candidate 稳定性 |
| Native DotProductAttention | native unfused core | A2 按 S 的 qualified 结果为 5.928x/1.167x/0.347x | not promoted | scaling 表现不一致；A3 correctness 失败 |
| SequentialMLP expert compute | native per-expert MLP | M4 correctness 通过；raw 结果为 1.2793x/1.4816x/0.9573x | not promoted | two-empty reference 稳定性 |

Transformer Engine（TE）路径没有用合成 fallback 替代。早期历史名称“Residual Add RMSNorm”、“Dense Fused Attention”和“MoE Grouped GEMM”不能被用来声称对应的不可用 fused backend 曾实际执行。
