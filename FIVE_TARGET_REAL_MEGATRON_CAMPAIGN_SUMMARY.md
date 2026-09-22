# 五个真实 Megatron target 的 campaign 汇总

> **历史状态说明**：本文是 Phase 19 historical campaign closure snapshot，保留当时 operator campaign 的正式终态，不是 2026-09-22 项目总体最新状态。当前项目状态以 [README.md](README.md)、[docs/PERFORMANCE_OVERVIEW.md](docs/PERFORMANCE_OVERVIEW.md) 和 [公开阶段快照](docs/audits/public_snapshot_20260922.md) 为准。

表格有六行是因为 Vocab-Parallel Cross Entropy 的 forward 和 backward 属于不同子 campaign，历史上分别记录；这不改变项目固定的五类研究目标计数。

| 真实研究目标 | 后端 | 当前最可靠结果 | 晋升状态 | 当前阻塞项 |
|---|---|---|---|---|
| MLP SwiGLU activation | Megatron/PyTorch activation | 独立 forward 约 2.8x，backward 约 3.46x；已完成 current-stream integration | completed, integration-limited | 周边 GEMM/autograd 开销 |
| Vocab-Parallel CE forward | native TP CE | Episode 7 2.4771907400291515x，CI95 [2.46543,2.48439] | promoted/frozen | 无；已冻结 |
| Vocab-Parallel CE backward | native rank-local backward | correctness-valid raw B1 约 2.56x | not promoted | 环境稳定性 |
| WrappedTorchNorm RMSNorm | torch.nn.RMSNorm non-TE | correctness-valid raw R1/R2 约 1.64x | not promoted | candidate 稳定性 |
| Native DotProductAttention | native unfused core | A2 按 S 的 qualified 结果为 5.928x/1.167x/0.347x | not promoted | scaling 表现不一致；A3 correctness 失败 |
| SequentialMLP expert compute | native per-expert MLP | M4 correctness 通过；raw 结果为 1.2793x/1.4816x/0.9573x | not promoted | two-empty reference 稳定性 |

Transformer Engine（TE）路径没有用合成 fallback 替代。早期历史名称“Residual Add RMSNorm”、“Dense Fused Attention”和“MoE Grouped GEMM”不能被用来声称对应的不可用 fused backend 曾实际执行。
