# 历史 Phase 报告索引

这些报告保存实验当时的边界、契约、结果和结论。文件名保持不变，目录按 Phase 分类；报告中的代码、命令、hash、状态枚举、原始日志和数值证据均保持原样。

| 目录 | 内容 |
|---|---|
| `phase-09/` | 早期完整性审计与 hardening。 |
| `phase-10/` | 平台冻结、外部复现与 CLI 完整性。 |
| `phase-11/` | 多算子 framework 与 contract hardening。 |
| `phase-12/` | Softmax、NSYS profiler feedback。 |
| `phase-13/` | Dense/Causal Attention 及 backward。 |
| `phase-14/` | Megatron SwiGLU target mapping、replay 与 integration。 |
| `phase-15/` | Vocab-Parallel Cross Entropy forward/backward。 |
| `phase-16/` | non-TE Torch RMSNorm。 |
| `phase-17/` | Native DotProductAttention。 |
| `phase-18/` | Native SequentialMLP expert compute。 |
| `phase-19/` | 五个真实 target 的收尾状态。 |
| `phase-20/` | Attention 的 vendor-GEMM-preserving 与 clean-window qualification。 |

当前状态请优先从根目录的 `FIVE_TARGET_REAL_MEGATRON_CAMPAIGN_SUMMARY.md`、`REAL_TARGET_CAMPAIGN_INDEX.md` 和 `five_target_campaign_state.json` 开始查看。
