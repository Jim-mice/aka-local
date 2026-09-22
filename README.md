# aka-local

`aka-local` 是一个研究 GPU/CUDA 算子自动优化、真实 Megatron 集成和证据驱动评价闭环的实验框架。它不是 Megatron-LM 的生产替代品，不是通用 fused-kernel 库，也不是已经完成的九格训练系统。

## 当前研究目标

当前固定研究目标及其当前真实执行边界如下：

| 项目目标名称 | 当前真实执行边界 |
|---|---|
| Dense Fused Attention | Native `DotProductAttention` dense/no-mask/p=0 forward core |
| Vocab-parallel Cross Entropy | Megatron Vocab-Parallel Cross Entropy forward / rank-local backward |
| SwiGLU | Megatron MLP `bias_swiglu_impl` activation boundary |
| Residual Add RMSNorm | non-TE `torch.nn.RMSNorm` / `WrappedTorchNorm` |
| MoE Grouped GEMM | Native `SequentialMLP` expert compute |

历史名称不一定对应真实 runtime 中存在同名 fused kernel；真实边界以 pinned Megatron source 为准。

## 当前完成到哪里

### Agent 机制推理能力

- V1：`REPEATABLE_MECHANISM_REASONING`
- V2：`CONSTRAINT_AWARE_MECHANISM_REASONING`
- V3：`HELD_OUT_MECHANISM_REASONING`

Planner benchmark 已关闭：`PLANNER_RESEARCH_STATUS = SUFFICIENT_FOR_OPTIMIZATION_LOOP`。这只是 mechanism-planning benchmark 结论，不是“Agent 已自动发现所有最优 kernel”的结论。详见 [Agent 机制推理能力证据](docs/AGENT_REASONING_EVIDENCE.md)。

### Megatron 本地端到端基础设施

真实路径已经打通：

```text
GPTModel → TransformerBlock → TransformerLayer → MLP
→ authentic bias_swiglu_impl → backward → optimizer-compatible step
```

已完成 real MLP L1 integration、TransformerLayer L2、TransformerBlock L2 和 Minimal GPTModel L2。`Megatron local E2E infrastructure = READY` 表示功能与集成链路已打通，不表示九格 representative performance 已完成；`Nine-grid E2E = NOT EXECUTED`。

### 当前真正 blocker

`REPRESENTATIVE_SHAPE_STATUS = PARTIAL`，`CONFIGURATION_IDENTITY = null`。当前缺少能够属于同一 authoritative configuration identity 的：

- `hidden_size`
- `ffn_hidden_size`
- `sequence_length`
- `micro_batch_size`
- `dtype`
- `TP`
- `SP`

这些字段足够决定 isolated SwiGLU shape replay；checkpoint、tokenizer、dataset 不阻塞单独 shape replay，但完整九格训练仍需要更完整资产。因此当前分支为 `NEXT_PROJECT_BRANCH = WAIT_FOR_AUTHORITATIVE_SHAPE`。

## 最新真实 optimization loop

- Loop 001：结构假设前提不足，先选择 measurement。
- Loop 002：证明 separate kernels 和 device intermediate 存在，但 HBM round-trip 仍未知。
- Loop 003：在 tiny local workload 上进一步测量，拒绝了 HBM-elimination causal story。
- Loop 004：寻找 representative shape，结果仍为 `PARTIAL`，停止继续优化 toy workload。

这表明 Agent 可以在证据不足时停止，也可以用测量证伪自己的 hypothesis；它不代表 tiny workload 的结果可以外推到九格。

## 性能结果

正式 promoted、stability-qualified、raw 和 diagnostic 结果严格分开。性能数字、实验口径和源 artifact 见 [性能结果概览](docs/PERFORMANCE_OVERVIEW.md)。特别注意：micro speedup 不等于 model speedup，graph boundary 不等于 HBM round-trip，toy shape 不等于 representative performance。

## 实验原则

- 固定 baseline identity、scope identity 和 source provenance。
- 优先使用 paired/same-process 测量，并先过 correctness gate。
- 记录 warmup、重复次数、统计摘要和 stability gate。
- 明确 instrumentation perturbation；raw ratio 不能冒充 promoted score。
- 不把 local、历史 V100 fixture 和九格配置混为一谈。

## 仓库入口

- [性能结果概览](docs/PERFORMANCE_OVERVIEW.md)
- [Agent 机制推理能力证据](docs/AGENT_REASONING_EVIDENCE.md)
- [文档索引](docs/README.md)
- [历史报告索引](docs/reports/README.md)
- [V3 最终审计](docs/audits/intuition_v3_final_3run_audit.md)
- [Real Optimization Loop 001](docs/audits/real_optimization_loop_001.md)
- [Real Optimization Loop 002](docs/audits/real_optimization_loop_002_measurement.md)
- [Real Optimization Loop 003](docs/audits/real_optimization_loop_003_cache_and_timing.md)
- [Real Optimization Loop 004](docs/audits/real_optimization_loop_004_shape_acquisition.md)
- [五个真实 Megatron target 汇总](FIVE_TARGET_REAL_MEGATRON_CAMPAIGN_SUMMARY.md)
- [真实 target campaign 索引](REAL_TARGET_CAMPAIGN_INDEX.md)

安装、只读查看命令和安全边界见 [docs/SETUP.md](docs/SETUP.md)。历史报告中的代码、命令、hash、原始日志和状态枚举保持原样，不应脱离其证据范围重新解释。
