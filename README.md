# aka-local

`aka-local` 是一个用于研究 CUDA 算子优化的实验框架。

它会把候选 CUDA / 算子实现放到真实的 Megatron-LM 执行边界中进行验证，并保存完整的实验依据。

这个项目主要用于研究和实验，不是 Megatron-LM 的生产替代品，也不是可以直接拿来替换所有算子的 fused-kernel 库。

## 范围与当前状态

仓库保存了实验所需的契约、候选代码、验证规则、campaign 记录、关键结果和阶段报告。

目前主要研究了 5 个真实的 Megatron-LM 目标：

1. Megatron MLP 中的 SwiGLU 激活计算；
2. Vocab-Parallel Cross Entropy，包括前向和 rank-local backward；
3. 非 Transformer Engine 路径下的 `torch.nn.RMSNorm` / `WrappedTorchNorm`；
4. 原生 `DotProductAttention` 的 dense / no-mask / p=0 前向核心；
5. 原生 `SequentialMLP` 的 expert compute。

项目当前的汇总状态见：

- [五个真实 Megatron target 的实验汇总](FIVE_TARGET_REAL_MEGATRON_CAMPAIGN_SUMMARY.md)
- [真实 target campaign 索引](REAL_TARGET_CAMPAIGN_INDEX.md)
- [Phase 19-A 五目标收尾报告](docs/reports/phase-19/PHASE19A_FIVE_TARGET_CAMPAIGN_CLOSURE.md)

这些报告会明确区分：

- 原始测量结果；
- 通过稳定性验证的结果；
- 正式晋升的性能结果。

因此，仓库中出现的 raw speedup 并不自动代表最终性能结论。

早期使用过的一些名称，例如：

- “Residual Add RMSNorm”
- “Dense Fused Attention”
- “MoE Grouped GEMM”

只是最初的研究目标名称，并不代表实际运行时一定存在一个同名的 fused operator。

后续阶段已经根据真实 Megatron 源码和运行路径重新确定了对应的研究边界。

## 性能结果概览

以下 Megatron 性能结果主要来自 V100 `sm_70` / CUDA 11.8 环境。结果具有不同证据等级，raw ratio 不能直接理解为正式 speedup。

| 目标 | 当前最可靠结果 | 状态 |
|---|---|---|
| SwiGLU | activation-only forward / backward 有明显局部加速；完整 training rerun 为 `0.90710x` | `COMPLETED_INTEGRATION_LIMITED` |
| Vocab-Parallel CE forward | Episode 7：`2.477191x`，CI95 `[2.465431x, 2.484392x]` | `PROMOTED_SCORE` |
| CE backward | B1 raw `2.557171x` | `ENVIRONMENT_STABILITY_BLOCKED` |
| Torch RMSNorm | R1/R2 raw `1.64629x` / `1.64134x` | `STABILITY_BLOCKED_R1_R2` |
| DotProductAttention | A2：`5.928x` / `1.167x` / `0.347x` | `AGGREGATE_WIN_PER_CONFIG_MIXED` |
| SequentialMLP MoE | M4 raw `1.2793x` / `1.4816x` / `0.9573x` | `STABILITY_BLOCKED` |

[查看完整性能报告](docs/PERFORMANCE_OVERVIEW.md)，其中逐项标出证据等级、benchmark scope 和机器可读 evidence 链接。

### 数据在哪里

- `campaigns/`：每个 Agent episode 的 candidate、result、decision 和 benchmark evidence；
- `targets/megatron_5be9626/`：真实 target 的 contract、baseline、replay 和 qualification evidence；
- `docs/reports/`：各 Phase 的完整实验报告；
- `knowledge/`：面向后续 Agent 的结构化经验总结。

## 架构

`lab/` 包含实验调度、运行时检查、evaluator 接口以及实验完整性规则。

`targets/megatron_5be9626/` 保存与真实 Megatron target 相关的契约、候选实现、诊断信息和精简后的实验结果。

`operators/`、`ops/` 和 `benchmarks/` 保存可复用的算子实现、辅助代码和 benchmark 组件。

历史阶段报告统一保存在 `docs/reports/`；审计和维护记录保存在 `docs/audits/`、`docs/maintenance/` 以及 `docs/archive/`。

整个实验流程大致为：

```text
确定真实执行边界
        ↓
冻结接口和实验契约
        ↓
Agent 生成候选实现
        ↓
接口 / ABI / 工具链检查
        ↓
编译
        ↓
正确性验证
        ↓
统计性能测试
        ↓
NSYS 性能分析
        ↓
接受 / 拒绝
        ↓
将经验反馈给下一轮 Agent
```

Agent/evaluator/promotion 流程有明确门槛：交付前检查源身份和 ABI 契约，随后检查正确性和基准范围，再进行环境和稳定性资格确认，最后才可能晋升分数。不要随意启动 campaign：它们可能触发 CUDA 构建或已配置的远程 evaluator。

## 安装与查看

请先阅读 [docs/SETUP.md](docs/SETUP.md)。以下命令只查看已有状态：

```powershell
py -3 -m json.tool .\five_target_campaign_state.json
py -3 -m json.tool .\real_target_campaign_index.json
.\lab.ps1 status
.\lab.ps1 list-ops
```

这里刻意不把 `run`、`evaluate` 和 Agent 命令作为示例，因为它们可能使用 CUDA 或远程基础设施。

## 远程执行

远程 V100 评估是可选项。将 `config/environments/v100.example.yaml` 复制为被 Git 忽略的 `v100.yaml`，再替换 `<REMOTE_HOST>`、`<REMOTE_USER>` 和路径占位符。请使用 SSH key 或本地交互式认证；不要提交密码、token、私有端点或本地配置。

## 仓库结构

```text
config/                    公开示例和 target 评估配置
lab/                       实验框架运行时、规则与 CLI
targets/megatron_5be9626/  真实 target 的契约、源码与精简证据
operators/, ops/           可复用算子与辅助源码
campaigns/                 保留的 campaign 溯源信息（不含生成物）
docs/reports/              历史阶段报告
docs/audits/               审计与完整性记录
docs/archive/              长时间运行和发布维护归档
```

大型原始 profiler 报告、虚拟环境、runtime bundle、外部源码 checkout、机器本地配置和编译产物均被有意排除在 Git 外。外部源码引用记录在 `lab/knowledge_sources/import_manifest.json`（如适用）。
