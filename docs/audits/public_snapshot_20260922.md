# 公开阶段快照审计

## 当前版本范围

本快照只整理公开文档和已有证据入口，不新增实验结论。真实 Megatron local E2E infrastructure 已打通到 Minimal GPTModel L2，但九格 E2E 未执行；representative operator performance 仍为 `BLOCKED / WAIT_FOR_AUTHORITATIVE_SHAPE`。

## 文档变更

- 更新根 `README.md`，明确项目范围、当前状态、性能口径和 blocker。
- 更新 `docs/PERFORMANCE_OVERVIEW.md`，区分 promoted、raw、diagnostic、local non-representative 和 inconclusive 证据。
- 新增 `docs/AGENT_REASONING_EVIDENCE.md`。
- 更新 `docs/README.md` 和 `docs/reports/README.md` 的当前阶段入口。
- 新增 README 中文化审计和本快照审计。

## 性能数字来源

公开主口径使用仓库 artifact：`artifacts/integration/swiglu/real_loop_003/batched_swiglu_timing.json` 的权威 `53.74621972441673 us`，并由 `real_loop_004/timing_evidence_reconciliation.json` 解释早期 `55.639 us` 的来源。local model fraction、NCU 数据、cache-pressure 数据和 representative-shape blocker 均链接到对应 artifact 或 audit。

## Agent benchmark 来源

Agent 结果来自 V1/V2/V3 final audit 和仓库内 benchmark package。V3 的停止规则在运行前冻结，`2/3` full HIT 得出 `HELD_OUT_MECHANISM_REASONING`。

## 当前 blocker

`REPRESENTATIVE_SHAPE_STATUS = PARTIAL`，缺少同一 authoritative configuration identity 下的 hidden size、FFN hidden size、sequence length、micro batch、dtype、TP 和 SP。不得用 historical V100 fixture 冒充九格配置。

## 未执行项目

- no new benchmark
- no remote GPU
- no Megatron modification
- no candidate creation
- no SSH、V100 或 BIV150
