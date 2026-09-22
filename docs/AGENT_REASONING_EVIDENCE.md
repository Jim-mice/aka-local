# Agent 机制推理能力证据

本文公开记录 mechanism-planning benchmark 的证据边界。它不是“AI 智商测试”，也不是性能保证。

## 结果摘要

| 版本 | 结果 | 状态 |
|---|---|---|
| V1 | RMSNorm lifetime `3/3`；GDN algebra/dataflow `3/3`；SwiGLU structural `3/3` | `REPEATABLE_MECHANISM_REASONING` |
| V2 | Case A `0 HIT / 3 PARTIAL`；Case B fusion rejection `3/3`；Case C lifetime rejection `3/3`；Case D system reasoning `3/3`；Case E evidence discipline `3/3` | `CONSTRAINT_AWARE_MECHANISM_REASONING` |
| V3 | run1 `HIT`；run2 `HIT`；run3 `PARTIAL`；`FULL_HITS = 2/3` | `HELD_OUT_MECHANISM_REASONING` |

V3 的 deterministic direct coverage 为 `NO`，接受的 measured-fact hallucinations 为 `0`。预先冻结的 STOP_RULE 是：`>= 2/3 full HIT → HELD_OUT_MECHANISM_REASONING`。

## 实验口径

- V1：9 次 independent fresh runs。
- V2：5 个 cases × 3 = 15 次 fresh runs。
- V3：1 个 held-out case × 3 = 3 次 fresh runs。
- 每次使用 fresh independent Codex session、blind public package、隔离的 evaluator-only 文件和 structured JSON schema。
- 结果经过 `evidence_refs` validation 和 UNKNOWN discipline 检查；benchmark 期间不实现 optimization candidate。

## 可复现性限制

- blind runs 使用当时的 current default Codex。
- 没有使用 `-m` 固定模型 override。
- artifact 证实本批次 CLI 为 `codex-cli 0.153.0`；这不构成固定模型环境的完整锁定。
- 因此这些结果不是固定模型排行榜，不能直接用于不同模型之间的排名比较。
- `HIT` / `PARTIAL` / `MISS` 来自 rubric 冻结后的 non-blind semantic audit，不是自动数学证明。

## 这些结果能说明什么

结果支持以下有限结论：当前 Codex 能进行 evidence-grounded mechanism planning，能使用反证约束，能在 held-out case 中提出算法状态重构方向。它不证明每次都能找到最优 kernel，不证明每个 hypothesis 都带来 speedup，也不替代真实 profiler、OJ 或 correctness gate。

## 审计入口

- [V1/V2/V3 最终审计](audits/intuition_v3_final_3run_audit.md)
- [V2 15-run 最终审计](audits/intuition_v2_15run_final_audit.md)
- [V3 benchmark 设计](audits/intuition_v3_final_target_design.md)
- [Generative planner 报告](audits/generative_planner_report.md)
