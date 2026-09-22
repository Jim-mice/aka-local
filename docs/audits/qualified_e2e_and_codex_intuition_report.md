# Qualified E2E and Codex Intuition Report

本报告对应本轮 Track A/Track B，详细结果同步于 `docs/audits/e2e_and_intuition_convergence_report.md`。

## Timing protocol

真实层级为 `GPTModel → TransformerBlock → TransformerLayer → MLP → authentic bias_swiglu_impl`。每个 scope 先完成构造、真实 fused forward/backward 编译和同步，再执行 20 warmups、120 个 paired CUDA Event measured steps。每条 raw sample 保存 `sample_id`、whole-step time、SwiGLU invocation count、SwiGLU boundary total time。首次编译不进入统计。

## Repeated scope evidence

来源：`artifacts/integration/swiglu/e2e_scope/qualified_scope_results.json`。

| Scope | Whole median ms | SwiGLU median ms | Ratio of medians | Median per-step fraction | Samples | Stability |
|---|---:|---:|---:|---:|---:|---|
| TRANSFORMER_LAYER | 1.910432 | 0.092112 | 0.048215 | 0.048010 | 120 | QUALIFICATION_UNSTABLE |
| TRANSFORMER_BLOCK | 2.059408 | 0.093856 | 0.045574 | 0.046682 | 120 | QUALIFICATION_UNSTABLE |
| MINIMAL_MODEL | 2.313872 | 0.093440 | 0.040383 | 0.041429 | 120 | QUALIFICATION_UNSTABLE |

三者均为 `MEASUREMENT_COMPLETE`，但 whole-step CV 分别为 0.1573、0.1364、0.1982，未通过稳定性门槛。因此最终 qualified timing 状态为 BLOCKED，而不是把 noisy repeated data 宣称为稳定性能资格。

## Scope ceilings

来源：`artifacts/integration/swiglu/e2e_scope/scope_ceiling_v2.json`。重复 model fraction `f=0.041429` 给出无限 SwiGLU 加速的 local minimal-model ceiling `1.04322x`；若 SwiGLU 仅 2x，加速约 `1.02119x`。这些不是九格结论；`NINE_GRID=null`。

## Nine-grid readiness

`artifacts/integration/swiglu/e2e_scope/nine_grid_readiness.json` 为只读审计，状态 `PARTIAL`。本地有 Megatron model code、generic training examples 和并行参数，但没有被证实的九格 config、checkpoint、tokenizer/data、topology 或 exact invocation；没有执行九格。

## Codex planning runtime

`artifacts/reasoning/codex_runtime/minimal_smoke.json` 记录了当前默认 `codex.exe` 的有界 smoke：收到 `thread.started`、`turn.started`，45 秒内未收到 assistant/final event，stderr 有 model refresh timeout 与 sampling request timeout。状态机停在 `WAITING_FOR_FIRST_EVENT`，所以 `CODEX_MINIMAL_PLANNING_SMOKE=BLOCKED`。

现有 `CodexAgentSession` 的 Python backend 还受当前环境缺少 `openai_codex` 阻塞；未修改全局配置、模型或用户设置。三类 blind case 均计为 0/3 completed，未伪造 LLM 输出。Retrieval-only 仍只能复用机制；deterministic generator 可产生 template-backed structural hypotheses；Codex intuition 尚未测得。

## Final status

```text
QUALIFIED_LAYER_TIMING = BLOCKED (measurement complete, qualification unstable)
QUALIFIED_BLOCK_TIMING = BLOCKED (measurement complete, qualification unstable)
QUALIFIED_MODEL_TIMING = BLOCKED (measurement complete, qualification unstable)
LAYER_OPERATOR_FRACTION = 0.048010
BLOCK_OPERATOR_FRACTION = 0.046682
MODEL_OPERATOR_FRACTION = 0.041429
MODEL_AMDAHL_CEILING = 1.04322x (unstable local-model evidence)
NINE_GRID_LOCAL_READINESS = PARTIAL
CODEX_MINIMAL_PLANNING_SMOKE = BLOCKED
CODEX_RMSNORM_COMPLETED_RUNS = 0
CODEX_GDN_COMPLETED_RUNS = 0
CODEX_SWIGLU_COMPLETED_RUNS = 0
LLM_RMSNORM_BLIND = BLOCKED
LLM_GDN_BLIND = BLOCKED
LLM_SWIGLU_BLIND = BLOCKED
LLM_EVIDENCE_GUARD = PASS
INTUITION_STATUS = NOT_YET_DEMONSTRATED
SELECTED_SWIGLU_EXPERIMENT_PLAN = NO
OPTIMIZATION_CANDIDATE_CREATED = NO
NINE_GRID_E2E_EXECUTED = NO
D_REPO_ONLY = PASS
C_AKA_MOTHER_MODIFIED = NO
MEGATRON_MODIFIED = NO
REMOTE_V100_USED = NO
BIV150_USED = NO
```
