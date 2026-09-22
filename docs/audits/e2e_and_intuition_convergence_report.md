# Qualified E2E and Codex Intuition Convergence Report

Date: 2026-09-22. Writable repository is the D repo; Megatron commit `5be9626709af2722333bf54797c954c09edeada3` was read-only.

# Part I — Qualified End-to-End Evidence

## Timing protocol

The real chain is `GPTModel → TransformerBlock → TransformerLayer → MLP → authentic bias_swiglu_impl`. Each scope was constructed, compiled through an authentic forward/backward warmup, synchronized, then run for 20 warmups and 120 paired CUDA Event samples. Every raw sample has `sample_id`, whole-scope time, SwiGLU call count, and summed boundary time. No first-compilation sample or CPU wall clock was used. Evidence: `artifacts/integration/swiglu/e2e_scope/qualified_scope_results.json`.

## Layer repeated results

`TRANSFORMER_LAYER`: `MEASUREMENT_COMPLETE`, 120 samples, whole median 1.910432 ms, SwiGLU median 0.092112 ms, ratio of medians 0.048215, median per-step fraction 0.048010. Whole CV 0.1573 and boundary CV 0.3784, so `QUALIFICATION_UNSTABLE`.

## Block repeated results

`TRANSFORMER_BLOCK`: `MEASUREMENT_COMPLETE`, 120 samples, whole median 2.059408 ms, SwiGLU median 0.093856 ms, ratio of medians 0.045574, median per-step fraction 0.046682. Whole CV 0.1364 and boundary CV 0.2113, so `QUALIFICATION_UNSTABLE`.

## Minimal model repeated results

`MINIMAL_MODEL`: `MEASUREMENT_COMPLETE`, 120 samples, whole median 2.313872 ms, SwiGLU median 0.093440 ms, ratio of medians 0.040383, median per-step fraction 0.041429. Whole CV 0.1982 and boundary CV 0.2694, so `QUALIFICATION_UNSTABLE`; the prior one-shot 1.86% is not used as the model conclusion.

## Operator fraction and Amdahl

| Scope | Ratio of medians | Median per-step fraction | Samples | Result |
|---|---:|---:|---:|---|
| LOCAL_MLP | 0.141029 | prior authentic profile | 120 | prior PASS |
| TRANSFORMER_LAYER | 0.048215 | 0.048010 | 120 | measured, unstable |
| TRANSFORMER_BLOCK | 0.045574 | 0.046682 | 120 | measured, unstable |
| MINIMAL_MODEL | 0.040383 | 0.041429 | 120 | measured, unstable |
| NINE_GRID | null | null | 0 | not executed |

The repeated data shows fraction dilution from MLP to higher scopes, but not stable qualification. With model `f=0.041429`, infinite SwiGLU speedup gives `1.04322x`; a 2x SwiGLU speedup gives about `1.02119x`, both local minimal-model scope only. The prior one-shot layer ceiling (~2.90x) is explicitly discarded.

`scope_ceiling_v2.json` contains the repeated values and `NINE_GRID: null`.

## Nine-grid readiness

`artifacts/integration/swiglu/e2e_scope/nine_grid_readiness.json` records a read-only audit: `PARTIAL`. Real Megatron model code, generic training examples, and parallelism arguments exist; an authoritative nine-grid config, checkpoint, tokenizer/data assets, validated topology, and exact invocation are absent from inspected local artifacts. No nine-grid run was attempted.

# Part II — Codex Planning Runtime

## Previous failure and root cause

The explicit smoke used the current `codex.exe`, default configuration, no model override, `--ephemeral`, read-only sandbox, `--skip-git-repo-check`, and the D repo as cwd. It returned `thread.started` and `turn.started`, then no assistant/final event within 45 seconds. `minimal_smoke.json` captures model-refresh timeout and `sampling request timed out` retry. Status is `WAITING_FOR_FIRST_EVENT`, not trust failure.

The existing `CodexAgentSession` exposes planning-only transport, but the selected Python environment cannot import its `openai_codex` backend package. The bounded CLI smoke therefore is the only valid transport evidence this turn; no global config, model, or user setting was changed. State: `CREATED → PROMPT_SENT → FIRST_EVENT → WAITING_FOR_FIRST_EVENT`; no `FINAL_RECEIVED` or `VALIDATED`.

# Part III — Blind Intuition Benchmark

No blind turn is counted because the minimal smoke was blocked and no hypothesis was fabricated.

| Case | Completed | Status | Result |
|---|---:|---|---|
| RMSNorm | 0/3 | BLOCKED | no Codex output |
| GDN algebra | 0/3 | BLOCKED | no Codex output |
| Real SwiGLU | 0/3 | BLOCKED | no Codex output |

Retrieval-only remains empty with zero matching mechanism records. The deterministic generator produces structural lifetime/materialization/measurement candidates but is template-backed. The deterministic evidence guard passes its offline hallucination test; there were no LLM claims to validate, so no hallucination was accepted or rejected. Semantic repeatability is unmeasured.

`INTUITION_STATUS = NOT_YET_DEMONSTRATED`: the runtime has not returned even the minimal planning response, so no Codex intuition claim is justified.

# Part IV — Next experiment

No experiment plan was selected. No optimization candidate, CUDA/Triton candidate, or new SwiGLU mechanism record was created.

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
