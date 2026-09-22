# Robust E2E and Blind Benchmark Report

Date: 2026-09-22. All writes were limited to the D repository. Megatron commit `5be9626709af2722333bf54797c954c09edeada3` was read-only.

# Part I — Timing noise audit

## Instrumentation overhead

The new audit separates: A uninstrumented whole-step batched CUDA Event windows, B whole-step windows with authentic inner-boundary Event instrumentation, C independent authentic boundary micro-timing, and D paired per-step boundary/whole samples. It used 5 independent campaigns per scope, 20 warmup batches, 30 measured batches per campaign, 32 model steps per batch, and 512 authentic boundary invocations per micro batch. The batch sizes were selected from the earlier ~2 ms step and ~0.09 ms boundary observations to target approximately 50 ms windows; the choice and raw samples are recorded in `artifacts/integration/swiglu/e2e_scope/fraction_audit.json`.

Instrumentation overhead, median campaign estimates:

| Scope | A uninstrumented ms/step | B instrumented ms/step | Overhead |
|---|---:|---:|---:|
| TransformerLayer | 1.77 | 1.79 | 1.20% |
| TransformerBlock | 1.81 | 1.89 | 4.75% |
| Minimal GPTModel | 2.05 | 2.12 | 3.45% |

The overhead is not silently ignored. It exceeds 1% in every scope, so paired timing is retained as a separate estimator and is not treated as an uninstrumented whole-step measurement.

## Batched protocol and independent campaigns

All three scopes completed `MEASUREMENT_COMPLETE`: 5 campaigns × 30 measured batch windows, with 1200 underlying model steps per scope. The authentic fused call path remained the real Megatron path; no source file was modified. Boundary micro-timing was measured independently with CUDA Events around batches of 512 original calls and is not presented as model-level time by itself.

## Split estimator

`f_split = median(authentic boundary micro estimate) / median(uninstrumented whole-step estimate)`.

| Scope | f_split | Bootstrap 95% CI |
|---|---:|---:|
| TransformerLayer | 0.017922 | [0.016939, 0.018168] |
| TransformerBlock | 0.017372 | [0.016400, 0.017650] |
| Minimal GPTModel | 0.015571 | [0.014957, 0.015796] |

## Paired estimator

`f_paired = median(boundary_total_i / whole_step_i)` using per-step paired CUDA Event identities inside the instrumented run.

| Scope | f_paired | Bootstrap 95% CI |
|---|---:|---:|
| TransformerLayer | 0.050195 | [0.050083, 0.050253] |
| TransformerBlock | 0.047009 | [0.046687, 0.047337] |
| Minimal GPTModel | 0.041957 | [0.041793, 0.042108] |

The split and paired estimates differ materially. The most defensible interpretation is that inner Event instrumentation and paired boundary timing perturb the short whole-step path. The split estimator is the primary estimate for an uninstrumented model fraction; the paired estimator is retained as an instrumentation-affected upper comparison, not discarded.

## Fraction confidence interval and final status

`FRACTION_EVIDENCE_STATUS = DIRECTIONAL`. The campaign-level split estimates are tightly clustered, but the estimator disagreement and the measured instrumentation overhead prevent calling the result ROBUST. The previous absolute-step CV gate remains unchanged; no qualification threshold was lowered. This is evidence of direction and approximate scale, not system performance qualification.

## Local-model Amdahl table

Scope is strictly `LOCAL_MINIMAL_GPTMODEL`, never nine-grid. For the primary split fraction, theoretical speedups are:

| SwiGLU speedup | Model speedup, f_split=0.015571 |
|---:|---:|
| 1.25x | 1.003124x |
| 1.5x | 1.005218x |
| 2x | 1.007847x |
| 4x | 1.011817x |
| infinity | 1.015818x |

For comparison, the instrumentation-affected paired fraction gives 1.008462x, 1.014184x, 1.021428x, 1.032490x, and 1.043794x respectively. These paired values must not be mistaken for unbiased model timing.

# Part II — Codex runtime conclusion

## Nested runtime evidence

The prior nested invocation reached `thread.started → turn.started`, then produced no assistant/final event. The explicit bounded smoke recorded model refresh timeout and `sampling request timed out` retry after 45 seconds. It was not a repository trust failure after `--skip-git-repo-check` was added.

## Runtime classification

`NESTED_CODEX_STATUS = BACKEND_SAMPLING_TIMEOUT`. `NESTED_RUNTIME_UNSUPPORTED` remains a possible architectural explanation, but the directly observed failure is sampling timeout after turn start. No additional recursive Codex attempts were made.

## Why this is not an intelligence result

No assistant response was returned, so no LLM hypothesis exists to score. This is a transport/backend execution block, not an LLM intelligence FAIL. `INTUITION_STATUS` remains `NOT_YET_DEMONSTRATED`.

# Part III — Blind benchmark package

## RMSNorm case

`benchmarks/intuition/rmsnorm/` contains only anonymized dataflow facts, unknown fields, contract, and case metadata. It does not contain a proposed transformation or mechanism label.

## GDN case

`benchmarks/intuition/gdn/` contains raw symbolic relations and operation-graph facts only. No simplified formula or known solution is included.

## SwiGLU case

`benchmarks/intuition/swiglu/` contains authentic source identity plus the latest split/paired measured evidence and explicit unknowns. It contains no new mechanism record and no deterministic planner output.

## Prompt and answer-key isolation

`BLIND_PROMPT.md` requests at most three structured hypotheses and forbids code, new facts, promotion decisions, and prior answers. The prompt contains no case-specific expected mechanism. `evaluator_only/` contains no answer key; human semantic review is intentionally required for the first external runs. `verify_blind_package.py` scanned the public package and passed.

## Collector and evidence guard

`scripts/collect_intuition_result.py` validates case identity, required fields, and evidence references against facts. Unknown evidence references are rejected as `REJECT_EVIDENCE_HALLUCINATION`. It does not call a model. `scripts/evaluate_intuition_result.py` creates per-dimension human-review records without a total score.

## Run slots

Nine slots exist under `benchmarks/intuition/results/`: three each for RMSNorm, GDN, and SwiGLU. Every slot is `PENDING_EXTERNAL_CODEX`; no result is fabricated or prefilled.

# Part IV — Nine-grid frozen requirements

`artifacts/integration/swiglu/e2e_scope/nine_grid_external_requirements.json` freezes the missing external inputs: authoritative model config, checkpoint, tokenizer, dataset/data split, distributed topology, exact invocation/environment lock, and acceptance window. Local readiness remains `PARTIAL`; nine-grid execution remains `NO`.

## Final status

```text
INSTRUMENTATION_OVERHEAD_AUDITED = PASS
BATCHED_LAYER_TIMING = PASS
BATCHED_BLOCK_TIMING = PASS
BATCHED_MODEL_TIMING = PASS
MODEL_OPERATOR_FRACTION = 0.015571 (split primary) / 0.041957 (paired instrumentation-affected)
MODEL_FRACTION_95CI = [0.014957, 0.015796] split; [0.041793, 0.042108] paired
FRACTION_EVIDENCE_STATUS = DIRECTIONAL
MODEL_AMDAHL_INFINITY = 1.015818x split primary; 1.043794x paired comparison
NESTED_CODEX_STATUS = BACKEND_SAMPLING_TIMEOUT
BLIND_BENCHMARK_PACKAGE = PASS
BLIND_PACKAGE_INTEGRITY = PASS
EXTERNAL_CODEX_RUNS_COMPLETED = 0
INTUITION_STATUS = NOT_YET_DEMONSTRATED
NINE_GRID_LOCAL_READINESS = PARTIAL
NINE_GRID_EXTERNAL_REQUIREMENTS = COMPLETE
OPTIMIZATION_CANDIDATE_CREATED = NO
NINE_GRID_E2E_EXECUTED = NO
D_REPO_ONLY = PASS
C_AKA_MOTHER_MODIFIED = NO
MEGATRON_MODIFIED = NO
REMOTE_V100_USED = NO
BIV150_USED = NO
```
