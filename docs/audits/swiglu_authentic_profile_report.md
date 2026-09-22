# SwiGLU Authentic Profile Audit

Date: 2026-09-22
Writable repository: `D:\Users\38154\Downloads\aka-local-main\aka-local-main`
Megatron source: `C:\Users\38154\projects\megatron-lm` (read-only)
Megatron commit: `5be9626709af2722333bf54797c954c09edeada3`

## Triton/MSVC root cause

The original call path was reproduced and fully recorded in `artifacts/integration/swiglu/authentic_debug/original_failure.txt`.

The path was:

`Megatron MLP.forward -> megatron.core.fusions.fused_bias_swiglu.bias_swiglu_impl -> BiasSwiGLUFunction -> Triton/Inductor`

The generated source was:

`D:\Users\38154\Downloads\aka-local-main\aka-local-main\.runtime_deps\triton\backends\nvidia\driver.c`

The reported lines are:

```text
1079: CUlaunchAttribute clusterAttr = {};
1087: CUlaunchAttribute clusterSchedulingAttr = {};
```

The failing command used `cl.EXE` from MSVC 14.29.30133 / compiler 19.29.30159, `/std:c11`, `/LD`, `/O2`, and Triton/CUDA/Python include and library paths. The command and full traceback are preserved in the artifact.

The default process also mixed the 14.29 compiler path with MSVC 14.51 include paths. This is direct evidence for `TOOLCHAIN_PATH` mismatch. The empty-brace initializers are also rejected by that generated-C/MSVC combination, so the immediate compiler symptom is `GENERATED_C_INCOMPATIBILITY` / `MSVC_LANGUAGE_MODE` rather than a Megatron source error.

The tested torch runtime was `2.11.0+cu128`, CUDA runtime `12.8`, Triton runtime `3.8.0` from the D-side isolated runtime, GPU capability `(12, 0)`. No evidence was found that requires labeling this as a Torch/Triton semantic version mismatch. No system compiler or dependency was changed.

## Fix or remaining blocker

The minimal successful workaround was process-local only: prepend the existing MSVC 14.51.36231 `cl.exe` directory to `PATH` for the Python child process. No persistent environment, system CUDA/MSVC installation, Megatron file, or C-drive repository was modified.

With that process-local PATH, the original Megatron fused path compiled and executed successfully. This is still authentic execution: the call used Megatron's original `bias_swiglu_impl` and original Triton-decorated fused callables; no eager replacement was installed.

## Authentic fused correctness

`AUTHENTIC_CORRECTNESS = PASS`.

Authentic fused output and backward gradient were compared with an eager mathematical reference on the same weights and input:

- shape: `[2, 3, 8]`
- dtype: `torch.float32`
- forward max absolute error: `0.0`
- forward max relative error: `0.0`
- loss absolute error: `0.0`
- input-gradient max absolute error: `0.0`
- input-gradient max relative error: `0.0`
- finite: true

Evidence: `artifacts/integration/swiglu/authentic_profile/authentic_correctness.json`.

## CUDA Event boundary timing

`CUDA_EVENT_BOUNDARY_TIMING = PASS`.

The D-side wrapper surrounds the original Megatron `bias_swiglu_impl` with `torch.cuda.Event(enable_timing=True)`. It records 20 warmups and 120 measured invocations. The wrapper recorded 120 original-boundary calls, one per local training step.

Authentic boundary timing:

- median: approximately `0.094704 ms` (`94.704 us`)
- mean: `0.098217 ms`
- stdev: `0.025861 ms`
- CV: `0.2633`
- min/max: `0.084864 / 0.336160 ms`
- heuristic bimodality: `NOT_DETECTED`
- stability threshold: not met because of outliers

Raw samples and the exact timing source are in `artifacts/integration/swiglu/authentic_profile/boundary_timing.json`. These are CUDA Event measurements around the authentic boundary, not CPU wall-clock timings.

## Whole-step timing

`WHOLE_STEP_TIMING = PASS`.

The same real Megatron MLP path measured forward, loss, backward, and optimizer-compatible `optimizer.step()` with 20 warmups and 120 samples:

- median: approximately `0.671520 ms`
- mean: `0.679152 ms`
- stdev: `0.073600 ms`
- CV: `0.10837`
- min/max: `0.574336 / 1.088480 ms`
- heuristic bimodality: `NOT_DETECTED`

The whole-step raw artifact is `artifacts/integration/swiglu/authentic_profile/whole_step_timing.json`. Stability is reported false at the strict 10% CV threshold, but the measurement set is complete and authentic.

## Operator fraction

`OPERATOR_FRACTION = PASS`.

There was one authentic SwiGLU invocation per measured step. Using the median CUDA Event values:

`T_swiglu_total_per_step / T_step = 0.094704 / 0.671520 = 0.141029`

Therefore the measured local MLP training-step operator fraction is approximately `14.10%`. This scope is explicitly `LOCAL_MLP_TRAINING_STEP`, not nine-grid E2E.

## Amdahl ceiling

`AMDAHL_CEILING = PASS`.

For the local training-step scope:

`S_max = 1 / (1 - 0.141029) = 1.164184x`

This is a theoretical local-step upper bound if the authentic SwiGLU boundary became infinitely fast. It is not a claim about the nine-grid model.

Artifact: `artifacts/integration/swiglu/authentic_profile/swiglu_e2e_ceiling_v2.json`.

## CUPTI status

`CUPTI_PROFILE = BLOCKED`.

`torch.profiler.profile(CPU, CUDA, record_shapes=True, profile_memory=True)` was the invocation under test. The runtime emitted `CUPTI_ERROR_INVALID_DEVICE`, and CUDA profiler activities were missing. The read-only capability inventory found no resolvable `cupti`, `cupti64_2026.1`, or `cupti64_12` library and `where.exe cupti64*.dll` returned no match. Device enumeration itself succeeds: RTX 5060 Laptop GPU, capability `(12, 0)`, driver reported by `nvidia-smi` as `610.88`.

This blocker does not invalidate the CUDA Event boundary timing. Kernel count, CUPTI kernel breakdown, registers, and profiler memory traffic remain `UNKNOWN`.

## PerformanceFacts v3

`REAL_PERFORMANCE_FACTS_V3 = PASS`.

`real_performance_facts_v3.json` contains authentic measured boundary latency, whole-step latency, one measured invocation per step, and derived local-step fraction and ceiling. Evidence status is explicit:

- `MEASURED`: boundary latency, whole-step latency, invocation count
- `DERIVED`: operator fraction and Amdahl ceiling
- `UNKNOWN`: registers per thread, kernel launch count, CUPTI kernel breakdown

No new schema was added.

## Raw planner v3 output

The existing `HypothesisPlanner` was called with v3 facts and the existing `knowledge/mechanisms.jsonl`. No SwiGLU mechanism record matched (`matching_mechanisms = 0`), and the raw result is an empty `ranked_opportunities_v3.json`.

`REASONER_RAW_OUTPUT_V3 = PASS` means the planner ran against authentic v3 evidence and its empty result was preserved. No mechanism was added and no planner rule was changed.

## Planner architecture audit

`PLANNER_ARCHITECTURE = RETRIEVAL_ONLY`.

Source evidence: `lab/runtime/reasoning/hypothesis_planner.py::HypothesisPlanner.rank` accepts `facts` and `mechanisms`, creates an empty candidate list, and only appends an `Opportunity` inside `for record in mechanisms`. `counterfactuals()` computes answers from facts, but those answers are used only to score existing records. There is no branch that constructs a new mechanism or hypothesis from facts alone, and there is no LLM generation call.

## Does the planner generate or only retrieve?

In the current implementation it only retrieves/ranks existing `MechanismRecord` entries. Therefore, with no pre-existing applicable SwiGLU mechanism, it cannot propose a novel mechanism hypothesis even when authentic facts contain a producer/consumer boundary and measured timing. This is an architectural limitation observed in this audit, not modified in this run.

No optimization candidate was implemented.

## Final status

```text
AUTHENTIC_SWIGLU_BASELINE = PASS
AUTHENTIC_CORRECTNESS = PASS

CUDA_EVENT_BOUNDARY_TIMING = PASS
WHOLE_STEP_TIMING = PASS
OPERATOR_FRACTION = PASS
AMDAHL_CEILING = PASS

CUPTI_PROFILE = BLOCKED

REAL_PERFORMANCE_FACTS_V3 = PASS
REASONER_RAW_OUTPUT_V3 = PASS
PLANNER_ARCHITECTURE = RETRIEVAL_ONLY

D_REPO_ONLY = PASS
C_AKA_MOTHER_MODIFIED = NO
MEGATRON_MODIFIED = NO
REMOTE_V100_USED = NO
BIV150_USED = NO
```
