# SwiGLU Local L2 and Profile Audit

Date: 2026-09-22
Repository: `D:\Users\38154\Downloads\aka-local-main\aka-local-main`
Megatron source: `C:\Users\38154\projects\megatron-lm` (read-only)
Megatron commit: `5be9626709af2722333bf54797c954c09edeada3`

## Authentic baseline status

`AUTHENTIC_SWIGLU_BASELINE = BLOCKED`.

The original Megatron fused SwiGLU path was invoked without the eager adapter. It reached the Triton/Inductor driver compilation path, but Windows MSVC failed compiling Triton's `driver.c` with C2059 syntax errors at lines 1079 and 1087. The resulting exception is recorded in `artifacts/integration/swiglu/l2_local/authentic_swiglu_baseline.json`.

The subsequent local baseline is explicitly `EAGER_COMPAT_BASELINE`; it is a real Megatron MLP training path, but it is not an authentic fused-kernel performance baseline. No eager substitution was used to claim authentic performance.

## Local training-step L2 definition

This L2 is `LOCAL_TRAINING_STEP_L2`, not nine-grid E2E. It instantiates the real Megatron `megatron.core.transformer.mlp.MLP`, with deterministic synthetic input, TP=1, sequence parallel disabled, batch=2, sequence=3, hidden=8, FFN hidden=16, float32. The path includes MLP forward, scalar loss, backward, finite-gradient checks, and an optimizer-compatible `optimizer.step()` with learning rate zero.

The same path was run for the eager-compatible baseline and the D-side `aka-local-swiglu-reference-v1` replacement. Each used 5 warmups and 10 measured repeats. Raw evidence is in `artifacts/integration/swiglu/l2_local/baseline_run.json` and `reference_replacement_run.json`.

## Baseline L2 evidence

The eager-compatible baseline completed. Median step time was approximately 0.70035 ms, mean 0.69879 ms, CV 0.0380, and peak memory was 17,058,304 bytes. Loss was finite and repeat-consistent. These numbers are compatibility-path measurements only, not authentic Megatron fused-kernel measurements.

## Reference replacement L2 evidence

The D-side reference replacement completed on the same real Megatron MLP path. Replacement invocation count was 15, and `no_silent_fallback` was true. Median step time was approximately 0.56845 ms, mean 0.58272 ms, CV 0.1007, and peak memory was 17,055,744 bytes.

The apparent timing difference must not be interpreted as an optimization result: the authentic fused baseline is blocked, the compared baseline is eager-compatible, and the replacement qualification was unstable.

## L2 correctness

`L2_CORRECTNESS = PASS` for the local training path. The recorded comparison shows:

- loss absolute and relative error: 0
- output shape equal, max absolute and relative error: 0
- gradient max absolute error: `1.8189894035458565e-12`
- gradient max relative error: `4.923731400604635e-07`
- all finite checks: true
- replacement invocations: 1 in the paired correctness run

The artifact is `artifacts/integration/swiglu/l2_local/l2_correctness.json`.

## EndToEndOJ result

The existing `lab/runtime/evaluators/end_to_end_oj.py` consumed the real baseline/replacement L2 evidence; no fourth evaluator was created. Its verdict was `SYSTEM_REJECT`, with the only failed check being `qualification_stable`. Correctness, required metrics, finite checks, comparable inputs, and artifact references all passed.

Therefore `END_TO_END_OJ_WIRING = PASS`, but this run did not qualify as a system acceptance. The reference replacement is not a performance candidate, so this rejection is expected evidence of the strict stability gate rather than a correctness failure.

## Real SwiGLU profile

The reference-replacement profile produced 65 profiler events and CPU-side `aten::silu`/`SiluBackward0` observations. CUDA profiler initialization failed with `CUPTI_ERROR_INVALID_DEVICE`, so CUDA kernel time, kernel count, and launch count are unavailable. The measured `latency_us` is explicitly scoped to the whole local training step and marked `REFERENCE_REPLACEMENT_ONLY`; it is not SwiGLU-boundary latency.

The profile artifact `swiglu_profile_facts.json` distinguishes evidence types:

- `MEASURED`: one SwiGLU invocation per local step; whole-step reference-only latency; profiler event counts
- `DERIVED`: input/target byte estimate only
- `UNKNOWN`: authentic fused boundary latency, registers per thread, kernel launch count

Thus `REAL_SWIGLU_PROFILE = BLOCKED` for an authentic performance profile. A reference-only CPU event profile exists.

## PerformanceFacts v2

`real_performance_facts_v2.json` was generated from the real Megatron shape and L2 invocation evidence. It records batch 2, sequence 3, hidden FC1 width 32, `fc1 -> swiglu -> fc2`, tensor lifetimes, producer/consumer boundaries, one measured invocation per step, and explicit unknown fields for authentic timing, registers, and kernel launches.

This is a successful evidence-to-`PerformanceFacts` wiring result, not proof that all performance facts are measured. `REAL_PERFORMANCE_FACTS_V2 = PASS` with the documented unknowns.

## Raw planner output

The existing `HypothesisPlanner` was called directly with the generated facts and no newly injected mechanisms. Its raw output is `ranked_opportunities_v2.json` and is an empty list.

No mechanism-memory answer was manually inserted to make the planner appear intelligent, and no planner rules were changed.

## Reasoner quality assessment

The raw result is conservative but not yet useful for SwiGLU optimization. It did not hallucinate a generic opportunity from insufficient evidence, but it also produced no ranked hypothesis because there is no applicable SwiGLU mechanism record in the supplied memory set. It therefore cannot yet demonstrate recognition of data lifetime, materialization, fusion boundary, call frequency, or E2E impact for this real observation.

`REASONER_RAW_OUTPUT = PASS` means the real evidence reached the planner and the empty result is genuine; it does not mean the reasoner solved the optimization problem.

## Amdahl ceiling

`AMDAHL_CEILING = BLOCKED`. The artifact `swiglu_e2e_ceiling.json` intentionally contains null operator fraction and null maximum speedup. Only whole-step timing was measured, and no authentic SwiGLU boundary timing was available. Dividing whole-step timing by whole-step timing would have produced a false 100% operator fraction and has not been done.

## Remaining gap to true nine-grid E2E

The current result is a local TP=1 training-step L2. It does not execute the nine-grid model, distributed tensor/sequence parallel contexts, real training data, or an authentic fused SwiGLU CUDA baseline. CUPTI/kernel evidence also remains unavailable. The next legitimate L2 expansion requires resolving the Windows Triton/MSVC driver compilation blocker and then measuring the authentic operator boundary before any performance comparison.

No performance optimization candidate was implemented in this run.

## Final status

```text
AUTHENTIC_SWIGLU_BASELINE = BLOCKED
LOCAL_TRAINING_STEP_L2 = PASS
REFERENCE_REPLACEMENT_L2 = PASS
L2_CORRECTNESS = PASS
END_TO_END_OJ_WIRING = PASS
REAL_SWIGLU_PROFILE = BLOCKED
REAL_PERFORMANCE_FACTS_V2 = PASS
REASONER_RAW_OUTPUT = PASS
AMDAHL_CEILING = BLOCKED
HYPOTHESIS_READY = PASS
NINE_GRID_E2E_EXECUTED = NO

D_REPO_ONLY = PASS
C_AKA_MOTHER_MODIFIED = NO
MEGATRON_MODIFIED = NO
REMOTE_V100_USED = NO
BIV150_USED = NO
```
