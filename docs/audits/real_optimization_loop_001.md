# Real Optimization Loop 001 — SwiGLU

Date: 2026-09-22. Scope was D-repo-only with read-only Megatron source. No remote GPU, SSH, BIV150, V100, or Mechanism Memory write was used.

## Authentic dataflow reconstruction

Source commit: `5be9626709af2722333bf54797c954c09edeada3`.

The exact source path is:

`MLP.forward → linear_fc1 → intermediate_parallel,bias_parallel → bias_swiglu_impl → BiasSwiGLUFunction.apply → bias_swiglu (bias add + swiglu) → activation output → linear_fc2`.

Source evidence is captured in `artifacts/integration/swiglu/real_loop_001/authentic_dataflow.json` with source hashes and inspected symbol bodies.

Forward findings:

- `MLP.forward` receives the fc1 output and bias as separate tensors.
- `bias_swiglu_impl` performs a shape view, invokes the custom autograd function, and restores rank; the Python wrapper itself is not proof of a GPU materialization.
- `BiasSwiGLUFunction.forward` saves input/bias for backward and calls Megatron’s fused bias-plus-SwiGLU callable.
- The fused activation result is returned from the activation path and passed as the input to a subsequent `linear_fc2` module call.
- The authentic implementation therefore fuses bias handling with SwiGLU activation, but source does not prove that it fuses activation with `linear_fc2`.

Backward findings:

- `BiasSwiGLUFunction.backward` reads saved input and bias and computes gradients through `bias_swiglu_back`.
- Saved tensors are real autograd state; exact device storage and any offload behavior depend on configuration and are not generalized beyond the observed run.
- The fc2 backward path is outside the inspected fused SwiGLU callable.

Confirmed versus unknown:

| Fact | Status |
|---|---|
| activation result is a real graph value consumed by fc2 | SOURCE_DERIVED |
| bias + SwiGLU callable fusion | SOURCE_DERIVED |
| activation output must incur HBM round-trip | UNKNOWN |
| exact GPU kernel launch count | UNKNOWN |
| activation bytes moved | UNKNOWN |
| registers/shared memory/occupancy | UNKNOWN |
| CUPTI kernel breakdown | UNKNOWN |

## PerformanceFacts v5

`artifacts/integration/swiglu/real_loop_001/performance_facts_v5.json` combines source-derived dataflow, prior real shape/L1/L2 evidence, and repeated timing. It preserves evidence status rather than upgrading inference to measurement:

- shape: batch 2, sequence 3, hidden 8, fc1 output 32, FP32, contiguous `[sequence,batch,hidden]` capture;
- authentic boundary median: 0.09344 ms from existing authentic profile;
- local minimal-model split fraction: 0.015571;
- paired fraction: 0.041957, explicitly instrumentation-affected;
- split Amdahl ceiling: 1.015818x;
- launch count, bytes, registers, shared memory, occupancy, CUPTI and backward kernel breakdown: UNKNOWN.

## Planner output

The real `GenerativeHypothesisPlanner` ran with zero matching SwiGLU MechanismRecords. It produced 3 raw, validated, and ranked hypotheses, with no evidence rejections:

1. `measurement-kernel-boundary-cost` — `EXPLORATORY`, measure launch/memory/boundary attribution;
2. `measurement-low-e2e-ceiling` — `FIRST_PRINCIPLES`, account for the low local system ceiling;
3. `fp-producer-consumer-boundary` — `FIRST_PRINCIPLES`, investigate possible activation/fc2 boundary elimination.

No Codex planning result or manually injected SwiGLU answer was used.

## Hypothesis selection

The structural producer-consumer hypothesis was not selected for implementation. Its graph-level observation is supported, but its decisive precondition—an artificial kernel boundary with eliminable materialization/HBM traversal—is not established. Source proves a returned activation tensor and a subsequent fc2 call; it does not prove the physical memory or launch behavior.

The selected hypothesis is `measurement-kernel-boundary-cost`, origin `EXPLORATORY`, because it directly measures the missing preconditions before any implementation. This is a measurement episode, not an optimization candidate.

## Precondition gate

`fp-producer-consumer-boundary = REJECTED_BEFORE_IMPLEMENTATION` for this episode due to unknown physical boundary evidence. One hypothesis was rejected at hypothesis level; no implementation retry was attempted.

`measurement-kernel-boundary-cost = SUPPORTED`: the local authentic workload exists, prior CUDA Event evidence exists, and its required fields are explicitly unknown in PerformanceFacts v5.

## Candidate and same-session repairs

No `REAL_OPT_LOOP_001` optimization candidate was created. Creating a reference wrapper at the existing SwiGLU boundary would not test the selected measurement hypothesis or prove activation/fc2 fusion, so it would be a false candidate. `HypothesisAttemptController` was not invoked because there was no implementation candidate requiring compile/correctness repair.

## L0 / L1 / L2

No new candidate means no new OJ verdict is claimed:

- L0: `NOT_RUN`;
- L1: `NOT_RUN`;
- TransformerLayer L2: `NOT_RUN`;
- TransformerBlock L2: `NOT_RUN`;
- Minimal GPTModel L2: `NOT_RUN`.

Existing authentic L1/L2 results remain historical baseline evidence and are not relabeled as candidate results.

## Amdahl prediction versus observation

No candidate speedup exists to compare. The existing local model ceiling remains approximately 1.015818x for infinite speedup under the split estimator, with paired 0.041957 fraction explicitly affected by inner instrumentation. This makes a micro-kernel improvement scientifically useful only if it tests a causal mechanism or supplies missing evidence; it does not justify a model-throughput claim.

## Causal interpretation

`CAUSAL_RESULT = MEASUREMENT_INCONCLUSIVE`.

The possible activation/fc2 structural opportunity is neither confirmed nor refuted. Its physical precondition is unmeasured. No performance gain, candidate correctness, or OJ promotion is claimed.

## Proposed knowledge delta

`artifacts/integration/swiglu/real_loop_001/knowledge_delta.json` is `INCONCLUSIVE` and explicitly not written to Mechanism Memory. It records that source-derived graph adjacency is insufficient to establish an HBM round-trip or eliminable materialization.

## What the Agent learned

The first real optimization episode correctly stopped before implementation when source evidence established a graph boundary but not the physical cost mechanism. The planner selected measurement rather than fabricating a fusion candidate. The next valid engineering step is to obtain local kernel/traffic/resource attribution, then revisit the structural hypothesis under the same evidence guard. That future step must remain local and must not infer nine-grid value from this minimal workload.

## Final status

```text
AUTHENTIC_DATAFLOW_RECONSTRUCTED = PASS
PERFORMANCE_FACTS_V5 = PASS
PLANNER_REAL_RUN = PASS
PLANNER_HYPOTHESES = 3
SELECTED_HYPOTHESIS_ID = measurement-kernel-boundary-cost
SELECTED_HYPOTHESIS_ORIGIN = EXPLORATORY
PRECONDITION_STATUS = SUPPORTED
HYPOTHESES_REJECTED_BEFORE_IMPLEMENTATION = 1
OPTIMIZATION_CANDIDATE_CREATED = NO
IMPLEMENTATION_ATTEMPTS = 0
L0_STATUS = NOT_RUN
L1_STATUS = NOT_RUN
L2_LAYER_STATUS = NOT_RUN
L2_BLOCK_STATUS = NOT_RUN
L2_MODEL_STATUS = NOT_RUN
OPERATOR_SPEEDUP = null
MODEL_SPEEDUP = null
CAUSAL_RESULT = MEASUREMENT_INCONCLUSIVE
KNOWLEDGE_DELTA = INCONCLUSIVE
MECHANISM_MEMORY_MODIFIED = NO
NINE_GRID_E2E_EXECUTED = NO
REMOTE_V100_USED = NO
BIV150_USED = NO
D_REPO_ONLY = PASS
C_AKA_MOTHER_MODIFIED = NO
MEGATRON_MODIFIED = NO
```
