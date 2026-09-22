# Intuition v3 Final 3-Run Audit

Date: 2026-09-22. This is a final non-blind audit of the three existing v3 results. No result, answer key, STOP_RULE, planner, Mechanism Memory, or Megatron source was modified.

## Executive result

All three runs are structurally valid. Runs 1 and 2 independently produced the same mechanism family: a compact running representation containing the row maximum, normalization contribution, and weighted numerator, with rescaling when a later partition changes the maximum. They also handled causal prefixes and numerical/resource risks. Run 3 proposed a useful tiled direct-accumulation decomposition but did not provide the required cross-partition state-combination rule.

Final classification: 2/3 HIT, 1/3 PARTIAL.

Under the frozen STOP_RULE, this is `INTUITION_V3_STATUS = HELD_OUT_MECHANISM_REASONING`. Planner benchmark research is closed regardless of the result.

## Validation

| Check | run1 | run2 | run3 |
|---|---|---|---|
| JSON parse | VALID | VALID | VALID |
| case_id | VALID | VALID | VALID |
| closed schema | VALID | VALID | VALID |
| evidence_refs | VALID | VALID | VALID |
| UNKNOWN discipline | VALID | VALID | VALID |
| evaluator leakage | none observed | none observed | none observed |
| generic slogan only | no | no | no |

Each result has exactly the required root/hypothesis fields, three hypotheses, and evidence references drawn from public `f1`–`f11`. The event traces show reads of public prompt/case/facts/contract/schema only. The string `answer_key` appears only because public `case.json` contains the explicit null isolation field; no evaluator file or evaluator path was read. No run asserts an unprovided bandwidth, occupancy, register count, latency, kernel count, or measured speedup.

## Run 1

`HIT`.

The first hypothesis discusses materialization/resource pressure, but the decisive second hypothesis goes beyond it: it maintains running `M`, `D`, and a 128-wide numerator, rescales prior accumulators when a later tile has a larger maximum, applies the row-specific mask, and keeps the transformation conditional on FP32 accuracy, register/scratch fit, and backward correctness. That is a concrete computation-state and reduction-decomposition change, not merely fusion. It cites public facts and labels profile/backward/launch information unknown.

State-reformulation checks: A yes; B yes; C yes through tile processing and rescaling of prior partial accumulators; D yes; E yes through the rescaling recurrence and normalized output semantics; F yes; G yes.

## Run 2

`HIT`.

The primary hypothesis explicitly maintains a running maximum, normalization sum, and weighted accumulator, rescales prior state on a larger tile maximum, skips masked suffixes, and normalizes after the valid prefix. The second hypothesis supplies an alternative two-pass decomposition and correctly identifies it as a different tradeoff rather than pretending it is the same mechanism. Numerical error, register/occupancy, scratch, mask, and backward risks are explicit.

State-reformulation checks: A yes; B yes; C yes via tile-wise state update and rescaling; D yes; E yes; F yes; G yes.

## Run 3

`PARTIAL`.

The first two hypotheses are technically meaningful: they use row-level FP32 accumulators, bounded 64-element tiles, direct denominator/numerator accumulation, causal-mask handling, and numerical/resource risks. However, they do not specify how two arbitrary partitions or prefixes compose when their maxima differ. The second hypothesis remains a two-stage tiled decomposition after storing the maximum, and the third is measurement-first. This is an implementable decomposition proposal, but it lacks the compact partial-state/merge rule required for a full HIT.

State-reformulation checks: A yes; B partial; C no explicit cross-partition combination; D yes; E partial; F yes; G yes. The missing C item prevents HIT.

## State-reformulation analysis

The answer key requires all of: observation of the current graph and output semantics, a causal explanation for avoidable intermediate work, a concrete changed representation or reduction structure, a mathematical equivalence precondition including causal-prefix handling, and evidence-grounded numerical/resource risk.

Runs 1 and 2 satisfy the complete chain. Their shared family is not just “fuse phases”: it carries a partial aggregate whose maximum can change and rescales the existing denominator and weighted numerator consistently. This supplies the missing composition logic. Run 3 removes materialization and changes work grouping, but does not establish how its row accumulators remain mathematically equivalent across partitions with different maxima.

## Semantic repeatability

Runs 1 and 2 are semantically repeatable members of the same family, even though their wording differs. Both preserve the same invariants: masked elements contribute zero, the maximum controls exponential scaling, the denominator and weighted numerator use the same scaling, and final division yields the required normalized output. Run 3 is an adjacent bounded two-pass/tiled alternative, not a third full state-reformulation hit.

## Template escape

The read-only deterministic planner audit remains unchanged: direct first-principles branches cover repeated-read/lifetime, producer-consumer materialization/fusion, missing-profile measurement, and low-fraction system reminders. There is no direct rule that creates a running maximum/normalization/numerator representation or its rescaling composition rule. Therefore:

`DETERMINISTIC_DIRECT_COVERAGE = NO`.

The two HIT outputs are not a direct dump of an existing deterministic candidate. They do use familiar concepts such as tiling and materialization reduction as implementation context, but the decisive mechanism is the new compact partial computation and its rescaling rule.

## Evidence hallucination audit

`MEASURED_FACT_HALLUCINATIONS_ACCEPTED = 0`.

Derived mathematical language about rescaling and normalization is not a measured claim. When the runs discuss profile detail, backward graph, launch count, occupancy after transformation, or tolerance under the new recurrence, they label those items as unknown or required evidence. Numerical/resource risks are tied to public `f5`–`f8`; causal-prefix statements are tied to `f6`.

## V1 → V2 → V3 capability progression

| Stage | Demonstrated capability |
|---|---|
| V1 | repeatable recovery of familiar mechanism families |
| V2 | constraint-aware rejection, system-value comparison, and unknown discipline |
| V3 | 2/3 independent full hits on a held-out computation-state/decomposition mechanism, plus one close partial |

This is stronger than pattern recovery: the successful runs construct a mathematically constrained representation not directly emitted by the deterministic templates. It is not evidence that every future novel mechanism will succeed, but it meets the frozen 2/3 criterion.

## Final intuition status

`INTUITION_V3_STATUS = HELD_OUT_MECHANISM_REASONING`.

This status is granted strictly by the precommitted 2/3 rule, not by relaxing the rubric. The full-hit threshold was applied to the state representation, partition-combination logic, semantics, causal masking, and risks.

## Planner research closure

`PLANNER_RESEARCH_STATUS = SUFFICIENT_FOR_OPTIMIZATION_LOOP`.

No further v4/v5 benchmark, prompt tuning, or template expansion is authorized by this conclusion. The planner benchmark phase is closed.

## Model-change decision

`MODEL_CHANGE_RECOMMENDED_NOW = NO`.

The current Codex has demonstrated the required evidence → mechanism → experiment-planning capability on this final held-out case. The remaining uncertainty belongs in real implementation/OJ feedback, not in another model comparison.

## Transition to real optimization loop

The next phase is the real optimization loop for authentic Megatron SwiGLU. Current scope facts remain binding:

- local minimal GPTModel split fraction is approximately 1.56%;
- paired fraction is approximately 4.20% and is instrumentation-affected;
- local-model Amdahl ceiling is therefore low;
- nine-grid assets remain unavailable;
- micro speedup must not be treated as system speedup.

## Recommended first real experiment

`NEXT_REAL_EXPERIMENT_SELECTED = YES` — validate one structural SwiGLU dataflow hypothesis at the authentic boundary: determine whether the current producer/consumer path materializes an intermediate that can be replaced by a bounded, numerically equivalent state-preserving computation without increasing register pressure or changing backward semantics.

Evidence: authentic Megatron path and shape capture are real, but current kernel count, transaction breakdown, register pressure, and backward graph detail remain unknown. Causal story: if an intermediate is only a boundary artifact, its write/read traversal may be avoidable; if it is required for reuse or resource balance, the hypothesis must be rejected. Expected scope: first SwiGLU boundary, with an explicitly low local-model ceiling and no nine-grid claim.

Required ablation: baseline authentic path; minimal state-preserving transformation; any combined variant only after the minimal result; compare forward-only and backward behavior separately. Required gates: L0 numerical/gradient correctness and invocation evidence, L1 real Megatron replacement and fallback detection, TransformerLayer/Block L2 correctness and paired/split timing, then Minimal Model L2 whole-step impact. No implementation was created in this audit.

## Final status

```text
VALID_RUNS = 3
RUN1 = HIT
RUN2 = HIT
RUN3 = PARTIAL
FULL_HITS = 2
MEASURED_FACT_HALLUCINATIONS_ACCEPTED = 0
DETERMINISTIC_DIRECT_COVERAGE = NO
INTUITION_V1_STATUS = REPEATABLE_MECHANISM_REASONING
INTUITION_V2_STATUS = CONSTRAINT_AWARE_MECHANISM_REASONING
INTUITION_V3_STATUS = HELD_OUT_MECHANISM_REASONING
PLANNER_RESEARCH_STATUS = SUFFICIENT_FOR_OPTIMIZATION_LOOP
MODEL_CHANGE_RECOMMENDED_NOW = NO
NEXT_REAL_EXPERIMENT_SELECTED = YES
OPTIMIZATION_CANDIDATE_CREATED = NO
D_REPO_ONLY = PASS
C_AKA_MOTHER_MODIFIED = NO
MEGATRON_MODIFIED = NO
```
