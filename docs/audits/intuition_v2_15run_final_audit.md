# Intuition v2 15-Run Final Audit

Date: 2026-09-22. This is a non-blind audit of the existing 15 result files. No result, planner, MechanismRecord, answer key, or Megatron source was modified.

## Executive result

All 15 `result.json` files are structurally valid and evidence-reference valid. The strongest result is not full held-out generation: the model reliably uses counterevidence to reject familiar positive patterns and preserves unknowns, but Case A did not produce a clear algorithmic/decomposition reformulation. The appropriate status is:

`INTUITION_V1_STATUS = REPEATABLE_MECHANISM_REASONING`
`INTUITION_V2_STATUS = CONSTRAINT_AWARE_MECHANISM_REASONING`
`PLANNER_RESEARCH_STATUS = ONE_MORE_TARGETED_TEST`

The last status is deliberately not `SUFFICIENT_FOR_OPTIMIZATION_LOOP`: the adversarial rejection capability is strong, but held-out mechanism generation outside the existing template families is not yet demonstrated.

## Validation

| Check | Result |
|---|---|
| Result count | 15/15 |
| JSON parse | 15/15 |
| case_id | 15/15 correct (`HELDOUT_*`) |
| hypothesis schema | 15/15 valid; 3 hypotheses each |
| evidence_refs | 15/15 use IDs from the corresponding public facts |
| UNKNOWN discipline | no unsupported measured value accepted |
| evaluator-only leakage | no `answer_keys.json` or evaluator-only path in event traces |
| generic-only slogan | 0 runs dominated by an unsupported generic slogan |
| accepted measured-fact hallucinations | 0 |

The event traces show the sessions reading the public prompt, case, facts, contract, and schema. The repeated `evaluator_only` wording is from the public instruction not to read that directory; no event shows an evaluator-only file read. Some runs contain transient transport reconnect messages, but each produced a complete valid result.

## Case A — Algorithmic held-out

Rubric target: a real change in reduction/decomposition algorithm that respects max-subtraction semantics, error tolerance, and scratch constraints. Simple phase fusion is not a full hit.

| Run | Validity | Classification | Audit reason |
|---|---|---|---|
| run1 | VALID | PARTIAL | identifies phase costs and numerical risk, but primary transformation is phase fusion; the measurement alternative is sound |
| run2 | VALID | PARTIAL | mentions streaming/online normalization and defers it correctly, but selected transformation remains fused/tiled scheduling rather than a concrete algorithmic reformulation |
| run3 | VALID | PARTIAL | proposes row-tiled streaming and controlled comparison, but does not establish a new reduction algorithm or decomposition as the causal choice |

`CASE_A_HITS = 0`; `CASE_A_PARTIALS = 3`. None of the three is a MISS because all preserve numerical constraints, acknowledge unknown profile detail, and avoid claiming a measured bottleneck. However, “phase fusion” is not being upgraded to an algorithmic HIT merely because it has a causal explanation.

## Case B — Fusion trap

| Run | Validity | Classification | Audit reason |
|---|---|---|---|
| run1 | VALID | HIT | explicitly defers full fusion, uses measured spill/register/occupancy facts, proposes consumer-local alternative |
| run2 | VALID | HIT | rejects unconditional fusion and proposes bounded consumer-local or partial-fusion measurement |
| run3 | VALID | HIT | rejects full fusion, explains resource-induced occupancy loss, preserves separate kernels pending evidence |

`CASE_B_FUSION_REJECTION_HITS = 3`. This is genuine counterevidence use, not a generic “fusion may be risky” disclaimer: every run makes the resource facts causal and changes the priority of fusion.

## Case C — Lifetime rejection

| Run | Validity | Classification | Audit reason |
|---|---|---|---|
| run1 | VALID | HIT | rejects full residency and proposes reload/recompute comparison |
| run2 | VALID | HIT | rejects register retention, proposes bounded staging or rematerialization |
| run3 | VALID | HIT | rejects the familiar lifetime extension and accounts for spill/occupancy tradeoff |

`CASE_C_LIFETIME_REJECTION_HITS = 3`. The model recognized the familiar repeated-use pattern but did not mechanically apply the v1 lifetime answer.

## Case D — System-value trap

| Run | Validity | Classification | Audit reason |
|---|---|---|---|
| run1 | VALID | HIT | compares dominant-path coverage against micro-kernel scope and preserves workload-mix uncertainty |
| run2 | VALID | HIT | explicitly uses Amdahl-style reasoning and stages structural work before micro work |
| run3 | VALID | HIT | rejects interpreting 8x local speedup as 8x system speedup and proposes incremental evaluation |

`CASE_D_SYSTEM_REASONING_HITS = 3`. This goes beyond merely repeating “low fraction is bad”: it compares two candidates, reasons about realized speedup and coverage, and keeps deployment mix as unknown.

## Case E — Insufficient evidence

| Run | Validity | Classification | Audit reason |
|---|---|---|---|
| run1 | VALID | HIT | defers bottleneck classification and requests boundary, traffic, cache, occupancy, and arithmetic evidence |
| run2 | VALID | HIT | explicitly keeps memory/compute/occupancy alternatives conditional on measurement |
| run3 | VALID | HIT | requests per-kernel and synchronization attribution before conditional fusion or layout/launch changes |

`CASE_E_EVIDENCE_DISCIPLINE_HITS = 3`. No run declared memory-bound or compute-bound from tensor size, operation count, or one ambiguous latency number.

## Cross-run mechanism families

| Case | Stable family across runs | Interpretation |
|---|---|---|
| A | numerical-preserving tiled/streaming or phase scheduling plus measurement | structurally sensible, but still adjacent to fusion/materialization templates; no held-out algorithmic HIT |
| B | resource-aware rejection of full fusion; consumer-local or bounded alternatives | repeatable counterfactual rejection |
| C | reject full lifetime extension; reload/recompute/split alternatives | repeatable suppression of familiar v1 instinct |
| D | scope-aware comparison and workload-mix measurement | repeatable system reasoning |
| E | measurement-first bottleneck classification | repeatable epistemic discipline |

## Counterexample rejection

B/C are the clearest evidence that the model can move from pattern matching to constraint-aware selection. It did not simply identify “producer-consumer” or “repeated use”; it used spills, register budget, occupancy, and negligible launch cost to reverse the default transformation. This is a real capability gain over v1 positive-example recovery.

## Evidence hallucination audit

No accepted hypothesis asserted a new measured number outside the supplied facts. Each result cited public fact IDs. Unknown fields stayed conditional or were listed as required measurements. The outputs contain derived language such as “may outweigh” and “upper bound,” but do not relabel those as measured facts. `MEASURED_FACT_HALLUCINATIONS_ACCEPTED = 0`.

## Template escape analysis

Case A is the limiting result. All three runs discuss fusion, tiling, streaming, or measurement; none commits to a distinct algorithmic/decomposition reformulation that can be credited as outside the current deterministic templates. Run2’s online-normalization discussion is cautious and numerically aware, but it is explicitly deferred rather than developed into a held-out mechanism hypothesis.

B/C/D/E show constraint use and alternative selection, but their correct responses can still be constructed by composing familiar resource, system-value, and measurement concepts. Therefore v2 demonstrates transfer under contradiction, not reliable novel mechanism invention.

## V1 vs V2

| Capability | V1 status | V2 evidence |
|---|---|---|
| PATTERN_RECOVERY | demonstrated 3/3 on familiar cases | retained |
| CONSTRAINT_AWARE_REASONING | not isolated | demonstrated strongly in B/C/E |
| COUNTERFACTUAL_REJECTION | not tested adversarially | B 3/3, C 3/3 |
| HELD_OUT_MECHANISM_GENERATION | not tested | A: 0/3 full hits |
| SYSTEM_REASONING | low-fraction template existed | D 3/3 comparative reasoning |
| EPISTEMIC_DISCIPLINE | deterministic guard existed | E 3/3, no accepted hallucinations |

The two benchmark generations must not be collapsed into one intelligence score. V1 proves repeatable recovery of known mechanism families. V2 proves reliable rejection and evidence discipline under several adversarial constraints, but not broad template-independent invention.

## Current Agent intuition capability

The current Codex can:

- identify a familiar opportunity;
- inspect explicit resource and system constraints;
- reject a familiar transformation when those constraints contradict its preconditions;
- preserve unknowns and request discriminating measurements;
- compare local speedup against system coverage.

It has not yet demonstrated a repeatable algorithmic reformulation outside the deterministic generator’s covered families. Thus the evidence supports `CONSTRAINT_AWARE_MECHANISM_REASONING`, not `HELD_OUT_MECHANISM_REASONING`.

## Remaining failure modes

1. Phase fusion and tiled scheduling can be described fluently without discovering a genuinely new reduction algorithm.
2. The benchmark does not yet separate a model’s independent algorithmic insight from a well-composed measurement plan.
3. Evidence references are valid, but semantic evaluator judgments remain human rubric judgments rather than automatically proven causal claims.
4. The results show no hallucinations on these cases, but this is not evidence that hallucination risk is zero for future schemas or longer outputs.

## Planner research stop/go decision

`PLANNER_RESEARCH_STATUS = ONE_MORE_TARGETED_TEST`.

Do not continue broad benchmark expansion. One focused follow-up is justified: a new algorithmic/decomposition held-out case whose acceptable answer cannot be expressed as the existing lifetime, fusion, materialization, measurement, or low-fraction families. If that case produces repeatable evidence-grounded hits, the system can move to the optimization loop; otherwise the remaining limitation is novel mechanism generation, not counterexample reasoning.

## Model-change decision

`MODEL_CHANGE_RECOMMENDED_NOW = NO`.

The current default Codex is already sufficient for evidence-grounded mechanism selection, counterexample rejection, and experiment planning. A model A/B comparison is not justified until the one targeted algorithmic case isolates a genuine generation failure.

## Next engineering step

Run one targeted held-out algorithmic/decomposition test using the existing public/evaluator separation. If it is not a hit, proceed with constrained optimization work while labeling the planner capability accurately; do not expand the general framework or claim autonomous novel mechanism discovery.

## Final status

```text
VALID_RUNS = 15
CASE_A_HITS = 0
CASE_A_PARTIALS = 3
CASE_B_FUSION_REJECTION_HITS = 3
CASE_C_LIFETIME_REJECTION_HITS = 3
CASE_D_SYSTEM_REASONING_HITS = 3
CASE_E_EVIDENCE_DISCIPLINE_HITS = 3
MEASURED_FACT_HALLUCINATIONS_ACCEPTED = 0
INTUITION_V1_STATUS = REPEATABLE_MECHANISM_REASONING
INTUITION_V2_STATUS = CONSTRAINT_AWARE_MECHANISM_REASONING
PLANNER_RESEARCH_STATUS = ONE_MORE_TARGETED_TEST
MODEL_CHANGE_RECOMMENDED_NOW = NO
OPTIMIZATION_CANDIDATE_CREATED = NO
D_REPO_ONLY = PASS
C_AKA_MOTHER_MODIFIED = NO
MEGATRON_MODIFIED = NO
```
