# Codex Blind Intuition 9-Run Audit

## Executive result

All nine `result.json` files are parseable and structurally valid.

- All case IDs match.
- All hypotheses satisfy the schema.
- All `evidence_refs` resolve to facts in the corresponding `facts.json`.
- No accepted hypothesis invents a measured fact.
- Explicitly unknown fields remain unknown.
- RMSNorm: all 3 runs hit the lifetime/redundant-traversal family.
- GDN: all 3 runs contain algebraic/dataflow simplification, although some proposals are generic fusion.
- SwiGLU: all 3 runs contain structural hypotheses; at least one non-trivial structural family appears independently in all 3.

The strongest justified status is:

`REPEATABLE_MECHANISM_REASONING`

This demonstrates repeatable mechanism-level planning, not proven optimization quality or physical performance improvement.

## Validation

| Run | Case ID | Schema | Evidence refs | Result |
|---|---|---:|---:|---|
| rmsnorm_run1 | Correct | PASS | All valid | Accepted |
| rmsnorm_run2 | Correct | PASS | All valid | Accepted |
| rmsnorm_run3 | Correct | PASS | All valid | Accepted |
| gdn_run1 | Correct | PASS | All valid | Accepted |
| gdn_run2 | Correct | PASS | All valid | Accepted |
| gdn_run3 | Correct | PASS | All valid | Accepted |
| swiglu_run1 | Correct | PASS | All valid | Accepted |
| swiglu_run2 | Correct | PASS | All valid | Accepted |
| swiglu_run3 | Correct | PASS | All valid | Accepted |

No result contains extra schema properties, missing required fields, invalid evidence references, or more than three hypotheses.

## RMSNorm

Historical target family: extending the lifetime of immutable input values across the reduction to eliminate a redundant traversal.

| Run | Mechanism evidence | Status | Other useful alternatives |
|---|---|---|---|
| run1 | “retain them in registers” and “eliminate a second global-memory read” | HIT | Shared-memory staging; reduction/elementwise fusion |
| run2 | “retain each thread’s input elements across the reduction” and “avoiding a second global-memory read” | HIT | Resource-adaptive register/shared-memory/reload variants |
| run3 | “reuse the input values across both consumers” and “input’s lifetime spans both uses” | HIT | On-chip staging; register retention; fused row execution |

All three independently identify the causal family, not merely generic vectorization or tuning.

`RMSNorm lifetime hits: 3/3`

The alternatives are also useful, but mostly represent implementation choices for the same lifetime/reuse mechanism.

## GDN

Target family: algebraic/dataflow simplification that reduces intermediate elementwise work.

| Run | Actual symbolic/dataflow simplification | Generic fusion/tuning | Status |
|---|---|---|---|
| run1 | Constant folding of invariant coefficients; producer-consumer elimination | Algebraic fusion of elementwise stages | HIT |
| run2 | Factoring/reusing common subexpressions; reciprocal substitution for invariant divisors | Kernel fusion | HIT |
| run3 | Folding invariant coefficients into adjacent arithmetic; producer-consumer fusion | Fewer elementwise stages | HIT |

The runs distinguish several mechanisms:

- Actual algebraic simplification: constant folding, factoring, reciprocal substitution.
- Dataflow simplification: keeping `new_v` transient and consuming it directly.
- Generic implementation change: fusing sequential elementwise kernels.

The first and third runs are especially clear about algebraic/dataflow transformation. Run2’s first hypothesis is generic fusion, but its second and third hypotheses provide independent algebraic transformations.

`GDN algebraic hits: 3/3`

## SwiGLU

There is no predefined correct answer. The hypotheses were evaluated against the supplied evidence and unknowns.

| Run | Main structural hypotheses | Evidence discipline | Assessment |
|---|---|---|---|
| run1 | Boundary fusion; launch/resource tuning; coordination across adjacent MLP operations | PASS; measured fractions and 3.45% overhead correctly used | Contains structural ideas |
| run2 | Boundary fusion; fusion-depth/resource tradeoff; measurement-attribution correction | PASS; unknown kernel breakdown remains unknown | Contains structural ideas |
| run3 | Boundary fusion; lower-overhead measurement/dispatch; fusion-boundary tuning | PASS; register pressure and kernel breakdown remain unknown | Contains structural ideas |

Repeated mechanism families:

- Producer-consumer/intermediate-materialization elimination: 3/3.
- Boundary fusion: 3/3.
- Register-pressure versus occupancy tradeoff: 3/3.
- Measurement and attribution uncertainty: 3/3.
- System-level impact constrained by the small measured model fraction: 3/3.

At least one non-trivial structural mechanism appears independently in `3/3` runs.

The most interesting run-specific idea is in `swiglu_run1` hypothesis 3: comparing isolated activation optimization with paired adjacent MLP optimization because the paired measured fraction is materially larger. This is not merely kernel tuning; it proposes changing the optimization scope based on system-level attribution. It remains unproven because the cause of the fraction difference is explicitly unknown.

## Mechanism repeatability

| Case | Criterion | Result |
|---|---|---|
| RMSNorm | At least 2/3 lifetime/redundant-traversal hits | 3/3 |
| GDN | At least 2/3 algebra/dataflow simplification hits | 3/3 |
| SwiGLU | At least 2/3 structural runs | 3/3 |
| Evidence discipline | Hallucinated measured facts not silently accepted | PASS |

The repeatability threshold is satisfied without weakening the rubric.

## Evidence hallucination audit

No accepted hypothesis invents a measured fact.

Correctly preserved unknowns include:

- RMSNorm register pressure and global traffic.
- GDN profile, launch cost, numerical behavior, and backend capability.
- SwiGLU boundary breakdown, register pressure, occupancy, launch count, and materialization details.

Questionable but not rejected claims:

- RMSNorm fusion hypotheses sometimes discuss intermediate materialization or separate kernel boundaries that are not directly present in `facts.json`; they are framed as possibilities and paired with unknowns or required evidence.
- GDN fusion language sometimes treats materialization as a practical optimization target, but this is directly supported by fact `f5`.
- SwiGLU hypotheses infer possible launch, synchronization, or intermediate-storage costs; these are explicitly marked as unknown rather than measured.

Therefore:

`EVIDENCE_HALLUCINATION_ACCEPTED = NO`

## Structural vs trivial hypotheses

Structural hypotheses include:

- Retaining immutable RMSNorm inputs across the reduction.
- Shared-memory/register staging.
- Eliminating GDN intermediate materialization.
- Algebraic factoring and reciprocal substitution.
- Fusing producer-consumer boundaries.
- Expanding SwiGLU optimization from an isolated boundary to adjacent MLP operations.

More trivial or template-like proposals include:

- Generic launch-configuration tuning.
- Generic occupancy/register balancing.
- Generic “fuse compatible operations” wording without a specific dataflow consequence.
- Measurement-only proposals.

The benchmark is not dominated by trivial tuning: every case contains at least one mechanism-level structural proposal. However, several hypotheses repeat standard fusion vocabulary without identifying a more specific causal bottleneck.

## Deterministic planner vs Codex LLM

### Retrieval-only

Retrieval-only planning can return nothing when no matching mechanism record exists. The planner documentation explicitly identifies this limitation.

It cannot reliably produce a new lifetime or algebraic mechanism when the mechanism memory has no matching record.

### Deterministic first-principles planner

The deterministic planner directly covers:

- Repeated-read/lifetime extension.
- Producer-consumer fusion.
- Intermediate materialization removal.
- Unknown kernel/profile measurement.
- Low end-to-end-fraction prioritization.

Therefore much of the repeated RMSNorm and SwiGLU output is already within deterministic template coverage.

### Codex blind runs

Codex repeatedly produced:

- RMSNorm lifetime extension.
- GDN algebraic simplification and producer-consumer elimination.
- SwiGLU structural boundary-fusion hypotheses.

The clearest ideas outside the directly shown deterministic templates are:

- GDN coefficient factoring and reciprocal substitution.
- SwiGLU paired adjacent-MLP scope expansion.

These are plausible, but not demonstrated as novel discoveries. They may be natural extrapolations from the supplied facts rather than genuinely new mechanisms.

### Hallucination comparison

The Codex outputs did not hallucinate measured facts. They were disciplined about unknowns and required evidence.

The deterministic path has stronger formal validation because it generates and validates its own structured candidates. Codex achieved comparable evidence discipline in these nine runs, but its prose contains more conditional extrapolation and more generic fusion language.

### Overall interpretation

The LLM adds some mechanism elaboration and alternative transformations, but the dominant successful families are already represented by deterministic first-principles templates.

The evidence supports:

- More than retrieval-only behavior.
- At least template-level mechanism reasoning.
- Some plausible extension beyond the templates.
- Not yet a strong claim of independent, novel mechanism discovery.

## Novel ideas not covered by templates

Most notable:

1. GDN invariant-divisor reciprocal substitution.
2. GDN common-subexpression factoring.
3. SwiGLU paired-scope optimization based on the difference between isolated and paired fractions.
4. SwiGLU resource-adaptive fusion-boundary selection.

These are interesting enough for follow-up measurement, but none is validated by implementation or performance evidence.

## Failure modes

- Fusion is frequently proposed without knowing whether the current implementation already fuses the relevant operations.
- Several hypotheses rely on unobserved kernel boundaries or materialization behavior.
- Potential benefits are generally qualitative and unmeasured.
- Novelty is difficult to separate from recombination of familiar optimization patterns.
- SwiGLU hypotheses correctly identify the small system-level ceiling, but none establishes a high-value optimization opportunity.
- The existing planner report predates these completed external runs and should not be treated as the final benchmark result.

## Final intuition status

The completed runs satisfy the previously agreed repeatability standard.

This means the current planning architecture gives the model a repeatable opportunity to reason about mechanisms. It does not establish that the model will find the best implementation, produce speedups, or replace measurement and human review.

The old 27-episode failure is therefore meaningfully addressed at the hypothesis-generation level: the current blind runs independently recover the historical RMSNorm lifetime mechanism in all three trials.

## Implication for the old 27-episode Agent failure

The old Agent’s repeated failure was not necessarily caused by lack of optimization vocabulary. It failed to discover or prioritize the data-lifetime transformation.

The current results show a materially better outcome:

- The lifetime mechanism appears independently in all RMSNorm runs.
- It is stated causally, not merely as “optimize memory.”
- Register pressure, occupancy, and spill risks are preserved as unknowns.
- Alternative implementations are proposed without claiming validation.

This is evidence that the newer planning architecture improves mechanism discovery opportunity. It is not evidence that the complete Agent loop would implement or validate the mechanism successfully.

## Recommended next experiment

Run a second adversarial blind suite with:

- Novel dataflow cases whose answer is not represented by current deterministic templates.
- At least one case where generic fusion is incorrect or harmful.
- Independent semantic grading by mechanism family.
- A held-out case testing whether the model recognizes when lifetime extension should be rejected because of register pressure.
- Separate scoring for template recovery, causal novelty, and evidence discipline.

Do not change the model or create an optimization candidate yet; first establish whether the apparent novelty survives template-held-out evaluation.

RMSNORM_LIFETIME_HITS = 3
GDN_ALGEBRA_HITS = 3
SWIGLU_STRUCTURAL_RUNS = 3

EVIDENCE_HALLUCINATION_ACCEPTED = NO

INTUITION_STATUS =
REPEATABLE_MECHANISM_REASONING

MODEL_CHANGE_RECOMMENDED_NOW = NO

OPTIMIZATION_CANDIDATE_CREATED = NO