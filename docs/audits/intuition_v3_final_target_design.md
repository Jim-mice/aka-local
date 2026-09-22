# Intuition v3 Final Target Design

Date: 2026-09-22. This turn designed one final held-out case only. No Codex trial was run, no planner or Mechanism Memory was modified, and no optimization candidate was created.

## Final stop rule

`benchmarks/intuition_v3/STOP_RULE.md` freezes this as the last Planner benchmark. After exactly three fresh independent external sessions: 2/3 or 3/3 full HIT means `HELD_OUT_MECHANISM_REASONING`; otherwise the status remains `CONSTRAINT_AWARE_MECHANISM_REASONING`. In both outcomes benchmark research stops and the project enters the real optimization loop. No v4/v5 may be added to improve the score.

## Target case

The public case is a row-wise weighted reduction. Its public facts define the input/output mathematics, current multi-phase graph, causal-prefix masking, FP16/FP32 numerical contract, row and batch sizes, on-chip scratch limits, register budget, and unknown profile fields. The current graph materializes per-element exponentials and weighted vectors before separate reductions.

The intended challenge is not to recognize a named optimization. A full answer must infer that the computation can be reorganized around a mathematically equivalent compact partial computation and a different reduction/work-ownership structure. The public facts do not name that representation or its combining rule. The evaluator-only key contains the expected family and acceptable equivalent descriptions.

## Why the case isolates algorithmic/decomposition generation

The output is not merely an elementwise transform: it couples a maximum, a denominator, and a weighted vector numerator under a causal prefix. The current graph exposes several phases, but the full-hit rubric requires more than deleting their boundaries. The candidate must explain:

1. what partial information is sufficient to represent a subset of the row;
2. how two disjoint valid prefixes/parts can be combined without changing the mathematical output;
3. how the work ownership and reduction structure change;
4. why the causal mask and FP16/FP32 error contract remain valid;
5. what register/scratch/backward risks must be tested.

That combination is a state-representation and decomposition judgment, not a keyword match. The public package never uses the evaluator’s names for the target family.

## Why familiar answers cannot receive full HIT

`Fusion alone = NO`: removing phase boundaries without introducing an equivalent partial computation representation is at most PARTIAL.

`Tiling alone = NO`: a 64-element tile or shared-memory staging plan that leaves the same mathematical phases and intermediate representation unchanged is at most PARTIAL.

`Measurement alone = NO`: a profiling plan is useful evidence discipline but does not generate the required algorithmic mechanism; it is at most PARTIAL.

Lifetime extension, materialization removal, vectorization, block/warp tuning, launch reduction, generic recomputation, and low-fraction prioritization are explicitly partial or miss patterns in `evaluator_only/answer_key.json`.

## Numerical validity

The public contract specifies FP16 inputs/outputs, optional FP32 accumulation, forward max-absolute error at most `3e-4`, backward relative error at most `2e-3`, and causal prefixes that may differ by row. A full hit must treat mathematical equivalence and numerical validation as preconditions, not as optional follow-up language. It must also remain grounded in the supplied register and scratch constraints.

## Deterministic-template audit

The read-only audit covered `lab/runtime/reasoning/hypothesis_planner.py`, `performance_model.py`, and `mechanism_memory.py`. Existing direct generation branches cover repeated reads/lifetime, producer-consumer materialization/fusion, missing-profile measurement, and low operator-fraction system reminders. Retrieval ranking adds scoring of existing records but does not create a new algorithmic state representation.

There is no direct branch for the final case’s required mathematically equivalent partial computation, reduction-state reformulation, or work-ownership change. Therefore:

`DETERMINISTIC_DIRECT_COVERAGE = NO`.

The planner was not changed to force this conclusion.

## Public/evaluator isolation

`benchmarks/intuition_v3/public/case_final/` contains only `case.json`, `facts.json`, and `contract.json`. The final mechanism family, acceptable equivalents, partial/miss patterns, hallucination criteria, and full-hit requirements are isolated in `evaluator_only/answer_key.json`. The schema has `additionalProperties: false` on every object schema. No blind result directory was created.

`verify_intuition_v3_package.py` passed with `INTUITION_V3_PACKAGE_PASS` and a complete SHA-256 manifest covering 7 package files. The checker also rejects v1/v2 answer identifiers, deterministic planner dumps, historical answer phrases, and evaluator sentinel leakage.

## External execution boundary

This turn performed no model call. The only future execution permitted for this benchmark is three fresh independent Codex CLI sessions using public case files only. Those results will be audited once; regardless of outcome, no further benchmark generation is allowed.

## Final status

```text
FINAL_TARGET_CASE = PASS
DETERMINISTIC_DIRECT_COVERAGE = NO
ALGORITHMIC_STATE_REFORMULATION_REQUIRED = YES
FUSION_ALONE_CAN_HIT = NO
TILING_ALONE_CAN_HIT = NO
MEASUREMENT_ALONE_CAN_HIT = NO
INTUITION_V3_PACKAGE = PASS
INTUITION_V3_PACKAGE_INTEGRITY = PASS
STOP_RULE_FROZEN = YES
EXTERNAL_CODEX_RUNS_COMPLETED = 0
OPTIMIZATION_CANDIDATE_CREATED = NO
D_REPO_ONLY = PASS
C_AKA_MOTHER_MODIFIED = NO
MEGATRON_MODIFIED = NO
```
