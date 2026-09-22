# Evidence-Grounded Generative Planner Audit

Date: 2026-09-22
Repository: `D:\Users\38154\Downloads\aka-local-main\aka-local-main`

## Previous retrieval-only limitation

The previous `HypothesisPlanner.rank()` created candidates only inside `for record in mechanisms`. `counterfactuals()` calculated useful answers, but those answers only changed the score of an existing `MechanismRecord`. With zero matching records, the result was always `[]`.

## New architecture

`GenerativeHypothesisPlanner` preserves the old path and adds a bounded generation path:

`PerformanceFacts + counterfactuals + matched MechanismRecords + explicit unknowns -> raw candidates -> evidence validation -> deduplication -> deterministic ranking -> ablation plan`

Candidates carry origin (`RETRIEVED`, `FIRST_PRINCIPLES`, `COMPOSED`, or `EXPLORATORY`), category, causal mechanism, transformation, risks, evidence references, unknowns, novelty, and score. Generation has no promotion, contract, benchmark, or implementation authority.

## Generation path

The deterministic first-principles generator checks repeated reads, producer-consumer boundaries, materialization, unknown launch/profile fields, and low E2E fractions. It emits measurement hypotheses when evidence is insufficient instead of asserting a bottleneck. An optional `llm_generator` callback accepts the existing Codex/Agent advisory output format; that output passes exactly the same supervisor checks.

`CodexAgentSession.run_planning_turn()` and `send_planning_context()` now expose a `PLANNING_ONLY` read-only path. The Agent can inspect generated hypotheses but cannot create candidates or decide promotion.

## Evidence validation

`validate_hypothesis_evidence()` validates every evidence reference against `PerformanceFacts`. Asserted measured values that target an explicit unknown field are rejected as `REJECT_EVIDENCE_HALLUCINATION`; mismatched or missing references are also rejected. Unknowns may remain in a hypothesis only as unknowns or measurement requirements.

## Retrieval vs generation

Retrieval remains intact and produces `RETRIEVED` candidates. First-principles candidates are generated even when the mechanism store returns zero records. When retrieval and generation coexist, a bounded two-part `COMPOSED` candidate may be created with interaction risk and an ablation plan. The novelty ratios and maximum candidate count are configurable through `NoveltyPolicy`, not fixed in the planner.

## Mechanism composition

Composition is limited to one retrieved candidate plus one generated candidate in this implementation. It records combined preconditions, expected effects, risks, unknowns, and mechanism IDs. Multi-transformation candidates receive a minimal ablation plan: baseline, each component, and full combination. No mechanism is written back to memory.

## Novelty policy

Default guidance is 50% exploit, 30% adjacent, 20% exploratory, with a configurable candidate cap. The policy is a budget and ranking input; zero retrieval matches do not suppress exploratory generation.

## Ranking

The deterministic supervisor ranks after validation. It considers prior retrieval score, novelty, measurement category, risk count, and the E2E ceiling. A low ceiling penalizes implementation-oriented candidates. Magnitude remains categorical (`LARGE`, `MEDIUM`, `SMALL`, `UNKNOWN`); no precise speedup is fabricated.

## Ablation planning

Any transformation expressed as multiple components receives a baseline/component/full plan. The plan is advisory and does not run experiments. It is included in P1 planning context for later human/controller review.

## Synthetic tests

`lab/tests/test_generative_planner.py` contains 10 tests covering:

- repeated-read first-principles generation with no memory;
- producer-consumer boundary generation;
- low E2E ceiling ranking penalty;
- unknown profile measurement hypothesis;
- retrieval plus composition without duplicates;
- Episode 28 blind reconstruction shape;
- hallucinated measured fact rejection;
- planning-only context hashing and authority;
- configurable novelty budget;
- advisory LLM candidate validation through the same supervisor.

No test seeds an answer string or writes a mechanism record.

## Episode 28 blind reconstruction

The independent fixture supplies only repeated reads, tensor lifetime metadata, and generic row/block facts. It does not contain Episode 28 text or a `retain x` answer. The planner produced an exploratory first-principles candidate with structured evidence references, risks, and unknowns. The test passes the structural requirement; it does not claim causal proof.

## Real SwiGLU blind generation

Input: `artifacts/integration/swiglu/authentic_profile/real_performance_facts_v3.json`.
Mechanism store: existing `knowledge/mechanisms.jsonl`, with zero matching SwiGLU records.
Output: `artifacts/integration/swiglu/authentic_profile/ranked_opportunities_v4.json`.

The raw generation contains two candidates and both validate:

1. `fp-producer-consumer-boundary` — `FIRST_PRINCIPLES`, exploratory; proposes measuring/validating producer-consumer materialization and possible fusion. It remains `UNKNOWN` magnitude and explicitly lists layout, traffic, and kernel-count unknowns.
2. `measurement-kernel-boundary-cost` — `EXPLORATORY`, measurement; asks for kernel-launch, boundary-latency, and memory-traffic evidence rather than claiming a memory-bound bottleneck.

No SwiGLU mechanism was manually added. No CUDA candidate was created.

## Failure cases

The planner rejects an advisory candidate when it claims a measured value for a field listed as unknown, references a missing facts path, or supplies a mismatched value. Duplicate observation/transformation pairs are rejected. The real SwiGLU run had zero rejected candidates and zero retrieval matches; its non-empty result comes only from first-principles generation.

## Remaining limitations

The first-principles generator is deterministic and template-backed; it is not itself an LLM. The optional existing Codex session adapter is planning-only and its output is untrusted until validation. The current composition policy is intentionally narrow. CUDA kernel launch count, registers, and CUPTI data remain unknown in the real SwiGLU facts. No implementation, benchmark, promotion, or candidate creation was performed.

## Final status

```text
PLANNER_RETRIEVAL_PATH = PASS
PLANNER_FIRST_PRINCIPLES_GENERATION = PASS
PLANNER_COMPOSITION = PASS
EVIDENCE_HALLUCINATION_GUARD = PASS
NOVELTY_WITH_ZERO_MEMORY_MATCHES = PASS
ABLATION_PLANNER = PASS

EP28_BLIND_HYPOTHESIS = PASS
SWIGLU_BLIND_HYPOTHESES = 2

P1_PLANNING_CONTEXT_WIRING = PASS
OPTIMIZATION_CANDIDATE_CREATED = NO

D_REPO_ONLY = PASS
C_AKA_MOTHER_MODIFIED = NO
MEGATRON_MODIFIED = NO
REMOTE_V100_USED = NO
BIV150_USED = NO
```
