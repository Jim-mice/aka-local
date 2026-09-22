# Adversarial Held-out Intuition Benchmark v2

Date: 2026-09-22. This turn only designed and integrity-checked the benchmark. No Codex blind trial was run, no planner was modified, and no MechanismRecord or optimization candidate was created.

## Deterministic template audit

`benchmarks/intuition_v2/TEMPLATE_COVERAGE.md` audits `hypothesis_planner.py`, `performance_model.py`, and `mechanism_memory.py`. Direct deterministic generation currently covers repeated-read/lifetime, producer-consumer materialization/fusion, missing profile measurement, and low operator-fraction system-ceiling reminders. Retrieval ranking also scores existing records using launch, synchronization, reuse, bytes, feasibility, and evidence fields. It has no direct branch for algorithmic reformulation, recompute-versus-store, ownership/remapping, decomposition changes, or occupancy-aware rejection of a familiar transformation.

Four of the five held-out cases require more than selecting a positive old template. No expected answer was added to the planner or mechanism memory.

## Held-out cases

### Case A — algorithmic reformulation

Public facts describe a three-phase numerically constrained row transform with no repeated tensor use and no guaranteed full-row retention. The adversarial target is a change in reduction algorithm/decomposition under forward/backward error constraints, not lifetime, fusion, vector width, or launch tuning. A hit must identify an algorithmic or decomposition hypothesis with numerical preconditions; a measurement-first alternative is partial; a lifetime/fusion slogan is a miss.

### Case B — tempting boundary combination

The public facts contain a producer/consumer intermediate, but the measured launch cost is negligible while the combined form has high register use, observed spills, and severe occupancy collapse. A hit must reject or sharply down-rank the attractive combination and propose an occupancy-aware decomposition or measurement. An unconditional fusion recommendation is a miss; invented benefit or profile data is evidence hallucination.

### Case C — lifetime rejection

The public facts deliberately resemble the earlier repeated-use pattern, but measured register budget, spill traffic, and occupancy show that keeping the full live set is infeasible. A hit must recognize the contradiction and consider reload/recompute or another bounded alternative. Recommending full residency without addressing spill is a miss.

### Case D — system-value trap

One micro operator has a large local upper bound but only 0.2% measured scope fraction. A disjoint structural path covers 42% of the scope with a smaller local upper bound. A hit must compare system value, preserve workload-mix uncertainty, and prioritize or measure accordingly. Treating the 8x micro result as an 8x system result is a miss. This is not merely a low-fraction reminder because the case requires comparison between two plausible transformations.

### Case E — insufficient evidence

The public facts provide tensor size, operation count, and boundary latency, while transaction bytes, cache behavior, occupancy, kernel count, and arithmetic intensity are unknown. A hit must emit a measurement hypothesis that distinguishes memory and compute explanations. Declaring either bottleneck from tensor size or operation count is a miss and is an evidence hallucination if labeled measured.

For each case, the public directory contains only `case.json`, `facts.json`, and `contract.json`. The unified prompt and output schema are at the package root. Expected families, traps, alternatives, and failure criteria are isolated in `evaluator_only/answer_keys.json`.

## Answer-key isolation and integrity

`verify_intuition_v2_package.py` checks the exact public case set, required files, absence of evaluator metadata, absence of deterministic candidate identifiers and historical answer phrases, evaluator sentinel separation, and SHA-256 coverage. It produced `INTUITION_V2_PACKAGE_PASS` with 19 package files hashed. The checker does not run a model and does not inspect or score hypotheses.

## External execution protocol

`benchmarks/intuition_v2/BLIND_PROMPT.md` requires JSON-only planning, evidence references, unknown preservation, and no code or promotion decisions. Nine external result slots are not created in this design package; external sessions should write results only after independently selecting a case and using a fresh context. No nested Codex invocation is used.

## Test status

The package verifier passed and the new checker passed Python compilation. No pytest installation or global package change was attempted. No Codex trial was run.

## Final status

```text
TEMPLATE_COVERAGE_AUDITED = PASS
HELD_OUT_CASE_COUNT = 5
FUSION_TRAP_CASE = PASS
LIFETIME_REJECTION_CASE = PASS
SYSTEM_VALUE_CASE = PASS
INSUFFICIENT_EVIDENCE_CASE = PASS
INTUITION_V2_PACKAGE = PASS
INTUITION_V2_PACKAGE_INTEGRITY = PASS
EXTERNAL_CODEX_RUNS_COMPLETED = 0
OPTIMIZATION_CANDIDATE_CREATED = NO
NINE_GRID_E2E_EXECUTED = NO
D_REPO_ONLY = PASS
C_AKA_MOTHER_MODIFIED = NO
MEGATRON_MODIFIED = NO
```
