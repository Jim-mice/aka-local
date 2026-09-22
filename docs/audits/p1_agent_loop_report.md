# P1 Agent Loop Implementation Report

## 1. Existing long-horizon capabilities reused

`LongHorizonRunner` already has one `session.start()` per local episode, controller-owned plan/edit/evaluator stages, candidate-path safety, diagnostic routing, a supervisor gate, and an earlier source lineage record. `CodexAgentSession` already retains a Codex thread object across `run_experiment_turn()` calls until `close()`.

The old V100 campaign differed materially: its CLI created an ephemeral Codex thread for one prompt, closed it, and only then invoked the evaluator. It had no attempt identity, no repair feedback delivery, and no candidate-level relationship between a compile/correctness failure and a later candidate.

The downloaded D-repository copy is incomplete for the older local runner: its `lab/core` package is absent although older long-horizon tests import it. P1 therefore adds a self-contained controller extension and a `LongHorizonRunner` factory hook, rather than reconstructing unrelated missing core modules.

## 2. Architecture changes

- `lab/runtime/agent/attempt_loop.py` adds the controller-owned `HypothesisAttemptController` extension.
- `LongHorizonRunner.build_hypothesis_attempt_controller()` exposes that extension through the existing persistent-session runner; it does not give an Agent evaluator, qualification, or promotion authority.
- `lab/runtime/agent/v100_attempt_adapter.py` adapts a deterministic V100 evaluator callback into compile/correctness/benchmark stages while caching its one measurement result.
- `remote_v100_campaign.py` exports `build_v100_attempt_adapter()` and disables its previous one-turn ephemeral Agent path. Deterministic `--skip-agent` audit/replay paths remain separate.

## 3. Hypothesis vs attempt model

`config/policies/agent_attempts.json` supplies the bounded policy:

- `max_attempts_per_hypothesis: 5`
- `max_compile_repairs: 2`
- `max_correctness_repairs: 2`
- `max_performance_refinements: 2`

Every append-only `campaigns/<operator>/lineage.jsonl` node contains `hypothesis_id`, `attempt_id`, episode, `parent_candidate_hash`, `candidate_hash`, status, failure class, timestamp, result reference, and persistent Agent thread ID. Rejected attempts are retained; a successful attempt points to the prior candidate hash.

## 4. Same-session repair loop

For one hypothesis, the controller runs the deterministic contract gate before starting the session. If it passes, it starts one session and repeatedly calls the supplied Agent step callback with structured feedback. Compile and correctness failures produce another attempt on the same `thread_id`; only explicit `ABANDON_HYPOTHESIS`, budget exhaustion, framework/contract stop, or qualification completion exits the loop. A subsequent hypothesis starts a new session/thread.

## 5. Feedback schemas

The controller returns structured feedback rather than prose-only failure messages:

- `compile_error`: attempt ID, candidate hash, compiler diagnostics.
- `correctness_error`: failed shapes, max error, tolerance, candidate hash.
- `benchmark_result`: per-shape data, aggregate, incumbent, stability.
- `profile_evidence`: registers/thread, spills, occupancy, DRAM bytes, kernel time, and explicit `MEASURED`, `UNAVAILABLE`, or `UNKNOWN` evidence status.

Missing profiler facts remain `null`/`UNAVAILABLE`; they are never converted into measured claims.

## 6. Attempt budgets

The policy is loaded from one JSON file and copied into each `AttemptResult`, so callers can persist the actual policy alongside future result artifacts. Budget counters are separate for compile repairs, correctness repairs, and performance refinements.

## 7. Candidate lineage

`LineageStore` is append-only. It records every rejected, framework-failed, contract-failed, unstable, and accepted attempt without modifying legacy episode files. It is hash-driven and has an explicit parent candidate hash rather than inferring parentage from episode number.

## 8. Knowledge changes

`StructuredKnowledgeSink` persists evidence-bearing records with explicit fields for mechanism, data lifetime, memory passes, vectorization, block/reduction/register/shared strategy, known failure mode, failure class, hashes/IDs, failed shapes, compiler diagnostics, correctness error, benchmark distribution, profile evidence, and evidence level. Unknown values remain explicit `null`/empty values.

`FRAMEWORK_ERROR`, `CONTRACT_ERROR`, and `ENVIRONMENT_ERROR` remain in lineage but are intentionally excluded from strategy knowledge. Thus framework defects such as dispatch/signature mistakes cannot create anti-strategy lessons.

## 9. V100 adapter boundary

`V100AttemptEvaluatorAdapter` accepts supervisor-owned `measure`, `qualify`, and `profile` callables. It validates the P0 semantic contract locally and caches the one existing combined V100 measurement while exposing it as compile/correctness/benchmark stages. This prevents the P1 loop from causing duplicate remote jobs and preserves P0 ownership of contract validation and qualification.

No real V100 call is wired or made in this change. The legacy CLI’s automatic ephemeral-Agent path now fails explicitly with `P1_SESSION_CONTROLLER_REQUIRED` rather than silently retaining the unsafe behavior.

## 10. Tests

P0 regression: `lab.tests.test_p0_integrity` — 11 passing tests.

P1 offline tests: `lab.tests.test_p1_attempt_loop` — 9 passing tests. They cover:

- CE → same-session repair → compile pass;
- WA → same-session repair → correctness pass;
- two CE repairs under one hypothesis;
- budget exhaustion and explicit abandon;
- lineage parentage, accepted and rejected preservation;
- framework errors excluded from strategy knowledge;
- profiler unavailable and non-bypassable qualification failure;
- contract gate before session start and new hypothesis/new thread;
- V100 adapter’s one-measurement cache.

Combined P0/P1 result: `Ran 20 tests ... OK`.

The pre-existing long-horizon test modules `test_execution_pipeline`, `test_diagnostic_actions`, and `test_filesystem_resume` cannot import in this downloaded copy because `lab.core` is absent. They fail at collection with `ModuleNotFoundError: No module named 'lab.core'`, before any P1 behavior runs. This is an archive completeness blocker, not a P1 test regression; it is intentionally not repaired in this focused phase.

## 11. Backward compatibility

Historical episode artifacts are not migrated or rewritten. The P1 lineage is new and append-only. Existing P0 result/contract readers remain unchanged. The V100 CLI retains deterministic `--skip-agent` paths, but autonomous Agent-driven invocation is deliberately stopped until a caller uses the new session controller boundary.

## 12. Remaining risks

- The P1 controller has fake-only tests; a later user-approved integration test must exercise it with a real local agent and evaluator.
- The D-repository’s missing `lab.core` prevents the old `LongHorizonRunner` suite from executing here.
- The exact scientific values of attempt budgets and qualification policy need owner review before a remote campaign uses them.
- The V100 evaluator still has a combined remote compile/correctness/benchmark protocol; its adapter avoids duplicate runs but cannot make remote `eval.sh` compile-only without a later evaluator protocol change.

## 13. Deferred P2

Not implemented: CAKE IR, verifier, cost model, profiler intelligence beyond a typed evidence interface, persistent remote integration, and broader knowledge-policy redesign.

## Production file change map

| File | Function/class | Before | After | Reason |
|---|---|---|---|---|
| `lab/runtime/agent/attempt_loop.py` | `HypothesisAttemptController` | No explicit hypothesis-attempt controller | Bounded same-session attempts, feedback, lineage, knowledge | Separate hypothesis from implementation candidates |
| `lab/runtime/agent/v100_attempt_adapter.py` | `V100AttemptEvaluatorAdapter` | No common V100/controller boundary | Cached deterministic evaluator adapter | Keep remote authority outside Agent |
| `lab/runtime/agent/long_horizon.py` | `build_hypothesis_attempt_controller` | Existing runner could not construct P1 loop | Reuses its session and campaign paths | Integrate rather than replace long horizon |
| `lab/runtime/evaluators/remote_v100_campaign.py` | `build_v100_attempt_adapter`, CLI branch | Ephemeral one-turn Agent lifecycle | P1 adapter boundary; unsafe autonomous path disabled | Prevent feedback discontinuity |
| `config/policies/agent_attempts.json` | policy | Limits scattered/implicit | Versioned, recorded limits | Bound runaway repairs |
| `lab/tests/test_p1_attempt_loop.py` | fake-only tests | No P1 coverage | Nine offline behavior tests | Protect P1 invariants |

## Required safety outcome

```text
D_REPO_ONLY = PASS
C_MOTHER_REPO_MODIFIED = NO
REMOTE_V100_USED = NO
CUDA_BENCHMARK_USED = NO
HISTORICAL_CANDIDATES_MODIFIED = NO
P0_REGRESSION = PASS
P1_OFFLINE_TESTS = PASS
```
