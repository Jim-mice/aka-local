# Architecture Audit: aka-local

**Date**: 2026-09-18
**Scope**: Read-only, code-level audit of `<PROJECT_ROOT>`
**Method**: Traced every module from entry points through Agent, evaluator, persistence, and knowledge. Did not guess from README. All conclusions cite file paths, class names, and function names as evidence.

---

## A. What This System Actually Is

aka-local is a **local GPU kernel optimization workbench with an event-sourced controller, a deterministic supervisor, and a bounded single-turn Codex Agent session runner**, wrapped in a Tkinter GUI and a PowerShell CLI.

It currently operates on exactly one operator (`rms_norm_train`), one GPU platform (`rtx5060_laptop_sm120`), one backend (`cuda_cpp`), and one campaign (`rms_norm_train__rtx5060_sm120__cuda_cpp`). The codebase contains generic abstractions (operators, platforms, backends, campaigns, knowledge cards, frontier directions) that point toward a multi-operator future, but only RMSNorm has real evaluator, knowledge, and episode data.

The system is **not** a fully autonomous kernel optimization agent. It is a **human-in-the-loop experiment manager** where:
- The human selects workspace configuration and issues directives.
- The Agent proposes plans (diagnostic or optimization) and implements code edits.
- A deterministic supervisor (`controller_policy.py:decide`) runs ABBA benchmarks and makes the final PROMOTE/REJECT decision.
- The Agent **never** self-declares success.

It is best described as: **experiment manager + event-sourced controller + deterministic supervisor + bounded Agent wrapper + static knowledge browser**. It is closest to a local, single-GPU subset of the Atrex orchestrator, ported from Linux/BIV150 to Windows/RTX5060.

---

## B. Current Architecture Diagram

```text
                           User
                            |
          +-----------------+-----------------+
          |                                   |
     lab.ps1 gui                         lab.ps1 ui
     (Tkinter App)                    (HTTP localhost:8765)
          |                                   |
          +--------- LabController -----------+
          |         (lab/core/controller.py)  |
          |                                   |
    +-----+--------+-------------+------------+-----+
    |              |             |                  |
[LIVE]          [LIVE]       [LIVE]            [LIVE]
Workbench     EventStore    Probe           Credential
(lab/core/    (lab/core/    (lab/core/       Store
workbench.py) events.py)    probe.py)        (.py)
    |              |             |
    |         events.jsonl    nvidia-smi
    |         (append-only     nvcc --version
    |          hash-chained)   torch.cuda.*
    |
    +---- LongHorizonRunner ----+
    |    (lab/runtime/agent/    |
    |     long_horizon.py)      |
    |                           |
    +-- CodexAgentSession ------+  [LIVE]
    |   (lab/runtime/agent/     |
    |    codex_session.py)      |
    |                           |
    +-- RTX5060LocalEvaluator --+  [LIVE]
    |   (lab/runtime/evaluators/|
    |    local.py)              |
    |                           |
    +-- controller_policy ------+  [LIVE]
    |   (lab/runtime/supervisor/|
    |    controller_policy.py)  |
    |                           |
    +-- CandidatePathPolicy ----+  [LIVE]
        (lab/runtime/
         path_policy.py)

  Knowledge System              Persistence           Campaign State
  [PARTIAL - static JSON]       [LIVE]                [LIVE]

  lab/knowledge/                lab/core/              lab/campaigns/
  - observations/               persistence.py         rms_norm_train__.../
  - anti_strategies/            - atomic_json()        - campaign.json
  - backend_quirks/             - read_json()          - frontier.json
  - external_references/        - sha256_json()        - episodes/e0001..6/
  - hypotheses/                                        - checkpoints/
  - open_questions/              journal.py            - memory/
  - supported_rules/             - append_experiment() - workspace/
                                - read_experiments()
  lab/core/                     - sync_live()         Run Store
  external_retrieval.py                                [LIVE]
  - retrieve_knowledge()         events.py             lab/runtime/runs/
                                - EventStore
                                - append/replay/ids    run_state.py
  lab/index.json                                        - RunStore
  (aggregated experiment                                - save()
   index)                                               - recoverable()

  lab/knowledge_sources/         [LIVE - git clones]
  - gpu_mode_resource_stream/
  - modal_gpu_glossary/

  Agent Backend                   External Tools         Continuous Runner
  [LIVE]                          [LIVE]                 [PARTIAL - legacy]
  agent_backends/                 benchmarks/            continuous_runner.py
  codex_appserver_agent.py        rmsnorm_abba.py        (separate state
  codex_appserver.py              rmsnorm_repeated.py     machine, different
  run_rmsnorm_episode.py          swiglu_abba.py          campaign root)
                                  swiglu_benchmark.py
  lab/runtime/agent/
  codex_session.py                profiles/
                                  swiglu_basic.ncu-rep
  Atrex Reference [reference]     swiglu_kernel_basic.
  atrex-kernel-agent-win/          ncu-rep
  - orchestrator/
  - long_horizon/
  - gpu-wiki/
  - skills/
```

### Flow Summary

**Control Flow**: User → GUI/CLI → LabController → LongHorizonRunner.run() → while loop: plan → (diagnostic | edit+compile+correctness+benchmark) → agent_interpretation → journal → gate (if ready_for_gate) → supervisor → promote/reject

**Data Flow**: campaign.json (config) → context_builder.build_authoritative_context() → CodexAgentSession.run_experiment_turn() → _reply() parses JSON → evaluator methods → evidence_dir/*.json → supervisor policy → canonical memory vNNN.json

**Event Flow**: Every state transition → EventStore.append() → lab/runtime/events.jsonl (hash-chained, append-only, secret-redacted). GUI polls events via LabController.status() → EventStore.replay().

---

## C. From GUI Click to Agent Behavior

### Path: Agent Console ASK

```
GUI: AgentConsole.ask()                         [lab/gui.py]
  → threading.Thread → ctrl.ask_advisory()       [lab/gui.py]
  → LabController.ask_advisory()                 [lab/core/controller.py]
    → builds advisory_agent_session(workspace)   [lab/gui.py:advisory_agent_session]
    → CodexAgentSession.ask_advisory()           [lab/runtime/agent/codex_session.py]
      → constructs prompt: "read-only advisory assistant" + context snapshot
      → self.thread.run(prompt, model="gpt-5.6-luna", effort="low",
                         sandbox=Sandbox.read_only, approval_mode=ApprovalMode.deny_all)
      → returns TurnResult
  → GUI renders answer in conversation panel
```

### Path: START OPTIMIZATION (full episode)

```
GUI: App.start()                                  [lab/gui.py]
  → LabController.start_optimization()            [lab/core/controller.py]
    → threading.Thread
    → LongHorizonRunner(campaign_dir, candidate_root, incumbent, session,
                        evaluator, events, ...)   [lab/runtime/agent/long_horizon.py]
    → runner.run()

  Inside run() loop:
    1. _knowledge(n)      → reads lab/knowledge/**/*.json, emits KNOWLEDGE_REFRESHED
    2. _context(n, ...)   → build_authoritative_context() from filesystem
    3. _plan(context)     → Agent: structured JSON plan (DIAGNOSTIC|OPTIMIZATION)
    4. If DIAGNOSTIC:
       _run_diagnostics() → evaluator.run_diagnostic_action() → subprocess
       _interpret_diagnostic() → Agent: classify results
    5. If OPTIMIZATION:
       _edit(plan)        → Agent: implement code in candidate_root
       evaluator.compile()  → subprocess (python correctness worker)
       evaluator.check_correctness() → subprocess (56 shapes)
       evaluator.development_benchmark() → subprocess (ABBA, warmup=2, repeats=5)
       Agent interprets → decision_request
    6. If "ready_for_gate":
       _gate()            → evaluator.authoritative_abba() (warmup=20, repeats=100)
                         → controller_policy.decide() [deterministic]
                         → if PROMOTE: shutil.copytree to incumbent dir
                         → write_canonical memory/vNNN.json
    7. append_experiment(journal.jsonl)
    8. If budget exhausted or blocked → _finish()
```

**BREAKPOINT**: There is NO automatic loop restart. After `_finish()`, the system returns to GUI overview. A new episode requires the human to press START or RESUME again. The `continuous_runner.py` has a separate state machine but uses a different campaign root and is disconnected from the main GUI/controller path.

---

## D. Where Is the Agent?

### 1. The actual reasoning/coding Agent

**Codex SDK** via `openai_codex.Codex` client, using model `gpt-5.6-luna` with reasoning effort `low`. The process is the Codex app-server binary at:
`<LOCAL_USER_HOME>\AppData\Roaming\npm\node_modules\@openai\codex\node_modules\@openai\codex-win32-x64\vendor\x86_64-pc-windows-msvc\bin\codex.exe`

Evidence: `lab/runtime/agent/codex_session.py:13-14`, `agent_backends/codex_appserver_agent.py:39-40`

### 2. CodexAgentSession creation

Created in `LongHorizonRunner.run()` via the session passed from `LabController.start_optimization()`. The session is instantiated as `CodexAgentSession(workspace, process_registry)`.

Evidence: `lab/runtime/agent/codex_session.py:15-25`

### 3. Agent session lifecycle

```
CodexAgentSession.__init__()
→ .start()      → Codex(CodexConfig(...)) → client.thread_start()
→ .send_context() → validates context_hash
→ .run_experiment_turn() → thread.run(prompt, ...) [may be called multiple times]
→ .close()      → client.close()
```

The session is ephemeral (`ephemeral=True`). Each new episode creates a new session.

### 4. Agent tools/commands

The Agent runs with `sandbox=Sandbox.workspace_write` (optimization turns) or `Sandbox.read_only` (ASK turns), with `approval_mode=ApprovalMode.deny_all`. This means the Agent's Codex SDK manages the sandbox; the Agent can use whatever tools Codex provides (file read/write, shell commands, etc.), constrained by the workspace sandbox and the prompt instructions.

Evidence: `lab/runtime/agent/codex_session.py:56-59, 63-65`

### 5. Agent capabilities (real vs instructed)

| Capability | Status | Evidence |
|---|---|---|
| Modify code | LIVE - Agent edits candidate files | `_edit()` at `long_horizon.py:227` |
| Build | INSTRUCTED NOT TO - compile runs via evaluator subprocess AFTER agent returns | `long_horizon.py:run()` |
| Run | INSTRUCTED NOT TO - benchmark runs via evaluator subprocess AFTER agent returns | same |
| Benchmark | INSTRUCTED NOT TO | same |
| Profile | INSTRUCTED NOT TO - NCU runs via evaluator after DIAGNOSTIC plan | `_run_diagnostics()` |
| Read profiler | PARTIAL - Agent receives structured results via context, not raw NCU | `_interpret_diagnostic()` |
| Commit | PARTIAL - `_commit()` runs git after gate, but Agent does not invoke it | `long_horizon.py:252` |
| Rollback | MISSING - no automated rollback; incumbent is `shutil.copytree` on promote | `_gate()` |

### 6. Human-required actions

- Select workspace / campaign configuration
- Issue directives
- Press START / RESUME / PAUSE / STOP
- Resolve WAITING_FOR_HUMAN states
- Interpret results and decide whether to start a new episode
- NO automatic multi-episode loop

### 7. Agent loop: CURRENT (real)

```
  +--------------------------------------------------+
  |  Human presses START                              |
  |        |                                          |
  |        v                                          |
  |  [AGENT] _plan() -> structured JSON plan           |
  |        |                                          |
  |        +-- DIAGNOSTIC branch                      |
  |        |   +-- _run_diagnostics() -> evaluator     |
  |        |   +-- _interpret_diagnostic() -> Agent    |
  |        |                                          |
  |        +-- OPTIMIZATION branch                    |
  |            +-- _edit() -> Agent modifies code      |
  |            +-- evaluator.compile()                |
  |            +-- evaluator.check_correctness()      |
  |            +-- evaluator.development_benchmark()  |
  |            +-- Agent interprets -> decision        |
  |        |                                          |
  |        v                                          |
  |  If READY_FOR_GATE:                               |
  |    +-- evaluator.authoritative_abba()             |
  |    +-- controller_policy.decide() [deterministic] |
  |    +-- PROMOTE or REJECT                          |
  |        |                                          |
  |        v                                          |
  |  append_experiment(journal.jsonl)                 |
  |        |                                          |
  |        v                                          |
  |  Loop back to _plan() for next experiment         |
  |  (within same episode, up to budget)              |
  |        |                                          |
  |        v                                          |
  |  _finish() -> returns to GUI                       |
  |        |                                          |
  |  BREAK: Human must press START/RESUME again       |
  +--------------------------------------------------+
```

---

## E. Supervisor / Controller Authority

### Who has final power?

**The deterministic supervisor code has final power.** The Agent's opinions are explicitly advisory.

Evidence chain:

1. **Agent cannot declare correctness PASS**: `controller_policy.py:decide()` checks `evaluation["compile"]["pass"]` and `evaluation["correctness"]["pass"]` from mechanical evaluator output, NOT from Agent claims.

2. **Agent cannot declare candidate faster**: `controller_policy.py:decide()` reads `authoritative_abba["metrics"]["arithmetic_mean_speedup"]` from the evaluator's benchmark output. If speedup is `None`, returns `INCONCLUSIVE`.

3. **KEEP/ROLLBACK is deterministic**: `decide()` returns `PROMOTE`, `REJECT_COMPILE`, `REJECT_CORRECTNESS`, `REJECT_PERFORMANCE`, `ROBUSTNESS_REQUIRED`, or `INCONCLUSIVE`. `finalize_after_robustness()` returns `PROMOTE` or `REJECT_ROBUSTNESS`. These are pure Python functions with no LLM calls.

Evidence: `lab/runtime/supervisor/controller_policy.py:1-29`

4. **Benchmark data source**: Subprocess calls to `benchmarks/rmsnorm_abba.py` and `benchmarks/rmsnorm_repeated.py`. Raw JSON output written to `evidence_dir/*.json`.

Evidence: `lab/runtime/evaluators/local.py:85-107`

5. **Candidate promotion**: `LongHorizonRunner._gate()` performs `shutil.copytree(self.candidate_root, target)` only when `action == "PROMOTE"`.

Evidence: `long_horizon.py:_gate()`

6. **Stop/Pause control**: `LongHorizonRunner` has `stop_requested` and `pause_requested` flags checked at the top of each loop iteration. The GUI buttons call `LabController.pause_optimization()`, `stop_optimization()`, and `emergency_stop()`.

Evidence: `long_horizon.py:run()` checks, `controller.py` methods

7. **Crash recovery**: `RunStore.recoverable()` scans `lab/runtime/runs/*/resume_manifest.json` for `PAUSED`, `INTERRUPTED`, or `RUNNING` states. Recovery requires environment fingerprint match. `LabController.resume_optimization()` rebuilds the filesystem picture and starts a new Codex session.

Evidence: `lab/runtime/run_state.py:15-17`, `lab/core/controller.py:filesystem_recovery_plan()`

### Dangerous path check: LLM says success -> system trusts it

**This path does NOT exist.** The Agent outputs a `decision_request` field in JSON (values: `keep_as_best`, `reject_and_continue`, `ready_for_gate`, `blocked`). These are advisory only:
- `ready_for_gate` -> triggers the deterministic `_gate()` which runs authoritative ABBA and `decide()`
- `keep_as_best` / `reject_and_continue` -> recorded as agent interpretation, loop continues
- `blocked` -> returns WAITING_FOR_HUMAN

The Agent CANNOT output `PROMOTE` or `ACCEPT`. The `decide()` function ignores agent opinion entirely.

Evidence: `long_horizon.py:_plan()` requires `decision_request` values, `controller_policy.py:decide()` uses only mechanical evidence.

---

## F. Experiment / Candidate Model

### How the project expresses campaign concepts:

| Concept | Expression | Location |
|---|---|---|
| Campaign | `campaign.json` with operator x platform x backend x objective x workload | `lab/campaigns/*/campaign.json` |
| Episode | `eNNNN` directory under campaign/episodes/ | `lab/campaigns/*/episodes/e0001..6/` |
| Experiment | One plan->edit->evaluate->interpret cycle; a journal.jsonl entry | `episodes/*/journal.jsonl` |
| Hypothesis | `plan.hypothesis` dict: `{question, claim, rationale, expected_effect, support_condition, refute_condition}` | Agent JSON output, stored in journal |
| Candidate | `episodes/*/candidate/` directory; file tree modified by Agent | Filesystem |
| Incumbent | `ops/rms_norm_v2/` (read-only, manually designated) | `campaign.json:incumbent.source` |
| Baseline | Not a formal type; incumbent serves as baseline | Implicit |
| Measurement | `development_benchmark`, `authoritative_abba`, `robustness`, `correctness` JSON outputs | `episodes/*/evidence/` |
| Profile | NCU CSV output, parsed via `ncu_profile.py:parsed_profile()` | `episodes/*/evidence/profiler/` |
| Decision | `supervisor_decision` in canonical memory: PROMOTE, REJECT_*, INCONCLUSIVE | `memory/vNNN.json` |

Evidence: `lab/core/models.py` (Experiment, Handoff dataclasses), `campaign.json`, `long_horizon.py:_gate()`, `memory.py:write_canonical()`

### RMSNorm campaign trace:

- **Incumbent source**: `ops/rms_norm_v2/candidate.py` — manually placed, NOT generated by the system
- **Candidate location**: `campaigns/rms_norm_train__rtx5060_sm120__cuda_cpp/episodes/eNNNN/candidate/`
- **Candidate ID**: `e0004-x001` format (episode-experiment)
- **Parent lineage**: `candidate_lineage.jsonl` in each episode dir, tracking `parent_candidate_id`, `source_hash`, `changed_files`
- **Metrics**: Stored in `episodes/*/evidence/{compile,correctness,development_abba,authoritative_abba,robustness}_*.json`
- **ABBA implementation**: `benchmarks/rmsnorm_abba.py` (called via subprocess) — REAL, LIVE
- **Frontier**: `campaigns/*/frontier.json` — a file with `directions: []` — SKELETON (empty array, never populated by the runner)
- **KEEP/ROLLBACK**: REAL via `_gate()` -> `decide()` -> `shutil.copytree` for PROMOTE

---

## G. Event Sourcing / Persistence

### Event Store

**File**: `lab/runtime/events.jsonl`

**Schema** (per event record):
```json
{
  "event_id": "UUID4",
  "sequence": int (monotonic),
  "timestamp": "ISO8601 UTC",
  "type": "EVENT_TYPE",
  "workbench_id": "local_rtx5060",
  "payload": {...},
  "previous_event_hash": "SHA256 of previous record",
  "event_hash": "SHA256 of this record"
}
```

Evidence: `lab/core/events.py:EventStore.append()`

- **Event ID**: UUID4, idempotent (re-append with same ID returns existing)
- **Sequence**: Monotonic integer from `len(existing)+1`
- **Previous hash**: SHA256 of the last line's JSON
- **Event hash**: SHA256 of canonical JSON of the full record
- **Replay**: `EventStore.replay()` reads all lines, returns list sorted by insertion order
- **Dedup**: By `event_id` (checked before append)
- **Secret redaction**: `redact_secrets()` replaces values for keys in `SENSITIVE` set with `[REDACTED]`

Evidence: `lab/core/events.py:1-34`

### Recovery without GUI

If the GUI is deleted, can the system recover from events + checkpoint?

**PARTIAL**. The event store is hash-chained and append-only, providing an audit trail. `RunStore` writes checkpoint JSON. `journal.jsonl` contains immutable experiment records. `live.json` tracks episode state. However:

- The actual candidate code lives in the filesystem, not events
- Recovery requires the original source directories to still exist
- The `LabController.filesystem_recovery_plan()` reads filesystem state, not just events

### Persistence mechanisms

| Mechanism | Used for | Implementation |
|---|---|---|
| `atomic_json()` | All JSON writes except journal/events | tempfile + os.fsync + os.replace |
| `append_experiment()` | journal.jsonl | `open("a")` + `write()` (NOT atomic, NOT fsync) |
| `EventStore.append()` | events.jsonl | `open("a")` + `write()` + `flush()` (NOT atomic, NOT fsync) |
| `sync_live()` | live.json | Uses `atomic_json()` (atomic) |
| `write_canonical()` | memory/vNNN.json | `write_text()` (NOT atomic, NOT fsync) |

Evidence: `lab/core/persistence.py`, `lab/core/journal.py`, `lab/core/events.py`, `lab/core/memory.py`

**Note**: `journal.jsonl` and `events.jsonl` use simple append writes without `fsync` or atomic replace, meaning a crash during append could corrupt the last line. `live.json` and checkpoint state use proper `atomic_json`.

---

## H. Knowledge System

### What knowledge actually is

**Static JSON files** organized by category under `lab/knowledge/`:
- `observations/` — local empirical findings (e.g., `rmsnorm-v2.json`)
- `anti_strategies/` — approaches proven NOT to work (e.g., `swiglu-v3.json`)
- `backend_quirks/` — platform-specific gotchas
- `external_references/` — imported from CUTLASS docs, GPU Mode lectures, GPU glossary
- `hypotheses/` — untested hypotheses
- `open_questions/` — research questions
- `supported_rules/` — confirmed optimization rules

Plus `lab/index.json` — an aggregated experiment index.

Evidence: `lab/knowledge/` directory structure, `lab/index.json`

### Knowledge flow

**Refresh**: `LabController.refresh_knowledge(query)` emits a `KNOWLEDGE_REFRESHED` event but does NOT actually re-read knowledge files. The actual loading happens in:
- `LongHorizonRunner._knowledge(n)` — reads `lab/knowledge/**/*.json`, filters by operator/platform/backend
- `context_builder.py:retrieve_knowledge()` — deterministic ranking by scope match, architecture match, authority score

**Into Agent context**: Knowledge goes into the authoritative context snapshot passed to the Agent as part of the plan/edit prompt.

**Experiment -> knowledge**: After a gate, `write_knowledge_candidates()` writes to `lab/knowledge/wiki_candidates/candidate_knowledge.jsonl`. The canonical memory `vNNN.json` contains `knowledge_candidates` paths.

Evidence: `lab/runtime/agent/episode_artifacts.py:write_knowledge_candidates()`

### Experiment -> structured lesson -> knowledge -> next Agent retrieval: CURRENT STATUS

```
Experiment completes -> canonical memory written -> knowledge_candidates appended
                                                    |
                                                    v
                                              [BREAK: No automatic
                                               promotion to knowledge cards.
                                               Human must review and
                                               manually create JSON files.]
```

The system writes candidate knowledge but does NOT automatically create new knowledge cards. There is no automated "lesson learned" pipeline. The knowledge retrieval (`retrieve_knowledge()`) reads static JSON files, not the output of previous experiments directly.

---

## I. Profiler / Evaluator

### Auto-evaluation pipeline: status per step

| Step | Status | Evidence |
|---|---|---|
| candidate -> build (compile) | LIVE | `RTX5060LocalEvaluator.compile()` -> subprocess runs correctness worker which triggers JIT compile |
| build -> correctness | LIVE | `rmsnorm_correctness.py:main()` — 56-shape allclose check, rtol=1e-2, atol=2e-2 |
| correctness -> benchmark (dev) | LIVE | `development_benchmark()` — subprocess `rmsnorm_abba.py`, warmup=2, repeats=5 |
| benchmark -> authoritative ABBA | LIVE | `authoritative_abba()` — warmup=20, repeats=100 |
| ABBA -> NCU profiler | LIVE (DIAGNOSTIC only) | `profile()` in `local.py` — calls `ncu.exe` with `--csv --page raw --metrics ...` |
| NCU -> parse metrics | LIVE | `ncu_profile.py:parse_ncu_csv()` + `parsed_profile()` |
| parse -> Agent diagnosis | LIVE | `_interpret_diagnostic()` sends parsed results to Agent |
| robustness (repeated ABBA) | LIVE | `robustness()` — 5 batches, warmup=20, repeats=100 |

### Detailed findings:

- **NCU**: Actually called via `subprocess.run([ncu_path, ...])`. The path is `C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.3.0\...\ncu.exe`. REAL.
- **Output parser**: `parse_ncu_csv()` handles two NCU CSV layout formats. BASIC_DIAGNOSTIC_METRICS = `gpu__time_duration.avg`, `dram__throughput.avg.pct_of_peak_sustained_elapsed`, `sm__warps_active.avg.pct_of_peak_sustained_active`. REAL.
- **Benchmark raw samples**: Development and authoritative ABBA output saved as JSON in `evidence/` directory. Per-shape repeated benchmark writes CSV. REAL.
- **Authoritative evaluator**: `controller_policy.decide()` is the single deterministic evaluator for promotion decisions. REAL.
- **ABBA**: Real same-process A/B/B/A via `benchmarks/rmsnorm_abba.py`. Not a schema placeholder. REAL.

---

## J. GUI Real Role

### GUI panels and their real function

| Panel | Displays | Connected to backend? |
|---|---|---|
| Overview (main) | Workbench state, episode count, candidate path | YES — `LabController.status()`, `reconcile_episode_state()` |
| Workspace Manager | Hardware probe, project selection, operator/platform/backend dropdowns | YES — writes to `workspace_draft.json`, calls `probe_local()` |
| Agent Console | Conversation with Agent (ASK), event stream, context status | YES — `CodexAgentSession.ask_advisory()` on worker thread |
| Knowledge Browser | Knowledge cards by operator->platform, search, filter | YES — reads `lab/knowledge/`, `lab/experiments/`, `lab/index.json` |
| Experiment Report | Per-experiment markdown reports | YES — reads `experiment_archive/` |
| START/RESUME/PAUSE/STOP | Episode lifecycle control | YES — `LabController.start_optimization()`, etc. |
| EMERGENCY STOP | Kill process tree | YES — `LabController.emergency_stop()` |
| HARVEST RESULTS | Read canonical memory | YES — reads `memory/v*.json` |

### GUI classification

**CONTROL PLANE + OBSERVABILITY + DEBUG TOOL**

The GUI is the primary human interface for everything: configuration, starting episodes, monitoring progress, browsing knowledge, and reviewing results. Without the GUI, the system would function only through the `lab.ps1` CLI, which currently only supports `status`, `knowledge`, `validate`, and `ui` (a minimal web server at `localhost:8765`).

The GUI is NOT a "demo UI" — it controls real Agent sessions, real evaluator subprocesses, and real GPU benchmarks. But it IS the only full-featured interface.

---

## K. Current System Maturity

### REAL (working code, exercised)

- `LabController` — configuration, probe, event emission, recovery planning
- `EventStore` — append-only hash-chained event log with dedup and secret redaction
- `Workbench` — identity, state machine, environment fingerprinting
- `RTX5060LocalEvaluator` — compile, correctness (56-shape), dev ABBA, authoritative ABBA, robustness, NCU profiling, static evidence, repeated per-shape benchmark, regime comparison
- `LongHorizonRunner` — full episode loop: plan->diagnostic/optimize->evaluate->interpret->gate->memory
- `controller_policy.decide()` — deterministic supervisor
- `CandidatePathPolicy` — filesystem sandbox enforcement
- `CodexAgentSession` — persistent Codex session wrapper
- `context_builder.build_authoritative_context()` — filesystem-derived context
- `persistence.atomic_json()` — crash-tolerant writes
- `run_state.RunStore` — checkpoint/resume manifests
- `journal` — append-only experiment records
- `memory.write_canonical()` — versioned canonical memory
- `episode_artifacts` — archive generation, knowledge candidate export
- `knowledge` system — static JSON cards with retrieval ranking
- GUI — all panels connected to real backend

### SKELETON (interface exists, core logic incomplete)

- `frontier.py`/`frontier.json` — data structure exists, `directions` array is empty, never populated by runner
- `continuous_runner.py` — separate state machine, different campaign root, disconnected from main controller
- `lab.ps1 optimize/resume/pause/stop` — all return hardcoded "no active session" messages
- `lab/runtime/executors/remote.py` — exists but not used (local-only)
- `lab/core/handoff.py` — module exists but minimal
- `lab/runtime/supervisor/promotion_policy.yaml` — file exists, controller_policy.py hardcodes thresholds
- `lab/registry/` — YAML files for operators/platforms/backends exist but used mainly for label lookup
- Multi-operator support — data structures generic, but only `rms_norm_train` has evaluator, knowledge, episodes

### MISSING (needed for autonomous kernel optimization)

- **Autonomous multi-episode loop** — no automatic "start next episode"
- **Structured task contracts** — no formal hypothesis->candidate->measurement contract
- **Knowledge auto-promotion** — experiment results do not automatically become knowledge cards
- **Profile-driven iteration** — Agent receives parsed profile, but cannot request follow-up profile in same episode
- **Canonical measurement suite** — ABBA scripts exist but are not formally registered; evaluator capabilities are hardcoded
- **Git worktree isolation** — candidate edits happen in-place; no copy-on-write worktree
- **Remote sandbox** — `executors/remote.py` exists but is not wired
- **Cost/budget model** — no token cost tracking, no GPU-hour tracking
- **Cross-campaign learning** — knowledge is scoped but no cross-operator transfer

---

## L. Relationship to Atrex / KDA / CAKE

| aka-local module | Atrex capability | KDA concept | CAKE concept | Current gap |
|---|---|---|---|---|
| `LabController` | `orchestrator/campaign.py` | Task contract issuer | Orchestrator | No production mode, no multi-campaign |
| `LongHorizonRunner` | `long_horizon/` engine | Episode state machine | Agent loop | No worktree isolation, no tiered episodes |
| `controller_policy.decide()` | `optimization_policy.py` | Supervisor authority | Verifier | Hardcoded thresholds, no policy agent review |
| `CandidatePathPolicy` | `tools/sandbox.py` | Sandbox boundary | Sandbox | No remote sandbox, local-only |
| `RTX5060LocalEvaluator` | Evaluator adapters | Canonical measurements | Structured diagnostics | No evaluator integrity guard (harness can be edited) |
| `CodexAgentSession` | `session_io.py` | Agent runtime | Agent interface | No multi-model fallback, no reviewer agent |
| `RunStore` / `EventStore` | `workspace_state.py` / telemetry | Checkpoint/recovery | Recovery | No Git-based recovery, filesystem-only |
| `knowledge/` system | `gpu-wiki/` | KernelWiki / NCU skill | Knowledge -> machine rule | Static JSON, no vector retrieval, no auto-update |
| `context_builder` | `prompts/` | Plan -> evidence | Structured IR | No iterative context refinement |
| `external_retrieval` | `gpu-wiki/tools/` | Query bridge | Cost model | Ranking is deterministic keyword, no embedding |
| `frontier.json` | Campaign directions | Frontier tracking | Optimization search | Empty directions array, never populated |
| `memory/vNNN.json` | `memory_manager.py` | Candidate evidence | Knowledge base | No aggregation across campaigns |
| `benchmarks/rmsnorm_abba.py` | Evaluator ABBA | Profile-driven iteration | Benchmark verifier | Same-process measurement, no isolated sandbox |

---

## M. Current Real Loop and Minimum Target Loop

### CURRENT REAL LOOP

```
Human -> GUI -> LabController.start_optimization()
  -> LongHorizonRunner.run()
    -> [while budget remaining]
      -> Agent._plan() -> JSON plan
      -> DIAGNOSTIC: evaluator.run_diagnostic_action() -> Agent interprets
         OPTIMIZATION: Agent._edit() -> evaluator.compile/correctness/benchmark -> Agent interprets
      -> If READY_FOR_GATE:
          evaluator.authoritative_abba() -> controller_policy.decide() -> PROMOTE/REJECT
      -> journal.jsonl record
      -> Loop back (if not terminal)
    -> _finish()
-> BREAK: Human must press START for next episode
-> Knowledge: manual review of vNNN.json, manual card creation
```

### MINIMUM TARGET LOOP (using existing modules)

```
+-------------------------------------------------------------+
|  Task Contract (new, small):                                |
|  operator + objective + budget + measurement suite          |
|      |                                                      |
|      v                                                      |
|  LongHorizonRunner.run() [EXISTING, EXTENDED]               |
|      |                                                      |
|      +-- Agent._plan() [EXISTING]                           |
|      |     ^ enriched with structured knowledge retrieval   |
|      |                                                      |
|      +-- DIAGNOSTIC branch [EXISTING]                       |
|      |   +-- Profile -> parse -> Agent diagnosis [EXISTING] |
|      |                                                      |
|      +-- OPTIMIZATION branch [EXISTING]                     |
|      |   +-- Edit -> Compile -> Correctness -> Benchmark    |
|      |                                                      |
|      +-- Gate [EXISTING]                                    |
|      |   +-- Authoritative ABBA -> supervisor.decide()      |
|      |                                                      |
|      +-- PROMOTE -> promote candidate [EXISTING]            |
|      |                                                      |
|      +-- write_canonical [EXISTING]                         |
|      |                                                      |
|      +-- auto-promote knowledge card [NEW, SMALL]           |
|      |   +-- structured lesson -> knowledge/*.json          |
|      |                                                      |
|      +-- update frontier directions [NEW, SMALL]            |
|      |                                                      |
|      +-- auto-continue [NEW, SMALL]                         |
|          +-- if budget remains, start next experiment       |
|              with enriched knowledge context                |
+-------------------------------------------------------------+
```

This adds only 3 small new capabilities to the existing system:
1. Auto-continue loop (remove the human-press-START breakpoint between experiments)
2. Auto-promote knowledge cards from canonical memory
3. Populate frontier directions from experiment outcomes

---

## N. Final Output

### 1. Current architecture

Event-sourced local GPU kernel optimization workbench: Tkinter GUI -> LabController -> LongHorizonRunner -> CodexAgentSession (GPT-5.6 Luna) + RTX5060LocalEvaluator + deterministic supervisor. Static JSON knowledge base. Single-operator (RMSNorm), single-GPU (RTX 5060 sm_120), local-only.

### 2. Current real execution loop

Human-started bounded episode: Plan -> (Diagnostic|Optimize) -> Evaluate -> Agent interprets -> Gate (if ready) -> Supervisor decides -> Memory written -> Loop within episode budget -> Return to GUI.

### 3. Biggest architectural gap

**No autonomous multi-episode loop.** Each episode requires human START. There is no automated "observe result -> formulate next hypothesis -> start next experiment." The system is a powerful experiment manager but not an optimization agent.

### 4. Top 5 modules worth keeping

1. `LongHorizonRunner` — well-structured episode state machine with clean diagnostic/optimization dispatch
2. `controller_policy.decide()` — deterministic supervisor is architecturally correct and safety-critical
3. `RTX5060LocalEvaluator` — comprehensive compile/correctness/ABBA/NCU/robustness surface
4. `EventStore` — hash-chained append-only event log with dedup and redaction
5. `context_builder.build_authoritative_context()` — single source of truth for Agent context

### 5. Top 5 missing capabilities

1. Autonomous multi-episode continuation (remove human START breakpoint)
2. Structured knowledge auto-promotion from experiment results
3. Git worktree isolation for candidate edits
4. Frontier direction tracking and population
5. Cross-campaign knowledge transfer

### 6. What NOT to rewrite

- The deterministic supervisor (`controller_policy.py`)
- The evaluator (`local.py`, `rmsnorm_correctness.py`, `ncu_profile.py`)
- The event store (`events.py`)
- The persistence layer (`persistence.py`, `journal.py`)
- The path policy (`path_policy.py`)

### 7. UI illusions (things that appear to work but are stubs)

- `lab.ps1 optimize` — prints "Planning only: no Agent/GPU execution is enabled in this scaffold."
- `lab.ps1 resume/pause/stop` — all print "No active session."
- `frontier.json` `directions: []` — the frontier system has structure but no content
- `continuous_runner.py` — separate legacy state machine, disconnected from main controller
- Multi-operator dropdowns in GUI — only `rms_norm_train` has real backend

### 8. Minimum next-stage development boundary

**Keep**: All existing modules. **Add only**:
1. An `auto_continue` flag in `LongHorizonRunner` that, when enabled, loops episodes without returning to GUI
2. A `knowledge_card_from_memory()` function that creates observation cards from canonical memory
3. A `frontier_update_from_episode()` that populates `frontier.json` directions from experiment outcomes
4. Episode-to-episode knowledge context enrichment (already partially wired via `retrieve_knowledge()`)

These 4 additions, built on the existing architecture, would close the loop from "human-operated experiment manager" to "supervised autonomous optimization system."
