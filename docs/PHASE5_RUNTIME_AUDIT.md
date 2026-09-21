# aka-local V2 Phase 5-A: Runtime Reliability Audit

## 1. Current Runtime Architecture

### 1.1 Experiment Lifecycle Call Chain

`
LabController.start_optimization()
  |
  +-- validates: candidate_root exists, incumbent exists, probe READY
  +-- creates: RunStore, ProcessRegistry, RTX5060LocalEvaluator
  +-- creates: CodexAgentSession(candidate_root)  <-- workspace = candidate_root
  +-- creates: LongHorizonRunner(...)
  +-- spawns: daemon thread -> LongHorizonRunner.run()
  |
LongHorizonRunner.run()
  |
  +-- evaluator.prepare()           # build/compile environment setup
  +-- probe_local()                 # GPU availability check
  +-- session.start()               # Agent session starts
  |
  +-- while True:                   # experiment loop
  |    |
  |    +-- _knowledge(n)            # loads external knowledge
  |    +-- _context(n, knowledge)   # builds authoritative context
  |    +-- _load_validated_plan()   # recovery: load persisted plan
  |    +-- _plan(context)           # ask Agent for plan
  |    +-- _persist_validated_plan()# save plan to disk
  |    |
  |    +-- if DIAGNOSTIC:
  |    |     _run_diagnostics() -> evaluator.run_diagnostic_action()
  |    |
  |    +-- if OPTIMIZATION:
  |    |     _edit(plan, context)   # Agent modifies files in candidate_root
  |    |     _candidate_lineage()   # write candidate_lineage.jsonl
  |    |     evaluator.compile()
  |    |     evaluator.check_correctness()
  |    |     evaluator.development_benchmark()
  |    |     Agent interprets results -> decision
  |    |
  |    +-- append_experiment_with_hypothesis(journal, rec)
  |    +-- if READY_FOR_GATE: _gate() -> authoritative ABBA -> supervisor.decide()
  |
  +-- finally: session.close()
`

### 1.2 Workspace Layout

`
lab/campaigns/{campaign_id}/
  +-- campaign.json
  +-- contract.json
  +-- incumbent/                   <-- READ-ONLY baseline
  |     +-- {operator}.cu
  |     +-- ...
  +-- memory/
  |     +-- v0.json, v1.json, ...
  +-- e001/                        <-- EPISODE directory
  |     +-- candidate/             <-- WRITABLE candidate workspace
  |     |     +-- {operator}.cu    <-- Agent edits THIS file
  |     |     +-- ...
  |     +-- journal.jsonl
  |     +-- live.json
  |     +-- candidate_lineage.jsonl
  |     +-- evidence/
  |     +-- validated_plan_e001.json
  +-- e002/
        +-- ...
`

### 1.3 Key Observations

| Question | Answer |
|----------|--------|
| Where is workspace created? | **Controller does NOT create it.** Workbench/GUI must materialize candidate_root first. Controller validates it exists (line 223). |
| Agent modifies which files? | Files inside candidate_root/. Protected by CandidatePathPolicy which: (a) requires all writes stay inside candidate_root; (b) prevents writes to protected_roots (incumbent, memory, knowledge, registry); (c) has source_line_guard against destructive edits. |
| How is baseline saved? | incumbent/ is read-only by policy. On PROMOTE: shutil.copytree(candidate, incumbent/{id}), AFTER authoritative ABBA confirms. Old incumbent is NOT deleted. |
| Candidate failure handling? | On REJECT_COMPILE/CORRECTNESS/PERFORMANCE: record written to journal, candidate directory left as-is for inspection, loop continues. No automatic cleanup. |
| Promotion replacement? | _gate() copies candidate to incumbent/{experiment_id}-{hash} (line 335). Old incumbent directory kept. |
| Pollution risk? | **Medium.** Candidate directory IS inside the campaign directory. No git worktree isolation. Agent has filesystem write access to entire candidate_root subtree. Python imports during diagnostics create __pycache__ (excluded from hashing). |

---

## 2. Atrex Comparison

### 2.1 Workspace Isolation

| Capability | Atrex | aka-local |
|-----------|-------|-----------|
| Git worktree | **YES** - Each episode in a separate git worktree | **NO** - Episode directory under campaign/ |
| Branch isolation | **YES** - Each episode on its own git branch | **NO** - No branch per episode |
| Candidate snapshot | **YES** - Git commit after every edit | **PARTIAL** - SHA256 hash only, no git commit |
| Immutable baseline | **YES** - PROTECTED_PATHS + git; definition.json, eference.py, 	est_kernel.py locked | **PARTIAL** - CandidatePathPolicy blocks writes to protected_roots; no git-level enforcement |
| Agent writable scope | **YES** - Only kernel.py + evidence files | **PARTIAL** - Entire candidate_root subtree except protected_roots |
| Post-experiment cleanup | **YES** - git reset --hard to baseline for next iteration | **NO** - Candidate directory left as-is; next experiment edits over it |
| Rollback after failure | **YES** - git checkout -- . or git reset --hard | **NO** - No rollback mechanism beyond supervisor REJECT |

### 2.2 Recovery

| Capability | Atrex | aka-local |
|-----------|-------|-----------|
| Safe point | **YES** - durable_state.py: atomic fsync'd writes, esume_manifest.json with PID ownership | **PARTIAL** - RunStore.save() writes esume_manifest.json but no PID-reuse guard |
| Crash detection | **YES** - environment_recovery.py: ATREX_ENVIRONMENT_STATE_FILE, PID lock files, ACTIVE_MARKER | **MINIMAL** - Thread-level try/finally only. No filesystem crash marker. |
| Resume after crash | **YES** - ecovery_processes.py: full process registry, kill stale processes, handoff protocol | **PARTIAL** - ecoverable_runs() lists PAUSED/INTERRUPTED/RUNNING runs; resume requires manual controller call |
| Infrastructure retry | **YES** - infrastructure_retry.py: bounded retry with backoff | **NO** - No retry logic for transient failures |
| Environment probe on resume | **YES** - environment_recovery.py: aise_if_environment_blocked(), SSH health check | **MINIMAL** - probe_local() at start only |
| Session recovery | **YES** - session.py: session token tracking, usage subtraction | **NO** - No session state recovery |

### 2.3 Lifecycle

| Capability | Atrex | aka-local |
|-----------|-------|-----------|
| Campaign store | **YES** - store.py: CampaignStore with durable write, git excludes | **NO** - No centralized campaign state |
| Telemetry | **YES** - 	elemetry.py: episode briefs, timeline probes | **NO** |
| Supervisor state | **YES** - models.py: SupervisorState enum, gate transitions | **PARTIAL** - String-based decisions |
| Stall detection | **YES** - workspace_state.py: STALL_STATE_FILE, consecutive-failure counter | **PARTIAL** - max_consecutive_failures budget only |
| Framework baseline | **YES** - ramework_baseline_progress.py: separate baseline measurement workflow | **NO** - Baseline is implicit (V0 memory) |

---

## 3. Checkpoint Audit

### 3.1 Recovery Scenarios

| Scenario | What survives | What is lost |
|----------|--------------|--------------|
| Agent timeout/error | journal.jsonl (all completed experiments), validated_plan for current slot, candidate directory state, live.json | In-progress Agent prompt/response, in-progress compile/benchmark |
| Python crash | journal.jsonl, validated_plan, candidate_lineage.jsonl, events.jsonl, resume_manifest.json, live.json | In-memory runner state, active evaluator subprocess, session thread |
| Machine reboot | journal.jsonl, validated_plan files, candidate_lineage.jsonl, events.jsonl, resume_manifest.json (if fsync'd) | All in-memory state, Agent session, evaluator process tree |

### 3.2 Current Recovery Flow

`
Crash/Interrupt
  |
  +-- controller.emergency_stop()
  |     +-- events.append("EMERGENCY_STOP_REQUESTED")
  |     +-- runner.stop()
  |     +-- process_registry.terminate_tree()
  |     +-- run_store.save(state="INTERRUPTED")
  |
  +-- On restart: controller.recoverable_runs()
  |     +-- scans runtime/runs/*/resume_manifest.json
  |     +-- returns PAUSED/INTERRUPTED runs
  |
  +-- Human decides: controller.resume_optimization()
        +-- reads resume_manifest for candidate_root, incumbent, budget
        +-- creates new LongHorizonRunner
        +-- runner reads existing journal.jsonl
        +-- accounting skips completed experiments
        +-- _load_validated_plan() recovers persisted plan
        +-- continues from next slot
`

### 3.3 Gaps

1. **No filesystem crash marker.** On power loss, resume_manifest may be partially written.
2. **No PID-reuse guard.** If PID is reused between crash and recovery scan, stale processes may be incorrectly identified.
3. **No environment re-probe on resume.** GPU may have changed state after reboot.
4. **No automatic resume.** Human must manually invoke esume_optimization().
5. **Evaluator state not checkpointed.** In-progress compile/benchmark cannot resume mid-execution.

---

## 4. Event Audit

### 4.1 Current Event Coverage

Event types emitted during one experiment:

| Phase | Events | Coverage |
|-------|--------|----------|
| Experiment start | EXPERIMENT_STARTED | Yes |
| Knowledge load | KNOWLEDGE_REFRESHED | Yes |
| Plan generation | OPTIMIZATION_PLAN_READY, DIAGNOSTIC_PLAN_READY | Yes |
| Hypothesis | HYPOTHESIS_RECORDED, HYPOTHESIS_VERIFIED | Yes (Phase 2) |
| Candidate edit | **NO explicit event** | **MISSING** |
| Candidate lineage | **NO explicit event** | **MISSING** |
| Compile | COMPILE_STARTED, COMPILE_RESULT | Yes |
| Correctness | CORRECTNESS_STARTED, CORRECTNESS_RESULT | Yes |
| Benchmark | BENCHMARK_STARTED, BENCHMARK_RESULT | Yes |
| Agent interpretation | **NO explicit event** | **MISSING** (only via AGENT_DECISION) |
| Decision | AGENT_DECISION, EXPERIMENT_FINISHED | Yes |
| Supervisor gate | SUPERVISOR_STARTED, SUPERVISOR_RESULT | Yes |
| Promotion | INCUMBENT_PROMOTED | Yes |

### 4.2 Missing Events for Replay

To fully replay an experiment, these events are needed:

1. CANDIDATE_EDIT_STARTED / CANDIDATE_EDIT_COMPLETED — before/after Agent file modifications
2. CANDIDATE_SNAPSHOT — file-level hash list after edit
3. AGENT_INTERPRETATION — the Agent's reasoning about benchmark results
4. SESSION_TOKEN_USAGE — token consumption per turn

---

## 5. Promotion Safety Audit

### 5.1 Agent Writable vs System Protected Boundary

| Resource | Agent Writable? | Protection |
|----------|----------------|------------|
| candidate_root/* | **YES** | CandidatePathPolicy containment check |
| incumbent/ | **NO** | protected_roots in CandidatePathPolicy |
| lab/knowledge/ | **NO** | protected_roots |
| lab/registry/ | **NO** | protected_roots |
| campaign/memory/ | **NO** | protected_roots |
| evaluator code | **NO** | Not in candidate_root |
| benchmark code | **NO** | Not in candidate_root |
| supervisor code | **NO** | Not in candidate_root |
| campaign.json | **NO** | Not in candidate_root |
| journal.jsonl | **NO** (theoretically) | In episode dir, but outside candidate_root. Agent could theoretically write here via session.run_experiment_turn() which has workspace_write sandbox. |

### 5.2 Risk Assessment

**Risk: Agent can write files outside candidate_root**

CodexAgentSession.run_experiment_turn() uses Sandbox.workspace_write with cwd=str(self.workspace) where workspace = candidate_root. But the Codex sandbox controls are external to aka-local. The effective boundary is:

1. **CandidatePathPolicy** — Post-hoc enforcement. Checks after Agent edit. If Agent writes outside candidate_root, alidate_snapshot_delta() raises PathPolicyViolation and the experiment is aborted.

2. **source_line_guard** — Prevents destructive edits (e.g., emptying a source file).

3. **protected_roots** — Prevents writes to incumbent, memory, knowledge, registry.

**Gap:** Protection is reactive (post-edit validation), not preventive (Agent prompt + filesystem permissions). The Agent prompt says "modify ONLY ACTIVE WORKBENCH ROOT" but there is no OS-level enforcement.

### 5.3 Recommended Boundary

`
SYSTEM PROTECTED (no Agent access, enforced by OS permissions):
  +-- incumbent/
  +-- campaign/memory/
  +-- lab/knowledge/
  +-- lab/registry/
  +-- campaign.json, contract.json
  +-- evaluator/ code
  +-- supervisor/ code

AGENT WRITABLE (read-write, validated post-hoc):
  +-- candidate_root/*.cu, *.cuh, *.py

EVIDENCE (Agent cannot modify, evaluator writes):
  +-- episode/evidence/
  +-- episode/journal.jsonl
`

---

## 6. Atrex Capability Mapping

| Atrex Capability | aka-local Status | Gap | Priority |
|-----------------|-----------------|-----|----------|
| Controller | IMPLEMENTED (LabController) | Single-threaded, no campaign queue | P2 |
| Agent Runtime | IMPLEMENTED (LongHorizonRunner) | No session token tracking | P2 |
| Workspace | PARTIAL (CandidatePathPolicy) | No git worktree, no atomic reset | **P0** |
| Git isolation | MISSING | No branch per episode, no git commit per candidate | **P0** |
| Evaluation | IMPLEMENTED (RTX5060LocalEvaluator) | Good coverage | - |
| Telemetry | MISSING | No GPU metrics, no timeline probes | P1 |
| Checkpoint | PARTIAL (RunStore) | No crash marker, no PID guard | **P0** |
| Recovery | PARTIAL (ecoverable_runs) | No automatic resume, no environment re-probe | **P0** |
| Rollback | MISSING | No candidate reset after REJECT | **P0** |
| Promotion | IMPLEMENTED (supervisor) | Good mechanical rules | - |
| Memory | IMPLEMENTED (write_canonical) | Good canonical memory | - |
| Durable state | PARTIAL (persistence.py) | tomic_json exists, but no fsync_directory | P1 |
| Stall detection | PARTIAL (budget only) | No consecutive-no-progress detection | P1 |
| Infrastructure retry | MISSING | No retry for transient GPU/compile failures | P1 |

---

## 7. Phase 5 Implementation Plan

### 7.1 Design Principles

1. **No refactoring** of existing evaluator, benchmark, supervisor, or GUI.
2. **Additive** — new modules wrap or extend existing ones.
3. **Backward compatible** — existing RMSNorm campaign must still run.
4. **Mechanical** — all protections are enforced by code, not Agent prompt.

### 7.2 New Modules

| File | Purpose | Priority |
|------|---------|----------|
| lab/runtime/workspace.py | CandidateWorkspace class: atomic reset, git isolation, safe point tracking | **P0** |
| lab/runtime/recovery.py | Crash-safe state marker, PID-reuse guard, environment re-probe | **P0** |
| lab/runtime/rollback.py | Rollback manager: reset candidate to baseline, preserve evidence | **P0** |
| lab/runtime/telemetry.py | GPU telemetry reader: temperature, power, clock during benchmark | P1 |

### 7.3 Modified Files

| File | Change | Priority |
|------|--------|----------|
| lab/runtime/agent/long_horizon.py | Use CandidateWorkspace for all candidate operations; add rollback on REJECT; emit new events | **P0** |
| lab/core/controller.py | Use RecoveryManager on start/resume/emergency_stop | **P0** |
| lab/runtime/path_policy.py | Minor: integrated into CandidateWorkspace | P1 |

### 7.4 CandidateWorkspace Design

`python
class CandidateWorkspace:
    """Git-isolated candidate workspace with atomic operations."""

    def create(self, episode_dir, incumbent) -> CandidateWorkspace
        # Copy incumbent -> candidate_root
        # git init if needed, create branch

    def snapshot(self) -> dict
        # SHA256 hash of all files

    def reset_to_baseline(self) -> None
        # Discard all candidate changes, restore from incumbent

    def commit_candidate(self, experiment_id) -> str
        # Git commit the current candidate state

    def safe_point(self) -> dict
        # Write checkpoint marker for crash recovery

    def promote(self, target_dir) -> Path
        # Copy to incumbent with experiment_id suffix
`

### 7.5 Rollback Flow

`
REJECT_COMPILE / REJECT_CORRECTNESS / REJECT_PERFORMANCE
  |
  +-- workspace.reset_to_baseline()    # discard failed candidate
  +-- journal preserved                # evidence retained
  +-- candidate_lineage preserved      # hypothesis tracking intact
  +-- continue to next experiment
`

### 7.6 Recovery Enhancement

`
Any crash/interrupt
  |
  +-- RecoveryManager.detect()
  |     +-- Read .atrex_recovery/crash_marker
  |     +-- Check PID validity (PID-reuse guard via start_token)
  |     +-- Kill stale evaluator processes
  |
  +-- RecoveryManager.recover()
  |     +-- re-probe_local()
  |     +-- validate candidate_root integrity
  |     +-- resume from last safe_point
  |
  +-- resume_optimization() continues
`

### 7.7 Compatibility

- Existing campaigns without git: CandidateWorkspace falls back to file-copy semantics (current behavior)
- Existing journal format: unchanged
- Existing event store: unchanged, new events are additive
- Existing supervisor: unchanged
- Existing GUI: unchanged, new recovery status via existing events

### 7.8 Implementation Order

1. **CandidateWorkspace** (2-3 files) — The foundation. Replaces inline file operations in long_horizon.py.
2. **Rollback on REJECT** — One method call in long_horizon.py loop. Immediate safety win.
3. **Recovery markers** — Crash-safe state file. Enables reliable resume.
4. **Telemetry** (P1) — Read-only GPU sensor data during benchmark.
