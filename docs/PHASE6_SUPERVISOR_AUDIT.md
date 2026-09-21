# Phase 6-A Supervisor & Gate Audit

> Generated: 2026-09-18
> Scope: `lab/runtime/supervisor/`, `lab/runtime/agent/long_horizon.py::_gate()`, `lab/core/evidence.py`, `lab/core/hypothesis.py`
> Status: READ-ONLY AUDIT — no code changes

---

## 1. Decision Flow (Call Chain)

```
┌─────────────────────────────────────────────┐
│  Evaluator                                  │
│  ├── compile()          → {pass, metrics}   │
│  ├── check_correctness()→ {pass, metrics}   │
│  ├── development_benchmark() → {metrics}    │
│  └── authoritative_abba()   → {metrics}     │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  Evidence.from_evaluation(eval_dict, cid)    │
│  ├── CORRECTNESS.FAIL     (compile fail)     │
│  ├── CORRECTNESS.FAIL     (correctness fail) │
│  ├── PERFORMANCE.INCONCLUSIVE (no speedup)   │
│  ├── PERFORMANCE.PASS     (speedup > 1.0)    │
│  └── PERFORMANCE.FAIL     (speedup ≤ 1.0)    │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  Hypothesis.verify(evidence)                 │
│  ├── SUPPORTED    (speedup > 1.0)            │
│  ├── REFUTED      (speedup < 1.0)            │
│  └── INCONCLUSIVE (no metrics / no change)   │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  decide({compile, correctness, abba})        │
│  controller_policy.py                        │
│                                              │
│  compile.pass == False?                      │
│    → REJECT_COMPILE                          │
│                                              │
│  correctness.pass == False?                  │
│    → REJECT_CORRECTNESS                      │
│                                              │
│  speedup == None?                            │
│    → INCONCLUSIVE                            │
│                                              │
│  speedup ≤ 1.0?                              │
│    → REJECT_PERFORMANCE                      │
│                                              │
│  1.0 < speedup ≤ 1.0 + weak_gain_pct?        │
│    → ROBUSTNESS_REQUIRED                     │
│                                              │
│  speedup > 1.0 + weak_gain_pct?              │
│    → PROMOTE                                 │
└──────────────────┬──────────────────────────┘
                   ↓
        ┌──────────┴──────────┐
        ↓                     ↓
  ROBUSTNESS_REQUIRED    PROMOTE / REJECT_*
        ↓
  finalize_after_robustness()
  ├── robustness.pass + speedup > 1.0 → PROMOTE
  └── else → REJECT_ROBUSTNESS
        ↓
┌─────────────────────────────────────────────┐
│  _gate(selected, records)                    │
│  long_horizon.py                             │
│                                              │
│  PROMOTE:                                    │
│    copy candidate → incumbent/<exp>-<hash>/   │
│    write canonical memory                    │
│    archive episode                           │
│                                              │
│  REJECT_*:                                   │
│    write canonical memory (rejection)         │
│    return _finish(action)                    │
└─────────────────────────────────────────────┘
```

---

## 2. Current Rules (Detailed)

### 2.1 PROMOTE Conditions

All of the following must hold:

| Gate | Condition | Source |
|------|-----------|--------|
| Compile | `compile["pass"] == True` | `decide()` L:17 |
| Correctness | `correctness["pass"] == True` | `decide()` L:19 |
| Performance | `arithmetic_mean_speedup > 1.0 + weak_gain_pct` | `decide()` L:26-29 |
| Robustness (if triggered) | `robustness["pass"] == True` AND `speedup > 1.0` | `finalize_after_robustness()` L:33-36 |

### 2.2 REJECT Conditions

| Action | Condition |
|--------|-----------|
| `REJECT_COMPILE` | `compile["pass"] == False` |
| `REJECT_CORRECTNESS` | `correctness["pass"] == False` |
| `REJECT_PERFORMANCE` | `speedup ≤ 1.0` |
| `REJECT_ROBUSTNESS` | robustness failed OR robustness speedup ≤ 1.0 |

### 2.3 UNKNOWN / INCONCLUSIVE

| Trigger | Condition |
|---------|-----------|
| Missing speedup | `arithmetic_mean_speedup is None` |
| Zero-change | `speedup == 1.0` (mapped to `REJECT_PERFORMANCE`, not INCONCLUSIVE) |

### 2.4 Answering the 6 Questions

| # | Question | Answer |
|---|----------|--------|
| 1 | 什么条件触发PROMOTE？ | compile ✓ + correctness ✓ + speedup > weak_gain_pct (2.0%) |
| 2 | 什么条件触发REJECT？ | compile ✗ / correctness ✗ / speedup ≤ 0 / robustness ✗ |
| 3 | 什么情况下保持UNKNOWN？ | speedup is None (no benchmark data) |
| 4 | correctness失败是否一定阻止promotion？ | **YES** — hard gate in `decide()` L:19, no bypass |
| 5 | performance提升是否有最低阈值？ | **YES** — `weak_gain_pct=2.0%`; 0-2% triggers ROBUSTNESS_REQUIRED |
| 6 | 多个evidence冲突如何处理？ | **Not handled** — only `authoritative_abba.arithmetic_mean_speedup` is used. No conflict resolution exists. |

---

## 3. Failure Analysis

### 3.1 `test_execution_pipeline.py::test_three_experiment_fixture`

**Symptom**: `AssertionError: 'BUDGET_EXHAUSTED' != 'PROMOTE'`

**Root Cause**: **B — Old logic bug**

In `long_horizon.py::run()`, the OPTIMIZATION path at the evidence verification stage:

```python
# Happy path (Evidence.from_evaluation succeeds):
try:
    evidence = Evidence.from_evaluation(...)
    rec["evidence"] = evidence.to_dict()
    hyp = Hypothesis.from_dict(...)
    rec["hypothesis_verdict"] = hyp.verify(...)
    # BUG: rec["decision"] is NEVER set here!
except Exception:
    # Fallback path (only reached on error):
    rec["decision"] = "READY_FOR_GATE" if ... else ...
```

When `Evidence.from_evaluation()` succeeds (normal for a good optimization),
`rec["decision"]` remains unset. The downstream check at the end of the
experiment loop uses `rec.get("decision","")` which returns `""`, so:

- `"" != "READY_FOR_GATE"` → no gate
- `"" != "BLOCKED"` → no human wait
- Experiment counted toward budget
- After 3 experiments, budget exhausted → `BUDGET_EXHAUSTED`

**Severity**: **HIGH** — all successful optimizations silently fail to reach gate.

**Introduced by**: Phase 5-D `rec.get("decision","")` change exposed this — 
before that, `rec["decision"]` threw `KeyError`, which masked the bug.

### 3.2 `test_process_recovery.py::test_environment_mismatch_blocks_resume`

**Symptom**: `AttributeError: 'NoneType' object has no attribute 'workbench_id'`

**Root Cause**: **B — Old logic bug**

In `controller.py::resume_recovered()` L:276:
```python
self.events.append("RECOVERY_VALIDATED", report.to_dict(),
    workbench_id=self.workbench.workbench_id)
```

The test creates `LabController(temp)` without a workbench config, so
`self.workbench` is `None`. The `workbench_id` access crashes.

**Severity**: **MEDIUM** — only affects resume path when workbench not configured.

**Introduced by**: Pre-existing, not related to Phase 5 changes.

### Summary

| Test | Type | Severity | Phase Introduced |
|------|------|----------|------------------|
| `test_three_experiment_fixture` | Logic bug (missing decision) | HIGH | Pre-existing, exposed by 5-D |
| `test_environment_mismatch_blocks_resume` | Logic bug (None guard) | MEDIUM | Pre-existing |

---

## 4. Missing Safeguards

### 4.1 Decision Contract Gaps

Current decision flow does NOT check:

| Gap | Risk |
|-----|------|
| **No minimum absolute speedup** | 2% of a noisy baseline is unreliable |
| **No variance/stdev check** | A 2.1% speedup with ±3% noise is meaningless |
| **No regression check** | Only `arithmetic_mean_speedup` checked; individual shapes may regress |
| **No correctness scope** | "56 tests pass" says nothing about what was tested |
| **No conflict resolution** | Multiple evidence sources are not combined or prioritized |
| **Agent opinion ignored** | `ready_for_gate` in plan is advisory; only mechanical evidence counts |
| **No human override** | Once `_gate()` runs, decision is final |

### 4.2 Decision Contract — Current vs Ideal

**Current**:
```json
{
  "action": "PROMOTE | REJECT_* | INCONCLUSIVE",
  "reason": "(implicit — derived from action)",
  "evidence": "(split across compile/correctness/abba dicts)",
  "confidence": "(NOT PRESENT)"
}
```

**Missing**: `confidence` field, explicit `reason` string, `evidence` aggregation.

### 4.3 Existing Safeguards (Positive)

| Safeguard | Location |
|-----------|----------|
| Correctness is hard gate | `decide()` |
| Weak gains require robustness | `decide()` → `ROBUSTNESS_REQUIRED` |
| Robustness is a second gate | `finalize_after_robustness()` |
| Agent cannot promote | `promotion_policy.yaml: agent_can_promote: false` |
| Compile failure blocks everything | `decide()` first check |

---

## 5. Phase 6 Implementation Plan

### 6-A (Current): Audit ✅
- [x] Read all supervisor/gate files
- [x] Map full decision chain
- [x] Identify 2 failing tests + root causes
- [x] Document missing safeguards
- [x] Output this report

### 6-B: Bug Fixes
- [ ] Fix `rec["decision"]` not set in happy-path evidence verification (`long_horizon.py`)
- [ ] Fix `self.workbench` None-guard in `resume_recovered()` (`controller.py:276`)
- [ ] Verify all tests pass after fixes

### 6-C: Decision Contract Hardening
- [ ] Add `confidence` field to decision output
- [ ] Add explicit `reason` string
- [ ] Aggregate evidence into decision payload
- [ ] Consider: per-shape regression check (not just mean speedup)
- [ ] Consider: minimum absolute improvement threshold

### 6-D: Supervisor Tests
- [ ] Test PROMOTE path end-to-end (fix existing fixture test)
- [ ] Test each REJECT branch with edge cases
- [ ] Test ROBUSTNESS_REQUIRED → PROMOTE chain
- [ ] Test ROBUSTNESS_REQUIRED → REJECT_ROBUSTNESS chain
- [ ] Test INCONCLUSIVE path

### 6-E: Documentation
- [ ] Update `STATUS.md` with Phase 6 completion
- [ ] Document decision contract in `docs/`

---

## Appendix A: File Map

| File | Role |
|------|------|
| `lab/runtime/supervisor/controller_policy.py` | `decide()`, `finalize_after_robustness()`, `decide_from_evidence()` |
| `lab/runtime/supervisor/policy.py` | Legacy `decide()` (simpler, unused?) |
| `lab/runtime/supervisor/promotion_policy.yaml` | Config: thresholds, flags |
| `lab/core/evidence.py` | `Evidence`, `EvidenceType`, `Verdict`, `from_evaluation()` |
| `lab/core/hypothesis.py` | `Hypothesis`, `verify()` |
| `lab/runtime/agent/long_horizon.py::_gate()` | Supervisor invocation, promotion copy, canonical write |
| `lab/runtime/agent/long_horizon.py::run()` | Experiment loop, evidence verification, decision dispatch |
| `lab/core/controller.py::resume_recovered()` | Recovery resume with environment check |
| `lab/tests/test_supervisor_policy.py` | Unit tests for `decide()` / `finalize_after_robustness()` |
| `lab/tests/test_execution_pipeline.py` | Integration test: 3-experiment → PROMOTE |
| `lab/tests/test_process_recovery.py` | Process recovery + environment mismatch tests |

## Appendix B: Key Constants

| Constant | Value | Source |
|----------|-------|--------|
| `weak_gain_pct` | 2.0 | `controller_policy.py` default |
| `weak_improvement_threshold_pct` | 2.0 | `promotion_policy.yaml` |
| `strict_positive_metric` | true | `promotion_policy.yaml` |
| `correctness_required` | true | `promotion_policy.yaml` |
| `agent_can_promote` | false | `promotion_policy.yaml` |
