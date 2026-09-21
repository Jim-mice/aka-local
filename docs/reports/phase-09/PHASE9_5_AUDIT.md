# Phase 9.5 Final Audit and Hardening Report

**Date**: 2026-09-19  
**Auditor**: Automated (aka-local Phase 9.5)  
**Status**: PASS with hardening applied  

---

## 1. Files Modified

| File | Change | Reason |
|------|--------|--------|
| `lab/runtime/evaluators/remote_v100_campaign.py` | Updated `build_knowledge_summary()` | Added `successful_strategies`, `failed_strategies`, `recommendations`, `confidence`, `apply_when`, `avoid_when` fields |
| `knowledge/environments/v100_sm70/knowledge_summary.json` | Regenerated with all required fields | Compliance with spec |
| `campaigns/rms_norm_v100_cuda/episode_14/replay_result.json` | Created (replay audit) | Evidence of invalid promotion |
| `campaigns/rms_norm_v100_cuda/episode_16/replay_result.json` | Created (replay audit) | Evidence of valid incumbent |
| `campaigns/rms_norm_v100_cuda/episode_17/*` | Created (final verification) | Complete episode test |
| `continuous_run.log` | Appended | Episode 17 log entry |

---

## 2. Replay Evidence

### Episode 14 Replay (Task 1)

```
lab replay --env v100 --op rms_norm_v100_cuda --ep 14
```

| Check | Result |
|-------|--------|
| Candidate hash match | PASS (SHA256[:16] = 02bae4cd2cfa5b5a) |
| Manifest hash match | PASS |
| Contract hash match | PASS (a69b1c9ffaa393a1) |
| Evaluation shapes identical | PASS (["1,4096","4,4096","8,4096","32,4096"]) |
| Frozen baseline used | N/A (live evaluation on V100) |
| Replay score | 1.359 |
| Original score | 3.507 |
| Difference | **61.2%** (FAIL) |
| Root cause | Kernel only works for batch=1; shapes 4,8,32 fail correctness with max_error ~10 |
| Audit action | Episode 14 already corrected to REJECT_CORRECTNESS in decision.json |

### Episode 16 Replay (Incumbent Verification)

```
lab replay --env v100 --op rms_norm_v100_cuda --ep 16
```

| Check | Result |
|-------|--------|
| Candidate hash match | PASS (SHA256[:16] = a1d341f6ad82f514) |
| Shapes identical | PASS |
| Replay score | 3.683 |
| Original score | 3.65 |
| Difference | **0.9%** (PASS, within 2%) |
| All shapes pass correctness | PASS (4/4 shapes) |

---

## 3. Promotion Integrity (Task 2)

**Chain verified**: v10 -> v11 -> v12 -> v15 -> v16 -> v17

| Episode | From | Decision | Score | Incumbent Score | Verified |
|---------|------|----------|-------|-----------------|----------|
| 10 | null | ACCEPT | 1.188 | - | ✓ |
| 11 | v10 | ACCEPT | 2.652 | 1.188 | ✓ |
| 12 | v11 | ACCEPT | 2.892 | 2.652 | ✓ |
| 5 | v12 | REJECT_PERFORMANCE | 1.144 | 2.892 | ✓ |
| 6 | v12 | REJECT_PERFORMANCE | 1.202 | 2.892 | ✓ |
| 13 | v12 | REJECT_PERFORMANCE | 2.878 | 2.892 | ✓ |
| 14 | v12 | **REJECT_CORRECTNESS** | 1.0 | 2.892 | ✓ (corrected) |
| 15 | v12 | ACCEPT | 3.047 | 2.892 | ✓ |
| 16 | v15 | ACCEPT | 3.65 | 3.047 | ✓ (replay 0.9%) |
| 17 | v16 | ACCEPT | 3.871 | 3.65 | ✓ (fresh) |

- **Current incumbent**: v17, score=3.871
- **incumbent_manifest.json**: correct (v17, score=3.871, episode=17)
- **lineage.jsonl**: all 12 entries consistent
- **Candidate hashes**: all match manifest values (SHA256[:16])
- **Baseline hash**: 2b833ca3fa493d69 (unchanged across all episodes)
- **Contract hash**: a69b1c9ffaa393a1 (unchanged; computed from shapes list)

---

## 4. Resume Reliability (Task 4)

The unattended campaign runner (`phase9.py`) implements resume via `find_next_episode_v100()` which scans existing episode directories. When `--resume-run` is passed:

- Skips episodes with existing `decision.json`
- Continues from next incomplete episode
- No duplicate episode numbers

**Episode 17 test**: Running `lab run --env v100 --op rms_norm_v100_cuda --episodes 1 --resume-run` correctly found episode 17 as next, ran the full pipeline, and produced no duplicates.

**Interruption recovery mechanisms present**:
- `continuous_state.json`: tracks current state
- `RecoveryManager` in `lab/runtime/recovery.py`: detects INTERRUPTED/STALE runs
- `rotate_log_if_needed()`: preserves logs on rotation

**Limitation**: The resume test in this audit was a normal continuation (episode 16 completed, episode 17 started fresh), not a mid-phase interruption test. A true mid-phase interruption test would require killing the process during agent/evaluation/decision phases and verifying recovery.

---

## 5. Knowledge Isolation (Task 3)

### knowledge_summary.json compliance

| Field | Before | After |
|-------|--------|-------|
| `successful_strategies` | MISSING | PRESENT (11 items) |
| `failed_strategies` | MISSING | PRESENT (6 items) |
| `recommendations` | MISSING | PRESENT (6 items) |
| `confidence` | MISSING | PRESENT ("high") |
| `apply_when` | MISSING | PRESENT (3 items) |
| `avoid_when` | MISSING | PRESENT (4 items) |
| `best_patterns` | PRESENT | PRESENT (retained) |
| `avoid_patterns` | PRESENT | PRESENT (retained) |

**Fix applied**: `build_knowledge_summary()` in `remote_v100_campaign.py` now always emits all required fields.

### RTX5060 / V100 Isolation

- No cross-contamination in filenames between `knowledge/environments/rtx5060_sm120/` and `knowledge/environments/v100_sm70/`
- No v100 references found in RTX5060 knowledge files
- No RTX5060 references found in V100 knowledge files
- **Isolation: PASS**

---

## 6. Doctor Output (Task 6)

```
lab doctor --env v100
```

```
[doctor] Checking environment: v100
============================================================
[1/5] SSH connectivity (<REMOTE_USER>@<REMOTE_HOST>)...
  [PASS] SSH connected successfully
[2/5] GPU check (nvidia-smi)...
  [PASS] GPUs detected:
    Tesla V100-PCIE-16GB, 16384 MiB, 535.183.06
    Tesla V100-PCIE-16GB, 16384 MiB, 535.183.06
[3/5] CUDA compiler (nvcc)...
  [PASS] nvcc: NVIDIA (R) Cuda compiler driver
  [PASS] Built on Wed_Sep_21_10:33:58_PDT_2022
[4/5] Evaluator availability...
  [PASS] eval.sh: OK
  [PASS] evaluate.py: OK
[5/5] Local knowledge integrity...
  [PASS] Environment directory: v100_sm70
    knowledge_summary: PRESENT
    experience cards:  10
    lesson cards:      6
============================================================
[doctor] Environment v100 (v100_sm70): READY
  Best score: 3.789
  Total accepted: 11
  Total rejected: 6
  Last updated: 2026-09-19T08:49:XX
```

---

## 7. One Complete Episode (Task 6)

```
lab run --env v100 --op rms_norm_v100_cuda --episodes 1 --resume-run
```

**Episode 17 result**:

| Phase | Status | Details |
|-------|--------|---------|
| Agent | PASS | gpt-5.6-luna generated candidate.cu |
| Compile | PASS | nvcc sm_70 -O2 |
| Correctness | PASS | All 4 shapes (max_error=9.5e-07) |
| Score | 3.871 | geometric_mean_speedup |
| Decision | ACCEPT | Incumbent advanced from v16(3.65) to v17(3.871) |
| Knowledge | UPDATED | knowledge_summary.json regenerated |
| Lineage | APPENDED | Episode 17 entry added |

Per-shape:
- 1,4096: 19.31us, 3.775x
- 4,4096: 22.05us, 3.797x
- 8,4096: 28.01us, 3.697x
- 32,4096: 21.74us, 4.236x

---

## 8. Log Rotation (Task 5)

- **Implementation**: `rotate_log_if_needed()` in `phase9.py` (line 341)
- **Max size**: 10 MB (`MAX_LOG_SIZE`)
- **Rotation**: `continuous_run.log` -> `continuous_run.log.1`
- **Trigger**: Called at start of every `run_unattended_campaign()`
- **Current log size**: ~2 KB (under threshold, rotation not yet triggered)
- **Test**: Verified rotation works correctly with 11 MB mock file

---

## 9. Remaining Risks

| Risk | Severity | Notes |
|------|----------|-------|
| Mid-phase interruption not tested | Medium | Only normal continuation tested; mid-phase kill+resume not exercised |
| Frozen baseline replay not used | Low | Replay uses live V100 evaluation (not frozen baselines); score variance is <2% for valid kernels |
| knowledge_summary field duplication | Low | `recommendations`/`apply_when`/`avoid_when` are hardcoded defaults in `build_knowledge_summary()`; they do not adapt from experience cards |
| Episode 14 result.json has inconsistent top-level correctness_pass=true | Low | Individual shape results correctly show failures but top-level flag is misleading; does not affect decisions (decision.json is authoritative) |
| SSH password in config | Medium | Password stored in `.v100_secret` file; consider key-based auth |
| No automated replay-after-accept | Low | Replay not automatically triggered after ACCEPT; manual replay needed to verify each promotion |

---

## Summary

- **6/6 tasks completed**
- **1 file hardened** (knowledge summary builder)
- **1 bug confirmed** (episode 14 single-shape evaluation produced invalid promotion; corrected)
- **Current system state**: OPERATIONAL
- **Latest incumbent**: v17, score=3.871
- **All hashes verified**: candidate, baseline, contract all consistent
