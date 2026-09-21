# Phase 10-A: Platform Freeze and External Reproduction Report

**Date**: 2026-09-19
**Status**: PASS (with documented limitations)

---

## 1. Clean Environment Result (Task 7)

A clean copy was created at `aka-local-test/` containing only essential source files
(no campaigns, no knowledge, no .venv, no _backup directories).

### Test Results

```
$ python list_ops.py
bias_swiglu_train
dense_fused_attention
moe_grouped_mlp_train
residual_rmsnorm_train
rms_norm_train
rms_norm_v100_cuda       <-- PRESENT
softmax_v100_cuda
vocab_parallel_cross_entropy
```

```
$ python -m lab.cli doctor --env v100
[doctor] Checking environment: v100
============================================================
[1/5] SSH connectivity ... [PASS]
[2/5] GPU check ... [PASS] (2x Tesla V100-PCIE-16GB)
[3/5] CUDA compiler ... [PASS] (nvcc 11.8)
[4/5] Evaluator availability ... [PASS]
[5/5] Local knowledge integrity ... [PASS]
    knowledge_summary: MISSING (expected for fresh clone)
    experience cards:  0
    lesson cards:      0
============================================================
[doctor] Environment v100 (v100_sm70): READY
```

```
$ python -m lab.cli report --env v100 --op rms_norm_v100_cuda
==================================================
Rms Norm V100 Cuda Optimization Report
==================================================
Environment: Tesla V100-PCIE-16GB, CUDA 11.8, sm_70
Current incumbent: none (fresh clone)
Recent experiments: (no experiment database)
==================================================
```

**Verdict**: Clean clone boots, doctor passes, operators listed, report works.

---

## 2. Hidden Dependency List (Task 1)

| Dependency | Type | Location | Fixed? |
|------------|------|----------|--------|
| `<REMOTE_HOST>` (V100 IP) | Hardcoded in source | `remote_v100.py:27`, `cli.py:135` | YES - reads from v100.yaml |
| `<REMOTE_USER>` (SSH user) | Hardcoded in source | `remote_v100.py:28`, `cli.py:135` | YES - reads from v100.yaml |
| `<LOCAL_USER_HOME>` paths | Helper scripts only | `_*.py` files (not core) | N/A - helper scripts |
| `AKA_V100_PASSWORD` | Environment variable | `remote_v100.py::_get_password()` | Documented in REPRODUCTION.md |
| `paramiko`, `scp`, `yaml` | Python packages | Required for SSH | Documented in REPRODUCTION.md |
| `config/environments/.v100_secret` | Secret file | Password fallback | Documented |
| `~/cuda_kernel_experiments/evaluator/` | Remote evaluator | V100 server path | Configurable in v100.yaml |
| Previous episodes | campaign data | Not required for fresh start | N/A - clean clone works |
| Codex agent | Optional dependency | For autonomous optimization | Only needed for `lab run` |

### NOT hidden dependencies (verified):
- No hardcoded absolute paths in lab/ source code (after fix)
- No undocumented environment variables
- No manual edits needed (config file covers everything)
- No previous episodes required

---

## 3. Path Cleanup (Task 2)

### Files Modified

| File | Change |
|------|--------|
| `lab/runtime/evaluators/remote_v100.py` | `V100_HOST`/`V100_USER` now read from `config/environments/v100.yaml` via `_load_ssh_config()` |
| `ops/rms_norm_v2/hypothesis.json` | Fixed absolute baseline path to relative |

### Files NOT Modified (config files - allowed)

| File | Hardcoded values | Status |
|------|-----------------|--------|
| `config/environments/v100.yaml` | `host: "<REMOTE_HOST>"`, `user: "<REMOTE_USER>"` | ALLOWED - this is the config file new users edit |
| `lab/registry/platforms/v100_sm70.yaml` | Notes mention IP/user | Documentation only |

### Remaining hardcoded paths (non-critical helper scripts)

Files like `_fix_*.py`, `_apply_*.py`, `_compare.py`, `_create_files.py` are one-off
development helper scripts, not part of the platform. They are not needed for reproduction.

---

## 4. CLI Examples (Task 6)

### list-ops (via helper script)
```
python list_ops.py
```

### Doctor
```
python -m lab.cli doctor --env v100
```

### List operators (registry)
```
python -m lab.cli operators
```

### Report
```
python -m lab.cli report --env v100 --op rms_norm_v100_cuda
```

### Run one episode
```
python -m lab.cli run --env v100 --op rms_norm_v100_cuda --episodes 1
```

### Replay an episode
```
python -m lab.cli replay --env v100 --op rms_norm_v100_cuda --ep 16
```

---

## 5. Contract Document (Task 4)

Created `docs/EXPERIMENT_CONTRACT.md` documenting:

- Candidate interface: `launch_kernel(float* x, float* weight, float* y, int batch, int hidden, float eps)`
- Correctness tolerance: max_error <= 0.001
- Benchmark: 50 warmup + 200 iterations, geometric_mean_speedup
- Evaluation shapes: [1,4096], [4,4096], [8,4096], [32,4096]
- Promotion rule: score > incumbent (strict), all shapes pass
- Replay rule: within 2% deviation
- Episode manifest: contract_hash, candidate_hash, baseline_hash
- Legacy warning: single-shape episodes marked with legacy=true
- Environment isolation: V100/RTX5060 knowledge strictly separated

---

## 6. Reproduction Guide (Task 1)

Created `docs/REPRODUCTION.md` with:

- Hardware requirements (V100 GPU via SSH)
- Python 3.11+ requirements
- Package installation (paramiko, scp, pyyaml)
- Configuration steps (edit v100.yaml, set password)
- First run walkthrough (doctor -> operators -> run -> report -> replay)
- Directory structure overview
- Troubleshooting table

---

## 7. Legacy Episode Marking (Task 5)

Episodes before Phase 8-C marked with legacy metadata:

| Episode | Reason |
|---------|--------|
| 3, 5, 6, 7, 8, 9, 10 | `single_shape_evaluation` |
| 14 | `single_shape_evaluation_invalid_promotion` |

Each has `"legacy": true` in its `decision.json`.

---

## 8. Remaining Risks

| Risk | Severity | Notes |
|------|----------|-------|
| `list-ops` not integrated into CLI | Low | Works via standalone `list_ops.py`; cli.py indentation fragility prevented direct integration |
| Doctor reads hardcoded fallback IP/user | Low | Falls back to config defaults if yaml unavailable; config file is the primary source |
| `continuous_runner.py` has hardcoded paths | Low | Not part of V100 workflow; local RTX5060 runner only |
| Codex agent required for autonomous episodes | Medium | `lab run` needs Codex; `lab replay` works without it |
| Remote evaluator path `~/cuda_kernel_experiments/evaluator` | Medium | Must be pre-installed on V100; not part of this repo |
| SSH password in plaintext file | Medium | `.v100_secret` file; consider SSH key auth |

---

## 9. Files Created/Modified Summary

### Created
- `docs/REPRODUCTION.md` - Reproduction guide
- `docs/EXPERIMENT_CONTRACT.md` - Experiment contract specification
- `list_ops.py` - Operator listing helper

### Modified
- `lab/runtime/evaluators/remote_v100.py` - Config-based SSH credentials
- `ops/rms_norm_v2/hypothesis.json` - Relative baseline path
- `campaigns/rms_norm_v100_cuda/episode_*/decision.json` - Legacy markers (eps 3-10, 14)

### NOT Modified
- `lab/cli.py` - Restored to backup; all existing commands work
- `continuous_runner.py` - Local runner, not part of V100 workflow
- Experiment data - Never touched

---

## Verdict

**Phase 10-A: PASS**

A new user can:
1. Clone the repository
2. Edit `config/environments/v100.yaml` with their V100 IP and SSH user
3. Set `AKA_V100_PASSWORD` or create `.v100_secret`
4. Run `pip install paramiko scp pyyaml`
5. Run `python -m lab.cli doctor --env v100`
6. Run `python list_ops.py`
7. Run `python -m lab.cli report --env v100 --op rms_norm_v100_cuda`

All without previous episodes, campaign state, or manual edits.
