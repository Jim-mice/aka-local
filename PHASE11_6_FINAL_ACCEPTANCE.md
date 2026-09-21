# PHASE11_6_FINAL_ACCEPTANCE.md
## Typed Operator Contract Hardening — Final Verdict

**Date**: 2026-09-19
**Status**: PHASE 11.6 — FULL PASS

---

## 1. Episode 24 Compile Failure Root Cause

**Error**: `identifier "uintptr_t" is undefined` at candidate.cu line 32.

**Classification**: **Category A — Ordinary Agent CUDA generation error.**

The agent used `uintptr_t` for alignment checking without `#include <stdint.h>`. The contract marker (`// AKA_CONTRACT: x, weight, y, batch, hidden, eps`) and hypothesis contract fields were all correct. No contract infrastructure issue.

---

## 2. Retry Commands

```
# Attempt 1 (Episode 25):
lab run --env v100 --op rms_norm_v100_cuda --episodes 1

Result: PASS on first retry.
```

---

## 3. Successful RMSNorm Regression (Episode 25)

### Contract Artifacts

| Artifact | Value | Match |
|---|---|---|
| candidate.cu marker | `// AKA_CONTRACT: x, weight, y, batch, hidden, eps` | ✅ |
| hypothesis.json operator | `rms_norm_v100_cuda` | ✅ |
| hypothesis.json contract_version | `1` | ✅ |
| hypothesis.json contract_hash | `0a3d176112c36cee` | ✅ |
| hypothesis.json interface.arguments | `["x","weight","y","batch","hidden","eps"]` | ✅ |
| episode_manifest.json operator | `rms_norm_v100_cuda` | ✅ |
| decision.json operator | `rms_norm_v100_cuda` | ✅ |

### Evaluation Result

| Check | Result |
|---|---|
| Contract validation | PASS (contract 0a3d176112c36cee matched) |
| Compile | True |
| Correctness | True |
| Geo Mean Speedup | 3.775x |
| Decision | REJECT_PERFORMANCE (3.775 < incumbent 4.077) |

---

## 4. LayerNorm Knowledge Isolation

| Check | Value |
|---|---|
| LayerNorm hash before RMSNorm retries | `c048e24deceb857c` |
| LayerNorm hash after RMSNorm retries | `c048e24deceb857c` |
| LayerNorm unchanged | ✅ TRUE |
| Cross-contamination | CLEAN |

---

## 5. Full Evidence Matrix

### LayerNorm (Episode 7)

| Stage | Result |
|---|---|
| Agent invoked | ✅ Real Codex |
| Contract validation | ✅ hash 135fb486548236a3 |
| candidate.cu marker | ✅ x, gamma, beta, y, batch, hidden, eps |
| Compile | ✅ True |
| Correctness | ✅ True |
| Benchmark (geo_mean) | ✅ 1.126x |
| Decision | ✅ ACCEPT |

### RMSNorm (Episode 25)

| Stage | Result |
|---|---|
| Agent invoked | ✅ Real Codex |
| Contract validation | ✅ hash 0a3d176112c36cee |
| candidate.cu marker | ✅ x, weight, y, batch, hidden, eps |
| Compile | ✅ True |
| Correctness | ✅ True |
| Benchmark (geo_mean) | ✅ 3.775x |
| Decision | ✅ REJECT_PERFORMANCE |

---

## 6. Contract Hashes Summary

| Operator | Hash | Args |
|---|---|---|
| layer_norm_v100_cuda | `135fb486548236a3` | x, gamma, beta, y, batch, hidden, eps |
| rms_norm_v100_cuda | `0a3d176112c36cee` | x, weight, y, batch, hidden, eps |

Both hashes stable and verified across hypothesis → manifest → decision.

---

## 7. Final Phase 11.6 Verdict

### PHASE 11.6 — FULL PASS

Both operators have demonstrated complete end-to-end typed-contract agent-driven optimization:

```
LayerNorm:  Agent → contract validation → compile ✅ → correctness ✅ → benchmark ✅ → decision ✅
RMSNorm:    Agent → contract validation → compile ✅ → correctness ✅ → benchmark ✅ → decision ✅
```

### What was proven:

1. **Parameter names survive** metadata → contract → prompt → agent → candidate.cu
2. **Argument order preserved** — agent uses exact order from metadata, no guessing
3. **Roles and semantics** are operator-specific and metadata-driven
4. **Contract version/hash** recorded in hypothesis, manifest, and decision
5. **Malformed contracts rejected locally** — 5 failure injection tests all REJECT_CONTRACT
6. **Knowledge isolation maintained** — zero cross-contamination across all runs
7. **No contract infrastructure caused any regression** — Episode 24 compile failure was ordinary agent CUDA error