# Phase 13-A: Final Profiler Acceptance Report

**Date:** 2026-09-19
**Status:** FULL PASS

---

## 1. Recovery Audit

On arrival, Episode 5 had compile+correctness+benchmark results but no profiling. The profile/ directory and diagnostic.json did not exist. No partial profiling work was present. All Phase 13-A implementation was validated and preserved.

---

## 2. Episode 5 Verification

| Artifact | Status |
|----------|--------|
| candidate.cu | Valid, 3438 bytes |
| hypothesis.json | Valid, strategy: warp-shuffle + shared memory + block-per-row |
| decision.json | REJECT_PERFORMANCE, score=0.028x |
| result.json | compile=TRUE, correct=TRUE |
| contract marker | // AKA_CONTRACT: q, k, v, o, batch, heads, seq, head_dim, scale |

---

## 3. Exact Episode 5 NSYS Command

```bash
<REMOTE_HOME>/tools/nsys-2026.2.1/target-linux-x64/nsys profile \
  --stats=true \
  -o /home/<REMOTE_USER>/aka_remote_jobs/nsys_attn_ep5/nsys_attn \
  <REMOTE_HOME>/venvs/lerobot-act/bin/python \
  /home/<REMOTE_USER>/aka_remote_jobs/nsys_attn_ep5/harness.py
```

NSYS version: 2026.2.1

---

## 4. Episode 5 Parsed Profiler Evidence

**GPU Kernels:**
| Kernel | Time% | Avg (us) | Min (us) | Max (us) | Instances |
|--------|-------|----------|----------|----------|-----------|
| dense_attention_kernel | 99.9% | 864.0 | 354.1 | 3282.7 | 50 |
| distribution_elementwise | 0.1% | 8.2 | 7.8 | 9.1 | 3 |

**CUDA API:**
| API | Time% | Calls |
|-----|-------|-------|
| cudaDeviceSynchronize | 87.5% | 1 |
| cudaLaunchKernel | 11.3% | 53 |

**Key observation:** Single kernel dominates GPU time (99.9%). Launch overhead is ~109us per launch. Total GPU time for 50 iterations: 43.2ms.

---

## 5. Episode 5 diagnostic.json

```json
{
  "operator": "dense_attention_v100_cuda",
  "episode": 5,
  "category": "KERNEL_LATENCY",
  "confidence": 0.6,
  "message": "NSYS-derived diagnosis for dense_attention_v100_cuda_episode_5",
  "suggested_actions": [
    "Kernel latency is the bottleneck -- single kernel is slower than reference",
    "Focus on instruction-level optimization within the kernel",
    "Investigate reduction efficiency and memory access patterns"
  ]
}
```

Evidence-backed: 99.9% GPU time in a single kernel, avg 864us per launch.

---

## 6. Profile-Guided Agent Prompt Sections

Episode 7 Agent prompt included:

```
=== CURRENT PERFORMANCE ===
  geo_mean_speedup: 0.028
  episode: 5
  shapes: ['1,4,64,64', '1,8,128,64', '2,8,256,64', '1,16,512,64']

=== PROFILE EVIDENCE ===
  dense_attention_kernel: 99.9% GPU time
  avg kernel time: 863.99 us

=== CURRENT DIAGNOSIS ===
  category: KERNEL_LATENCY
  confidence: 0.6
  message: NSYS-derived diagnosis...

=== PROFILE-GUIDED RECOMMENDATIONS ===
  - Kernel latency is the bottleneck -- single kernel is slower than reference
  - Focus on instruction-level optimization within the kernel
  - Investigate reduction efficiency and memory access patterns
```

---

## 7. New Agent Episode: Episode 7

Profile-guided Codex Agent episode.

**Strategy:** "single-pass online-softmax attention kernel with warp-shuffle plus shared-memory dot-product reduction, shared query staging"

**Strategy evolution from Episode 5:**
- Episode 5: 3-pass (max/sum/output) with recomputed dot products
- Episode 7: single-pass online-softmax, fusing score/softmax/V accumulation

The Agent responded to the KERNEL_LATENCY diagnostic by fusing passes.

---

## 8. Episode 7 Results

| Criterion | Result |
|-----------|--------|
| Contract validation | PASS |
| Compile | PASS |
| Correctness (all 4 shapes) | PASS |
| Benchmark | PASS |
| Geometric mean speedup | 0.049x |
| vs Episode 5 (0.028x) | 1.75x improvement |
| Decision | REJECT_PERFORMANCE |

---

## 9. Episode 7 NSYS Evidence

**GPU Kernels:**
| Kernel | Avg (us) | Min (us) |
|--------|----------|----------|
| dense_attention_kernel | 501.9 | 214.1 |

**Kernel-level speedup vs Episode 5:** 864.0 / 501.9 = **1.72x**

The profile-guided Agent successfully reduced average kernel time from 864us to 502us through pass fusion.

---

## 10. Episode 7 diagnostic.json

```json
{
  "category": "KERNEL_LATENCY",
  "confidence": 0.6,
  "strategy_evolution": "Ep5: 3-pass (max/sum/output) -> Ep7: single-pass online-softmax",
  "kernel_speedup_vs_ep5": 1.72
}
```

---

## 11. Episode 5 Profile -> New Agent Strategy Trace

```
Ep5 benchmark (0.028x)
  -> REAL NSYS on Ep5
  -> diagnostic: KERNEL_LATENCY (99.9% GPU in one kernel, 864us avg)
  -> Agent prompt includes: PROFILE EVIDENCE + CURRENT DIAGNOSIS + PROFILE-GUIDED RECOMMENDATIONS
  -> Episode 7 Agent: "single-pass online-softmax" (strategy change: pass fusion)
  -> Ep7 benchmark (0.049x, 1.75x faster)
  -> REAL NSYS on Ep7: avg kernel 502us (1.72x faster at kernel level)
```

---

## 12. Newest Diagnostic Reaches Next Context

Verified: Episode 7 diagnostic.json is saved at:
`campaigns/dense_attention_v100_cuda/episode_7/diagnostic.json`

The `build_profile_context` function in `nsys_profiler.py` will read this for any subsequent attention episodes. The `build_knowledge_context` function in `remote_v100_campaign.py` includes profile evidence from the most recent profiled episode.

---

## 13. Knowledge Hashes

| Operator | Hash | Accepted | Rejected |
|----------|------|----------|----------|
| rms_norm_v100_cuda | 45abe3a4... | 12 | 13 |
| layer_norm_v100_cuda | 97ee3796... | 1 | 7 |
| softmax_v100_cuda | bbba0b88... | 1 | 6 |
| dense_attention_v100_cuda | 141c6cc2... | 0 | 6 |

Only dense_attention knowledge changed during this phase. RMSNorm, LayerNorm, and Softmax knowledge unchanged.

---

## 14. Regression Results

| Operator | Compile | Correctness | Status |
|----------|---------|-------------|--------|
| rms_norm_v100_cuda | PASS | PASS | PASS |
| layer_norm_v100_cuda | PASS | PASS | PASS |
| softmax_v100_cuda | PASS | PASS | PASS |

---

## 15. Final PASS/FAIL Verdict

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Episode 5 validity verified | PASS |
| 2 | REAL NSYS invoked on Episode 5 | PASS |
| 3 | Attention profile artifacts saved | PASS |
| 4 | Attention diagnostic.json generated | PASS |
| 5 | diagnostic is evidence-backed | PASS |
| 6 | Episode 5 profile reaches next real Agent prompt | PASS |
| 7 | real profile-guided Attention Codex episode executed | PASS |
| 8 | at least one NEW profile-guided candidate passes compile+correctness+benchmark | PASS (Ep 7) |
| 9 | REAL NSYS profiles that new candidate | PASS |
| 10 | new diagnostic generated | PASS |
| 11 | newest diagnostic is available to subsequent Attention context | PASS |
| 12 | Attention knowledge updated | PASS |
| 13 | RMSNorm knowledge unchanged | PASS |
| 14 | LayerNorm knowledge unchanged | PASS |
| 15 | Softmax knowledge unchanged | PASS |
| 16 | all three regressions pass | PASS |

**Phase 13-A: FULL PASS — 16/16 criteria met.**

---

## 16. Key Metrics Summary

| Metric | Episode 5 (baseline) | Episode 7 (profile-guided) | Improvement |
|--------|---------------------|---------------------------|-------------|
| Geo mean speedup | 0.028x | 0.049x | 1.75x |
| Avg kernel time | 864.0 us | 501.9 us | 1.72x |
| Strategy | 3-pass (max/sum/output) | single-pass online-softmax | pass fusion |
| Diagnostic | KERNEL_LATENCY | KERNEL_LATENCY | consistent |
