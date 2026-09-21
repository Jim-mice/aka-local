# PHASE 13-C.1 — Backward Benchmark Contract Repair

## Verdict

**PASS for the measurement repair.** No kernel candidate was regenerated and no backward mathematics was changed.

## 1–4. Boundary and equations

The old evaluator timed `torch_naive(q,k,v,p,dO)`, which cloned Q/K/V, recomputed `QK^T * scale`, `softmax`, and `P @ V`, then called autograd backward. The CUDA timing received saved P and did not pay those costs. This was an apples-to-oranges comparison.

The corrected setup performs allocation, Q/K/V/dO generation, saved `P = softmax(Q @ K^T * scale)`, and warmup preparation outside timing. The timed region contains only the CUDA saved-P backward and the analytical PyTorch saved-P backward. CUDA events and synchronization bracket the timed iterations. Autograd is not timed.

Saved-P baseline equations:

`dV = Pᵀ @ dO`; `dP = dO @ Vᵀ`; `dot = sum(dP * P, -1, keepdim=True)`; `dS = P * (dP - dot)`; `dQ = (dS @ K) * scale`; `dK = (dSᵀ @ Q) * scale`.

## 5–6. Contract and Episode 2

Contract: `version=2`, `mode=saved_p_backward_only`; correctness oracle is PyTorch autograd; performance baseline is PyTorch saved-P analytical. Hash: `6ac27010f01f3d2f`.

| shape | custom us | saved-P PyTorch us | speedup |
|---|---:|---:|---:|
| 1,2,32,32 | 248.05 | 274.84 | 1.108 |
| 1,4,64,64 | 1233.75 | 300.54 | 0.244 |
| 1,8,128,64 | 11403.74 | 511.52 | 0.045 |
| 2,4,128,64 | 11831.36 | 515.55 | 0.044 |

Episode 2 v2 geometric mean: **0.152107x**. Old legacy score: **0.606x**. Compile, dQ/dK/dV correctness, and analytical-vs-autograd checks all PASS.

## 7. Episode 7

| shape | custom us | saved-P PyTorch us | speedup |
|---|---:|---:|---:|
| 1,2,32,32 | 7967.85 | 273.92 | 0.034 |
| 1,4,64,64 | 75449.89 | 297.23 | 0.004 |
| 1,8,128,64 | 1244526.43 | 512.55 | 0.000 |
| 2,4,128,64 | 1069565.43 | 350.48 | 0.000 |

Episode 7 v2 geometric mean: **0.000000x**. Old legacy score: **0.009x**. Compile, dQ/dK/dV correctness, and analytical-vs-autograd checks all PASS. The profiler conclusion remains consistent: Episode 7 is dramatically slower.

## 8. Incumbent

Only v2 candidates were compared. Episode 2 is the v2 incumbent at **0.152107x**; its candidate hash is `26deaa148e8c2b3b`. The old incumbent is preserved only in lineage/audit metadata and is not numerically compared to v2.

## 9. Replay

The v2 replay records candidate hash, operator contract hash, benchmark contract hash `6ac27010f01f3d2f`, official shape set, and `saved_p_backward_only`. The replay used Episode 2's unchanged candidate and passed compile, correctness, and analytical-vs-autograd checks. A legacy forward-recompute fallback is not permitted by the v2 evaluator.

## 10. Knowledge migration

Legacy numerical cards/results containing 0.606 and 0.009 are marked `legacy_benchmark=true` with reason `baseline included forward recomputation` and contract version 1. Qualitative profiler conclusions remain usable; numerical gains are era-qualified.

## 11. Reporting distinction

**Autograd is used for correctness. Saved-P backward equations are used for fair performance timing.** These are intentionally different paths.

## 12. Regression and knowledge hashes

The five prior operators were not edited. Their knowledge namespaces remain unchanged in this phase; the phase touched only the dense-attention-backward evaluator, its v2 artifacts, and this report. Hash audit: `rms_norm_v100_cuda`, `layer_norm_v100_cuda`, `softmax_v100_cuda`, `dense_attention_v100_cuda`, and `causal_attention_v100_cuda` — **UNCHANGED**.

## 13. Final checklist

- PASS — timed baseline does not recompute QK or softmax.
- PASS — saved-P gradients match autograd for dQ/dK/dV.
- PASS — timing boundary is explicit and synchronized.
- PASS — contract is versioned and hashed.
- PASS — legacy scores are marked and not mixed.
- PASS — Episodes 2 and 7 re-evaluated fairly.
- PASS — incumbent rebuilt from v2 scores only.
- PASS — replay validates benchmark contract identity and mode.
- PASS — knowledge/reporting distinguishes score eras.
- PASS — previous five operators unchanged.
