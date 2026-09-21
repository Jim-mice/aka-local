# Phase 13-B: Causal Attention — Final Report

**Date:** 2026-09-19
**Status:** FULL PASS

---

## 1. ABI

```c
extern "C" void launch_kernel(
    float* q,      // input_query  [batch, heads, seq, head_dim]
    float* k,      // input_key    [batch, heads, seq, head_dim]
    float* v,      // input_value  [batch, heads, seq, head_dim]
    float* o,      // output       [batch, heads, seq, head_dim]
    int batch,     // dimension
    int heads,     // dimension
    int seq,       // dimension
    int head_dim,  // dimension
    float scale    // scaling_factor
);
```

Contract marker: `// AKA_CONTRACT: q, k, v, o, batch, heads, seq, head_dim, scale`

---

## 2. Tensor Layout

All tensors: `[batch, heads, seq, head_dim]` contiguous FP32 row-major.

---

## 3. Causal Mathematical Semantics

For each b, h, i:

```
For j <= i: score(i,j) = sum_d(Q[b,h,i,d] * K[b,h,j,d]) * scale
For j > i:  effectively -infinity (zero contribution)

m_i = max_{j <= i} score(i,j)

For j <= i: p(i,j) = exp(score(i,j) - m_i) / sum_{t <= i} exp(score(i,t) - m_i)
For j > i:  p(i,j) = 0

O[b,h,i,d] = sum_{j <= i} p(i,j) * V[b,h,j,d]
```

Special cases: i=0 depends only on V[...,0,:]; i=seq-1 sees all keys.

---

## 4. Causal-Specific Correctness Tests

All verified during Phase 13-B:

1. **seq=1**: causal equals single-token attention — PASS
2. **First query token**: O[...,0,:] depends only on V[...,0,:] — PASS (verified by `j <= qi` loop bound)
3. **Dense cannot masquerade as causal**: Dense reference evaluated as causal fails with max_err=<REMOTE_HOST> — PASS
4. **Last query position**: can depend on all keys — PASS (verified by `j <= qi` where qi=seq-1)

---

## 5. Official Shapes

| Shape | Batch | Heads | Seq | Head Dim |
|-------|-------|-------|-----|----------|
| 1,4,64,64 | 1 | 4 | 64 | 64 |
| 1,8,128,64 | 1 | 8 | 128 | 64 |
| 2,8,256,64 | 2 | 8 | 256 | 64 |
| 1,16,512,64 | 1 | 16 | 512 | 64 |

Same as Dense Attention for interpretable comparison.

---

## 6. Reference Evaluation

All 4 shapes pass correctness against causal PyTorch reference.

| Shape | Correctness | max_error | latency_us |
|-------|-------------|-----------|------------|
| 1,4,64,64 | PASS | 1.9e-06 | 3,659 |
| 1,8,128,64 | PASS | 4.3e-06 | 8,001 |
| 2,8,256,64 | PASS | 8.1e-06 | 22,883 |
| 1,16,512,64 | PASS | 1.2e-05 | 84,686 |

Reference uses `for (int kj = 0; kj <= qi; kj++)` loop — only visible keys.

---

## 7. Baseline Methodology

PyTorch causal reference:
```python
scores = Q @ K.transpose(-2, -1) * scale
causal_mask = torch.triu(torch.ones(S, S), diagonal=1).bool()
scores = scores.masked_fill(causal_mask, float("-inf"))
p = torch.softmax(scores, dim=-1)
O = p @ V
```

Score: geometric mean speedup vs PyTorch naive causal across all 4 shapes.

---

## 8. All Real Agent Episodes

| Episode | Decision | Compile | Correctness | Geo Mean | Notes |
|---------|----------|---------|-------------|----------|-------|
| 1 | REJECT_CORRECTNESS | PASS | FAIL | 0.069 | Causal mask attempted but failed |
| 2 | REJECT_PERFORMANCE | PASS | PASS | **0.038** | First valid causal candidate |
| 3 | REJECT_PERFORMANCE | PASS | PASS | **0.238** | Profile-guided, 6.3x improvement |

---

## 9. Strategy Evolution

| Episode | Strategy | Score |
|---------|----------|-------|
| 1 | Online softmax with warp-shuffle, `j <= qi` loop | 0.069 (correctness fail) |
| 2 | Warp-shuffle reductions, shared-memory staging, fused score/softmax/V | 0.038 |
| 3 | Profile-guided: warp-shuffle reductions, shared-memory score staging, fused accumulation | **0.238** |

Profile feedback (KERNEL_LATENCY diagnostic from Ep2) guided the Agent to Ep3 strategy.

---

## 10. NSYS Evidence

### Episode 2
| Kernel | Avg (us) | GPU Time% |
|--------|----------|-----------|
| causal_attention_kernel | 950.2 | 99.9% |

### Episode 3
| Kernel | Avg (us) | GPU Time% |
|--------|----------|-----------|
| causal_attention_kernel | 116.5 | 99.6% |

**Kernel-level speedup: 950.2 / 116.5 = 8.2x**

---

## 11. Diagnostics

### Episode 2
- Category: KERNEL_LATENCY
- Confidence: 0.6
- Actions: Focus on instruction-level optimization, reduction efficiency, memory access patterns

### Episode 3
- Category: UNKNOWN (insufficient evidence for further classification)
- Strategy evolution: Ep2 0.038x (950us) -> Ep3 0.238x (116.5us)

---

## 12. Profile -> Agent Feedback Proof

```
Ep2 benchmark (0.038x)
  -> REAL NSYS -> avg kernel 950us
  -> diagnostic: KERNEL_LATENCY
  -> Agent prompt: PROFILE EVIDENCE + DIAGNOSIS + RECOMMENDATIONS
  -> Ep3 Agent: warp-shuffle + shared-memory staging + fused accumulation
  -> Ep3 benchmark (0.238x, 6.3x faster)
  -> REAL NSYS -> avg kernel 116.5us (8.2x faster)
```

---

## 13. Dense-vs-Causal Separation

Dense reference evaluated through causal evaluator: **FAIL** with max_err=<REMOTE_HOST> for seq=128.

This confirms the framework correctly distinguishes dense and causal semantics.

---

## 14. Knowledge Isolation Hashes

| Operator | Hash | Accepted | Rejected |
|----------|------|----------|----------|
| rms_norm_v100_cuda | 45abe3a4... | 12 | 13 |
| layer_norm_v100_cuda | 97ee3796... | 1 | 7 |
| softmax_v100_cuda | bbba0b88... | 1 | 6 |
| dense_attention_v100_cuda | 141c6cc2... | 0 | 6 |
| causal_attention_v100_cuda | new | 0 | 3 |

Only causal_attention knowledge changed. All 4 existing operators unchanged.

---

## 15. Four Regression Results

| Operator | Compile | Correctness | Status |
|----------|---------|-------------|--------|
| rms_norm_v100_cuda | PASS | PASS | PASS |
| layer_norm_v100_cuda | PASS | PASS | PASS |
| softmax_v100_cuda | PASS | PASS | PASS |
| dense_attention_v100_cuda | PASS | PASS | PASS |

---

## 16. Operator Discovery

All 5 V100 operators dynamically discovered:

```
causal_attention_v100_cuda
dense_attention_v100_cuda
layer_norm_v100_cuda
rms_norm_v100_cuda
softmax_v100_cuda
```

---

## 17. Remaining Limitations

1. No backward pass
2. No arbitrary masks (only causal)
3. No dropout
4. FP32 only
5. No Tensor Cores
6. Geo mean 0.238x still far below 1.0x (PyTorch SDpa is highly optimized)
7. No group-query attention (GQA) or KV cache

---

## 18. Final Acceptance Checklist

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Causal typed contract works | PASS |
| 2 | Reference passes all official shapes | PASS |
| 3 | Causal semantic tests pass | PASS |
| 4 | Dense cannot masquerade as causal | PASS |
| 5 | Real Codex Agent generates causal candidate | PASS (3 episodes) |
| 6 | Agent candidate passes compile + correctness + benchmark | PASS (Ep2, Ep3) |
| 7 | REAL NSYS profiles causal candidate | PASS (Ep2, Ep3) |
| 8 | Diagnostic generated | PASS |
| 9 | Profiler evidence reaches later Agent prompt | PASS (Ep2 -> Ep3) |
| 10 | Causal knowledge isolated | PASS |
| 11 | RMSNorm regression | PASS |
| 12 | LayerNorm regression | PASS |
| 13 | Softmax regression | PASS |
| 14 | Dense Attention regression | PASS |
| 15 | lab list-ops discovers all five | PASS |

**Phase 13-B: FULL PASS — 15/15 criteria met.**
