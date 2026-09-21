# Phase 13-C: Dense Attention Backward — Final Report

**Date:** 2026-09-20
**Status:** FULL PASS

---

## 1. Backward Equations

Forward:
```
S = Q @ K^T * scale              [B, H, S, S]
P = softmax(S, dim=-1)           [B, H, S, S]
O = P @ V                        [B, H, S, D]
```

Given upstream gradient dO [B, H, S, D]:
```
dV = P^T @ dO                    [B, H, S, D]
dP = dO @ V^T                    [B, H, S, S]

For each row i (softmax backward):
  dot_i = sum_j(dP[i,j] * P[i,j])
  dS[i,j] = P[i,j] * (dP[i,j] - dot_i)

dQ = dS @ K * scale              [B, H, S, D]
dK = dS^T @ Q * scale            [B, H, S, D]
```

---

## 2. ABI

```c
extern "C" void launch_kernel(
    float* q,      // input_query          [B, H, S, D]
    float* k,      // input_key            [B, H, S, D]
    float* v,      // input_value          [B, H, S, D]
    float* p,      // saved_forward_tensor  [B, H, S, S]
    float* grad_o, // upstream_gradient     [B, H, S, D]
    float* grad_q, // gradient_output       [B, H, S, D]
    float* grad_k, // gradient_output       [B, H, S, D]
    float* grad_v, // gradient_output       [B, H, S, D]
    int batch,     // dimension
    int heads,     // dimension
    int seq,       // dimension
    int head_dim,  // dimension
    float scale    // scaling_factor (1/sqrt(D))
);
```

Contract marker: `// AKA_CONTRACT: q, k, v, p, grad_o, grad_q, grad_k, grad_v, batch, heads, seq, head_dim, scale`

---

## 3. Tensor Layouts

| Tensor | Layout | dtype |
|--------|--------|-------|
| Q, K, V, dO, dQ, dK, dV | [B, H, S, D] | FP32 contiguous |
| P | [B, H, S, S] | FP32 contiguous |

---

## 4. Saved-Tensor Contract

P is EXPLICITLY PROVIDED from forward. The backward kernel does NOT recompute softmax.
PyTorch autograd reference computes gradients from scratch (Q, K, V, dO), recomputing S and P internally, providing the ground truth for correctness.

---

## 5. Official Shapes

| Shape | B | H | S | D |
|-------|---|---|---|---|
| 1,2,32,32 | 1 | 2 | 32 | 32 |
| 1,4,64,64 | 1 | 4 | 64 | 64 |
| 1,8,128,64 | 1 | 8 | 128 | 64 |
| 2,4,128,64 | 2 | 4 | 128 | 64 |

Smaller than forward shapes since backward is O(S^2 × D) per head.

---

## 6. PyTorch Autograd Reference

```python
qq = q.clone().requires_grad_(True)
kk = k.clone().requires_grad_(True)
vv = v.clone().requires_grad_(True)

scores = qq @ kk.transpose(-2, -1) * scale
pp = torch.softmax(scores, dim=-1)
oo = pp @ vv

oo.backward(grad_o)

# Reference gradients: qq.grad, kk.grad, vv.grad
```

---

## 7. Finite-Difference Check

Not performed. Reference autograd is sufficient for correctness.

---

## 8. Manual CUDA Reference Results

| Shape | Correctness | dQ max_err | dK max_err | dV max_err | Latency (us) | Speedup |
|-------|-------------|------------|------------|------------|-------------|---------|
| 1,2,32,32 | PASS | < 2e-3 | < 2e-3 | < 2e-3 | 3,133 | 0.359x |
| 1,4,64,64 | PASS | < 2e-3 | < 2e-3 | < 2e-3 | 8,573 | 0.118x |
| 1,8,128,64 | PASS | < 2e-3 | < 2e-3 | < 2e-3 | 28,037 | 0.039x |
| 2,4,128,64 | PASS | < 2e-3 | < 2e-3 | < 2e-3 | 28,038 | 0.038x |

**Geo mean speedup: 0.089x**

Reference uses one kernel with atomicAdd for dK accumulation and __syncthreads for dV.

---

## 9. All Real Agent Episodes

| Episode | Decision | Compile | Correctness | Geo Mean | Notes |
|---------|----------|---------|-------------|----------|-------|
| 1 | REJECT_CORRECTNESS | PASS | FAIL | 0.000 | max_err=39.76; confused Q and V in dP |
| 2 | REJECT_PERFORMANCE | PASS | PASS | **0.606** | Best: row-major, warp-shuffle, atomic dK/dV |
| 3 | REJECT_COMPILE | FAIL | — | 0.000 | Pointer type error at line 80 |
| 4 | REJECT_PERFORMANCE | PASS | PASS | **0.007** | Correct but much slower |
| 5 | REJECT_CORRECTNESS | PASS | FAIL | 0.000 | Correctness failure |

**Two candidates (Eps 2, 4) passed compile + dQ + dK + dV + benchmark.**

---

## 10. dQ/dK/dV Errors (Episode 2)

| Shape | dQ max_err | dK max_err | dV max_err |
|-------|-----------|-----------|-----------|
| 1,2,32,32 | < 2e-3 | < 2e-3 | < 2e-3 |
| 1,4,64,64 | < 2e-3 | < 2e-3 | < 2e-3 |
| 1,8,128,64 | < 2e-3 | < 2e-3 | < 2e-3 |
| 2,4,128,64 | < 2e-3 | < 2e-3 | < 2e-3 |

All within tolerance (atol=2e-3, rtol=2e-3).

---

## 11. Benchmark Results (Episode 2 — Best)

| Shape | Latency (us) | Speedup vs Torch |
|-------|-------------|-----------------|
| 1,2,32,32 | 171 | **6.55x** |
| 1,4,64,64 | 885 | 1.22x |
| 1,8,128,64 | 7,520 | 0.15x |
| 2,4,128,64 | 7,497 | 0.12x |

**Geo mean: 0.606x**

Fast on small shapes (< 1ms at S=32); bottlenecked by atomic contention on large shapes.

---

## 12. NSYS Evidence

NSYS profiling was attempted but unavailable (NSYS binary not accessible through default mechanism).

Profile cards were written as "unavailable" for all episodes.

---

## 13. Diagnostics

Knowledge-based diagnostics were generated for all episodes. Key patterns:

- **Episode 1**: Correctness failure — Agent confused Q and V in dP computation
- **Episode 2**: Performance (geo=0.606) — atomicAdd contention on dK and dV
- **Episode 3**: Compile failure — pointer type error
- **Episode 4**: Performance (geo=0.007) — excessive kernel overhead
- **Episode 5**: Correctness failure

---

## 14. Profile-Guided Strategy Evolution

NSYS was unavailable, so profile-guided feedback was limited to performance numbers from knowledge context. Episode 2's strategy (row-major mapping, warp-shuffle dot_i, atomic dK/dV) achieved 0.606x geo mean. Episode 3's compile error prevented further profile guidance.

---

## 15. Knowledge Isolation Hashes

| Operator | Knowledge File | State |
|----------|---------------|-------|
| rms_norm_v100_cuda | knowledge_summary.json | UNCHANGED |
| layer_norm_v100_cuda | knowledge_summary.json | UNCHANGED |
| softmax_v100_cuda | knowledge_summary.json | UNCHANGED |
| dense_attention_v100_cuda | knowledge_summary.json | UNCHANGED |
| causal_attention_v100_cuda | knowledge_summary.json | UNCHANGED |
| **dense_attention_backward_v100_cuda** | **knowledge_summary.json** | **NEW** |

Only the backward operator's knowledge directory was populated.

---

## 16. Five Regressions

| Operator | Compile | Correctness | Status |
|----------|---------|-------------|--------|
| rms_norm_v100_cuda | PASS | PASS | OK |
| layer_norm_v100_cuda | PASS | PASS | OK |
| softmax_v100_cuda | PASS | PASS | OK |
| dense_attention_v100_cuda | PASS | PASS | OK |
| causal_attention_v100_cuda | PASS | PASS | OK |

All 5 operators regress cleanly (reference kernels compile and pass correctness).

---

## 17. Operator Discovery

```
lab list-ops → 6 V100 operators:
  causal_attention_v100_cuda
  dense_attention_backward_v100_cuda  ← NEW
  dense_attention_v100_cuda
  layer_norm_v100_cuda
  rms_norm_v100_cuda
  softmax_v100_cuda
```

No hardcoded operator list — dynamically discovered from operators/ directory.

---

## 18. Remaining Limitations

1. No causal backward
2. No dropout
3. FP32 only
4. No Tensor Cores
5. No Megatron integration
6. No NSYS profiling (binary unavailable)
7. AtomicAdd contention limits large-shape performance
8. P tensor must be consistent with Q,K (provided from forward)
9. geo mean 0.606x best (Episode 2), below 1.0x PyTorch reference
10. No fused forward+backward API

---

## 19. Final Acceptance Checklist

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Backward math explicitly documented | PASS |
| 2 | Typed backward contract works | PASS |
| 3 | CUDA reference passes dQ/dK/dV | PASS |
| 4 | PyTorch autograd oracle works | PASS |
| 5 | Real Codex Agent generates backward candidate | PASS (5 episodes) |
| 6 | Agent candidate passes all three gradient checks | PASS (Eps 2, 4) |
| 7 | Benchmark produced | PASS |
| 8 | All 5 previous operators regress cleanly | PASS |
| 9 | lab list-ops discovers 6 operators | PASS |
| 10 | Backward knowledge isolated | PASS |
| 11 | Contract validation (AKA_CONTRACT) | PASS |
| 12 | Dense-vs-causal distinction preserved | PASS |

**Phase 13-C: FULL PASS — 12/12 criteria met.**

---

## 20. Infrastructure Changes

- **lab/cli.py**: Fixed shape parsing for multi-dimensional shapes (was 2D-only)
- **lab/runtime/evaluators/remote_v100_campaign.py**: Fixed multi-dimensional shape serialization
- **Remote evaluate.py**: Added `dense_attention_backward_v100_cuda` with PyTorch autograd reference and P-from-QK computation fix
