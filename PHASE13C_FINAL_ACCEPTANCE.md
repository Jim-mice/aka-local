# Phase 13-C: Final Acceptance — Dense Attention Backward

**Date:** 2026-09-20
**Status:** FULL PASS

---

## 1. ABI Audit

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

**Argument count: 13** (8 float* + 4 int + 1 float). Consistent across metadata.json, reference.cu, AKA_CONTRACT, evaluator invocation, and documentation.

Contract marker: `// AKA_CONTRACT: q, k, v, p, grad_o, grad_q, grad_k, grad_v, batch, heads, seq, head_dim, scale`

---

## 2. Saved-P Verification

P is computed from the **same Q and K** used for backward:

```python
if op == "dense_attention_backward_v100_cuda":
    q_in, k_in = inputs[0], inputs[1]
    scale_val = 1.0 / math.sqrt(float(shape[3]))
    scores = torch.matmul(q_in, k_in.transpose(-2, -1)) * scale_val
    inputs[3] = torch.softmax(scores, dim=-1)
```

This runs inside `main()` in evaluate.py AFTER all tensors are generated. The factory returns `None` placeholder for P.

Layouts confirmed:
- Q, K, V, dO, dQ, dK, dV: [B, H, S, D] contiguous FP32
- P: [B, H, S, S] contiguous FP32

---

## 3. dQ/dK/dV Error Evidence

### Episode 2 (Best Agent Candidate)

Verified independently via remote evaluation:

| Gradient | max_abs_error | Correctness |
|----------|--------------|-------------|
| dQ | 3.81e-06 | PASS |
| dK | 3.81e-06 | PASS |
| dV | 3.81e-06 | PASS |

Tolerance: atol=2e-3, rtol=2e-3. All three pass.

### Episode 7 (Profile-Guided Candidate)

All three gradients pass correctness (max_error = 1.34e-05).

---

## 4. Finite-Difference Sanity Check

Shape: B=1, H=1, S=4, D=4, dtype=float64

| Tensor | Element | Autograd | Finite Diff | Rel Diff |
|--------|---------|----------|-------------|----------|
| dQ | (0,0,1,2) | 0.87495001 | 0.87495001 | 1.51e-09 |
| dK | (0,0,2,1) | 0.74358141 | 0.74358141 | 7.44e-10 |
| dV | (0,0,3,0) | 0.17771312 | 0.17771312 | 3.45e-09 |

**PASS** — all relative differences < 1e-8. PyTorch autograd and analytical backward are consistent.

---

## 5. Benchmark Timing Contract

### Documented Mismatch

The PyTorch reference (`torch_naive`) recomputes the full forward pass (QK^T, softmax, P @ V) before calling `.backward()`. The CUDA kernel receives saved P directly and only computes backward work.

This means CUDA-vs-PyTorch speedup numbers are **inflated** — the CUDA kernel appears faster because it skips forward recomputation.

| Component | CUDA Kernel | PyTorch Reference |
|-----------|-------------|-------------------|
| QK^T * scale | not timed | included |
| softmax(S) | not timed | included |
| P @ V (forward O) | not timed | included |
| dV = P^T @ dO | timed | included |
| dP = dO @ V^T | timed | included |
| Softmax backward | timed | included |
| dQ = dS @ K * scale | timed | included |
| dK = dS^T @ Q * scale | timed | included |

For Phase 13-C, this mismatch is **documented but accepted** because:
- The CUDA kernel contract receives saved P (matching real-world attention backprop)
- Torch benchmarking without autograd separation is a known limitation
- The primary goal is correctness and profiling, not raw speedup comparison

Future phases should separate forward/backward timing.

---

## 6. Episode 2 NSYS Command

```bash
cd /home/<REMOTE_USER>/aka_remote_jobs/nsys_ep2_v3 && \
<REMOTE_HOME>/tools/nsys-2026.2.1/target-linux-x64/nsys profile \
  --trace=cuda,osrt -o nsys_candidate \
  <REMOTE_HOME>/venvs/lerobot-act/bin/python runner.py
```

Shape: 1,2,32,32. NSYS version: 2026.2.1.

---

## 7. Episode 2 Profile Evidence

### GPU Kernels
| Kernel | Time% | Avg (ns) | Instances |
|--------|-------|----------|-----------|
| backward_rows | 97.5% | 149,311 | 210 |
| clear_kernel | 2.2% | 3,369 | 210 |

### CUDA API
| API Call | Time% | Calls |
|----------|-------|-------|
| cudaLaunchKernel | 51.5% | 435 |
| cudaDeviceSynchronize | 31.4% | 2 |

---

## 8. Episode 2 Diagnostic

- **Category**: KERNEL_LATENCY (confidence 0.85)
- **Dominant kernel**: backward_rows at 149 μs avg (97.5% GPU time)
- **Secondary**: clear_kernel at 3.4 μs (2.2%)
- **Synchronization**: 31.4% API time in cudaDeviceSynchronize
- **Actions**: Fuse clear_kernel, reduce sync overhead, consider warp-shuffle instead of atomics

---

## 9. Profile-Guided Agent Prompt Excerpt

The Agent prompt for Episode 6/7 contained:

```
=== CURRENT PERFORMANCE ===
  geo_mean_speedup: 0.606
  episode: episode_2

=== PROFILE EVIDENCE ===
  dominant_kernel_pct=97.5
  dominant_kernel_name=backward_rows
  dominant_kernel_avg_ns=149311
  kernel_count=2 (backward_rows + clear_kernel)
  sync_overhead_pct=31.4

=== CURRENT DIAGNOSIS ===
  KERNEL_LATENCY: backward_rows kernel at 149us dominates GPU time

=== PROFILE-GUIDED RECOMMENDATIONS ===
  - Fuse clear_kernel into backward_rows
  - Reduce cudaDeviceSynchronize frequency
  - Consider warp-shuffle reductions instead of atomicAdd for dK
  - Try shared-memory staging to reduce repeated global memory accesses

=== WHAT WORKED ===
  - Compile passed
  - Correctness passed (dQ, dK, dV all within tolerance)
  - geo_mean=0.606

=== WHAT FAILED ===
  - Episode 1: REJECT_CORRECTNESS
  - Episode 3: REJECT_COMPILE
  - Episode 5: REJECT_CORRECTNESS
```

---

## 10. New Agent Episode Result (Episode 7)

| Metric | Value |
|--------|-------|
| Compile | PASS |
| Correctness (dQ/dK/dV) | PASS |
| Benchmark | PASS |
| geo_mean_speedup | 0.009 |
| Decision | REJECT_PERFORMANCE |

---

## 11. Episode 7 dQ/dK/dV Errors

| Gradient | max_abs_error |
|----------|--------------|
| dQ | < 2e-3 |
| dK | < 2e-3 |
| dV | < 2e-3 |

All within tolerance.

---

## 12. Episode 7 NSYS Profile

### GPU Kernels
| Kernel | Time% | Avg | Instances |
|--------|-------|-----|-----------|
| dense_dkv | 96.5% | **5.08 ms** | 210 |
| dense_backward_rows | 3.4% | 181 μs | 210 |

### CUDA API
| API Call | Time% | Calls |
|----------|-------|-------|
| cudaDeviceSynchronize | 93.4% | 2 (551 ms each) |
| cudaLaunchKernel | 4.8% | 435 |

---

## 13. Strategy Evolution

| Episode | Strategy | Kernels | Atomic | Geo Mean | dK+dV Kernel |
|---------|----------|---------|--------|----------|--------------|
| 2 | Row-major, warp-shuffle dot_i, atomic dK/dV | 1 + clear | Yes | **0.606** | 149 μs |
| 7 | Two-kernel fused: dense_dkv + dense_backward_rows | 2 | No | 0.009 | **5,080 μs** |

**Analysis**: Profile feedback (KERNEL_LATENCY, fuse clear_kernel, reduce atomics) led the Agent to a two-kernel design. The Agent eliminated atomics and the separate clear kernel. However, the dK+dV kernel (`dense_dkv`) became 34x slower than the single-kernel approach. The profile-guided recommendations backfired for this operator because the cost of avoiding atomics (O(S^2 * D) per-thread accumulation) exceeded the atomic contention cost.

**Lesson**: For backward attention with small seq lengths (S=32), atomicAdd contention is negligible compared to recomputation cost. Profile-guided recommendations must be operator-aware.

---

## 14. Knowledge Isolation Hashes

| Operator | Knowledge File | Status |
|----------|---------------|--------|
| rms_norm_v100_cuda | knowledge_summary.json | UNCHANGED |
| layer_norm_v100_cuda | knowledge_summary.json | UNCHANGED |
| softmax_v100_cuda | knowledge_summary.json | UNCHANGED |
| dense_attention_v100_cuda | knowledge_summary.json | UNCHANGED |
| causal_attention_v100_cuda | knowledge_summary.json | UNCHANGED |
| dense_attention_backward_v100_cuda | knowledge_summary.json | UPDATED |

Only the backward operator's knowledge changed. All 5 existing namespaces remain unchanged.

---

## 15. Regression Results

Already verified in Phase 13-C:

| Operator | Compile | Correctness | Status |
|----------|---------|-------------|--------|
| rms_norm_v100_cuda | PASS | PASS | OK |
| layer_norm_v100_cuda | PASS | PASS | OK |
| softmax_v100_cuda | PASS | PASS | OK |
| dense_attention_v100_cuda | PASS | PASS | OK |
| causal_attention_v100_cuda | PASS | PASS | OK |

---

## 16. Final PASS/FAIL Verdict

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | ABI argument count/documentation consistent | PASS | 13 args, all artifacts agree |
| 2 | Saved P comes from actual Q/K | PASS | evaluate.py lines 284-288 |
| 3 | dQ/dK/dV independently validated | PASS | Ep2 verified: 3.81e-06 each |
| 4 | Finite-difference sanity check | PASS | rel_diff < 1e-8 |
| 5 | Benchmark timing boundary documented | PASS | Mismatch documented in §5 |
| 6 | REAL NSYS profiles valid backward candidate | PASS | Episode 2 profiled |
| 7 | Structured diagnostic generated | PASS | Episode 2 diagnostic.json |
| 8 | Profiler evidence reaches later Agent prompt | PASS | Prompt excerpt in §9 |
| 9 | New Agent candidate passes all three gradients | PASS | Episode 7: dQ/dK/dV PASS |
| 10 | REAL NSYS profiles new candidate | PASS | Episode 7 profiled |
| 11 | Newest diagnostic reaches subsequent context | PASS | Episode 7 diagnostic.json |
| 12 | Knowledge isolation remains clean | PASS | 5 namespaces unchanged |
| 13 | All 5 previous operators regress cleanly | PASS | Verified in Phase 13-C |

**Phase 13-C Final Acceptance: FULL PASS — 13/13 criteria met.**

---

## 17. Remaining Limitations

1. Benchmark timing includes forward recomputation in PyTorch reference
2. No NSYS for shapes > S=32 (backward kernel too slow)
3. Agent strategies diverge from profile guidance (atomics vs recomputation)
4. No causal backward yet
5. No mixed precision
6. No Tensor Cores
7. No Megatron integration
8. Profile-guided strategy actually regressed performance (0.606 -> 0.009)

---

## 18. Infrastructure Changes

- `lab/cli.py`: Fixed multi-dimensional shape parsing
- `lab/runtime/evaluators/remote_v100_campaign.py`: Fixed shape serialization
- Remote `evaluate.py`: Added `dense_attention_backward_v100_cuda` with P-from-QK fix and per-gradient error tracking
- New files: `operators/dense_attention_backward_v100_cuda/`, `config/environments/v100_sm70/evaluation_dense_attention_backward_v100_cuda.json`, campaign episodes 1-7
- NSYS profiles: Episodes 2 and 7 (campaign directories)
- Diagnostics: Episodes 2 and 7
