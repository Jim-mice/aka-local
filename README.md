# V100 RMSNorm Agent vs Human — Controlled Megatron E2E

> **实验分支标识**
>
> 本分支是 **2026-09-22 至 2026-09-23** 的 V100 RMSNorm 专项实验快照。
> Agent-vs-Human 对照仅针对 **RMSNorm**。
> 它**不是**整个 GPU 优化 Agent 的通用胜负结论，也**不是** Nine-grid E2E。
>
> `RMSNORM_ONLY = YES`
> `NINE_GRID_E2E = NO`
>
> This README intentionally differs from `main`. `main` contains the general
> aka-local project overview. This branch is a frozen V100 RMSNorm
> Agent-vs-Human experimental record.

---

## 1. 一句话结论

在冻结的 V100 Controlled Megatron E2E + Snapshot C measurement protocol 下，
**Current Agent (H003)** 与 **Human (v2)** 都通过 L0 / L1 / L2 全部 correctness 与 stability gate；
两次独立正式 L2 中，Run 1 Agent 数值较高，Run 2 Human 数值较高。
当前证据支持的结论是**二者处于同一性能量级**，frozen OJ 没有定义跨两个 independent runs 的 aggregate winner。

---

## 2. Final Results

### 2.1 主表（冻结最终 L2 Snapshot C）

| System | L0 | L1 | L2 Run 1 | L2 Run 2 | Status |
|---|---:|---|---:|---:|---|
| Current Agent H003 | `3.6734935121x` | PASS | `1.1967401230x` | `1.1815262131x` | PASS |
| Human v2 | 见 §6 | PASS | `1.1735x` | `1.1990x` | PASS |

**Official aggregate winner: NOT DEFINED**（frozen OJ 未定义跨两个 independent L2 runs 的 aggregate winner）。

可记录的事实（不是 winner claim）：

- Run 1：Agent numerical result is higher (`1.1967401230x` vs `1.1735x`)。
- Run 2：Human numerical result is higher (`1.1990x` vs `1.1815262131x`)。
- 因此：`CURRENT_AGENT_AND_HUMAN_IN_SAME_PERFORMANCE_REGIME`。

禁止：算均值/geometric mean 后排名 Agent 或 Human、"Agent beats Human" / "Human beats Agent"。

---

## 3. What is now implemented

### 3.1 Agent mechanism reasoning loop

Current Agent 已经不只是做 standalone kernel score。当前 RMSNorm loop 已跑通：

```
PerformanceFacts
   → mechanism-level hypothesis
   → precondition / evidence
   → implementation attempt
   → L0 Operator OJ
   → L1 Megatron Integration OJ
   → L2 Controlled Megatron E2E OJ
```

### 3.2 Same-hypothesis repair

Hypothesis 与 implementation attempt 分离。同一个 hypothesis 下可以有多次 attempt，便于在不重写 hypothesis 的情况下做修复（Human v2 的 compile repair 就是同 hypothesis 的纯语法修复示例）。

### 3.3 E2E reward

最终不只看 standalone kernel：L0 → L1 → L2。

### 3.4 Evidence discipline

Measurement failure 会阻止 promotion。Snapshot B 就是实例：观测到 speedup，但 CV gate failure，因此不 promotion。

> 关于 planner / mechanism-reasoning benchmark 的 caveat：这些 benchmark 衡量的是 mechanism-level reasoning capability，**不是** optimal CUDA generation guarantee；历史 caveat 见既有 audit（不在本 README 展开）。

---

## 4. Frozen Experimental Environment

| 项 | 值 |
|---|---|
| GPU | Tesla V100-PCIE-16GB |
| GPU UUID | `GPU-88be8e63-dd61-3d8f-2f44-e1f93204654d` |
| Megatron-LM commit | `5be9626709af2722333bf54797c954c09edeada3` |
| Workload | Controlled Megatron GPTModel training step |
| dtype | FP16 |
| RMSNorm epsilon | `1e-5` |
| Hidden size | 1024 |
| Tensor Parallel | 1 |
| Pipeline Parallel | 1 |
| Snapshot | `V100_RMSNORM_E2E_SNAPSHOT_C` |
| Benchmark base commit | `6d8d3549b3f37d0be8d87126ac6b5581bfdfd829` |

不暴露：SSH host、SSH username、密码、内部 IP。

---

## 5. Frozen L0 Shapes

| S | B | H |
|---:|---:|---:|
| 16 | 1 | 1024 |
| 64 | 2 | 1024 |
| 128 | 2 | 1024 |

L0 验证：

- forward correctness
- input gradient correctness
- weight gradient correctness
- forward performance
- backward performance

---

## 6. Evaluation Stack

### 6.1 L0 — Operator OJ

RMSNorm 算子级：

- forward
- backward
- dx
- dw
- correctness
- performance

### 6.2 L1 — Megatron Integration OJ

真实：

```
WrappedTorchNorm → candidate RMSNorm
```

检查：

- replacement
- invocation
- no fallback
- forward
- backward
- loss / gradient gate

### 6.3 L2 — Controlled Megatron E2E

完整：

```
GPTModel
   → forward
   → loss
   → backward
   → optimizer-compatible step
```

Primary timing：**outer whole-step CUDA Event**。这不是只测 RMSNorm kernel。

---

## 7. Measurement Protocol (Snapshot C)

| 项 | 值 |
|---|---|
| Order | same-process Reference / Candidate |
| Pairing | ABBA / BAAB paired ordering |
| K (whole-step iterations per window) | 8 |
| Target outer measurement window | ≈ 100 ms |
| Measured windows per side | ≥ 50 |
| CV gate | CV ≤ 0.20 |
| Outlier trimming | NONE |
| GPU clean gate | before every official run |
| Raw samples retention | full |

### 7.1 Snapshot B stability failure

Reference / Candidate 的 bimodality + drift + common-mode 触发了正式 STABILITY_FAILURE：

- `reference CV = 0.2578826165`
- `candidate CV = 0.2733676548`
- Pearson common-mode correlation = `0.9680155499`

50/50 raw samples 完整保留（`snapshot_b_l2_raw_diagnosis.json`），未做任何 trimming。

### 7.2 Snapshot C measurement repair

唯一 measurement change 是 L2 采用**冻结 K=8** outer whole-step window（约 100 ms target）。

Reference-only calibration（独立两次）：

| Calibration | target_ms | K | CV |
|---|---:|---:|---:|
| `k8_1` | 100 | 8 | `0.0501405476` |
| `k8_2` | 100 | 8 | `0.0166824822` |

两条均无明显 bimodality 或单调 drift。

`CV_THRESHOLD_CHANGED = NO`（CV gate 仍为 0.20，未为任何一方放宽）。

---

## 8. Agent Results

### 8.1 搜索过程

| Hypothesis | Attempt | Result |
|---|---:|---|
| H001 | 1 | valid |
| H001 | 2 | valid / tuning |
| H002 | 1 | correctness PASS, performance reject |
| H003 | 1 | final promoted candidate |

Mechanism family（来自既有 hypothesis.json）：

- H001：Triton 融合 + analytic 反向 + 列规约
- H002：反向 producer/consumer 融合 + FP32 atomic dweight
- H003：native CUDA / block reduction / coalesced mapping（三 kernel autograd）

### 8.2 Agent final candidate

- SHA256：`028a576d2220922e76a80bd6f1cbaff7b6054c87b7d45bd50d0ad6acc6cbaedb`
- L0：`3.6734935121x`
- L1：PASS
- Snapshot C L2：

| Run | Speedup | Reference CV | Candidate CV |
|---|---:|---:|---:|
| Run 1 | `1.1967401230x` | `0.0630404816` | `0.1613387697` |
| Run 2 | `1.1815262131x` | `0.1405883001` | `0.1586078379` |

历史 lineage（仅记录，不作为正式 Human 对照）：

- Snapshot A L2 `1.1643486648x`（旧 measurement protocol）
- Snapshot B L2 `1.1422839342x` — **STABILITY_FAILURE**（ref CV `0.2578826165`，cand CV `0.2733676548`）

---

## 9. Human v1

- Candidate SHA256：`cf404cc23b3ea4b0be6ea0e8ea18d1d27d952864bb7a9cfcdf26e81fbd18b477`

### 9.1 L0

Forward：

| Shape | Speedup |
|---|---:|
| `[16,1,1024]` | `2.818x` |
| `[64,2,1024]` | `2.823x` |
| `[128,2,1024]` | `2.856x` |

Backward：

| Shape | Speedup |
|---|---:|
| `[16,1,1024]` | `3.409x` |
| `[64,2,1024]` | `3.439x` |
| `[128,2,1024]` | `3.580x` |

### 9.2 L1 与 L2

- L1：PASS
- L2 observed：`1.2120129033294296x`
- L2 reference CV：`0.1413`
- L2 candidate CV：`0.2180`

→ **STABILITY_BLOCKED**。`1.212x` **不能**被展示为正式通过成绩。

---

## 10. Human v1 Profiler

最大 shape backward breakdown：

| Operation | Time (us) | Share |
|---|---:|---:|
| `rms_backward_kernel` | 19.20 | 57.3% |
| FP32 → FP16 copy | 7.65 | 22.8% |
| zero / fill | 6.66 | 19.9% |

NCU：

| 指标 | 值 |
|---|---|
| Main kernel duration | 21.44 us |
| Registers / thread | 28 |
| Achieved occupancy | 39.74% |
| L1 hit | 22.07% |
| L2 hit | 61.87% |
| DRAM | 5.67% of peak |
| DRAM write | 49.25 GB/s (21.94% of peak) |
| L2 sector utilization | 2.0 / 4 |
| Max-shell FP32 `atomicAdd` count | 262144 (= 256 × 1024) |

结论：

- `ATOMIC_DW_BOTTLENECK = LIKELY`
- `DW_ZERO_OVERHEAD = SIGNIFICANT`
- `DW_CAST_OVERHEAD = SIGNIFICANT`

---

## 11. Human v2 — Profiler-driven Structural Redesign

v2 是 profiler-driven structural redesign，**不是**随机 parameter sweep。

| 阶段 | v1 | v2 |
|---|---|---|
| Backward step 1 | `zeros` 占位 | Kernel A：row-wise dx |
| Backward step 2 | fused dx + atomic dweight | Kernel B：row-tiled / column-tiled FP32 partial dweight |
| Backward step 3 | FP32 → FP16 cast | Kernel C：partial reduction 直接写最终 FP16 dweight |
| Forward | row-wise reduction | 与 v1 基本一致 |

固定：`ROW_TILE = 8`，`BLOCK = 256`。

目标：

- 消除 dweight global atomics
- 消除 zero kernel
- 消除 separate cast kernel

初始 v2 compile failure 是**纯 Python 语法换行 bug**，不是新 optimization hypothesis；随后的 compile repair 只做 Python syntax repair，零算法语义变化。

Final v2 SHA256：`eee09b4fd9eec5796f8079b4ed7492c3470c08ffb2f056ca9f4aafee66e5e29e`

---

## 12. Human v2 L0

Forward：

| Shape | Speedup | Reference CV | Cand CV |
|---|---:|---:|---:|
| `[16,1,1024]` | `2.7280x` | `0.0139` | `0.0190` |
| `[64,2,1024]` | `2.7273x` | `0.0125` | `0.0146` |
| `[128,2,1024]` | `2.9805x` | `0.0105` | `0.0117` |

Backward：

| Shape | Speedup | Reference CV | Cand CV |
|---|---:|---:|---:|
| `[16,1,1024]` | `4.0535x` | `0.0966` | `0.0991` |
| `[64,2,1024]` | `3.7930x` | `0.0349` | `0.0678` |
| `[128,2,1024]` | `4.0020x` | `0.0706` | `0.1407` |

`FORWARD_CAUSAL_CONTROL = MIXED`（不声称 forward 一定改善）
`BACKWARD_IMPROVEMENT_VS_V1 = YES`（三个 shape backward 全部高于 v1）

---

## 13. Human v2 L1 / L2

- L1：PASS
- Snapshot C L2：

| Run | Speedup | Reference CV | Cand CV |
|---|---:|---:|---:|
| Run 1 | `1.1735x` | `0.1050` | `0.1405` |
| Run 2 | `1.1990x` | `0.0433` | `0.1230` |

`HUMAN_L2_STATUS = PASS`

---

## 14. Final Comparison

| System | Run 1 | Run 2 | Both stability-qualified |
|---|---:|---:|---|
| Current Agent H003 | `1.1967401230x` | `1.1815262131x` | YES |
| Human v2 | `1.1735x` | `1.1990x` | YES |

**No official aggregate winner is defined by the frozen OJ.**

- Run 1：Agent numerical result is higher。
- Run 2：Human numerical result is higher。
- 因此：`CURRENT_AGENT_AND_HUMAN_IN_SAME_PERFORMANCE_REGIME`。

---

## 15. What this does NOT prove / 这不代表什么

- 只测试 RMSNorm
- 不代表 Attention
- 不代表 SwiGLU
- 不代表 Cross Entropy
- 不代表 MoE Grouped GEMM
- 不代表所有 CUDA kernel
- 不代表 Agent 普遍等于 / 超过人类工程师
- 不代表 Nine-grid workload
- 不代表生产环境 Megatron throughput
- 不代表多 GPU / TP / PP / DP scalability

Human 也不是严格 blind：

- `HUMAN_BLIND = NO`（Human 此前知道历史 v28 lifetime idea）
- 但 `CURRENT_AGENT_SOLUTION_READ_BY_HUMAN = NO`（Human 没看 H003 source / mechanism）

因此这是 **controlled comparison**，**不是** strict double-blind comparison。

---

## 16. Reproducibility Artifacts

只列当前仓库真实存在的文件路径（相对 `D:\Users\38154\Downloads\aka-local-publish`）：

### 16.1 顶层报告 / 总表

- `docs/audits/V100_RMSNORM_AGENT_VS_HUMAN_FINAL_20260923.md`
- `docs/audits/V100_RMSNORM_ARTIFACT_INDEX_20260923.md`
- `overnight/v100_rmsnorm_e2e_20260922/FINAL_AGENT_VS_HUMAN_SUMMARY.json`

### 16.2 Agent 冻结与 Snapshot C

- `overnight/v100_rmsnorm_e2e_20260922/CURRENT_AGENT_FROZEN.json`
- `overnight/v100_rmsnorm_e2e_20260922/SNAPSHOT_C_MANIFEST.json`
- `overnight/v100_rmsnorm_e2e_20260922/snapshot_c_calibration/reference_calibration_k8_1.json`
- `overnight/v100_rmsnorm_e2e_20260922/snapshot_c_calibration/reference_calibration_k8_2.json`
- `overnight/v100_rmsnorm_e2e_20260922/snapshot_c_requalification_run1.json`
- `overnight/v100_rmsnorm_e2e_20260922/snapshot_c_requalification_run2.json`

### 16.3 Stability repair

- `docs/audits/V100_RMSNORM_L2_STABILITY_DIAGNOSIS.md`
- `docs/audits/V100_RMSNORM_L2_STABILITY_REPAIR.md`
- `docs/audits/V100_RMSNORM_OJ_REPAIR_AND_SNAPSHOT_B.md`
- `overnight/v100_rmsnorm_e2e_20260922/snapshot_b_l2_raw_diagnosis.json`

### 16.4 Human v1

- `docs/audits/V100_RMSNORM_HUMAN_ATTEMPT_01.md`
- `docs/audits/V100_RMSNORM_HUMAN_V1_PROFILE.md`
- `overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_01/HUMAN_ATTEMPT_01_SUMMARY.json`
- `overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_01/profile/PROFILE_SUMMARY.json`

### 16.5 Human v2

- `docs/audits/V100_RMSNORM_HUMAN_ATTEMPT_02.md`
- `docs/audits/V100_RMSNORM_HUMAN_ATTEMPT_02_REPAIR_01.md`
- `overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_02/HUMAN_ATTEMPT_02_SUMMARY.json`
- `overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_02_repair_01/HUMAN_ATTEMPT_02_REPAIR_01_SUMMARY.json`
- `overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_02_repair_01/official_l2_run1.json`
- `overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_02_repair_01/official_l2_run2.json`

### 16.6 Performance overview

- `docs/PERFORMANCE_OVERVIEW.md`

---

## 17. 总结

- 本次实验是 **V100 RMSNorm** 专项快照。
- Agent 走 `facts / hypothesis / precondition / attempt / OJ` 闭环，独立得到 H003。
- Human 走 `correctness → stability block → profiler → redesign → compile repair → PASS`。
- 两者在冻结 Snapshot C + K=8 + 50 windows/side + CV ≤ 0.20 protocol 下达到相同性能量级。
- 没有为真人、不读为假人、不跨多 GPU、不跨其他 workload。
