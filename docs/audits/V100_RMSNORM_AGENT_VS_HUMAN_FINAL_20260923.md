# V100 RMSNorm Agent vs Human 最终实验报告

> **范围**：2026-09-22 夜间 ~ 2026-09-23 Human v2 完成
> **主题**：V100 RMSNorm · Current Agent vs Human · Controlled Megatron E2E
> **NINE_GRID_E2E = NO**（本实验不是九格真实训练 workload）

## 1. 实验目标

在 V100 上，用同一份 frozen Controlled Megatron E2E 协议对照两条独立路径：

1. **Current Agent**：使用 `facts / hypothesis / precondition / attempt / OJ` 闭环，独立搜索 RMSNorm 实现，最终锁定 H003。
2. **Human**：手动设计 v1 → profiler → 结构化 v2 redesign → compile-only repair。

目标不是为某一方"宣称"胜出，而是验证：在冻结 OJ 与 measurement protocol 下，两条独立路径是否收敛到同一性能量级。

## 2. 硬件与 Frozen Snapshot

- 设备：Tesla V100-PCIE-16GB
- GPU UUID：`GPU-88be8e63-dd61-3d8f-2f44-e1f93204654d`
- 驱动：535.183.06
- Megatron commit：`5be9626709af2722333bf54797c954c09edeada3`（目录 `megatron-lm-5be9626`）
- Contract：`v2`，SHA256 `1042caf509cb37c2defe2c230cd6e0b38ef8e784268ba0cf1de29c9893505f97`

Benchmark lineage：

| Snapshot | 用途 |
|----------|------|
| V100_RMSNORM_E2E_SNAPSHOT_A | 首版 OJ，固化 L0/L1/L2 协议与 agent 历史 score |
| V100_RMSNORM_E2E_SNAPSHOT_B | Snapshot A 的延续；50/50 raw samples 显示 bimodal/drift/common-mode |
| V100_RMSNORM_E2E_SNAPSHOT_C | 唯一 measurement change：L2 使用冻结 K=8 outer whole-step window（约 100 ms target） |

## 3. OJ 层级

- **L0**：Candidate 与 `torch.nn.RMSNorm` reference 的 forward / dx / dw 比对；50 windows/side，FP16，eps=1e-5。
- **L1**：真实 Megatron GPTModel 替换 5 个 `torch.nn.RMSNorm`；剩余 reference RMSNorm 必须为 0；invocation > 0；loss_abs ≤ 0.005；norm_grad_max_abs ≤ 0.005；gradients finite。
- **L2**：Controlled Megatron E2E，K=8 whole-step CUDA Event，A/B 同 process，ABBA/BAAB，≥ 50 windows/side，CV ≤ 0.20。

## 4. Current Agent 搜索过程

Current Agent 在不知 historical sources、不读 Human v1/v2 source 的前提下，独立构造了三个 hypothesis family：

| Hypothesis | Mechanism family | 出发点 |
|------------|------------------|--------|
| H001 | Triton 融合 + analytic 反向 + 列规约 | source-derived from current agent |
| H002 | 反向 producer/consumer 融合 + FP32 atomic dweight | derived from H001 profile-free dataflow review |
| H003 | 原生 CUDA block reduction + 三 kernel autograd | current agent mapping reformulation |

每个 hypothesis 都在 `hypotheses/Hxxx/attempts/{attempt}/` 下保留 `candidate.py` / `decision.json` / `feedback.json`。

## 5. Agent H003 最终结果

- **Candidate SHA256**：`028a576d2220922e76a80bd6f1cbaff7b6054c87b7d45bd50d0ad6acc6cbaedb`
- **L0**：3.6734935121x（官方 3 shapes，50 windows/side）
- **L1**：PASS（replacements=5，remaining reference RMSNorm=0，loss_abs=3.0e-5）
- **Snapshot A L2**：1.1643486648x（历史旧 measurement protocol，仅作 lineage 保留，不参与 Human 对照）
- **Snapshot B L2**：1.1422839342x — reference CV=0.2578826165，candidate CV=0.2733676548 → **STABILITY_FAILURE**
- **Snapshot C L2**（正式 Agent E2E 数据）：

| Run | Speedup | reference CV | candidate CV |
|-----|---------|--------------|--------------|
| Run 1 | 1.1967401230x | 0.0630404816 | 0.1613387697 |
| Run 2 | 1.1815262131x | 0.1405883001 | 0.1586078379 |

`H003_REQUALIFICATION = PASS`。

## 6. Snapshot B 稳定性问题

Snapshot B 的 50/50 raw samples 显示：

- Pearson correlation（reference vs candidate）：**0.9680155499**（强 common-mode）
- 模式分类：**BIMODAL + DRIFT + COMMON_MODE**
- reference CV：0.2578826165
- candidate CV：0.2733676548
- 原始 samples 完整保留在 `snapshot_b_l2_raw_diagnosis.json`，未做 trimming。

按官方 CV ≤ 0.20 gate，Snapshot B 正式 L2 失败是 fail-closed stability failure，并非性能回归。

## 7. Snapshot C measurement repair

在不修改 H003 source、megatron、shape、dtype、CV gate 的前提下，仅对 measurement protocol 做一处修改：

- L2 改用冻结 K=8 outer whole-step window（target ≈ 100 ms）
- 使用两次独立 reference-only calibration（无 trim）：

| Calibration | target_ms | K | CV |
|-------------|-----------|---|----|
| k8_1 | 100 | 8 | 0.0501405476 |
| k8_2 | 100 | 8 | 0.0166824822 |

CV threshold 仍为 0.20，未为 H003 放宽。

## 8. Human v1

- **Candidate SHA256**：`cf404cc23b3ea4b0be6ea0e8ea18d1d27d952864bb7a9cfcdf26e81fbd18b477`
- 编译：PASS
- Quick L0 / Official correctness：PASS

Forward：

| Shape | Speedup | ref CV | cand CV |
|-------|---------|--------|---------|
| [16,1,1024] | 2.818x | 0.076 | 0.0979 |
| [64,2,1024] | 2.823x | 0.0533 | 0.0614 |
| [128,2,1024] | 2.856x | 0.0158 | 0.0253 |

Backward：

| Shape | Speedup |
|-------|---------|
| [16,1,1024] | 3.409x |
| [64,2,1024] | 3.439x |
| [128,2,1024] | 3.580x |

- L1：PASS
- L2 observed：**1.212x**，reference CV=0.1413，candidate CV=0.2180 → **`HUMAN_V1_L2 = STABILITY_BLOCKED`**

1.212x 不作为 Human 正式成绩。

## 9. Human v1 profiler

最大 shape backward breakdown：

| Kernel | Duration | Percent |
|--------|----------|---------|
| rms_backward_kernel | 19.20 us | 57.31% |
| FP32→FP16 cast (float16_copy_kernel) | 7.65 us | 22.83% |
| FillFunctor zero_memset | 6.66 us | 19.86% |

NCU 主 kernel：21.44 us；registers/thread=28；occupancy=39.74%；L1 hit=22.07%；L2 hit=61.87%；DRAM=5.67% of peak；DRAM write 49.25 GB/s（21.94% peak）；L2 sector utilization=2.0/4。

结论：

- `ATOMIC_DW_BOTTLENECK = LIKELY`（最大 shape 理论 FP32 atomicAdd = 256 × 1024 = 262144）
- `DW_ZERO_OVERHEAD = SIGNIFICANT`
- `DW_CAST_OVERHEAD = SIGNIFICANT`
- `CANDIDATE_VARIABILITY_SOURCE = MAIN_KERNEL`

V2 priority：
1. REDUCE_ATOMIC_DW
2. REMOVE_TEMP_ZERO_ALLOC
3. REMOVE_DW_CAST

## 10. Human v2 redesign

基于 profiler，Human v2 是一次 structural redesign（非参数搜索）：

| 阶段 | v1 | v2 |
|------|----|----|
| Forward | row-wise reduction | 与 v1 基本一致 |
| Backward step 1 | `zeros` 占位 | Kernel A：row-wise dx |
| Backward step 2 | fused dx + atomic dweight | Kernel B：row-tiled / column-tiled FP32 partial dweight |
| Backward step 3 | FP32 → FP16 cast | Kernel C：partial reduction 直接写最终 FP16 dweight |

固定：`ROW_TILE = 8`，`BLOCK = 256`。没有做参数搜索。

初始 Human v2 因 Python 语法（`x, weight, inv_rms =` 后非法换行）导致 COMPILE_FAILURE。
Compile repair 发现共 6 处同种非法换行，全部仅做 Python syntax repair，零算法语义变化。这仍属于 Human v2 implementation compile repair，不是 Human v3。

## 11. Human v2 最终结果

- **Repaired Candidate SHA256**：`eee09b4fd9eec5796f8079b4ed7492c3470c08ffb2f056ca9f4aafee66e5e29e`
- Compile：PASS；Quick L0：PASS；Official L0 correctness：PASS
- L1：PASS

Forward：

| Shape | Speedup | ref CV | cand CV |
|-------|---------|--------|---------|
| [16,1,1024] | 2.7280x | 0.0139 | 0.0190 |
| [64,2,1024] | 2.7273x | 0.0125 | 0.0146 |
| [128,2,1024] | 2.9805x | 0.0105 | 0.0117 |

Backward：

| Shape | Speedup | ref CV | cand CV |
|-------|---------|--------|---------|
| [16,1,1024] | 4.0535x | 0.0966 | 0.0991 |
| [64,2,1024] | 3.7930x | 0.0349 | 0.0678 |
| [128,2,1024] | 4.0020x | 0.0706 | 0.1407 |

`FORWARD_CAUSAL_CONTROL = MIXED`（两个 shape 略降，一个 shape 提升）。
`BACKWARD_IMPROVEMENT_VS_V1 = YES`（三个 shape 全部提升）。

Snapshot C L2：

| Run | Speedup | reference CV | candidate CV |
|-----|---------|--------------|--------------|
| Run 1 | 1.1735x | 0.1050 | 0.1405 |
| Run 2 | 1.1990x | 0.0433 | 0.1230 |

`HUMAN_L2_STATUS = PASS`。

## 12. Agent vs Human

| System | Run 1 | Run 2 | Status |
|--------|-------|-------|--------|
| Agent H003 | 1.1967401230x | 1.1815262131x | PASS |
| Human v2 | 1.1735x | 1.1990x | PASS |

Run1 Agent 数值较高；Run2 Human 数值较高。两者处于同一性能量级。

**No official aggregate winner is defined**（frozen OJ 未定义跨两个 independent runs 的 aggregate winner）。
Descriptive-only：

- mean(H003 Run1, Run2) = 1.18913x
- mean(Human v2 Run1, Run2) = 1.18625x

标 `DESCRIPTIVE_ONLY` / `NOT_AN_OFFICIAL_SCORE`。

## 13. 机制与工程过程比较

| 维度 | Current Agent (H003) | Human (v2) |
|------|----------------------|------------|
| 决策循环 | facts / hypothesis / precondition / attempt / OJ | 手动 profiler + redesign |
| 实现语言 | 原生 CUDA + nvcc | 原生 CUDA + nvcc |
| 起点 | Source-derived from current agent | profiler-driven from v1 |
| 视角 | 重写 mapping，降低 compiler/register 压力 | 移除 atomic + 移除 zero + 移除 cast |
| 优化目标 | dweight 合并 tile / coalesced | row-tile / column-tile + partial reduction 直接出 FP16 |
| 后台 OJ 透明度 | `HUMAN_BLIND = NO`（Human 此前知道 historical v28 lifetime idea），但 `AGENT_CURRENT_SOLUTION_READ_BY_HUMAN = NO` |

## 14. 可信结论

- 在 frozen Controlled Megatron E2E + Snapshot C measurement protocol 下，Current Agent (H003) 与 Human (v2) 均通过 L0/L1/L2 correctness 与 stability gate。
- 两者 E2E speedup 处于同一数量级。
- 该结论支持：Current Agent 的 `facts / hypothesis / precondition / attempt / OJ` 闭环在 RMSNorm standalone 上独立达到人类 profiler-driven 重设计的系统性能量级。

## 15. 不能声称的结论

- 不能宣布 `AGENT_WIN` 或 `HUMAN_WIN`（frozen OJ 未定义 aggregate winner）。
- 不能将 1.212x（Human v1）作为 Human 正式成绩（其 L2 STABILITY_BLOCKED）。
- 不能将 Snapshot A L2 1.1643486648x 作为 Agent 最终成绩（旧 measurement protocol）。
- 不能声称 strict double-blind comparison（Human 此前知道历史 v28 lifetime idea）。

## 16. 实验限制

- NINE_GRID_E2E = NO（不是九格真实训练 workload）。
- 仅 V100 单卡，sm70。H003 / Human v2 在其他架构上未验证。
- 协议是 frozen Controlled Megatron E2E，未引入多卡 / FSDP / 真实 large-scale 训练分布。
- Human v2 仅 NUM_WARPS=4 official shot；快速 L0 中 4 与 8 双 shot（不在协议边界）。
- CV gate = 0.20 未为任一方放宽。

## 17. Reproducibility / Artifact Index

详见 `docs/audits/V100_RMSNORM_ARTIFACT_INDEX_20260923.md`。

机器可读总表：`overnight/v100_rmsnorm_e2e_20260922/FINAL_AGENT_VS_HUMAN_SUMMARY.json`。
