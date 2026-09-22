# 性能结果概览

本文汇总 `aka-local` 当前最重要的性能结果，并链接到可追溯的原始 benchmark、机器可读汇总和 Phase 报告。不同结果的证据等级不同：不能把所有 speedup 都当作正式性能结论。

除非某个 artifact 另有说明，以下 Megatron 性能实验主要来自 Tesla V100-PCIE-16GB、`sm_70`、CUDA 11.8、PyTorch `2.7.1+cu118` 环境。这些结果不能直接外推到 RTX 5060、A100、H100 或其他 CUDA 架构。

## 证据等级

### `PROMOTED_SCORE`

已通过正确性、冻结的 benchmark scope、reference/candidate stability、统计门禁和 promotion evidence，并成为正式 incumbent 的结果。

### `STABILITY_QUALIFIED_SCORE`

已通过统计稳定性测试，但因 per-config regression、promotion rule、profile gate 或其他明确原因，没有成为正式 incumbent。

### `RAW_RATIO`

完整或部分 benchmark 产生的原始比值。若 stability gate 未通过，它不能作为正式性能结论。

### `DIAGNOSTIC_ONLY`

NSYS、microbenchmark、kernel latency、Amdahl upper bound 等定位数据。它们用于解释瓶颈，不能当作 end-to-end speedup。

### `LOCAL_NON_REPRESENTATIVE`

真实本地 runtime 上已经执行，但 shape、硬件或 scope 不能代表目标九格 workload。此等级可用于 correctness、integration 和诊断，不能外推为生产性能。

### `MEASUREMENT_INCONCLUSIVE`

测量已经执行，但关键归因证据不足，不能支持结构性优化结论。

### `REFUTED_FOR_LOCAL_WORKLOAD`

在明确冻结的本地 workload 上，某条 causal story 已被证据拒绝。这个结论只约束该 workload，不外推到所有 shape 或所有硬件。

## 历史 V100 campaign 性能证据

本节保留历史 V100 campaign 的正式终态、raw ratio 和 diagnostic evidence；这些数字不代表当前 RTX 5060 本地诊断，也不代表九格性能。

## 顶部总览

| 真实目标 | 当前最佳性能证据 | 证据等级 | 当前状态 | 主要数据位置 |
|---|---|---|---|---|
| SwiGLU activation | sidecar forward `2.799422x`；sidecar backward `3.457830x` | `RAW_RATIO` | `COMPLETED_INTEGRATION_LIMITED` | [forward incumbent](../campaigns/targets/megatron_5be9626/swiglu/incumbent_manifest.json)、[backward incumbent](../campaigns/targets/megatron_5be9626/swiglu_backward/incumbent_manifest.json) |
| Vocab-Parallel CE forward | Episode 7：`2.477191x`，CI95 `[2.465431x, 2.484392x]` | `PROMOTED_SCORE` | `FORWARD_CAMPAIGN_FROZEN` | [incumbent manifest](../campaigns/targets/megatron_5be9626/vocab_parallel_cross_entropy/incumbent_manifest.json) |
| Vocab-Parallel CE backward | B1 raw geometric mean `2.557171x` | `RAW_RATIO` | `ENVIRONMENT_STABILITY_BLOCKED` | [B1 summary](../campaigns/targets/megatron_5be9626/vocab_parallel_cross_entropy_backward/episode_B1/paired_benchmark_summary.json) |
| Torch RMSNorm | R1 raw `1.64629x`；R2 raw `1.64134x` | `RAW_RATIO` | `STABILITY_BLOCKED_R1_R2` | [R1 raw benchmark](../campaigns/targets/megatron_5be9626/residual_rmsnorm/episode_R1/paired_benchmark_raw.json) |
| Native DotProductAttention | A2：`5.928x` / `1.167x` / `0.347x`；aggregate `1.338883x` | `STABILITY_QUALIFIED_SCORE` | `AGGREGATE_WIN_PER_CONFIG_MIXED` | [A2 summary](../campaigns/targets/megatron_5be9626/megatron_native_dot_product_attention/episode_A2/paired_summary.json) |
| Native SequentialMLP expert compute | M4 raw：`1.2793x` / `1.4816x` / `0.9573x` | `RAW_RATIO` | `STABILITY_BLOCKED` | [M4 raw benchmark](../campaigns/targets/megatron_5be9626/moe_native_sequential_expert_compute/episode_M4/paired_benchmark_raw.json) |

## 怎么看这些结果

- `PROMOTED_SCORE` 是目前最严格、可作为正式 performance score 的结果。
- `RAW_RATIO` 表示“看起来可能更快”，但当前证据不足以形成正式结论。
- `STABILITY_BLOCKED` 表示正确性可以通过，但 benchmark 波动超过预先冻结的门限。
- `PER_CONFIG_MIXED` 表示不同 shape 的表现差异明显，不能用一个 aggregate 平均数掩盖 regression。

## 当前最重要的结论

1. CE forward Episode 7 是目前证据最完整的正式优化结果。
2. SwiGLU 的局部 activation kernel 有显著收益，但完整 MLP training path 受 GEMM、autograd 和 integration/allocation overhead 限制。
3. full-core custom Attention 在小 `S` 很强，但在 `S=128` 明显失去 vendor GEMM 优势。
4. CE backward、RMSNorm 与 MoE 都遇到冻结稳定性门禁；不能把它们的 raw ratio 写成 promoted score。
5. NSYS/profile 的 kernel 观察与正式 benchmark scope 是不同证据，必须分开解释。

## SwiGLU activation

### 当前结果与范围

当前保存的最佳 valid sidecar candidate 是 Episode 2：activation-only forward geometric mean 为 `2.7994219276258265x`，activation-only backward geometric mean 为 `3.4578303034730578x`。这两个数字只描述保存的 sidecar activation boundary，不是完整 Megatron MLP speedup，也没有 Megatron patch。

Phase 14-F 的完整 TP=1 MLP forward + backward integration rerun 为 reference `2133.61 µs`、integrated `2352.13 µs`，即 `0.90710x`。同一报告还将 `[128,2,1024]` 的 isolated forward/backward component latency 与完整 graph timing 分开列出；局部 kernel 加速不能直接相加或改写为 full-training 加速。

### 数据位置

#### 原始 benchmark

- [forward Episode 2 result](../campaigns/targets/megatron_5be9626/swiglu/episode_2/result.json)
- [backward Episode 2 result](../campaigns/targets/megatron_5be9626/swiglu_backward/episode_2/result.json)
- [integration remote result](../targets/megatron_5be9626/swiglu/remote_result.json)

#### 汇总结果

- [forward incumbent manifest](../campaigns/targets/megatron_5be9626/swiglu/incumbent_manifest.json)
- [backward incumbent manifest](../campaigns/targets/megatron_5be9626/swiglu_backward/incumbent_manifest.json)
- [integration contract](../targets/megatron_5be9626/swiglu/integration_contract.json)

#### NSYS / profile

- [SwiGLU NSYS text summary](../targets/megatron_5be9626/swiglu/nsys_stats.txt)

### 进一步阅读

- [Phase 14-C：SwiGLU campaign](reports/phase-14/PHASE14C_MEGATRON_SWIGLU_AGENT_CAMPAIGN.md)
- [Phase 14-E：SwiGLU backward](reports/phase-14/PHASE14E_MEGATRON_SWIGLU_BACKWARD.md)
- [Phase 14-F：training overhead analysis](reports/phase-14/PHASE14F_SWIGLU_TRAINING_OVERHEAD_ANALYSIS.md)

## Vocab-Parallel Cross Entropy forward

### 正式性能结论

Episode 7 是正式 incumbent，稳定 TP=2 geometric-mean speedup 为 `2.4771907400291515x`，bootstrap CI95 为 `[2.465431157986491x, 2.484391708222999x]`。其 collective structure 为 1 次 `MAX` all-reduce 与 2 次 `SUM` all-reduce；campaign 已冻结为 `FORWARD_CAMPAIGN_FROZEN`。这是本仓库当前唯一 `PROMOTED_SCORE`。

### 数据位置

#### 原始 benchmark

- [Episode 7 incumbent manifest](../campaigns/targets/megatron_5be9626/vocab_parallel_cross_entropy/incumbent_manifest.json)
- [final replay](../targets/megatron_5be9626/vocab_parallel_cross_entropy/final_replay_15b8.json)
- [Episode 7 stability analysis](../targets/megatron_5be9626/vocab_parallel_cross_entropy/stability_analysis_15b7_ep7.json)

#### 汇总结果

- [Episode 7 incumbent manifest](../campaigns/targets/megatron_5be9626/vocab_parallel_cross_entropy/incumbent_manifest.json)
- [campaign state 15-B.8](../targets/megatron_5be9626/vocab_parallel_cross_entropy/campaign_state_15b8.json)

#### NSYS / profile

- [promotion NSYS summary](../targets/megatron_5be9626/vocab_parallel_cross_entropy/promotion_nsys_summary_15b8.json)

### 进一步阅读

- [Phase 15-B.8：CE forward campaign closure](reports/phase-15/PHASE15B8_CE_FORWARD_CAMPAIGN_CLOSURE.md)

## Vocab-Parallel Cross Entropy backward

### 当前证据：raw ratio，不是正式 speedup

B1 candidate hash 为 `008155d13eda72ab1feb267a06d360a13ad2730128a4aca4cfe8a3af70e374ab`。其 official rank-local backward boundary（saved state 到 local logits gradient，timed range 内 0 collective）给出了 raw geometric mean `2.5571710921067x`。

原始 B1 的 candidate CV 为 `0.107`、`0.167`、`0.185`，但 reference rank 0 在 `8x1x32` 和 `32x2x64` 的 CV 分别为 `0.209` 与 `0.223`，超过冻结的 `0.20` 门限。后续稳定性恢复中又出现 candidate rank 0 于 `32x2x64` 的 `0.20803`。因此终态是 `ENVIRONMENT_STABILITY_BLOCKED`；raw ratio 不能作为 promotion score，也没有 CI 或 incumbent。

### 数据位置

#### 原始 benchmark

- [B1 paired raw benchmark](../campaigns/targets/megatron_5be9626/vocab_parallel_cross_entropy_backward/episode_B1/paired_benchmark_raw.json)
- [B1 paired benchmark summary](../campaigns/targets/megatron_5be9626/vocab_parallel_cross_entropy_backward/episode_B1/paired_benchmark_summary.json)

#### 汇总结果

- [B1 result](../campaigns/targets/megatron_5be9626/vocab_parallel_cross_entropy_backward/episode_B1/result.json)
- [backward optimization contract](../targets/megatron_5be9626/vocab_parallel_cross_entropy/backward_optimization_contract.json)

#### NSYS / profile

- [backward NSYS summary](../targets/megatron_5be9626/vocab_parallel_cross_entropy/backward_nsys_summary.json)（诊断，不是 promotion evidence）

### 进一步阅读

- [Phase 15-C.2：CE backward B1](reports/phase-15/PHASE15C2_CE_BACKWARD_AGENT_CAMPAIGN.md)
- [Phase 15-C.3：CE backward stability recovery](reports/phase-15/PHASE15C3_CE_BACKWARD_STABILITY_RECOVERY.md)

## Torch RMSNorm

### 当前证据：raw ratio，不是正式 speedup

真实目标是 non-TE `WrappedTorchNorm -> torch.nn.RMSNorm` forward，而不是历史名称“Residual Add RMSNorm”。R1 raw geometric mean 为 `1.64629x`，R2 为 `1.64134x`。两者都在 `64x2x1024` 触发 candidate CV 门禁：R1 为 `0.2174`，R2 为 `0.2063`，超过 `CV <= 0.20`。

因此状态为 `STABILITY_BLOCKED_R1_R2`。没有 CI、promotion score 或 incumbent；“RMSNorm 加速 1.64x”不是允许的正式表述。

### 数据位置

#### 原始 benchmark

- [R1 paired raw benchmark](../campaigns/targets/megatron_5be9626/residual_rmsnorm/episode_R1/paired_benchmark_raw.json)
- [R2 paired raw benchmark](../campaigns/targets/megatron_5be9626/residual_rmsnorm/episode_R2/paired_benchmark_raw.json)

#### 汇总结果

- [baseline manifest](../targets/megatron_5be9626/residual_rmsnorm/baseline_manifest.json)
- [R1 result](../campaigns/targets/megatron_5be9626/residual_rmsnorm/episode_R1/result.json)
- [R2 result](../campaigns/targets/megatron_5be9626/residual_rmsnorm/episode_R2/result.json)

#### NSYS / profile

- [Phase 16-A NSYS summary](../targets/megatron_5be9626/residual_rmsnorm/phase16a_nsys_summary.json)（reference profile；没有 candidate promotion NSYS）

### 进一步阅读

- [Phase 16-B：Torch RMSNorm campaign](reports/phase-16/PHASE16B_MEGATRON_TORCH_RMSNORM_AGENT_CAMPAIGN.md)

## Native DotProductAttention

### Phase 17 full-core A2：稳定性合格，但 per-config mixed

A2 通过 correctness 与 stability。官方 per-config speedup 分别为：`S=16` `5.928x`、`S=64` `1.167x`、`S=128` `0.347x`；aggregate geometric mean 为 `1.338883x`，CI95 为 `[1.307531x, 1.368066x]`。

`S=128` 的 `0.347x` 表示约为 reference 的 `0.347x`，即明显变慢。最终状态 `AGGREGATE_WIN_PER_CONFIG_MIXED`，没有 unconditional incumbent。因此它属于 `STABILITY_QUALIFIED_SCORE`，必须同时标出 `PER_CONFIG_MIXED`，不能简写为“Attention 加速 1.34x”。

### Phase 20 vendor-GEMM-preserving：仅诊断

Phase 20-A/20-B.0 没有 candidate score。保留的 non-GEMM local GPU graph 时间为 `45.536–80.928 µs`；在 event scope 内，即使消除所有 local kernel 的上界也仅为 `1.055x–1.099x`。这是 `THEORETICAL UPPER BOUND` / `DIAGNOSTIC_ONLY`，不是 speedup claim。

### 数据位置

#### 原始 benchmark

- [A2 paired raw benchmark](../campaigns/targets/megatron_5be9626/megatron_native_dot_product_attention/episode_A2/paired_raw.json)
- [A2 paired summary](../campaigns/targets/megatron_5be9626/megatron_native_dot_product_attention/episode_A2/paired_summary.json)

#### 汇总结果

- [A2 paired summary](../campaigns/targets/megatron_5be9626/megatron_native_dot_product_attention/episode_A2/paired_summary.json)
- [Phase 20-A kernel decomposition](../targets/megatron_5be9626/megatron_native_dot_product_attention/phase20a_kernel_decomposition.json)
- [Phase 20-B.0 timing diagnostic](../targets/megatron_5be9626/megatron_native_dot_product_attention/phase20b0_timing_diagnostic.json)

#### NSYS / profile

- [A2 NSYS raw summary](../campaigns/targets/megatron_5be9626/megatron_native_dot_product_attention/episode_A2/nsys_raw.json)
- [Phase 20-A hybrid NSYS summary](../targets/megatron_5be9626/megatron_native_dot_product_attention/phase20a_hybrid_nsys.json)

### 进一步阅读

- [Phase 17-B.1：A2 continuation](reports/phase-17/PHASE17B1_ATTENTION_TOOLCHAIN_RECOVERY_AND_CONTINUATION.md)
- [Phase 17-B.2：profile-guided closure](reports/phase-17/PHASE17B2_ATTENTION_PROFILE_GUIDED_CLOSURE.md)
- [Phase 20-A：vendor-GEMM-preserving qualification](reports/phase-20/PHASE20A_ATTENTION_VENDOR_GEMM_PRESERVING_QUALIFICATION.md)
- [Phase 20-B.0：scoring-era qualification](reports/phase-20/PHASE20B0_ATTENTION_SCORING_ERA_QUALIFICATION.md)

## Native SequentialMLP expert compute

### 当前证据：raw ratio，不是正式 speedup

真实目标是 Megatron Native `SequentialMLP` expert compute，不是实际执行的 Transformer Engine Grouped GEMM。M4 correctness 为 `PASS`，三个 distribution 的 raw ratio 分别为：balanced `[4,4,4,4]` `1.2793x`、moderate imbalance `[1,3,5,7]` `1.4816x`、two-empty `[0,0,8,8]` `0.9573x`。

two-empty 的 reference CV 为 `0.3733`，超过冻结门限 `0.20`。故状态保持 `STABILITY_BLOCKED`；没有 aggregate promoted score、CI 或 incumbent。`0.9573x` 没有加速，不能只展示前两个较好的 distribution。

### NSYS 诊断

`DIAGNOSTIC_ONLY` 的 NSYS summary 显示 reference kernel count 为 `13 / 13 / 7`，而 M4 为 `1 / 1 / 1` 个 custom kernel；M4 kernel duration 约为 `1.645 ms`、`1.650 ms`、`1.596 ms`。这些数据帮助解释 kernel graph，不是正式 operator benchmark 结论。

### 数据位置

#### 原始 benchmark

- [M4 paired raw benchmark](../campaigns/targets/megatron_5be9626/moe_native_sequential_expert_compute/episode_M4/paired_benchmark_raw.json)

#### 汇总结果

- [M4 decision](../campaigns/targets/megatron_5be9626/moe_native_sequential_expert_compute/episode_M4/decision.json)
- [target qualification manifest](../targets/megatron_5be9626/moe_native_sequential_expert_compute/qualification_manifest.json)

#### NSYS / profile

- [M4 diagnostic NSYS summary](../campaigns/targets/megatron_5be9626/moe_native_sequential_expert_compute/episode_M4/diagnostic_nsys_summary.json)

### 进一步阅读

- [Phase 18-B.1：MoE delivery hardening](reports/phase-18/PHASE18B1_MOE_AGENT_DELIVERY_HARDENING.md)
- [Phase 19-A：五目标收尾](reports/phase-19/PHASE19A_FIVE_TARGET_CAMPAIGN_CLOSURE.md)

## 全局状态入口

- [五个真实 Megatron target 的实验汇总](../FIVE_TARGET_REAL_MEGATRON_CAMPAIGN_SUMMARY.md)
- [真实 target campaign 索引](../REAL_TARGET_CAMPAIGN_INDEX.md)
- [机器可读五目标状态](../five_target_campaign_state.json)

## RTX 5060 本地真实 Megatron 诊断

本节记录 2026-09-22 的本地真实 Megatron 证据。它不是代表性九格性能结果。

### 环境与冻结 workload

环境来自 [timing attribution](../artifacts/integration/swiglu/real_loop_003/timing_attribution.json) 和 [shape provenance](../artifacts/integration/swiglu/real_loop_004/nine_grid_shape_provenance.json)：

| 项目 | 值 |
|---|---|
| Python | `C:\Users\38154\.venvs\urban6-stgcn\Scripts\python.exe` |
| PyTorch | `2.11.0+cu128` |
| CUDA runtime | `12.8` |
| GPU | `NVIDIA GeForce RTX 5060 Laptop GPU` |
| Megatron commit | `5be9626709af2722333bf54797c954c09edeada3` |
| batch / sequence / hidden | `2 / 3 / 8` |
| FFN hidden | `16` |
| dtype | `float32` |
| TP / SP | `1 / false` |

真实路径为 `MLP.forward → linear_fc1 → bias_swiglu_impl → BiasSwiGLUFunction → linear_fc2`。该 workload 用于 correctness、integration 和物理测量，不是 production representative shape。

### 权威 timing reconciliation

权威原始 artifact 为 [batched_swiglu_timing.json](../artifacts/integration/swiglu/real_loop_003/batched_swiglu_timing.json)，reconciliation 记录在 [timing_evidence_reconciliation.json](../artifacts/integration/swiglu/real_loop_004/timing_evidence_reconciliation.json)。协议是外层 CUDA Event、`K=512`、20 个 warmup batches、50 个 measured batches；中位 batch window 为 `27.518064498901367 ms`，权威 amortized authentic callable 为 `53.74621972441673 us`。

历史 `55.63945323228836 us` 来自早期 `K=8192` diagnostic batch，不是当前权威值；它只保留在 reconciliation 的历史来源说明中。

三种 timing 不是同一测量对象：

| 对象 | 值 | 证据等级 |
|---|---:|---|
| NCU active SwiGLU kernel duration | `3.648 us` | `DIAGNOSTIC_ONLY` |
| batched authentic callable | `53.74621972441673 us` | `LOCAL_NON_REPRESENTATIVE` |
| old inner CUDA Event region | `72.512 us` | `DIAGNOSTIC_ONLY`，已知受扰动 |
| synchronized Python wall | `65.10000093840063 us` median | `DIAGNOSTIC_ONLY` |

NCU active kernel duration、authentic callable GPU timeline、inner Event region 和 host wall time 测量的是不同层级。结论不是“CUDA Event 不准”，而是微秒级 kernel 上 inner Event instrumentation 会改变路径。

### instrumentation perturbation

Loop 002 artifact 显示 uninstrumented whole MLP median 为 `0.19195199757814407 ms`，instrumented median 为 `0.2633120119571686 ms`，扰动约 `+37.18%`。因此 `EVENT_INSTRUMENTATION = PERTURBING`；受 inner Event 影响的 paired fraction 不能作为无扰动模型占比。

### local model fraction 与 Amdahl 上界

robust fraction audit 的 split estimator 为 `0.015571`，95% CI `[0.014957, 0.015796]`；paired estimator 为 `0.041957`，95% CI `[0.041793, 0.042108]`，后者受 instrumentation 影响。公开主口径采用 split estimator：`LOCAL_MINIMAL_GPTMODEL` 中 SwiGLU 约占 `1.56%`。

对应的无限加速上界约为 `1.015818x`；若 SwiGLU 加速为 `2x`，理论 local-model 上界约为 `1.007847x`。这不是九格模型结论。

### 物理 profiler 数据

NCU 记录的 authentic SwiGLU kernel 为：`16 registers/thread`、动态/静态 shared memory 均为 `0 B`、grid 为 `1 block`、block 为 `128 threads`、设备为 `26 SM`、achieved occupancy 为 `7.25%`、kernel-total DRAM read 为 `8448 B`、DRAM write 为 `0 B`。`8448 B` 是 kernel-total read，不是 activation-only bytes。单 block 在 26-SM GPU 上支持 `GRID_UNDERSUBSCRIBED`，不能把低 achieved occupancy 解释成 register pressure。

### activation、cache-pressure 与因果结论

真实 activation shape 为 `[3, 2, 16]`，`96` 个 float32 元素，payload 为 `384 B`；详见 [measurement_facts_003.json](../artifacts/integration/swiglu/real_loop_003/measurement_facts_003.json)。FC2 baseline 的 total DRAM read 为 `25088 B`、L2 request 为 `82912 B`；64 MiB pressure buffer（设备 L2 为 32 MiB）后分别为 `24832 B` 和 `82912 B`，FC2 duration 为 `6.368 us` / `6.304 us`。这些是 FC2 total traffic，不能直接当 activation-only traffic。

因此层级结论必须分开保存：`GRAPH_VALUE_EXISTS = YES`、`DEVICE_TENSOR_MATERIALIZED = YES`、`SEPARATE_KERNELS = YES`、`CACHE_RESIDENCY = INCONCLUSIVE`、`DRAM_ROUND_TRIP = UNKNOWN`。当前 tiny local workload 上，“存在值得消除的 SwiGLU→FC2 activation HBM round-trip”被标为 `REFUTED_FOR_LOCAL_WORKLOAD`，不是“所有 SwiGLU fusion 都无效”。

### 为什么停止继续优化 toy workload

Loop 004 的 `REPRESENTATIVE_SHAPE_STATUS = PARTIAL`，`CONFIGURATION_IDENTITY = null`。现有 historical fixture 的 `S=128, B=2, H=1024, FP16, TP=1` 只能作为 historical V100 fixture；同一 identity 下缺 `FFN hidden` 和 `SP`，不能标为九格 representative。因此 `REPRESENTATIVE_REPLAY_SPEC = NOT_CREATED`，下一阶段为 `WAIT_FOR_AUTHORITATIVE_SHAPE`。

来源： [Loop 003 audit](audits/real_optimization_loop_003_cache_and_timing.md)、[Loop 004 audit](audits/real_optimization_loop_004_shape_acquisition.md)、[Loop 004 provenance](../artifacts/integration/swiglu/real_loop_004/nine_grid_shape_provenance.json)。
