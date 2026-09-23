# Human Attempt 01

## Candidate

- **SHA256**: `cf404cc23b3ea4b0be6ea0e8ea18d1d27d952864bb7a9cfcdf26e81fbd18b477`
- **设计**: Human v1（RMSNorm CUDA kernel + PyTorch custom op）
- **类型**: Python adapter + torch.utils.cpp_extension native CUDA extension

## GPU

- **型号**: Tesla V100-PCIE-16GB
- **UUID**: `GPU-88be8e63-dd61-3d8f-2f44-e1f93204654d`
- **状态**: 评测前后均无其他 compute process，GPU 空闲
- **GPU CLEAN**: YES

## Compile

- **状态**: PASS
- **方式**: torch.utils.cpp_extension load_inline，Ninja + NVCC 编译
- **编译产物**: `human_rmsnorm_v1_ext.so` (216 KB)

## L0 Correctness

- **状态**: PASS（全部 3 个 frozen shapes）

| Shape | Forward max_abs | Forward max_rel | dx max_abs | dx max_rel | dw max_abs |
|-------|----------------|-----------------|------------|------------|------------|
| [16,1,1024] | 0.001953 | 0.000724 | 0.000001 | 0.002004 | 0.000031 |
| [64,2,1024] | 0.000977 | 0.000720 | 0.000977 | 0.001068 | 0.000031 |
| [128,2,1024] | 0.000977 | 0.000845 | 0.000977 | 0.002232 | 0.003906 |

## L0 Forward

| Shape | Reference (ms) | Candidate (ms) | Speedup | Ref CV | Cand CV |
|-------|---------------|----------------|---------|--------|---------|
| [16,1,1024] | 0.120 | 0.043 | 2.818x | 0.0760 | 0.0979 |
| [64,2,1024] | 0.344 | 0.122 | 2.823x | 0.0533 | 0.0614 |
| [128,2,1024] | 0.675 | 0.236 | 2.856x | 0.0158 | 0.0253 |

## L0 Backward

| Shape | Reference (ms) | Candidate (ms) | Speedup |
|-------|---------------|----------------|---------|
| [16,1,1024] | 0.457 | 0.134 | 3.409x |
| [64,2,1024] | 0.806 | 0.234 | 3.439x |
| [128,2,1024] | 1.556 | 0.434 | 3.580x |

## L1

- **状态**: PASS
- **替换数量**: 5/5 torch.nn.RMSNorm 被替换
- **残留 torch.nn.RMSNorm**: 0
- **Forward**: PASS
- **Backward**: PASS
- **Loss correctness**: PASS (abs error = 3.0e-05)
- **RMSNorm gradient correctness**: PASS
- **Input/parameter gradients**: finite

## L2 Run 1

- **状态**: **CANDIDATE_STABILITY_FAILURE**
- **Speedup**: 1.212x
- **Reference**: mean=20.07ms, CV=0.1413 (PASS)
- **Candidate**: mean=16.56ms, CV=0.2180 (**FAIL** — 超过 0.20 阈值)

## L2 Run 2

- **状态**: NOT_RUN（Run 1 stability 未通过，未执行 Run 2）

## Stability

- 候选实现存在稳定性问题：candidate CV 0.218 > 0.20 threshold
- L0 层面 CV 正常，L2 E2E 层面出现波动
- 可能原因：kernel launch 参数或 memory access pattern 在 full GPTModel 上下文中不稳定

## Human vs Frozen Agent 数值表

| 指标 | Human v1 | Frozen Agent |
|------|----------|-------------|
| L0 Forward (approx) | ~2.83x | 3.673x |
| L2 Run 1 | 1.212x | 1.197x |
| L2 Run 2 | N/A | 1.182x |

## 当前结论

Human v1 在 L0 层面 correctness 全部通过，forward/backward speedup 约 2.8-3.6x。
L1 Megatron 集成验证通过（5 层 RMSNorm 全部替换，loss/gradient 正确）。

但在 L2 CONTROLLED_MEGATRON_E2E 中，candidate 稳定性未达标：
- candidate CV 0.218 > 0.20 threshold
- speedup 1.212x 低于 Agent L2 目标

**下一步建议**: PROFILE_BEFORE_V2 — 对 Human v1 进行 profiling，分析 L2 稳定性根因后再编写 Human v2。
