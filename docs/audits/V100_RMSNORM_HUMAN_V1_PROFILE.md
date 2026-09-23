# V100 RMSNorm Human v1 Profile

## Profile Status: PASS

## Backward Execution Graph

Human v1 backward consists of exactly **3 GPU kernels** per call:

| # | Kernel | Duration (max shape) | % Backward GPU |
|---|--------|---------------------|----------------|
| 1 | `FillFunctor<float>` (zero/memset) | 6.66 µs | 19.9% |
| 2 | `rms_backward_kernel` | 19.20 µs | 57.3% |
| 3 | `float16_copy_kernel` (FP32→FP16 cast) | 7.65 µs | 22.8% |

Total backward GPU time: ~33.5 µs per call (max shape [128,2,1024]).

### Root Causes

1. **`torch::zeros({1024}, float32)`** — triggers a `FillFunctor` memset kernel every backward call
2. **`atomicAdd` for dweight** — 262,144 atomic operations (256 rows × 1024 weight dims) per backward
3. **`grad_weight_fp32.to(float16)`** — triggers a separate `float16_copy_kernel` every backward call

## Main Kernel NCU Analysis (rms_backward_kernel)

| Metric | Value |
|--------|-------|
| Duration | 21.44 µs |
| Grid | 256 blocks × 256 threads = 65,536 threads |
| Registers/Thread | 28 |
| Shared Memory | 128 bytes static |
| Theoretical Occupancy | 100% |
| Achieved Occupancy | 39.74% |
| Waves Per SM | 0.48 |
| Active Warps Per SM | 25.44 / 64 |
| Eligible Warps Per Scheduler | 0.46 / 16 |
| No Eligible Cycles | 77.15% |
| Executed IPC (active) | 0.87 |
| Executed Instructions | 221,184 |
| L1/TEX Hit Rate | 22.07% |
| L2 Hit Rate | 61.87% |
| DRAM Throughput | 5.67% of peak |
| Memory Throughput | 49.25 GB/s (21.94% peak) |
| SM Frequency (NCU) | 245.72 MHz (low — likely clock transition artifact) |

### Key NCU Warnings

- **Grid too small**: "This kernel grid is too small to fill the available resources on this device, resulting in only 0.5 full waves across all SMs."
- **All compute pipelines under-utilized**: "Either this kernel is very small or it doesn't issue enough warps."
- **Issue slot utilization**: "Each scheduler only issues an instruction every 4.4 cycles."
- **Memory access pattern**: "L1TEX→L2 stores/loads only access 2.0 sectors out of possible 4 per cache line."

## Atomic DW Bottleneck: LIKELY

Evidence:
- 262,144 `atomicAdd` operations per backward call (256 rows × 1024 weight elements)
- Each row-block's 256 threads contend on the same 1024 FP32 dweight elements
- Warps stalled 77.15% of cycles — consistent with memory/atomic dependency stalls
- Kernel too small for GPU (0.48 waves) — cannot hide atomic latency with other work
- Memory throughput only 21.94% of peak despite atomic-heavy computation

## DW Zero Overhead: SIGNIFICANT

- `torch::zeros({1024}, float32)` → `FillFunctor` kernel: 6.66 µs (19.9% of backward GPU time)
- Fixed overhead independent of row count
- For small shapes ([16,1,1024]), this is 27.9% of backward — proportionally worse

## DW Cast Overhead: SIGNIFICANT

- `grad_weight_fp32.to(float16)` → `float16_copy_kernel`: 7.65 µs (22.8% of backward GPU time)
- Fixed overhead independent of row count
- Combined zero+cast = 42.7% of backward GPU time — nearly half the backward is overhead

## Candidate Variability: MAIN_KERNEL

- **Diagnostic 120-iteration backward**: CV=0.079 (7.9%), bimodal with outlier tail
  - Tight cluster: 0.176-0.203 ms
  - Outliers: 0.206-0.285 ms (single spike at 0.285)
- **L0 CV**: 0.025-0.098 (in-window timing, controlled), all well under 0.20
- **L2 CV**: 0.218 (K=8 outer windows, full GPTModel) — exceeds 0.20 threshold
- **Root cause**: Small kernel (0.48 waves) is sensitive to GPU clock/power state transitions. The V100 SM clock transitions from idle (37 MHz) to P0, and very short kernels may not see stable clock. Large K=8 outer windows in L2 capture this variability.

## V2 Optimization Priority

1. **A. REDUCE_ATOMIC_DW** — Replace per-thread atomicAdd with block-level reduction then single atomic, or warp-level reduction. This would reduce 262K atomics to at most 1024 (or fewer with hierarchical reduction).
2. **B. REMOVE_TEMP_ZERO_ALLOC** — Initialize dweight inside the main kernel instead of a separate memset. Saves 6.66 µs (19.9% of backward).
3. **C. REMOVE_DW_CAST** — Either accumulate dweight directly in FP16 (with appropriate care) or fuse the cast into the main kernel's final write. Saves 7.65 µs (22.8% of backward).

Items 2+3 alone would recover ~43% of backward GPU time and eliminate 2 kernel launches.
