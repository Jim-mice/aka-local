# V100 RMSNorm L2 Stability Repair

Snapshot A/B 保持 immutable。本轮只对 Snapshot B 做离线 raw-sample 审计、reference-only calibration，并建立 Snapshot C；没有修改 H003 source、Megatron、数学 contract、CV gate、shape 或 dtype。

## 诊断

Snapshot B 的 50/50 raw samples 显示明显 bimodal/time-drift，并且 reference/candidate 有强 common-mode 相关性（Pearson correlation 0.9680155499）。完整样本未删除、未 trimming。Snapshot B reference CV=0.2578826165，candidate CV=0.2733676548，因此原正式 L2 失败是有效的 stability failure。

## Measurement protocol

Reference-only calibration 使用同一 model/input、outer CUDA Event、20 warmup windows、30 measured windows、无 timed-region allocation/rebuild。固定 K=8、target=100 ms 的两次独立 calibration：CV=0.0501405476 和 0.0166824822，均无明显双峰或单调 drift，满足内部 CV<=0.10；正式 gate 仍为 CV<=0.20。

## Snapshot C qualification

Snapshot C 继承 Snapshot B，唯一执行语义变化是 L2 使用冻结 K=8 的 outer whole-step measurement。H003 source SHA256 保持 `028a576d2220922e76a80bd6f1cbaff7b6054c87b7d45bd50d0ad6acc6cbaedb`。

| run | L1 | L2 | speedup | reference CV | candidate CV | windows/side |
|---|---|---|---:|---:|---:|---:|
| 1 | PASS | PASS | 1.1967401230x | 0.0630404816 | 0.1613387697 | 50 |
| 2 | PASS | PASS | 1.1815262131x | 0.1405883001 | 0.1586078379 | 50 |

两次 run 均为同一 H003 hash、同一 V100 UUID、同一 process 内 paired ABBA/BAAB、outer CUDA Event；两次均通过正式 CV gate 和 correctness/L1 gate。未使用 trimmed score 或 profiler timing。

## 结论

SNAPSHOT_B_RAW_PATTERN = MIXED (BIMODAL + DRIFT + COMMON_MODE)

REFERENCE_ONLY_CALIBRATION = PASS

SELECTED_WINDOW_TARGET_MS = 100

SELECTED_K = 8

REFERENCE_CALIBRATION_CV_1 = 0.0501405476

REFERENCE_CALIBRATION_CV_2 = 0.0166824822

SNAPSHOT_C_CREATED = YES

CV_THRESHOLD_CHANGED = NO

H003_SOURCE_MODIFIED = NO

H003_SNAPSHOT_C_RUN1_L2 = 1.1967401230x

H003_SNAPSHOT_C_RUN1_REF_CV = 0.0630404816

H003_SNAPSHOT_C_RUN1_CAND_CV = 0.1613387697

H003_SNAPSHOT_C_RUN2_L2 = 1.1815262131x

H003_SNAPSHOT_C_RUN2_REF_CV = 0.1405883001

H003_SNAPSHOT_C_RUN2_CAND_CV = 0.1586078379

H003_REQUALIFICATION = PASS

COMMON_MODE_NOISE = YES

HUMAN_CHALLENGE_PACKAGE_V3 = PASS

COMPARISON_CONTRACT_IDENTICAL = YES

NEW_AGENT_OPTIMIZATION = NO

NEW_CANDIDATE = NO

MEGATRON_MODIFIED = NO
