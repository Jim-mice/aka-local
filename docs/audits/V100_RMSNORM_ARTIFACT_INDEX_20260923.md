# V100 RMSNorm Artifact Index (2026-09-23)

> 仅列出当前仓库中真实存在的可追溯文件。路径相对于 `D:\Users\38154\Downloads\aka-local-publish`。

## 1. Frozen Benchmark / Contract / Agent Frozen

| 文件 | 用途 |
|------|------|
| `overnight/v100_rmsnorm_e2e_20260922/CURRENT_AGENT_FROZEN.json` | Agent 冻结声明：H001/a1, H001/a2, H002/a1, H003/a1 的 L0/L1/L2 与 decision |
| `overnight/v100_rmsnorm_e2e_20260922/CONTRACT_LOCK.json` | 契约锁 |
| `overnight/v100_rmsnorm_e2e_20260922/controlled_v100_rmsnorm_e2e_contract.json` | Controlled Megatron E2E contract v2 |
| `overnight/v100_rmsnorm_e2e_20260922/STATE.json` | Campaign 状态 |
| `overnight/v100_rmsnorm_e2e_20260922/CANDIDATE_REGISTRY.jsonl` | Candidate 注册表 |
| `overnight/v100_rmsnorm_e2e_20260922/RUN_INDEX.json` | 运行索引 |
| `overnight/v100_rmsnorm_e2e_20260922/JOURNAL.jsonl` | 事件流水 |
| `overnight/v100_rmsnorm_e2e_20260922/README_CN.md` | Campaign README |

## 2. Snapshots

| 文件 | 用途 |
|------|------|
| `overnight/v100_rmsnorm_e2e_20260922/snapshot_b_requalification/SNAPSHOT_B_*` | Snapshot B 现场重测（参见目录内全部文件） |
| `overnight/v100_rmsnorm_e2e_20260922/SNAPSHOT_B_REQUALIFICATION.json` | Snapshot B 综合 requalification 结果 |
| `overnight/v100_rmsnorm_e2e_20260922/snapshot_b_l2_raw_diagnosis.json` | Snapshot B 50/50 raw samples + Pearson 0.9680155499 + pattern: peak/GAS/DRIFT/COMMON_MODE |
| `overnight/v100_rmsnorm_e2e_20260922/snapshot_c_calibration/reference_calibration_k8_1.json` | Snapshot C calibration k8_1：CV=0.0501405476 |
| `overnight/v100_rmsnorm_e2e_20260922/snapshot_c_calibration/reference_calibration_k8_2.json` | Snapshot C calibration k8_2：CV=0.0166824822 |
| `overnight/v100_rmsnorm_e2e_20260922/snapshot_c_calibration/reference_calibration_k4_1.json` | Snapshot C 备选 K=4 calibration |
| `overnight/v100_rmsnorm_e2e_20260922/snapshot_c_calibration/reference_calibration_k4_2.json` | Snapshot C 备选 K=4 calibration |
| `overnight/v100_rmsnorm_e2e_20260922/snapshot_c_calibration/reference_calibration.json` | Snapshot C calibration 汇总 |
| `overnight/v100_rmsnorm_e2e_20260922/snapshot_c_calibration/reference_calibration_2.json` | Snapshot C calibration 汇总 2 |
| `overnight/v100_rmsnorm_e2e_20260922/snapshot_c_calibration/reference_calibration_3.json` | Snapshot C calibration 汇总 3 |
| `overnight/v100_rmsnorm_e2e_20260922/snapshot_c_requalification_run1.json` | H003 Snapshot C Run 1：1.1967401230x PASS |
| `overnight/v100_rmsnorm_e2e_20260922/snapshot_c_requalification_run2.json` | H003 Snapshot C Run 2：1.1815262131x PASS |
| `overnight/v100_rmsnorm_e2e_20260922/SNAPSHOT_C_MANIFEST.json` | Snapshot C manifest（来自 human_challenge_package_v3） |
| `overnight/v100_rmsnorm_e2e_20260922/SNAPSHOT_B_MANIFEST.json` | Snapshot B manifest |
| `overnight/v100_rmsnorm_e2e_20260922/snapshot_b_correctness_policy.json` | Snapshot B correctness policy |

## 3. Agent Hypothesis Lineage

| 文件 | 内容 |
|------|------|
| `overnight/v100_rmsnorm_e2e_20260922/hypotheses/H001/hypothesis.json` | H001 mechanism + fingerprint |
| `overnight/v100_rmsnorm_e2e_20260922/hypotheses/H001/attempts/1/candidate.py` | H001 attempt 1 source |
| `overnight/v100_rmsnorm_e2e_20260922/hypotheses/H001/attempts/1/decision.json` | H001 attempt 1 decision |
| `overnight/v100_rmsnorm_e2e_20260922/hypotheses/H001/attempts/1/feedback.json` | H001 attempt 1 feedback |
| `overnight/v100_rmsnorm_e2e_20260922/hypotheses/H001/attempts/2/candidate.py` | H001 attempt 2 source |
| `overnight/v100_rmsnorm_e2e_20260922/hypotheses/H001/attempts/2/decision.json` | H001 attempt 2 decision |
| `overnight/v100_rmsnorm_e2e_20260922/hypotheses/H001/attempts/2/feedback.json` | H001 attempt 2 feedback |
| `overnight/v100_rmsnorm_e2e_20260922/hypotheses/H002/hypothesis.json` | H002 mechanism |
| `overnight/v100_rmsnorm_e2e_20260922/hypotheses/H002/attempts/1/candidate.py` | H002 attempt 1 source |
| `overnight/v100_rmsnorm_e2e_20260922/hypotheses/H002/attempts/1/decision.json` | H002 attempt 1 decision |
| `overnight/v100_rmsnorm_e2e_20260922/hypotheses/H002/attempts/1/feedback.json` | H002 attempt 1 feedback |
| `overnight/v100_rmsnorm_e2e_20260922/hypotheses/H003/hypothesis.json` | H003 mechanism |
| `overnight/v100_rmsnorm_e2e_20260922/hypotheses/H003/attempts/1/candidate.py` | H003 attempt 1 source (final) |
| `overnight/v100_rmsnorm_e2e_20260922/hypotheses/H003/attempts/1/decision.json` | H003 attempt 1 decision |
| `overnight/v100_rmsnorm_e2e_20260922/hypotheses/H003/attempts/1/feedback.json` | H003 attempt 1 feedback |

## 4. Agent Final H003 Results

| 文件 | 内容 |
|------|------|
| `overnight/v100_rmsnorm_e2e_20260922/CURRENT_AGENT_FROZEN.json` | Agent 冻结综合（含 H003 L0/L1/Snapshot A L2） |
| `overnight/v100_rmsnorm_e2e_20260922/SNAPSHOT_B_REQUALIFICATION.json` | Snapshot B requalification（H003 在 Snapshot B STABILITY_FAILURE） |
| `overnight/v100_rmsnorm_e2e_20260922/snapshot_b_l2_raw_diagnosis.json` | Snapshot B raw samples |
| `overnight/v100_rmsnorm_e2e_20260922/snapshot_c_requalification_run1.json` | Snapshot C Run 1 |
| `overnight/v100_rmsnorm_e2e_20260922/snapshot_c_requalification_run2.json` | Snapshot C Run 2 |

## 5. Stability Repair Report

| 文件 | 内容 |
|------|------|
| `docs/audits/V100_RMSNORM_L2_STABILITY_DIAGNOSIS.md` | L2 稳定性诊断 |
| `docs/audits/V100_RMSNORM_L2_STABILITY_REPAIR.md` | Snapshot C measurement repair 报告 |
| `docs/audits/V100_RMSNORM_OJ_REPAIR_AND_SNAPSHOT_B.md` | OJ repair 记录 |
| `docs/audits/V100_RMSNORM_CURRENT_AGENT_RESULT_CN.md` | Current Agent 结果中文版 |

## 6. Human v1

| 文件 | 内容 |
|------|------|
| `overnight/v100_rmsnorm_e2e_20260922/human_challenge_package/human_candidate_v1.py` | Human v1 source（来自人类 challenge package） |
| `overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_01/candidate_sha256.txt` | Human v1 SHA |
| `overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_01/HUMAN_ATTEMPT_01_SUMMARY.json` | Human v1 综合 |
| `overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_01/quick_l0.json` | Quick L0 |
| `overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_01/official_l0.json` | Official L0 |
| `overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_01/l1_l2_smoke.json` | L1 smoke |
| `overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_01/official_l2_run1.json` | Human v1 L2 Run1（1.212x STABILITY_BLOCKED） |
| `overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_01/GPU_STATE_BEFORE.txt` | GPU state before |
| `overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_01/GPU_STATE_AFTER.txt` | GPU state after |
| `docs/audits/V100_RMSNORM_HUMAN_ATTEMPT_01.md` | Human v1 报告 |

## 7. Human v1 Profile

| 文件 | 内容 |
|------|------|
| `overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_01/profile/PROFILE_SUMMARY.json` | Profiler summary（19.20/7.65/6.66 us 等） |
| `docs/audits/V100_RMSNORM_HUMAN_V1_PROFILE.md` | Profiler 报告 |
| `0cd81c9`（commit） | NCU + timeline + variability diagnostic |

## 8. Human v2

| 文件 | 内容 |
|------|------|
| `overnight/v100_rmsnorm_e2e_20260922/human_challenge_package_v3/human_candidate_v2.py` | Human v2 source（repaired 状态，SHA256 `eee09b4fd9eec5796f8079b4ed7492c3470c08ffb2f056ca9f4aafee66e5e29e`） |
| `overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_02/candidate_sha256.txt` | Human v2 初始失败版本 SHA (`4c0da5ae5e1964d0d626fffd785e8120e0df6a110fc0d46183b4516b51f84416`) |
| `overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_02/HUMAN_ATTEMPT_02_SUMMARY.json` | Human v2 初始 COMPILE_FAILURE summary |
| `overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_02_repair_01/candidate_sha256.txt` | Human v2 repair1 SHA |
| `overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_02_repair_01/repair_diff.txt` | 6 处纯语法换行修复 diff |
| `overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_02_repair_01/compile.log` | 编译日志 |
| `overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_02_repair_01/quick_l0.json` | Quick L0 |
| `overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_02_repair_01/official_l0.json` | Official L0 |
| `overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_02_repair_01/l1_l2_smoke.json` | L1 smoke |
| `overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_02_repair_01/official_l2_run1.json` | Human v2 L2 Run1：1.1735x PASS |
| `overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_02_repair_01/official_l2_run2.json` | Human v2 L2 Run2：1.1990x PASS |
| `overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_02_repair_01/GPU_STATE_BEFORE.txt` | GPU state before |
| `overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_02_repair_01/GPU_STATE_AFTER.txt` | GPU state after |
| `overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_02_repair_01/HUMAN_ATTEMPT_02_REPAIR_01_SUMMARY.json` | 综合 summary |
| `docs/audits/V100_RMSNORM_HUMAN_ATTEMPT_02.md` | Human v2 初始失败报告 |
| `docs/audits/V100_RMSNORM_HUMAN_ATTEMPT_02_REPAIR_01.md` | Human v2 compile repair 报告 |

## 9. Final Summary

| 文件 | 内容 |
|------|------|
| `overnight/v100_rmsnorm_e2e_20260922/FINAL_AGENT_VS_HUMAN_SUMMARY.json` | 机器可读总表 |
| `docs/audits/V100_RMSNORM_AGENT_VS_HUMAN_FINAL_20260923.md` | 中文最终报告 |
| `docs/audits/V100_RMSNORM_ARTIFACT_INDEX_20260923.md` | 本文件 |

## 10. Recovery / Ingest (历史溯源)

| 文件 | 内容 |
|------|------|
| `overnight/v100_rmsnorm_e2e_20260922/recovery_audit/` | 历史恢复审计 |
| `overnight/v100_rmsnorm_e2e_20260922/recovery_ingest/` | 历史 ingest |
| `overnight/v100_rmsnorm_e2e_20260922/gpu_state_snapshots/` | GPU state 快照 |
| `overnight/v100_rmsnorm_e2e_20260922/artifacts/` | artifacts |
| `overnight/v100_rmsnorm_e2e_20260922/remote_artifacts/` | remote artifacts |
| `overnight/v100_rmsnorm_e2e_20260922/agent_blind_package/` | Agent blind package |
| `docs/audits/v100_rmsnorm_overnight_recovery_audit.md` | Overnight recovery audit |
| `docs/audits/v100_rmsnorm_recovery_repair.md` | Recovery repair |
| `docs/audits/STATUS.md` | Status |

## 11. Git Commit Trail

| Commit | Title |
|--------|-------|
| `aa702d4` | experiment: evaluate human RMSNorm v2 compile repair 1 |
| `fb03cf4` | experiment: evaluate human RMSNorm attempt 02 on snapshot C |
| `0cd81c9` | profile: human RMSNorm v1 backward diagnostic (NCU + timeline + variability) |
| `a21be2d` | experiment: evaluate human RMSNorm attempt 01 on snapshot C |
| `db56d9d` | docs: record Snapshot C protocol manifest |
| `2116c00` | checkpoint: freeze V100 RMSNorm Snapshot C stability repair |
| `ae46998` | checkpoint: freeze V100 RMSNorm agent challenge snapshot |
| `77208a6` | checkpoint: freeze V100 RMSNorm agent challenge snapshot |
| `1bbca47` | chore: checkpoint V100 RMSNorm overnight recovery |
| `6d8d354` | Add performance overview and evidence links |
