# V100 RMSNorm OJ 修复与 Snapshot B

## 结论

Snapshot A 保持 immutable。本轮创建独立 `V100_RMSNORM_E2E_SNAPSHOT_B`，只修复 OJ/packaging：official L0 从 28 windows/side 修复为至少 50；L1 correctness 改为 fail-closed；L2 增加 L1、protocol、finite timing、CV、invocation 和 no-reference gate；三个 shell script 统一使用 `contract_v2_snapshot.json`。

## 静态测试

Python syntax/AST、JSON parse、shell contract path、28-window arithmetic failure、zero-invocation L1 failure、CV>0.20 L2 failure、package leak scan 均 PASS。Snapshot A 的 `human_challenge_package/` 未发生变化。H003 source hash 未改变，也未读取 hypothesis/mechanism、v23/v28 或修改 Megatron。

## OJ gate gap

发现 Snapshot A 的 OJ gate gap：L1 已计算 model loss 与 RMSNorm gradient error，却未把它们纳入 PASS/FAIL。Snapshot B 使用 Snapshot A 已冻结的 0.005 correctness tolerance 作为同一类绝对误差阈值，来源写入 `snapshot_b_correctness_policy.json`，不是按 H003 误差卡线。

## H003 Snapshot B requalification

candidate SHA256=`028a576d2220922e76a80bd6f1cbaff7b6054c87b7d45bd50d0ad6acc6cbaedb`，未修改。

- L0：PASS；3 official shapes 均 correctness PASS，candidate 每个 shape 为 50 windows。
- L1：PASS；5 replacements，remaining reference RMSNorm=0，replacement invocations=5，loss error=`3.00407409667969e-05`，norm grad max error=`1.1444091796875e-05`。
- L2：FAIL-CLOSED，`REFERENCE_STABILITY_FAILURE`。reference n=50、CV=`0.257882616473316`；candidate n=50、CV=`0.273367654795072`；frozen threshold=0.20。观察到的 paired speedup=`1.14228393415353x`，但不构成正式 L2 成绩。

因此没有生成 `CURRENT_AGENT_FROZEN_SNAPSHOT_B.json`，没有把 Snapshot B 标为可发布 Human challenge，也没有放宽 stability policy 或修改 H003。

## 最终状态

`HUMAN_CHALLENGE_PACKAGE_V2=FAIL/PAUSED`。Snapshot B package 已保留，失败 result/status/GPU gate artifact 位于 `overnight/v100_rmsnorm_e2e_20260922/snapshot_b_requalification/`；明天 Human Challenge 暂停，直到用户另行决定如何处理稳定性问题。不得自动继续 optimization 或 rebenchmark。
