# V100 RMSNorm Snapshot B L2 raw-sample diagnosis

审计对象为 Snapshot B 的完整 50 个 reference 与 50 个 candidate measured windows；没有删除样本或进行 outlier trimming。Reference CV=0.2578826165，candidate CV=0.2733676548。

raw samples 在约 13--15 ms 与 20--25 ms 两个区间之间切换，且高值集中于早期/后期、低值集中于中段，构成明显 bimodality 与 time drift。Reference 与 candidate cycle time 的相关系数为 0.9680155499，说明存在 common-mode noise。paired ratio 比绝对时间更稳定，但不能绕过冻结的 reference/candidate CV<=0.20 gate。

逐样本数据、ABBA/BAAB position、cycle、paired delta 和 paired ratio 保存在 `overnight/v100_rmsnorm_e2e_20260922/snapshot_b_l2_raw_diagnosis.json`。
