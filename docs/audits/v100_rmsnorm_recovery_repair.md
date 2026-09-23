# V100 RMSNorm Recovery Repair

## 结果

- H003 remote status: `DONE`。只读 SSH 确认 PID 4027990 已退出，`status.json= DONE`，`result.json` 与 before/after GPU snapshot 存在。
- H003 provenance: `PASS`。candidate SHA256=`028a576d2220922e76a80bd6f1cbaff7b6054c87b7d45bd50d0ad6acc6cbaedb`；contract SHA256=`1042caf509cb37c2defe2c230cd6e0b38ef8e784268ba0cf1de29c9893505f97`；Megatron 路径为 `megatron-lm-5be9626`；GPU UUID=`GPU-88be8e63-dd61-3d8f-2f44-e1f93204654d`。
- H003 ingest: `PASS`。artifact 先下载到 `recovery_ingest/`，完成 hash、input hash、L1 replacement/invocation、无 reference RMSNorm 残留、forward/input/weight gradient、20 次 warmup、50 个 measured windows、ABBA/BAAB paired outer CUDA timing、CV 和 GPU gate 校验后登记。
- H003 L0 speedup=`3.6734935121x`；H003 L1=`PASS`；H003 L2 speedup=`1.1643486648x`；reference CV=`0.17793452996622466`；candidate CV=`0.1813704121420646`。
- H003 L2 包含 paired reference，但 `REFERENCE_ONLY_L2_ARTIFACT` 仍为 `NOT_RUN`。

## Promotion 与 provenance

H001/a2 仍是当前 incumbent：L0=`1.6639332616x`，L2=`1.0770773234x`。H003 数值更高，但本轮不自动改 incumbent；promotion 仍需遵守 frozen policy 和后续明确授权。

campaign branch 已创建：`overnight/v100-rmsnorm-e2e-20260922`，父历史为 `6d8d3549b3f37d0be8d87126ac6b5581bfdfd829`。对 `6d8d354..db8c2ea` 的比较发现执行相关变更，包括 `lab/runtime`、evaluators、contracts/policies、operator metadata 等，因此 `BASE_COMMIT_COMPATIBILITY=FAIL`，不能进入 READY_TO_RESUME。

STATE、RUN_INDEX、REMOTE_JOBS、CANDIDATE_REGISTRY 已追加 H003 DONE/validated-ingest 信息；JOURNAL 只追加 recovery reconciliation 事件，未重写历史。历史来源仍未打开，blind isolation 保持。

## 最终状态

H003 已合法 ingest，但 campaign 因 base compatibility FAIL 被阻断；没有创建新 hypothesis/candidate，也没有运行 NCU、benchmark 或 historical v23/v28。checkpoint commit `1bbca47badbbe3c2954479426a32c2c5b44213d0` 已真实存在，并属于 `overnight/v100-rmsnorm-e2e-20260922` 分支；此前“尚未创建 checkpoint commit”的表述已纠正。

下一步唯一动作：先由用户决定如何处理 `db8c2ea` 引入的 execution-relevant provenance 分叉；在此之前不得恢复新 candidate 或 benchmark。
