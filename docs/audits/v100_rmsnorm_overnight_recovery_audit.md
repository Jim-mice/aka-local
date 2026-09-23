# V100 RMSNorm Overnight 恢复审计

## 1. 审计范围

本轮仅恢复、审计、状态盘点。未执行 resume、benchmark、profiler、candidate 修改、远端 kill、远端写入或新 workload。原始 campaign artifact 未修改；本报告与 `recovery_audit/RECOVERY_AUDIT.json` 为新增文件。

仓库检查结果：当前分支为 `main`，HEAD=`6d8d354`，工作区干净；未发现可切换的 `overnight/v100-rmsnorm-e2e-20260922` 分支。STATE 内的 `git_status_snapshot` 记录的是未纳入仓库的 `overnight/` 与 resume script，因此 provenance 不能视为已由目标 branch 固化。

## 2. Codex 退出时状态

STATE 的 campaign 起点为 `2026-09-22T16:48:25Z`，deadline 为 `2026-09-23T01:48:25Z`。退出点是 `PHASE_E_L1_L2 / H003_attempt1_official_running`，active job 为 `official_H003_a1_20260922T184806Z`，GPU UUID 为 `GPU-88be8e63-dd61-3d8f-2f44-e1f93204654d`。STATE 的 `next_action` 是轮询该 job、ingest 并验证 Megatron replacement；本轮只报告，不执行。

STATE 统计为 completed=15、failed=5、invalidated=1、finalization_started=false。JOURNAL 最后一项也是 H003 official L2 launch，二者在退出点一致；但 STATE/JOURNAL 的 git 状态与当前本地 Git 不一致，且 STATE 的 active job 后来已有本地 L0/smoke artifact，形成待修复的状态滞后。

## 3. Campaign 时间线

| 阶段 | 已完成事实 |
|---|---|
| Reference | contract v1 reference 失败/后被 invalidated；contract v2 reference `2026-09-22T17:33:28Z` 完成，L0/L1 PASS |
| H001 attempt 1 | compile/correctness/official L0 PASS；L1 PASS；controlled Megatron E2E L2 PASS，speedup 1.046689x |
| H001 attempt 2 | H001 内参数调优，不是新机制；official L0 PASS，L2 PASS，成为当前 best L0/L2 |
| H002 attempt 1 | compile/correctness PASS；official L0 1.604084x，较 incumbent 1.663933x 回退，PERFORMANCE_REJECT；L1/L2 未进入 |
| H003 attempt 1 | native CUDA 机制；quick/official L0 与 L1/L2 smoke artifact 已有；退出时 official L2 仍 RUNNING |

假设数为 3，implementation attempt 数为 4。H001 两个 attempt 属于同一 mechanism family；H002 是 producer/consumer fusion + atomic weight gradient；H003 是 native CUDA block reduction/coalesced mapping。没有把 4/8/其他 warp 参数当成独立机制。

## 4. V100 环境与 GPU 状态

历史 snapshot 显示选中 GPU 为 `Tesla V100-PCIE-16GB`、index 1、UUID=`GPU-88be8e63-dd61-3d8f-2f44-e1f93204654d`、16384 MiB、PCI bus `84:00.0`。snapshot gate 为 PASS，无 compute app，只有 Xorg 图形进程；各 official artifact 的 metadata 使用同一选中 UUID。当前远端 nvidia-smi 未执行，因此不以当前状态覆盖历史 snapshot，也不能声称当前远端 job 已结束。

每个已 ingest official artifact 具有 before/after GPU 文件与 UUID；没有发现 campaign 内 GPU UUID 变化证据。驱动/triton fingerprint 在 environment.json 中仍有 PENDING 字段，因此环境证明不是完全封闭的。分类：已有 official run 以 CLEAN/证据充分为主；H003 official L2 因缺少完成 artifact 为 UNKNOWN。

## 5. Frozen contract

contract v2 hash 为 `1042caf509cb37c2defe2c230cd6e0b38ef8e784268ba0cf1de29c9893505f97`，LOCK 与实际 contract 一致。目标明确为 `non-TE megatron.core.transformer.torch_norm.WrappedTorchNorm -> torch.nn.RMSNorm`，Megatron commit=`5be9626709af2722333bf54797c954c09edeada3`，fp16、epsilon=1e-5、[S,B,H] contiguous、forward/input-gradient/weight-gradient/optimizer-compatible weight 均受控。

contract v2 新增 same-process interleaved evaluator，并要求 reference 在 official promotion 前重测。contract v1 reference 属旧 contract，RUN_INDEX 已标 `INVALIDATED_BY_CONTRACT_CHANGE`，不能与 v2 混算。v2 L0/L1/L2 protocol、warmup、50 measured windows、ABBA/BAAB 类交错、outer CUDA-event whole-step timing 与 CV<=0.20 规则均已写入。

## 6. Blind isolation

manifest 与 excluded_sources.json 明确排除 v23/v28、历史 RMSNorm source、human v28 mechanism；JOURNAL 也记录 blind package hash upload PASS，且 STATE 的 `historical_sources_opened=false`。因此没有证据表明当前 Agent 在 freeze 前读取历史材料。

但 `human_challenge_package/` 实际为空，且当前 Agent 未标记为 frozen；因此公平 human challenge package 尚未建立，整体结论为 `PARTIAL`，不能宣称已完成的盲测公平比较。未发现已访问 old-Agent/Human v28 的证据，历史 recheck 为 NOT_RUN。

## 7. Reference L0/L1/L2

contract v2 reference 有真实 result、correctness、50-sample timing、GPU/device 信息，L0 PASS；L1 记录 `torch.nn.modules.normalization.RMSNorm`，forward/backward/input-grad/weight-grad 均 invoked/finite，PASS。没有独立 reference L2 whole GPT training-step artifact，故 REFERENCE_L2=NOT_RUN。早期 v1 reference 不可信，已 invalidated。

## 8. Agent hypothesis lineage

| hypothesis | mechanism family | attempts | 结果 |
|---|---|---:|---|
| H001 | fusion + row reduction + analytic backward | 2 | attempt 1、2 均 correctness/L0/L1/L2 有效；attempt 2 为 tuning |
| H002 | backward producer-consumer fusion + FP32 atomic dweight | 1 | correctness PASS，L0 performance reject |
| H003 | native CUDA block reduction + coalesced column tile | 1 | L0/smoke artifact 已有；official L2 未完成/未 ingest |

统计：proposed=3，implementation attempts=4，valid implementations=3，official L0 runs=4，L1 runs=3，L2 runs=2。compile fail 未见已记录的正式 compile failure；H003 的 feedback 仍 PENDING 不能代替完整结果。correctness fail 未见；H002 是 performance reject。

## 9. Agent 全部 implementation attempts

H001/a1 hash=`ba79fe6c...e7cdb`，official L0 combined=1.6578598549x，L1 PASS，L2=1.0466891407x。H001/a2 hash=`e58b359d...107e15`，official L0 combined=1.6639332616x，L1 PASS，L2=1.0770773234x，当前 incumbent。H002/a1 hash=`dba00d1e...1a9a4`，L0=1.6040842845x，correctness/stability PASS 但 performance reject。H003/a1 hash=`028a576d...baedb`，official L0 artifact 与 L1/L2 smoke 已完成；official L2 在退出时仍 running，不能把 smoke 当 official L2。

## 10. Candidate 跑分

已确认的正式 v2 L0 数值：H001/a1=1.6578598549x，H001/a2=1.6639332616x，H002/a1=1.6040842845x。H001/a1 与 a2 的 L2 controlled Megatron E2E 分别为 1.0466891407x 与 1.0770773234x。H003 official L0 虽已落地，但其 H003 official L2 仍缺完成 evidence；不得自动据此 promotion。

所有 L2 primary timing 应以 outer whole-step CUDA event；已见 H001 L2 使用 50 windows、reference/candidate CV 均低于 0.20。未发现 inner Event 作为 primary reward 的证据。NINE_GRID_E2E 仍为 NO。

## 11. 当前 best L0

BEST_L0_OPERATOR 为 H001/a2，hash=`e58b359d...107e15`，combined L0=1.6639332616x。它同时有 L1/L2，但 L0 与 L2 仍应分开表达；其 L2 仅为 controlled Megatron E2E，不是 nine-grid。

## 12. 当前 best L2

BEST_L2_SYSTEM 同为 H001/a2，L2=1.0770773234x，L1 PASS，稳定性证据为 reference/candidate CV 通过 v2 gate。H001/a2 是当前正式 best L2；H003 不能在 official L2 artifact 缺失时取代它。

## 13. NCU/profiler 证据

未找到本 campaign 已完成的 NCU/NSYS profiler evidence 可支持正式 score；contract/feedback 对 profiler 多为 PENDING/NOT_RUN。因此 NCU_STATUS=NOT_RUN，任何 profiler timing 若后续存在也只能 diagnostic-only，不能替代 benchmark score。

## 14. 稳定性与 bimodality

H001 官方结果记录 all CV below 0.20；H001 L2 reference/candidate CV 也通过。H001/a1 quick 的短窗口异常已由 official long-window 重新评估。当前 artifact 未提供足够统一的全 campaign bimodality 判定，尤其不能确认或排除历史约 9 us/22 us 双峰；BIMODALITY_STATUS=UNKNOWN，不得把未知写成未检测。

## 15. Historical old-Agent / Human v28 状态

STATE 标记 historical_sources_opened=false，未发现 old-Agent best 与 Human v28 的昨晚 recheck artifact。两者均为 NOT_RUN；不能引用旧的 4.08x/8.7x 替代昨晚新跑数据。由于 Agent freeze 与 challenge package 尚未形成完整可验证链，HISTORICAL_OPEN_AFTER_AGENT_FREEZE=UNVERIFIABLE_NOT_OPENED。

## 16. Human challenge package

`human_challenge_package/` 当前无文件，状态 `NOT_CREATED`。因此尚未形成包含 baseline/contract/OJ/benchmark policy/environment/shape-config 且不含 Current Agent source/mechanism 的可交付挑战包。

## 17. 远端仍在运行的 job

REMOTE_JOBS.json 记录 `official_H003_a1_20260922T184806Z`，PID=4027990，remote workspace 为 `/home/bencheng/aka_v100_challenges/v100_rmsnorm_e2e_20260922`，expected artifact 为该 job 的 `result.json/status.json`，退出时 status=RUNNING。另有本地已完成 H003 official L0 与 smoke artifact，但 H003 official L2 没有本地 result；因此至少一个 remote job 仍需只读确认，且不得自动 ingest。

## 18. 自洽性问题

P0：H003 official L2 的 remote running/未完成 artifact 不能用于 score 或 promotion；若在恢复时误把它当 DONE，会使 E2E 结论失效。

P1：当前 Git branch 为 main 而非 requested overnight branch；STATE 的 git snapshot 与当前仓库 provenance 不一致。contract v1 与 v2 结果若混算会失效，虽 RUN_INDEX 已对旧 reference 标 invalidated。H003 L0/smoke 已有本地结果但 STATE 仍停在 running。environment fingerprint 含 PENDING 字段。human challenge package 缺失导致公平性证明不完整。

P2：部分结果 JSON schema 没有重复写入 candidate/contract/GPU 顶层字段，需依赖 RUN_INDEX/remote metadata 联结；报告展示上容易误读为缺 provenance。

## 19. 可信结论

可信的是：v2 frozen contract、选中的 16GB Tesla V100 UUID、v2 reference L0/L1、H001/a1 与 H001/a2 的 controlled Megatron E2E 证据、H002 correctness PASS 及 performance reject、以及 H001/a2 当前 L0/L2 incumbent。H001/a2 是“当前最好已完成系统结果”，但不是 nine-grid E2E 结果。

## 20. 不能声称的结论

不能声称 H003 已完成 official L2 或击败 incumbent；不能声称 NINE_GRID_E2E；不能声称 old-Agent/Human v28 已在昨晚完成公平 recheck；不能声称当前远端 job 已结束；不能声称 profiler 已证明性能原因；不能把旧 contract reference 与 v2 结果混合。

## 21. 下一步唯一推荐恢复点

只读确认 `official_H003_a1_20260922T184806Z` 的远端 PID/job 状态及 artifact 是否存在；若 DONE，先下载并做完整 hash、contract、candidate、GPU、correctness、stability provenance validation，再决定是否 ingest。若仍 RUNNING，等待外部状态变化，不启动 replacement。之后先修复 branch/provenance 与 challenge-package/freeze 审计缺口，再由用户明确授权是否继续 campaign。本轮已停止，未执行该下一步。
