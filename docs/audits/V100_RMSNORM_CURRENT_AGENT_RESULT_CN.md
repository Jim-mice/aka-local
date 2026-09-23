# V100 RMSNorm Current Agent 冻结结果

Snapshot A：`V100_RMSNORM_E2E_SNAPSHOT_A`

| Hypothesis | Attempt | Mechanism family | L0 | L1 | L2 | Decision |
|---|---:|---|---:|---|---:|---|
| H001 | 1 | fusion / row reduction / analytic backward | 1.6578598549x | PASS | 1.0466891407x | VALID |
| H001 | 2 | H001 参数调优（非新机制） | 1.6639332616x | PASS | 1.0770773234x | VALID |
| H002 | 1 | backward producer-consumer fusion / atomic dweight | 1.6040842845x | PASS correctness | — | PERFORMANCE_REJECT |
| H003 | 1 | native CUDA / block reduction / coalesced mapping | 3.6734935121x | PASS | 1.1643486648x | PROMOTION_ELIGIBLE |

H003 的 L2 是 `CONTROLLED_MEGATRON_E2E`：GPTModel whole-step、forward/backward、optimizer-compatible step、paired reference/candidate outer timing；不是 Nine-grid E2E。`NINE_GRID_E2E=NO`。
