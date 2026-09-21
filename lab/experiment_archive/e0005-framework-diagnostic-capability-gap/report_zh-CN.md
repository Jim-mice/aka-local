# e0005：诊断能力缺口

## 适用范围

- 分类：`FRAMEWORK_TOOLING_OBSERVATION`
- 本归档不是 GPU、kernel 或性能结论。
- Campaign：`rms_norm_train__rtx5060_sm120__cuda_cpp`
- Episode：`e0005`
- 已记录的固定优化流水线尝试：`5`

## 事实结果

该 Episode 的 directive 要求在修改 candidate 前获得 repeated per-shape statistics、cross-batch stability、regime grouping、static evidence 和 representative profiling。当时 Runner 只支持固定优化路径：`edit -> compile -> correctness -> development ABBA`，没有一等的 DIAGNOSTIC action space。

## Candidate 状态

- 依据 episode record，candidate 保持与 incumbent 相同。
- 本归档不产生 ANTI_STRATEGY、HARDWARE_FACT 或 RMSNorm 性能结论。

## 后续处理

该 Episode 之后已补上诊断 experiment routing 和 controller-owned diagnostic actions。未来只有用户批准的新 Episode 才能先收集所请求的证据，再提出优化改动。
