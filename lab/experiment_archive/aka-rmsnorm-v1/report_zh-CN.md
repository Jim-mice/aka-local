# Standalone RMSNorm V1 fused row reduction（中文归档）

- 实验 ID：`aka-rmsnorm-v1`
- 算子：`rms_norm_train`
- GPU/平台：`rtx5060_laptop_sm120`
- Backend：`cuda_cpp`
- 决策：`PROMOTE`

## 证据

以下内容是对原始结构化证据的中文说明；数字、ID、路径和技术符号保持不变。

56/56 correctness PASS; arithmetic mean speedup 7.202740x vs eager; one block per row, shared-memory tree reduction

## 来源

`<PROJECT_ROOT>\campaigns\rms_norm\backward\episode_1\decision.json`
