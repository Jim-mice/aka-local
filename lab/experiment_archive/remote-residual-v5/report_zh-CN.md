# Residual RMSNorm flat row-batched（中文归档）

- 实验 ID：`remote-residual-v5`
- 算子：`residual_rmsnorm_train`
- GPU/平台：`biv150_corex`
- Backend：`corex_triton`
- 决策：`ACCEPT`

## 证据

以下内容是对原始结构化证据的中文说明；数字、ID、路径和技术符号保持不变。

5 hidden workloads pass; 5-seed all-pass; score 19.75/19.81 vs V3 12.72; ~56 percent

## 来源

`/private/atrex-megatron/campaigns/kernel_opt_residual_rmsnorm_train_triton_bi_v150_production/memory/v5.json`
