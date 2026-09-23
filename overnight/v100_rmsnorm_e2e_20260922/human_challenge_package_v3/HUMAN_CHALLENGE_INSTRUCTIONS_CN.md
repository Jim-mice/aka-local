# V100 RMSNorm Human Challenge（Snapshot B）

候选必须提供 Python module，并导出冻结 ABI：`TritonRMSNorm(torch.nn.Module)`。Python adapter 不等于 GPU kernel 必须用 Python；内部可以使用 Triton、native CUDA extension 或 PyTorch custom op。不得修改 OJ、不得 silent fallback 到 reference RMSNorm，并必须满足 contract v2。

L0 是 operator forward/backward correctness 与官方计时；L1 是真实 Megatron `WrappedTorchNorm` replacement/invocation、forward/backward、loss、输入梯度和 weight 梯度检查；L2 是 `GPTModel` controlled whole-step，reference/candidate 同进程配对、outer CUDA Event 计时、至少 50 measured windows，并执行 CV<=0.20 与 fail-closed gates。

本赛场是 Controlled Megatron E2E，不是 Nine-grid E2E。Human 与 Agent 使用同一个 Snapshot B runner、contract、GPU、shape/config、correctness 与 stability policy。`HUMAN_BLIND=NO`，因为 Human 可以知道历史 v28；这不改变 `COMPARISON_CONTRACT_IDENTICAL=YES`。
