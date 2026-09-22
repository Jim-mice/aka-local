# aka-local 使用指南

## 先阅读安全边界

本仓库同时包含本地实验、历史 V100 证据和只读审计。默认只查看状态，不会自动启动 Agent、CUDA benchmark、SSH 或远程 evaluator。公开性能结论请先阅读 [性能结果概览](PERFORMANCE_OVERVIEW.md)。

## 非破坏性查看

```powershell
py -3 -m json.tool .\five_target_campaign_state.json
py -3 -m json.tool .\real_target_campaign_index.json
.\lab.ps1 status
.\lab.ps1 list-ops
python -m lab.cli status
```

这些命令用于查看已有状态和索引。执行 GPU campaign 前必须确认 target、baseline、scope、认证方式和远程权限，不要把历史报告中的命令当作本轮授权。

## 研究流程

```text
真实执行边界 → 冻结契约 → 生成候选 → 编译与正确性
→ 统计性能 → 稳定性门禁 → 接受或拒绝 → 经验反馈
```

候选必须先通过 correctness，再讨论性能；raw ratio 不等于 promoted score，micro speedup 不等于 model speedup。

## 文档入口

- [安装与安全复现](SETUP.md)
- [复现说明](REPRODUCTION.md)
- [性能结果概览](PERFORMANCE_OVERVIEW.md)
- [文档索引](README.md)
- [安全说明](../SECURITY.md)

历史 CLI 参数、代码块和状态枚举可能来自旧阶段；使用前请核对当前 campaign contract 和环境配置。远程 V100 说明只描述历史/可选能力，不是当前阶段的执行要求。
