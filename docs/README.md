# 文档索引

本目录面向项目使用者与维护者。科研结论以根目录的五目标汇总和索引为入口，详细历史记录按用途归档。

## 当前阶段入口

- [性能结果概览](PERFORMANCE_OVERVIEW.md)：严格区分 promoted、raw、diagnostic 和 local non-representative 结果。
- [Agent 机制推理能力证据](AGENT_REASONING_EVIDENCE.md)：V1/V2/V3 benchmark 的公开摘要和边界。
- [Representative shape acquisition](audits/real_optimization_loop_004_shape_acquisition.md)：当前继续实验前的 shape blocker。
- [公开阶段快照审计](audits/public_snapshot_20260922.md)：本次文档整理的范围和未执行项目。

- [安装与安全复现](SETUP.md)：本地环境、只读查看命令与可选远程配置。
- [使用指南](USER_GUIDE.md)：CLI 与日常操作说明。
- [复现说明](REPRODUCTION.md)：历史结果的复现边界。
- [报告索引](reports/README.md)：按 Phase 查找历史实验报告。
- `audits/`：完整性、状态、知识隔离和 promotion 规则审计。
- `maintenance/`：维护与 clean-environment requalification 计划。
- `archive/`：长时间运行记录和公开发布维护历史。

阅读历史报告时，请保留 raw result、stability-qualified score 与 promoted score 的区别；它们不能互相替代。
