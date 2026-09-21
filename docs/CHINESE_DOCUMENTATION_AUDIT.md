# 中文文档审计

## 扫描范围

本次扫描了 233 个 tracked Markdown 文件。中文化优先覆盖公开入口和日常说明：`README.md`、`SECURITY.md`、`docs/SETUP.md`、`docs/REPRODUCTION.md`、`docs/README.md`、`docs/reports/README.md`、五目标汇总/索引，以及 Phase 19-A 收尾报告。

## 处理规则

- 说明性 prose 使用中文；技术术语按工程习惯保留英文。
- code fence、命令、JSON/YAML 字段、ABI、symbol、marker、hash、状态枚举、数值与原始 compiler/terminal/NSYS evidence 不翻译。
- 历史 Agent prompt 与 campaign 内的输入记录保留英文原文，以维持实验可复现性。

## 保留英文的主要类别

- 历史 Phase 报告中作为原始 evidence 保存的命令、trace、错误输出、表内状态值和技术实现描述。
- `campaigns/**/AGENT.md`、`lab/experiment_archive/**` 等历史 Agent 输入/输出与 archive 原文。
- 文件名、路径、代码、CLI、hash、contract、状态枚举和行业专有名词。

## 链接与人工检查

Markdown 本地相对链接已扫描，未发现失效目标；未发现跨文档 heading anchor 链接。剩余历史报告以原始实验记录为主，若需要逐段中文注释，应在不改动 evidence 的前提下另行增加中文导读，而不改写原始输出。
