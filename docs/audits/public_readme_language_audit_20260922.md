# 公开 README 中文化审计

审计范围：当前 Git-backed 发布副本中的 tracked `README.md` 文件。判断只针对面向项目使用者或维护者的人类说明；fixture、第三方 vendored 内容和原始证据不为中文化而改写。

| path | 分类 | 是否中文 | 是否修改 | 原因 |
|---|---|---|---|---|
| `README.md` | A：项目入口 | 是 | 是 | 重写为中文公开阶段入口，保留路径、状态枚举和技术名词。 |
| `docs/README.md` | A：文档入口 | 是 | 是 | 增加当前阶段和最新审计入口。 |
| `docs/reports/README.md` | A：报告入口 | 是 | 是 | 增加当前阶段入口，保留历史报告文件名。 |
| `docs/reports/README.md` | A：报告入口 | 是 | 是 | 增加当前阶段入口，保留历史索引。 |
| `lab/README.md` | A：实验层入口 | 是 | 是 | 将面向人的说明中文化。 |
| `lab/runtime/agent/README.md` | A：Agent 边界说明 | 是 | 是 | 将边界和权限说明中文化。 |
| `lab/operators/*/README.md` | A：operator registry 入口 | 是 | 是 | 保留 operator 名和状态枚举，中文化说明。 |
| `lab/learning/README.md` | A：维护者说明 | 是 | 是 | 中文化解释层说明。 |
| `lab/i18n/zh-CN/README.md` | A：本地化说明 | 是 | 否 | 原已为中文。 |
| `benchmarks/intuition/evaluator_only/README.md` | B：evaluator-only fixture 说明 | 否 | 否 | 机器评估包的一部分；不改写以免改变 blind package。 |

结论：面向人的公开入口已使用中文；代码块、日志、hash、路径、JSON key、状态枚举和 API 名称按原样保留。
