# GPU Operator Lab 用户手册

> 当前真正可执行的路径是：**LOCAL · RTX 5060 Laptop · RMSNorm Training · CUDA C++**。远程优化尚未启用。

## 1. 这是什么

GPU Operator Lab 把 Workbench、知识库、Codex Agent、GPU evaluator、Supervisor 和归档放到同一条可追溯的实验链中。Agent 对话不是事实源；`context_snapshot.json`、journal、evidence、canonical memory 和 archive 才是。Agent 只能改批准的 candidate 根目录；incumbent、reference、benchmark、registry、knowledge 与其它 campaign 均受保护。Agent 只能建议继续、拒绝、请求 gate 或阻塞；Supervisor 才能 promotion。

## 2. 快速启动

```powershell
cd <PROJECT_ROOT>
.\lab.ps1 gui
```

顶部必须显示 `LOCAL`、`NVIDIA GeForce RTX 5060 Laptop GPU`、`sm_120` 与 `READY`。启动会实际 probe GPU、Torch/CUDA、driver、Python、编译器与 environment fingerprint；不匹配 registry 时 Start 被阻止。

只读知识库：

```powershell
.\lab.ps1 knowledge-gui
```

## 3. Workbench：先确认代码改在哪里

Workbench 是 `Project + Operator + Platform + Execution target + Backend + Campaign + candidate directory + environment fingerprint` 的明确组合。它不是随意的源代码目录。

1. 进入 **CHANGE WORKBENCH**，选 Project、Operator、Hardware、Backend、Campaign。
2. 生成 proposal，核对 source、只读 incumbent 和完整 candidate 路径。
3. 点击“批准并创建 Workbench”后，系统才创建 `lab\campaigns\<campaign>\episodes\eNNNN\candidate` 快照。
4. 点击“设为当前实验台”。此时 Agent 仍是 IDLE，未执行任何优化。

取消 proposal 不会创建目录、candidate 或 Git state。Controller 与 PathPolicy 强制 candidate root；candidate 外写入、symlink/junction escape、或把既有源码压缩为近乎空文件均会成为 `PATH_POLICY_VIOLATION`。

## 4. 第一次本地 RMSNorm 实验

> 点击 Start 会调用 `GPT-5.6 Luna / low`、修改已批准 candidate、编译 CUDA、运行 GPU correctness/benchmark。它不会改写 `ops\rms_norm_v2`、reference 或历史 knowledge；promotion 只会在 Lab campaign 内写新 incumbent snapshot。

1. 启动 GUI，确认 RTX5060 / `sm_120` / `READY`。
2. 选择 `RMSNorm Training`、`CUDA C++` 和本地 campaign。
3. 批准 Workbench，核对首页的 Agent 工作目录绝对路径。
4. 可先看 Knowledge Browser / Agent Console，确认 scope 与 context。
5. 点击 **START OPTIMIZATION**。
6. 在二次确认框核对 GPU、campaign、candidate、只读 incumbent、Agent 与预算，再点“开始优化”。
7. 观察 Agent 状态、timeline 与 evidence；运行时不要手改 candidate。
8. 随时 Pause、Safe Stop 或 Harvest。

## 5. Start 后真实发生什么

```text
用户确认 Start
 -> LabController（后台线程）
 -> probe + environment fingerprint
 -> KNOWLEDGE_REFRESHED / knowledge_snapshot.json
 -> context_snapshot.json
 -> 同一 Codex AgentSession：结构化计划
 -> candidate 内最小 edit + path/source guard
 -> compile
 -> 56-shape correctness
 -> development same-process A/B/B/A
 -> Agent 公开解释
 -> 下一 experiment / ready_for_gate / blocked
 -> Supervisor gate（仅 ready_for_gate）
 -> authoritative A/B/B/A；弱收益时 repeated robustness
 -> PROMOTE 或 REJECT
 -> canonical memory、archive、knowledge candidate、zh-CN localization
```

每一 experiment 必须先写 `KNOWLEDGE_REFRESHED`，再允许 `HYPOTHESIS_RECORDED`。同一 episode 可进行多个 experiment，但受 `max_experiments` 与连续失败预算约束。GUI/Agent 崩溃后，可用 journal、live、events、checkpoint、canonical memory 和新 session 恢复；不依赖旧聊天。

## 6. 指标、ABBA 与 Supervisor

- **compile**：触发既有 CUDA extension build/import。
- **correctness**：官方 eager oracle 下 56 个 shape。
- **development ABBA**：同进程 A/B/B/A，供 Agent 选择方向，不能 promotion。
- **authoritative ABBA**：仅 `ready_for_gate` 后运行。
- **repeated robustness**：本 campaign 的 0–2% 弱收益必须复测。

Supervisor 不调用 LLM。规则：compile fail → `REJECT_COMPILE`；correctness fail → `REJECT_CORRECTNESS`；不快于 incumbent → `REJECT_PERFORMANCE`；0–2% → `ROBUSTNESS_REQUIRED`；清晰可复现收益 → `PROMOTE`。Promotion 仅创建 `lab\campaigns\<campaign>\incumbent\...`，不会覆盖 `ops\rms_norm_v2`。

## 7. Pause、Stop 与 Harvest

- **PAUSE AFTER STEP**：当前原子操作结束后 `PAUSED`，不启动新 experiment。
- **RESUME**：重新 probe、确认 fingerprint、重建 knowledge/context 后继续。
- **STOP SAFELY**：不再开新 experiment，当前 evaluator 返回后 checkpoint 并关闭 Agent session。
- **EMERGENCY STOP**：当前版本没有可靠 Windows process-tree UI；优先 Safe Stop，必要时在系统进程管理器终止，并保留证据目录。
- **HARVEST RESULTS**：写 `checkpoints\<timestamp>\checkpoint.json` 和 `CHECKPOINT.md`，汇总环境、workbench、candidate/incumbent、journal、指标、blocker、frontier、知识和 Git lineage。

## 8. 知识库、归档与中文

Knowledge Browser 可按 operator/GPU/backend/type/关键词查找。中文是 localization overlay，不是第二套事实；数字、ID、代码、路径、shape、GPU/backend scope 不会被翻译改写。

Supervisor 完成后会写 `memory/vNNN.json`；从证据自动产生的知识先进入 `lab\knowledge\pending\`，不会凭一次实验自动成为 `SUPPORTED_RULE`。同时每个真实 experiment 归档到 `lab\experiment_archive\<experiment_id>\`，其中有 original/zh-CN report、analysis、metrics、provenance、knowledge、next directions 与 diff/source manifest。

知识类型：`OBSERVATION`、`HYPOTHESIS`、`SUPPORTED_RULE`、`ANTI_STRATEGY`、`BACKEND_QUIRK`、`HARDWARE_FACT`、`OPEN_QUESTION`。scope 永远包含硬件/backend/operator 条件，不能跨 GPU 自动泛化。

## 9. Agent Console

- **ASK**：只读咨询当前 knowledge/context；不改 candidate、不运行 evaluator、不新建 experiment。
- **DIRECTIVE**：保存 pending 指令，下一 experiment 的 context snapshot 自动包含；不改 canonical history。

用户可见回复默认简体中文，技术名、代码、指标、shape 与路径保留原文。

## 10. 数据位置与禁止手工改动的目录

```text
lab/campaigns/<campaign>/episodes/       candidate、live、journal、evidence
lab/campaigns/<campaign>/memory/         canonical memory
lab/campaigns/<campaign>/incumbent/      Lab-only promoted snapshots
lab/campaigns/<campaign>/checkpoints/    harvest/stop checkpoint
lab/knowledge/                           canonical cards/pending candidates
lab/experiment_archive/<experiment>/     可直接阅读的报告和指标
lab/i18n/zh-CN/                          中文 localization
lab/runtime/events.jsonl                 append-only 事件日志
ops/rms_norm_v2/                         历史只读 incumbent
```

不要手动编辑 `events.jsonl`、`memory/`、已完成 archive 或 `incumbent/`。要做新实验就新建 approved Workbench。

## 11. 常见故障

- `AGENT_BACKEND_ERROR` / `QUOTA_EXHAUSTED`：停止；不自动换模型；用文件状态新建 session 恢复。
- `PATH_POLICY_VIOLATION`：路径逃逸/破坏性写入被阻止；它是运行时问题，不是 GPU 反策略。
- `REJECT_COMPILE` / `REJECT_CORRECTNESS`：是 candidate evidence；保留代码，incumbent 不动。
- `ENVIRONMENT_CHANGED`：重新 probe 后人工确认，不能静默继续。
- 翻译缺失：查看 original；translation 不影响 canonical technical facts。

## 12. CLI 速查与 Remote 状态

```powershell
.\lab.ps1 gui
.\lab.ps1 knowledge-gui
.\lab.ps1 status
.\lab.ps1 validate
```

**REMOTE OPTIMIZATION NOT YET ENABLED。** 请勿将当前 remote probe 当作 remote execution/sync/recovery。下一阶段才会单独实现 SSH bundle、remote evaluator、event spool 与 reconcile；本地 control plane/knowledge 仍是 source of truth。

## 13. 紧急停止与跨进程恢复

`EMERGENCY STOP` 会二次确认，并只终止当前 `run_id` 在 `lab/runtime/runs/<run_id>/processes.json` 中登记的 Agent/evaluator 进程树。它先尝试终止，超时后以 Windows `taskkill /PID <pid> /T /F` 强制结束该树；不会扫描或终止其它 Codex、Python、nvcc 或 cl.exe 进程。已完成 evaluator 结果仍保留；未完成操作只记录为 `INTERRUPTED`，绝不会自动变成 `REJECT_PERFORMANCE`、`REJECT_CORRECTNESS` 或 GPU 反策略。

每个 run 均持久化 `run.json`、`resume_manifest.json`、`state.json`、`checkpoint.json` 和 process registry，关键 JSON 使用 atomic replace。重新打开 GUI 后，Controller 扫描 `lab/runtime/runs/` 中的 `PAUSED`、`INTERRUPTED` 或 orphaned `RUNNING` run；系统不会自动继续，用户必须确认 Resume。

Resume 会重新 probe RTX5060、验证 environment fingerprint、candidate/incumbent/hash/journal，然后刷新知识和 context 并创建**新的** Codex session。旧 Codex thread 不作为恢复依据；恢复只依赖 campaign、candidate、journal、events、canonical memory、knowledge、snapshots 和 resume manifest。
