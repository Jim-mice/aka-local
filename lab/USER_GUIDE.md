# GPU Operator Lab 用户说明

本手册说明当前 `aka-local/lab` 如何管理 GPU 算子实验：导入项目、定义算子、选择 GPU/Backend、创建 Campaign 和 Workbench、查知识、与 Agent 交流、保存 checkpoint，以及未来完整的自动优化和知识更新闭环。

> **当前边界（2026-09-16）**：LOCAL RTX5060 + RMSNorm + CUDA C++ 已接通为用户确认后才运行的执行链：受限 Codex Agent candidate edit、compile、56-shape correctness、development ABBA、Supervisor gate、canonical memory、archive 与 knowledge candidate。远程优化仍未启用。完整且以实际状态为准的操作说明见仓库根目录 `USER_MANUAL_CN.md`。

## 1. 核心对象与边界

```text
Project + Operator + Platform + Backend
                 ↓
              Campaign
                 ↓
        Workbench（批准的 candidate 目录）
                 ↓
Episode → Experiment → Handoff → Supervisor
                 ↓
Canonical memory → Knowledge → Experiment archive
```

| 对象 | 含义 | 当前状态 |
|---|---|---|
| Project | 本地目录或显式 clone 的 Git repository | 可用 |
| Operator | 计算合同，例如 RMSNorm Training | 可用 |
| Platform | GPU/架构，例如 `rtx5060_laptop_sm120` | 可用 |
| Backend | CUDA C++、Triton、CoreX Triton 等 | 可用 |
| Campaign | Operator × Platform × Backend 的实验线 | 可用 |
| Workbench | Agent 唯一允许修改的 candidate snapshot | 可用，需人工批准 |
| Episode / Experiment | 有预算的 Agent 会话/单次假设 | 数据模型已具备；执行未接通 |
| Supervisor | correctness、ABBA、robustness、promotion 的权威边界 | skeleton/设计；执行未接通 |
| Canonical memory / Knowledge | 已验证事实/带 scope 的长期经验 | 可读；自动写入未接通 |

必须始终记住：`选算子 ≠ 开始实验`、`创建 Campaign ≠ 创建 candidate`、`批准 Workbench ≠ 批准优化`、`Agent 建议 ≠ Supervisor promotion`、`runtime 错误 ≠ GPU 优化失败`。

## 2. 启动

在 PowerShell：

```powershell
cd <PROJECT_ROOT>
.\lab.ps1 gui
```

`lab.ps1` 优先使用 `.venv\Scripts\python.exe`。不要用系统 Python 替代它运行 GUI：系统 Python 可能没有 torch，GPU probe 会显示 `ENVIRONMENT_CHANGED`；`.venv` 才是当前已验证的实验 Python。

只打开全局知识浏览器：

```powershell
.\lab.ps1 knowledge-gui
```

常用只读命令：

```powershell
.\lab.ps1 status
.\lab.ps1 validate
.\lab.ps1 knowledge rms_norm_train
```

## 3. 主窗口：先确认当前环境

主窗口顶部横幅和“当前实验台”卡片应展示 Execution、GPU/architecture、Operator、Backend、Campaign、environment fingerprint、Agent 工作目录与 Workbench 状态。

当前预期本地 probe：

```text
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
Compute capability: [12, 0]
Torch: 2.14.0+cu130
Torch CUDA runtime: 13.0
Driver: 610.88
Status: READY
```

environment fingerprint 是由 GPU、driver、CUDA runtime、Torch、compiler、OS、Python、repo/source 等确定性生成的 SHA256，用于保证“这条性能证据到底在哪个环境测得”。如果实际 GPU 与 registry 不匹配，不应登记性能结果。

若“Agent 工作目录”显示“尚未创建”，这是安全默认值：此时没有可写 candidate，Agent 不应改任何源码。

## 4. 只查知识：Knowledge Browser

点击 `KNOWLEDGE`，或运行 `.\lab.ps1 knowledge-gui`。Knowledge Browser 是**全局**证据浏览器，不属于当前 Campaign；即使当前是 RTX 5060 RMSNorm，也能查 BI-V150 SwiGLU 或 V100。

### 知识地图

知识地图按真实 evidence scope 构建：

```text
RMSNorm
└─ RTX 5060 Laptop · sm_120
   ├─ 实验：V0 / V1 / V2
   └─ 知识：RMSNorm V2 ...

Residual RMSNorm
└─ BI-V150 · CoreX

SwiGLU
├─ RTX 5060 Laptop · sm_120
└─ BI-V150 · CoreX
```

`UNKNOWN ⚠ missing metadata` 仅表示某个原始记录确实没有完整 operator/platform scope，例如跨平台 coverage open-question；它不是正常资料被错误归类。

### 搜索与过滤

可组合搜索词、算子、GPU、Backend、类型和语言。

| 目标 | 建议操作 |
|---|---|
| RTX 5060 RMSNorm V2 | 搜 `warp`；算子 RMSNorm；GPU RTX 5060 |
| 归约经验 | 搜 `归约` 或 `warp reduction` |
| 稳定性证据 | 搜 `ABBA` |
| BI-V150 backend 问题 | 搜 `BI-V150`；类型 `BACKEND_QUIRK` |
| BI-V150 SwiGLU 失败方向 | 算子 SwiGLU；GPU BI-V150；类型 `ANTI_STRATEGY` |

搜索索引包括 canonical 原文、中文 localization、record ID、scope、operator/platform/backend ID 和技术别名。语言切换仅改变展示，不会改变 canonical metrics、ID、hash、路径或 evidence scope。

### 阅读实验档案

在“实验记录”页双击记录，打开 Experiment Report。archive 存在时，报告直接读取 `lab/experiment_archive/<experiment_id>/` 下的 `report_zh-CN.md`、`analysis_zh-CN.md`、`report_original.md`、`metrics.json` 与 `provenance.json`。不需要再手工定位历史 `report.json`。

## 5. 配置一个新实验：Workspace Manager

点击 `CHANGE WORKBENCH`。正确顺序：

```text
Project → Operator → Platform/Execution → Backend → Campaign
        → Workbench proposal → 人工批准创建 → Set Active Workbench
```

前半段选择只写入 `lab/runtime/workspace_draft.json`；不会自动创建 candidate、启动 Agent、运行 benchmark 或修改源码。

### 5.1 添加或选择 Project

在“项目”页点击“添加本地项目”，选择目录。系统只读检查路径、Git root、origin remote、HEAD、branch 与 dirty state，并把 metadata 写入 `lab/runtime/projects.json`。随后在列表中选择项目并点击“使用此项目”。这不会修改所选项目。

点击“添加 Git 项目”，输入任意 Git Repository URL。GUI 先展示 destination；只有确认后才执行一次：

```text
git clone <url> lab/sources/<source_id>
```

不会自动 `pull`、`fetch` 或更新来源项目。

### 5.2 选择、发现或手工添加 Operator

“算子 / Backend”页列出 RMSNorm、Residual RMSNorm、SwiGLU、Dense Fused Attention、Vocab Parallel Cross Entropy 与 MoE Grouped MLP。选择后点击“使用此算子”，只更新 draft，并清除不兼容的旧 Campaign selection。

“手工添加算子”可填写显示名称、Operator ID、小写 snake_case ID、类型、Project、reference/candidate boundary、correctness/benchmark command、inputs/outputs/dtypes/shapes。最后必须确认“批准注册”才写入 `lab/registry/operators/<operator_id>.yaml`；不会覆盖已有 manifest。

“发现新算子（只读）”要求先选择 Project 和一个项目内源文件。确认后才会向已有 Codex adapter 发出 advisory 请求。prompt 禁止修改文件、运行命令、编译、benchmark、profile 和启动实验。返回的 proposal 应包含 forward/backward symbols、inputs/outputs、dtype、shape、reference/candidate boundary、依赖和可能的 correctness/benchmark boundary。proposal 不会自动注册；仍须人工批准。

### 5.3 选择硬件和 Execution Target

“硬件 / Execution”页的“重新探测本地硬件”会真实读取 GPU、architecture、Torch、CUDA runtime、driver、nvcc、fingerprint 和状态。点击“使用本地 RTX 5060”仅写入 `platform_id=rtx5060_laptop_sm120` 与 `execution_target=local`。

对于 V100 或 BI-V150，点击“连接并只读探测 SSH”，输入 Host、Port、Username、Password。它只读收集 hostname、OS、GPU、Python、Torch/CUDA runtime、nvcc 和 disk。密码只存在当前 GUI 进程的 `EphemeralCredentialStore`，不会写入 JSON/YAML/event/Git/log，退出 GUI 后消失。

远端探测完成后状态是 `REMOTE WORKBENCH CANDIDATE / WAITING_APPROVAL`，但不会创建 remote directory、复制 source、上传 candidate、启动 Agent 或运行 benchmark。当前 remote workspace sync、remote evaluator、断线恢复与 remote optimization 尚未实现。

### 5.4 选择 Backend

可选 CUDA C++、PyTorch eager、Triton、CoreX Triton、Transformer Engine、cuBLAS、cuBLASLt。Backend 是 campaign identity 的组成部分；切换 Backend 会清除旧 Campaign selection，要求选择或创建兼容 Campaign。

## 6. 创建 Campaign

Project、Operator、Platform、Backend 都确定后，打开“Campaign / Workbench”。“兼容 Campaign”只列出 operator/platform/backend 同时匹配当前 draft 的 campaign。

没有兼容 Campaign 时，点击“新建 Campaign”。确认页展示 Project、Operator、GPU、Backend、source commit 与目标目录。只有“批准创建 Campaign”才建立：

```text
lab/campaigns/<campaign_id>/campaign.yaml
```

创建 Campaign 不会启动 Agent，也不会创建 candidate。

## 7. Workbench proposal 与批准创建

点击“提出 Workbench 方案”。系统生成并保存 `lab/campaigns/<campaign_id>/workbench_proposal.json`，展示 Project、Operator、GPU、Backend、Campaign、source、建议 candidate path 和 copy mode。

建议 path 形如：

```text
lab/campaigns/<campaign_id>/episodes/e0001/candidate/
```

点击“取消”不会创建目录、复制 source、初始化 Git、启动 Agent 或执行 benchmark。

点击“批准并创建 Workbench”后才会创建 candidate directory，从已选 Project 建立 snapshot，排除 `.git`、`.venv`、`__pycache__`、`build`、`dist`、`.pytest_cache`，并写入 `episode.json` 与 `WORKBENCH_PREPARED` event。

接着点击“设为当前实验台”。主窗口会显示真实 absolute candidate path，状态是：

```text
PREPARED
Agent: IDLE
Optimization: NOT STARTED
```

批准 Workbench 不等于批准优化；其目的就是确保 Agent 将来只改明确批准的 candidate root，而不会触及 incumbent、reference、knowledge 或 archive。

## 8. Agent 与用户交流

点击 `AGENT CONSOLE`。

### ASK：只咨询

例子：`RMSNorm 当前最大的优化空间是什么？`、`BI-V150 和 RTX5060 上关于 reduction 有哪些证据？`、`为什么 V3 没有 promotion？`

ASK 会记录 `HUMAN_MESSAGE`；只有用户显式点击后才调用 Codex backend；默认要求简体中文回答；禁止修改文件、运行工具、启动 experiment、编译、benchmark 或 profile；回复记录为 `AGENT_MESSAGE`。backend 不可用或 quota 失败时会显示错误，且不会自动换模型。

### DIRECTIVE：给下一次实验的约束

例子：`下一轮先检查 block size，不要改 reduction algorithm。`

DIRECTIVE 写入 `HUMAN_DIRECTIVE`，但不修改 candidate、不增加 experiment number、不跑 benchmark。设计上它会进入下一次 context snapshot。

## 9. 自动优化 → 知识更新：目标闭环

未来真实执行链应为：

```text
批准并激活 Workbench
  ↓ probe + environment fingerprint
  ↓ mandatory knowledge refresh
  ↓ context_snapshot.json
  ↓ 同一 Agent session 的多个 experiments
  ↓ Agent 仅改 approved candidate root
  ↓ development compile / correctness / benchmark
  ↓ keep_as_best / reject_and_continue / ready_for_gate / blocked
  ↓ handoff.json
  ↓ deterministic Supervisor
  ↓ official correctness + authoritative A/B/B/A + robustness
  ↓ ACCEPT / REJECT / PROMOTE
  ↓ memory/vNNN.json
  ↓ knowledge candidate → evidence/scope gate → card + zh-CN localization
  ↓ experiment archive
```

永远遵守：每个 experiment 前必须 `KNOWLEDGE_REFRESHED`；Agent 的 `keep_as_best` 不是 promotion；只有 Supervisor 能写 `supervisor_decision`；弱收益应进入 repeated robustness；`AGENT_TIMEOUT`、`AGENT_BACKEND_ERROR`、`QUOTA_EXHAUSTED`、`INVALID_HANDOFF`、`INFRA_FAILURE` 不是 GPU anti-strategy；一张 GPU 的结论不自动泛化到另一张 GPU。

### 真实实现状态

| 环节 | 状态 |
|---|---|
| Local GPU probe | 已实现，真实运行 |
| Project / Operator / Backend / Campaign / Workbench 配置 | 已实现 |
| Knowledge Browser / archive reader | 已实现 |
| Agent ASK advisory | 已实现为显式调用，依赖已有 Codex backend |
| Agent 只读 Operator discovery | 已实现为显式调用，需人工注册 proposal |
| 多 experiment long-horizon loop | 未接通 |
| candidate 自动修改 | 未接通 |
| evaluator / authoritative ABBA / robustness | 未接通 |
| deterministic supervisor promotion | skeleton/设计，未接通执行 |
| 自动 canonical memory / knowledge / localization 写入 | 未接通执行 |
| remote experiment / sync / recovery | 未接通 |

所以当前不要把 GUI 的“开始优化”理解成真的在跑优化；它是受控入口占位，不会运行 GPU workload。

## 10. 结束、checkpoint、归档和知识更新

完整 archive 的目标目录：

```text
lab/experiment_archive/<experiment_id>/
├─ metadata.json
├─ report_original.md
├─ report_zh-CN.md
├─ analysis_original.md
├─ analysis_zh-CN.md
├─ metrics.json
├─ provenance.json
├─ knowledge_used.json
├─ knowledge_produced.json
├─ next_directions.json
├─ source_manifest.json
├─ diff.patch
└─ code/ 或 artifact manifest
```

这应让人不依赖聊天记录就能回答：在哪张 GPU、哪个 Backend、哪个 fingerprint、candidate 在哪里、改了什么、correctness/每 shape 性能如何、ABBA/robustness 是否通过、为何接受/拒绝、下一步是什么。

当前“收获当前结果”会写：

```text
lab/campaigns/<campaign_id>/checkpoints/<timestamp>/
├─ checkpoint.json
└─ CHECKPOINT.md
```

它保存当前 controller 状态与事件，是 checkpoint，不等同于 authoritative canonical experiment。

知识更新的正确顺序：

```text
live experiment → Supervisor confirmed canonical experiment
→ knowledge candidate → evidence/scope review
→ knowledge card → zh-CN localization overlay
```

禁止 `live.json → SUPPORTED_RULE`。中文只是一层 localization，不能改 metrics、ID、hash、代码符号、硬件/Backend 名称、shape 或 evidence scope。

## 11. 日常工作方式与故障判断

只学习：打开 Knowledge Browser，筛选 operator/GPU/mechanism，打开实验报告。无需 Campaign 或 Agent session。

只讨论：打开 Agent Console，ASK 后阅读 scoped evidence；必要时写 DIRECTIVE；不启动实验。

准备新线：Workspace Manager → Project → Operator → GPU probe → Backend → Campaign → Workbench proposal → 批准 candidate snapshot → Set Active Workbench → 停在 `PREPARED / Agent IDLE`。

| 现象 | 含义 | 正确处理 |
|---|---|---|
| `ENVIRONMENT_CHANGED` | Python/GPU 与 registry 不匹配 | 用 `.\lab.ps1 gui` 启动，重新 probe；不记录性能结果 |
| `UNKNOWN ⚠ missing metadata` | 老 evidence 缺少 scope | 查看来源；不当作特定 operator 的结论 |
| Agent backend error | Codex backend/依赖/quota 不可用 | 不自动切模型；保留上下文后重试 |
| 没有 compatible Campaign | 当前组合没有实验线 | 显式创建新 Campaign；不混用旧数据 |
| 没有 Workbench | candidate 未获批准创建 | 审核 proposal 后批准；不要直接改 incumbent |
| Remote `WAITING_APPROVAL` | 已只读探测，尚未批准 | 当前不能运行 remote optimizer |

## 12. 目录速查与启动前检查

```text
lab/
├─ registry/                 # platform/operator/backend manifests
├─ campaigns/                # campaign、episode、workbench、checkpoint
├─ experiments/              # canonical/imported experiment records
├─ experiment_archive/       # 可读报告、分析、指标、来源
├─ knowledge/                # scoped knowledge cards
├─ i18n/zh-CN/               # localization overlay，不是第二事实源
├─ runtime/                  # events、projects、workspace draft
├─ core/                     # controller、probe、workbench 等
├─ gui.py                    # Tkinter GUI
└─ USER_GUIDE.md             # 本文
```

真正接通自动执行链前，至少检查：GPU/architecture/fingerprint 正确；Project 是正确源目录；Operator/Backend 对应真实代码边界；Campaign 未混入另一 GPU/Backend 数据；Workbench candidate path 已人工批准；incumbent/reference/knowledge/archive 被保护；每轮可强制 knowledge refresh；development 和 authoritative evaluator 分离；弱收益有 robustness policy；停止后可从落盘事件和 checkpoint 恢复。

在自动执行链真正接通前，正确使用方式是：**用 Lab 管理知识、项目和经批准的候选工作台；不要把“开始优化”误认为已经在运行真实长时域优化 Agent。**
