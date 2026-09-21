# AKA-LOCAL V2: 跨项目架构分析与迁移规划

> 基于 Atrex Kernel Agent、KDA (Kernel Design Agents)、CAKE (Compiler-Agent Co-Design)
> 的深度源码审计，对 aka-local 进行架构升级设计。
>
> 分析日期: 2026-09-18
> 分析范围: 四项目完整源码（非仅 README）

---

## 第一部分：四项目架构地图

使用统一概念体系：User / GUI / Controller / Task Contract / Agent / Candidate / Workspace /
Evaluator / Benchmark / Profiler / Diagnostics / Evidence / Knowledge / Memory / Promotion / Rollback

---

### 1.1 Atrex Architecture

`
User
 |
CLI (orchestrator/optimize.py)
 |
Controller (optimize.py::_race_main -> campaign.run())
 |
Task Contract -- IMPLEMENTED (agent_problem.json in operator directory)
 |
Agent -- IMPLEMENTED (orchestrator/agent_runtime/: Codex/Qoder/Claude/Pi adapters)
 |
Candidate -- IMPLEMENTED (Git worktree per episode, long_horizon/git_episode.py)
 |
Workspace -- IMPLEMENTED (isolated Git worktree, orchestrator/workspace_state.py)
 |
Evaluator -- IMPLEMENTED (long_horizon/verifier.py::GatewayABBAValidator, ABBA schedule)
 |
Benchmark -- IMPLEMENTED (remote sandbox ABBA, orchestrator/session_io.py)
 |
Profiler -- IMPLEMENTED (NCU profile driver, PROFILE_DRIVER constant, skills/ppu-acu-joint-profile)
 |
Diagnostics -- IMPLEMENTED (orchestrator/plan_reviewers.py, orchestrator/numerical_policy.py)
 |
Evidence -- IMPLEMENTED (long_horizon/journal.py, memory/v<N>.json per version)
 |
Knowledge -- IMPLEMENTED (gpu-wiki/: technique cards, symptom cards, hardware records)
 |
Memory -- IMPLEMENTED (orchestrator/workspace_state.py::read_memory, canonical v<N>.json)
 |
Promotion -- IMPLEMENTED (framework_baseline_progress.py, commit-based promotion with kernel blob check)
 |
Rollback -- IMPLEMENTED (optimization_policy.py::reject_production_commit, git reset --hard)
 |
Recovery -- IMPLEMENTED (orchestrator/environment_recovery.py, SSH health monitor)
`

**特征**: 全自动长周期优化引擎。每个 episode 在独立 Git worktree 中运行。Agent 由
orchestrator 驱动（而非 Agent 驱动 orchestrator）。Promotion 基于 git commit 的
kernel.py blob 变更，是机械事实而非模型报告。

---

### 1.2 KDA Architecture

`
User
 |
CLI (external: Claude Code / Codex)
 |
Controller -- MISSING (manual, human-driven)
 |
Task Contract -- IMPLEMENTED (prompts/basic-flow.md, structured fill-in)
 |
Agent -- CONCEPT (external coding agent session, workflow-governed)
 |
Candidate -- CONCEPT (task workspace worktree, docs/draft.md -> plan.md)
 |
Workspace -- CONCEPT (separate implementation workspace; no isolation mechanism)
 |
Evaluator -- CONCEPT (validation command in contract)
 |
Benchmark -- CONCEPT (evaluation command, benchmark.csv)
 |
Profiler -- CONCEPT (skills/ncu-report-skill, external skill binding)
 |
Diagnostics -- MISSING
 |
Evidence -- IMPLEMENTED (candidates.jsonl, benchmark.csv, profile/, evidence records)
 |
Knowledge -- IMPLEMENTED (skills/KernelWiki: technique/symptom cards)
 |
Memory -- MISSING
 |
Promotion -- IMPLEMENTED (promotion criteria in contract)
 |
Rollback -- MISSING
`

**特征**: 轻量级工作流模板。不是运行时。定义了 Agent 应该如何工作（contract →
draft → plan → experiment → evidence → decision），但不提供执行引擎。强调
"可复用工作流与任务工作区分离"。

---

### 1.3 CAKE Architecture

`
User
 |
CLI (src/cake_ir/cli.py)
 |
Controller -- MISSING
 |
Task Contract -- MISSING
 |
Agent -- MISSING (compiler, not agent system)
 |
Candidate -- CONCEPT (Kernel IR, src/cake_ir/ir.py)
 |
Workspace -- MISSING
 |
Evaluator -- MISSING (no runtime evaluation)
 |
Benchmark -- MISSING
 |
Profiler -- MISSING
 |
Diagnostics -- IMPLEMENTED (src/cake_ir/diagnostics.py: Diagnostic, VerificationResult, Severity)
 |
Evidence -- CONCEPT (cost.py::Calibration.source as evidence label)
 |
Knowledge -- MISSING
 |
Memory -- MISSING
 |
Promotion -- MISSING
 |
Rollback -- MISSING
 |
Structured IR -- IMPLEMENTED (ir.py: Kernel, Op, Buffer, Role, Pipeline, Barrier, Parameter)
 |
Verifier -- IMPLEMENTED (verifier.py: IR001-005, SCHED001-004, TYPE001, MEM001-002, etc.)
 |
Cost Model -- IMPLEMENTED (cost.py: CostModel, Calibration, roofline estimation)
 |
Compiler -- IMPLEMENTED (compiler.py: Compiler, CudaCodegen, CompilerOptions)
`

**特征**: 纯编译器基础设施。引入 GPU kernel schedule 的结构化 IR 和静态验证。
不涉及 Agent、优化循环、评估器。Cost model 以 calibration evidence 驱动。
"Compiler-Agent Co-Design" 的思想在于：编译器产生结构化输出，Agent 可以消费和推理。

---

### 1.4 aka-local Architecture

`
User
 |
GUI -- IMPLEMENTED (lab/gui.py: Tkinter AgentConsole + WorkspaceManager)
 |
Controller -- IMPLEMENTED (lab/core/controller.py: LabController)
 |
Task Contract -- MISSING
 |
Agent -- IMPLEMENTED (lab/runtime/agent/codex_session.py: CodexAgentSession, gpt-5.6-luna)
 |
Candidate -- IMPLEMENTED (lab/campaigns/<name>/episodes/e####/candidate/)
 |
Workspace -- IMPLEMENTED (per-episode candidate directory, lab/core/workbench.py: Workbench)
 |
Evaluator -- IMPLEMENTED (lab/runtime/evaluators/local.py: RTX5060LocalEvaluator)
 |
Benchmark -- IMPLEMENTED (benchmarks/rmsnorm_abba.py, ABBA protocol)
 |
Profiler -- IMPLEMENTED (lab/runtime/evaluators/ncu_profile.py, NCU profile capture)
 |
Diagnostics -- IMPLEMENTED (lab/runtime/evaluators/diagnostics.py: repeated per-shape, regime analysis, stability classification)
 |
Evidence -- CONCEPT (episode journal.jsonl, evidence directories; no formal evidence chain)
 |
Knowledge -- IMPLEMENTED (lab/knowledge/: anti_strategies, hardware_facts, observations, hypotheses)
 |
Memory -- IMPLEMENTED (lab/core/memory.py: canonical v<N>.json)
 |
Promotion -- IMPLEMENTED (lab/runtime/supervisor/controller_policy.py: mechanical decide())
 |
Rollback -- IMPLEMENTED (lab/runtime/run_state.py: RunStore, checkpoint recovery)
 |
Events -- IMPLEMENTED (lab/core/events.py: append-only sha256-chained event store)
 |
Checkpoint -- IMPLEMENTED (lab/runtime/run_state.py: resume_manifest.json, state.json)
`

**特征**: 基于 Tkinter 的个人 GPU kernel 优化实验平台。人工引导的 experiment 循环。
Controller 决定何时开始实验，Agent 执行编辑，Evaluator 在本地 RTX5060 上编译/正确性/ABBA/NCU。
Supervisor policy 是机械的（不可被 Agent 篡改）。

---

## 第二部分：aka-local 当前定位

### 回答: E. Human-in-the-loop performance engineering platform

**支持证据**:

| 特征 | 证据 |
|---|---|
| GUI | lab/gui.py:1-828 — Tkinter AgentConsole, 实时事件流, 知识卡片, 对话面板 |
| Human directives | lab/core/controller.py::start_optimization — 接受 human directives |
| Agent 为辅助 | lab/runtime/agent/codex_session.py::CodexAgentSession — 仅执行 instruction, 不自主决策 |
| 本地评估 | lab/runtime/evaluators/local.py::RTX5060LocalEvaluator — RTX5060 本地编译/正确性/ABBA/NCU |
| 人机协作 | lab/runtime/supervisor/controller_policy.py::decide() — 机械决策, 不可被 Agent 篡改 |
| 实验会计 | lab/core/experiment_accounting.py — 区分 valid/framework 实验计数 |

**次要定位 (D. Workflow orchestrator)**: 具有 episode 管理、campaign、frontier、journal
等编排能力，但编排决策权在人，Controller 是"人类控制的断路器"而非"自主调度器"。

**排除的理由**:

- **A (Agent wrapper)**: 不是简单 wrapper。Controller 拥有完整的实验生命周期管理、
  评估管道、supervisor 边界、证据收集。
- **B (Experiment manager)**: 实验管理是子集，但缺少 Task Contract 和 hypothesis
  驱动的实验设计。
- **C (Kernel optimization Agent runtime)**: Agent 不能自主驱动优化。没有 Atrex
  那样的自主 episode loop。Human 必须给每一步 directive。

---

## 第三部分：能力矩阵

| 能力 | Atrex | KDA | CAKE | aka-local |
|---|---|---|---|---|
| **Controller** | IMPLEMENTED | MISSING | MISSING | IMPLEMENTED |
| **Agent Runtime** | IMPLEMENTED | CONCEPT | MISSING | IMPLEMENTED |
| **Task Contract** | IMPLEMENTED | IMPLEMENTED | MISSING | MISSING |
| **Hypothesis** | CONCEPT | IMPLEMENTED | MISSING | CONCEPT |
| **Candidate** | IMPLEMENTED | CONCEPT | CONCEPT | IMPLEMENTED |
| **Workspace Isolation** | IMPLEMENTED | CONCEPT | MISSING | IMPLEMENTED |
| **Correctness** | IMPLEMENTED | CONCEPT | MISSING | IMPLEMENTED |
| **Benchmark** | IMPLEMENTED | CONCEPT | MISSING | IMPLEMENTED |
| **Profiler** | IMPLEMENTED | CONCEPT | MISSING | IMPLEMENTED |
| **Diagnostics** | IMPLEMENTED | MISSING | IMPLEMENTED | IMPLEMENTED |
| **Evidence** | IMPLEMENTED | IMPLEMENTED | CONCEPT | CONCEPT |
| **Knowledge** | IMPLEMENTED | IMPLEMENTED | MISSING | IMPLEMENTED |
| **Memory** | IMPLEMENTED | MISSING | MISSING | IMPLEMENTED |
| **Promotion** | IMPLEMENTED | IMPLEMENTED | MISSING | IMPLEMENTED |
| **Rollback** | IMPLEMENTED | MISSING | MISSING | IMPLEMENTED |
| **Structured IR** | MISSING | MISSING | IMPLEMENTED | MISSING |
| **Verifier** | MISSING | MISSING | IMPLEMENTED | MISSING |
| **Cost Model** | MISSING | MISSING | IMPLEMENTED | MISSING |
| **Recovery** | IMPLEMENTED | MISSING | MISSING | CONCEPT |

**解读**:

- aka-local 在 **Controller、Agent、Evaluator、Knowledge、Memory** 上已经全面实现。
- 核心缺口是 **Task Contract**（KDA 强项）、**Structured IR + Verifier + Cost Model**（CAKE 强项）、
  **形式化 Evidence 链**（Atrex + KDA 各有贡献）。
- Atrex 的 **Recovery**（SSH 环境恢复、监控进程）目前 aka-local 只有概念级支持。

---

## 第四部分：设计 aka-local V2

### 目标: Human-guided GPU Kernel Optimization Workbench

`
Human
 |
GUI (aka-local 原有)
 |
Controller (aka-local 原有)
 |
Task Contract (来自 KDA) -- 形式化实验目标、约束、验证命令、晋升标准
 |
Agent Runtime (aka-local 原有 + Atrex 改进)
 |
Candidate (aka-local 原有)
 |
Workspace (aka-local 原有)
 |
Evaluator (aka-local 原有)
 |
Benchmark (aka-local 原有)
 |
Profiler (aka-local 原有)
 |
Diagnostics (aka-local + CAKE 结构 -- 标准化诊断码/严重级别/路径)
 |
Evidence (来自 KDA -- 形式化证据链: Hypothesis -> Experiment -> Evidence -> Decision)
 |
Promotion / Rollback (aka-local 原有)
 |
Knowledge (aka-local 原有 + Atrex gpu-wiki 思想)
 |
Memory (aka-local + Atrex 改进)
`

**来源标注**:

| 层 | 来源 |
|---|---|
| GUI | aka-local 原有创新 (Tkinter AgentConsole) |
| Controller | aka-local 原有创新 (LabController + Workbench) |
| Task Contract | 来自 KDA (basic-flow.md 结构化 contract) |
| Agent Runtime | aka-local 原有 + Atrex 改进 (telemetry, phase markers, usage budget) |
| Candidate | aka-local 原有创新 |
| Workspace | aka-local 原有 + Atrex 改进 (git-backed workspace state) |
| Evaluator/Benchmark/Profiler | aka-local 原有创新 (RTX5060 本地评估) |
| Diagnostics | aka-local 原有 + CAKE 结构 (Diagnostic code + severity + path) |
| Evidence | 来自 KDA (Hypothesis → Experiment → Evidence 链) |
| Promotion/Rollback | aka-local 原有创新 (mechanical supervisor) |
| Knowledge | aka-local 原有 + Atrex gpu-wiki 思想 (technique/symptom cards) |
| Memory | aka-local 原有 + Atrex 改进 (canonical memory with git provenance) |
| Structured IR | 来自 CAKE (未来 P2 方向) |
| Verifier | 来自 CAKE (未来 P2 方向) |
| Cost Model | 来自 CAKE (未来 P2 方向) |

---

## 第五部分：当前缺口分析 (按优先级)

### P0 (必须补，否则架构不完整)

1. **Task Contract 缺失**
   - KDA 的 Task Contract 是实验形式化的基础
   - 当前 aka-local 缺少: 结构化目标定义、约束声明、晋升标准
   - 影响: Campaign 的 manifest (campaign.json/yaml) 包含部分信息但非正式 contract

2. **Hypothesis → Experiment → Evidence 链断裂**
   - KDA 要求: plan draft → hypothesis → experiment → evidence → decision
   - 当前 aka-local: journal.jsonl 记录了实验，但没有 hypothesis 识别和证据链追溯
   - 影响: 无法回答"为什么做了这个实验？期望什么结果？实际什么结果？"

3. **Evidence 形式化不足**
   - 当前: Evidence 以文件形式存在（compile.json, correctness.json），
     但没有统一的 Evidence 类型系统（CONCEPT 级别）
   - Atrex 和 KDA 都有更强的 evidence 概念

4. **Evaluator 未与 Agent 物理隔离**
   - 当前: Evaluator 运行在同一 Python 进程中（虽然通过 subprocess）
   - Atrex: Evaluator 在远程 sandbox（SSH gateway），Agent 无法接触裁判
   - 当前隔离程度足够（Agent 不直接运行 evaluator），但缺少明确的"Agent 不可修改裁判"声明

### P1 (明显增强)

5. **Agent 的 telemetry/phase markers 缺失**
   - Atrex: 完整的 TokenUsage, NormalizedAgentEvent, phase markers (start/end)
   - 当前 aka-local: 简单的 token 计数，无结构化事件流
   - 影响: 无法做精细的 token 预算控制

6. **Profiler 诊断未结构化**
   - CAKE: Diagnostic(code, severity, message, path, notes) 的严格结构
   - 当前: diagnostics.py 产生 summary.json 和 CSV，但诊断码非标准化
   - 改进: 采用 CAKE 的诊断结构（但保留 aka-local 的 per-shape regime 分析）

7. **Knowledge 无法形成规则**
   - 当前: knowledge.py 有 can_promote_to_rule(evidence_levels) 函数但未见使用
   - Atrex: gpu-wiki 的 technique/symptom cards 可被 Agent 查询
   - 改进: 实现 evidence_level 累积 -> rule 自动生成的闭环

8. **Candidate lineage 不完整**
   - 当前: journal.jsonl 按顺序记录，但 parent 关系弱
   - KDA: candidates.jsonl 记录候选名称、parent links、状态
   - 当前已有部分支持（episode 目录编号），但缺少显式 lineage DAG

### P2 (未来研究方向)

9. **Structured IR (CAKE)**
   - CAKE 的 Kernel IR 是 agent-compiler 接口的关键创新
   - 未来方向: Agent 输出 IR 而非直接写 CUDA，编译器生成代码 + 静态验证
   - 当前: Agent 直接编辑 candidate.py（CUDA inline）

10. **Cost Model 引导**
    - CAKE: 校准 roofline 模型预测瓶颈（compute vs memory）
    - 价值: 在编译前即可给 Agent 提供性能上限估计
    - 当前: 依赖实际 ABBA 测量，无先验估计

11. **Remote sandbox (Atrex)**
    - Atrex: SSH gateway 远程评估，环境恢复监控
    - 当前: 纯本地 RTX5060；未来可能的远程 GPU 支持

---

## 第六部分：代码修改计划（仅模块清单，不写代码）

### 新增模块

| 路径 | 作用 | 输入 | 输出 | 连接 |
|---|---|---|---|---|
| lab/core/contract.py | 任务契约定义 | campaign manifest, operator registry | TaskContract 数据结构 | Controller 读取，Agent 接收 |
| lab/core/hypothesis.py | 实验假设定义与追踪 | Agent 的 plan draft, human directive | Hypothesis 记录（claim, mechanism, expected effect） | journal.jsonl 附加字段 |
| lab/core/evidence.py | 形式化证据链 | compile result, correctness, benchmark, profile | Evidence 记录（type, verdict, metrics, provenance） | knowledge 形成规则的基础 |
| lab/core/lineage.py | Candidate 血缘关系 | journal.jsonl, episode 目录 | 显式 lineage DAG（parent, fork, merge） | Controller 决策依据 |
| lab/runtime/evaluators/structured_diagnostics.py | 采用 CAKE 诊断结构 | 诊断 raw 数据 | Diagnostic(code, severity, message, path, notes) 列表 | diagnostics.py 的输出增强 |
| lab/knowledge/rule_engine.py | Evidence level 累积 → rule | observations with evidence levels | 自动生成的规则（promotable rules） | knowledge.py 的 can_promote_to_rule 实现 |

### 修改模块

| 路径 | 改动 | 原因 |
|---|---|---|
| lab/core/controller.py | 集成 Task Contract 读取 | P0 - contract 缺失 |
| lab/runtime/agent/codex_session.py | 添加 telemetry (phase markers, usage deltas) | P1 - Atrex 对齐 |
| lab/runtime/supervisor/controller_policy.py | 增加 hypothesis 验证检查 | P0 - evidence 链 |
| lab/core/context_builder.py | 在 authoritative context 中包含 contract + hypothesis | P0 |

---

## 第七部分：本周目标

### 本周最合理的开发目标

**让 aka-local 成为 Human-guided GPU Kernel Optimization Workbench**

#### 具体任务（按执行顺序）

**Day 1-2: Task Contract (P0)**

- [ ] 在 lab/core/contract.py 中定义 TaskContract 数据结构
  - objective, constraints, validation_command, promotion_criteria, evaluation_command
- [ ] 在 lab/core/controller.py 中集成 contract 读取
  - 从 campaign manifest (campaign.yaml) 解析 contract
  - 在 context_builder 中将 contract 传给 Agent

**Day 3-4: Hypothesis → Evidence 链 (P0)**

- [ ] 在 lab/core/hypothesis.py 中定义 Hypothesis 数据结构
  - claim, mechanism, expected_effect, experiment_id
- [ ] 在 lab/core/evidence.py 中定义 Evidence 数据结构
  - type (CORRECTNESS, PERFORMANCE, DIAGNOSTIC), verdict, metrics, provenance
- [ ] 修改 journal.jsonl 写入逻辑，添加 hypothesis + evidence 字段
- [ ] 修改 controller_policy.py: decide() 接受 evidence 而非裸 dict

**Day 5: 结构化诊断 (P1)**

- [ ] 在 lab/runtime/evaluators/structured_diagnostics.py 中实现
  - 将现有的 per-shape regime 分析、stability classification 映射到 Diagnostic 结构
  - 采用 CAKE 的 code + severity + path 模式但不照搬

**Day 5 收尾: 文档**

- [ ] 更新 docs/ARCHITECTURE_V2_PLAN.md（本报告）
- [ ] 添加 Task Contract 模板示例

#### 不做的

- 不做自主 Agent（保持 human-in-the-loop）
- 不做 multi-agent
- 不重写系统
- 不引入新的大依赖（CAKE 的 IR/compiler 需要在 aka-local 环境编译）
- 不修改 GUI 布局

---

## 附录：四个项目的文件结构对比

### Atrex (关键文件)
`
atrex-kernel-agent/
├── orchestrator/
│   ├── optimize.py           # CLI 入口 + outer loop
│   ├── campaign.py           # 单 operator campaign
│   ├── workspace_state.py    # Git-based canonical state
│   ├── workspace_runtime.py  # Git worktree lifecycle
│   ├── environment_recovery.py  # SSH 恢复
│   ├── optimization_policy.py   # production/leaderboard mode
│   ├── agent_runtime/
│   │   ├── model.py          # AgentRunRequest/Result, TokenUsage
│   │   ├── runtime.py        # CliAgentRuntime
│   │   └── adapter.py        # Codex/Qoder/Claude/Pi adapters
│   └── numerical_policy.py   # 数值正确性门
├── long_horizon/
│   ├── campaign.py           # LongHorizonCampaign state machine
│   ├── verifier.py           # GatewayABBAValidator
│   ├── git_episode.py        # Episode worktree
│   ├── journal.py            # Episode journal
│   └── models.py             # SupervisorState, VerificationResult
├── gpu-wiki/                 # 知识库
└── skills/                   # Agent skills
`

### KDA (关键文件)
`
kda/
├── docs/agent-flow.md        # 工作流描述
├── prompts/basic-flow.md     # 结构化 task contract
├── CLAUDE.md                 # Agent 指令
├── skills/KernelWiki/        # GPU 知识库
└── skills/ncu-report-skill/  # NCU 分析
`

### CAKE (关键文件)
`
cake-ir/src/cake_ir/
├── ir.py                     # Kernel IR 定义
├── verifier.py               # 静态验证
├── diagnostics.py            # 结构化诊断
├── cost.py                   # Cost model
├── compiler.py               # 编译器 facade
├── backends/cuda.py          # CUDA 代码生成
├── passes/memory.py          # 内存规划
└── targets.py                # GPU target 定义
`

### aka-local (关键文件)
`
aka-local/lab/
├── gui.py                    # Tkinter GUI
├── core/
│   ├── controller.py         # LabController
│   ├── context_builder.py    # 权威上下文构建
│   ├── events.py             # 事件存储
│   ├── workbench.py          # Workbench 状态机
│   ├── knowledge.py          # 知识分类
│   ├── memory.py             # 规范内存
│   ├── journal.py            # 实验日志
│   ├── experiment_accounting.py  # 实验会计
│   └── persistence.py        # 原子 JSON 读写
├── runtime/
│   ├── agent/codex_session.py    # Codex Agent 会话
│   ├── evaluators/
│   │   ├── interface.py      # Evaluator Protocol
│   │   ├── local.py          # RTX5060 本地评估器
│   │   ├── diagnostics.py    # per-shape 诊断
│   │   └── ncu_profile.py    # NCU profiler
│   ├── supervisor/controller_policy.py  # 机械决策
│   └── run_state.py          # RunStore + checkpoint
├── knowledge/                # 知识库
├── campaigns/                # Campaign 目录
├── benchmarks/               # 基准脚本
└── tests/                    # 测试
`

---

> **本报告由架构审计生成。未修改任何代码。下一步等待确认后开始 P0 实施。**
