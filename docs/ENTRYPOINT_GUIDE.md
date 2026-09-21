# AKA-Local Entry Point Guide

> Generated: 2026-09-19
> Machine: RTX 5060 Laptop (sm_120), CUDA 13.4
> Status: READ-ONLY reconnaissance — no code changes

---

## 1. Agent 入口

### Agent 是谁？

```
LongHorizonRunner  (lab/runtime/agent/long_horizon.py)
  └── CodexAgentSession  (lab/runtime/agent/codex_session.py)
```

### 入口函数

| 层级 | 类/函数 | 说明 |
|------|---------|------|
| Agent Loop | `LongHorizonRunner.run()` | 完整 experiment loop: plan → edit → compile → correctness → benchmark → gate |
| Agent Session | `CodexAgentSession` | 封装 Codex API 调用 |

### 调用链

```
┌─────────────────────────────────────────┐
│  lab.ps1 gui       (PowerShell)         │  ← GUI 入口
│  lab.ps1 ui        (PowerShell)         │  ← Web UI 入口
│  continuous_runner.py                    │  ← 命令行批处理入口
│  run_episode.py                          │  ← 机械评估（无 Agent）入口
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  LabController        (lab/core/controller.py) │
│  ├── configure_local()                  │  ← 初始化 Workbench
│  ├── start_optimization()               │  ← 启动优化线程
│  ├── resume_recovered()                 │  ← 恢复中断任务
│  └── probe_local()                      │  ← GPU 环境探测
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  LongHorizonRunner   (lab/runtime/agent/long_horizon.py) │
│  └── run()                              │
│      ├── start marker (RecoveryManager) │
│      ├── probe environment              │
│      └── while budget:                  │
│          ├── refresh knowledge          │
│          ├── build context              │
│          ├── _plan() → Agent generates  │
│          ├── _edit() → Agent writes code│
│          ├── compile() → RTX5060Eval    │
│          ├── check_correctness()        │
│          ├── development_benchmark()    │
│          ├── evidence verification      │
│          └── _gate() → Supervisor decide│
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  RTX5060LocalEvaluator                   │
│  (lab/runtime/evaluators/local.py)       │
│  └── atrex-bench compile/correctness/ABBA│
└─────────────────────────────────────────┘
```

---

## 2. Campaign 入口

### 已有 Campaigns

```
campaigns/
  agent_smoke/        # 冒烟测试 (swiglu kernel)
  local_gateway_v0/   # 本地 gateway 实验
  luna/               # gpt-5.6-luna 实验 (episode_3)
  luna_episode_1/     # luna 首次实验
  luna_episode_2/     # luna 二次实验
  rms_norm/           # RMSNorm 连续实验 (episode_2)
```

### 创建 Campaign 的最小示例

**方法 A: 通过 continuous_runner.py (推荐)**

```powershell
# 生成下一集 candidate (Agent 写 kernel)
.\.venv\Scripts\python.exe continuous_runner.py --operator rms_norm --max-model-turns 1

# 如果 candidate 已存在，跳过 Agent 直接机械评估
.\.venv\Scripts\python.exe continuous_runner.py --operator rms_norm
```

**方法 B: 通过 LabController API**

```python
from lab.core.controller import LabController

ctrl = LabController(Path("C:/Users/38154/projects/aka-local"))
ctrl.configure_local("rms_norm_train", "cuda_cpp", "rms_norm_train__rtx5060_sm120__cuda_cpp")
ctrl.probe_local()  # 探测 GPU 环境

# 启动优化（需要先创建 candidate 目录）
ctrl.start_optimization(
    campaign_id="rms_norm_train__rtx5060_sm120__cuda_cpp",
    candidate_root=Path("campaigns/rms_norm/episode_3/candidate"),
    incumbent=Path("ops/rms_norm_v2"),
    budget={"max_experiments": 5, "max_consecutive_failures": 3}
)
```

**方法 C: 通过 run_episode.py (纯机械评估，无 Agent)**

```powershell
.\.venv\Scripts\python.exe run_episode.py --episode campaigns/luna/episode_3
```

---

## 3. GUI 入口

### 存在两个 GUI

| GUI | 文件 | 类型 | 启动命令 | 地址 |
|-----|------|------|----------|------|
| **Tkinter Desktop GUI** | `lab/gui.py` | 本地桌面 | `.\lab.ps1 gui` | 桌面窗口 |
| **Web UI (HTTP)** | `lab/ui_server.py` | 本地 Web | `.\lab.ps1 ui` | http://127.0.0.1:8765/ |

### Tkinter GUI

```powershell
.\lab.ps1 gui
```

功能：
- Campaign 浏览器
- Knowledge 浏览器（经验卡片 + 实验记录）
- 实验启动/暂停/恢复
- Human directive 发送

### Web UI

```powershell
.\lab.ps1 ui
```

功能：
- Dashboard: 当前 Workbench 状态
- Dashboard: 最近 events
- Human directive 输入框
- Harvest（收割实验结果）

### 状态

GUI 是完整的本地工具，非半成品。两者都通过 `LabController` 操作同一个后端。

---

## 4. CLI 入口

### PowerShell 包装器 (`lab.ps1`)

```powershell
.\lab.ps1 <command>
```

| 命令 | 说明 |
|------|------|
| `gui` | 启动 Tkinter 桌面 GUI |
| `ui` | 启动 Web UI (http://127.0.0.1:8765) |
| `status` | 显示 STATUS.md |
| `knowledge --operator <id>` | 查询 knowledge |
| `validate` | 验证 lab 结构 |
| `knowledge-gui` | Knowledge 浏览器 |
| `knowledge-import` | 导入外部 knowledge |
| `backfill` | 回填 archive |

### Python CLI (`lab/cli.py`)

```powershell
.\.venv\Scripts\python.exe -m lab.cli <command>
```

| 命令 | 说明 |
|------|------|
| `status` | 显示项目状态 |
| `platforms` | 列出已知平台 |
| `operators` | 列出已知算子 |
| `campaigns` | 列出 campaigns |
| `knowledge` | 查询 knowledge |
| `validate` | 验证项目结构 |
| `recover --list` | 列出可恢复的中断 runs |
| `recover --inspect <run_id>` | 检查 run 状态 |
| `recover --resume <run_id>` | 恢复 run（需人工确认） |

### continuous_runner.py

```powershell
.\.venv\Scripts\python.exe continuous_runner.py [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--operator` | `rms_norm` | 算子名称 |
| `--resume` | false | 恢复模式 |
| `--max-model-turns` | 1 | Agent 调用次数 |
| `--max-episodes` | 3 | 最多 episode 数 |
| `--stop-on-timeout` | false | 超时停止 |
| `--stop-on-quota` | false | 配额用尽停止 |

### run_episode.py

```powershell
.\.venv\Scripts\python.exe run_episode.py --episode <path> [--baseline <path>]
```

无 Agent 参与，纯机械评估（compile → correctness → ABBA benchmark → decision）。

---

## 5. First Experiment 流程

### 从零开始：RMSNorm CUDA Kernel Optimization

#### 步骤 0: 前提检查

```powershell
# 确认 GPU 可用
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.get_device_name(0))"
# 期望: NVIDIA GeForce RTX 5060 Laptop GPU

# 确认 atrex-bench 可用
dir <LOCAL_USER_HOME>\projects\atrex-bench\src\atrex_bench
```

#### 步骤 1: 确认 baseline (已有)

```powershell
# RMSNorm reference kernel (PyTorch 参考实现)
dir ops\rms_norm\

# 输出:
#   input.py          ← 输入生成
#   reference.py      ← PyTorch RMSNorm 参考
#   shapes.json       ← 56 种 shape (token_count × hidden_size)
#   metadata.json     ← 算子元数据
#   roofline.json     ← Roofline 模型

# 当前最优 incumbent (v2 — Agent 已优化过两轮)
dir ops\rms_norm_v2\
# 输出:
#   candidate.py      ← 当前最优 CUDA kernel
#   hypothesis.json   ← 优化假设
#   AGENT.md          ← Agent 优化说明
```

#### 步骤 2: 运行 continuous_runner（最简单）

```powershell
cd <PROJECT_ROOT>

# 让 Agent 生成下一个 episode 的 candidate
.\lab.ps1\.venv\Scripts\python.exe continuous_runner.py --operator rms_norm --max-model-turns 1
```

#### 步骤 3: 观察结果

```powershell
# 查看新 episode
dir campaigns\rms_norm\episode_3\

# 关键文件:
#   candidate.py      ← Agent 生成的新 kernel
#   AGENT.md          ← Agent 优化说明
#   hypothesis.json   ← 优化假设
#   decision.json     ← 机械评估结果 (compile/correctness/benchmark/gate)
```

#### 步骤 4: 用 GUI 监控

```powershell
# 终端 1: 启动 Web UI
.\lab.ps1 ui

# 浏览器打开 http://127.0.0.1:8765/
# 查看 Dashboard 和 Events
```

---

## 项目架构总览

```
aka-local/
├── lab.ps1                 ← PowerShell CLI 包装器
├── continuous_runner.py    ← 批处理 Agent Runner
├── run_episode.py          ← 纯机械评估（无 Agent）
│
├── lab/
│   ├── cli.py              ← Python CLI (lab recover, lab status...)
│   ├── gui.py              ← Tkinter 桌面 GUI
│   ├── ui_server.py        ← HTTP Web UI (localhost:8765)
│   ├── core/
│   │   ├── controller.py   ← LabController (总控)
│   │   ├── decision.py     ← Decision 数据类
│   │   ├── evidence.py     ← Evidence 数据类
│   │   ├── hypothesis.py   ← Hypothesis 数据类
│   │   └── events.py       ← EventStore
│   └── runtime/
│       ├── agent/
│       │   ├── long_horizon.py   ← LongHorizonRunner (Agent loop)
│       │   └── codex_session.py  ← Codex API session
│       ├── supervisor/
│       │   └── controller_policy.py  ← decide() / Supervisor
│       ├── evaluators/
│       │   └── local.py     ← RTX5060LocalEvaluator
│       ├── recovery.py      ← RecoveryManager
│       ├── workspace.py     ← CandidateWorkspace
│       └── run_state.py     ← RunStore
│
├── ops/
│   ├── rms_norm/            ← RMSNorm 参考实现 (PyTorch)
│   ├── rms_norm_v1/         ← Agent 第1轮优化结果
│   ├── rms_norm_v2/         ← Agent 第2轮优化结果 (当前incumbent)
│   └── swiglu_forward_v2b/  ← SwiGLU 参考实现
│
├── campaigns/
│   ├── rms_norm/            ← RMSNorm 连续实验
│   │   ├── episode_2/       ← 上轮实验
│   │   └── episode_3/       ← 下一轮 (continuous_runner 创建)
│   └── luna*/               ← gpt-5.6-luna 历史实验
│
├── knowledge/
│   ├── experience/          ← 经验卡片 (.json)
│   ├── frontier/            ← 前沿方向
│   └── lessons/             ← 经验教训
│
└── benchmarks/              ← ABBA benchmark 脚本
```

---

## 快速命令速查

```powershell
# === GUI ===
.\lab.ps1 gui          # 桌面 GUI
.\lab.ps1 ui           # Web UI → http://127.0.0.1:8765

# === CLI ===
.\lab.ps1 status       # 查看状态
.\lab.ps1 validate     # 验证项目
.\lab.ps1 knowledge --operator rms_norm   # 查看 RMSNorm 知识

# === Recovery ===
.\.venv\Scripts\python.exe -m lab.cli recover --list
.\.venv\Scripts\python.exe -m lab.cli recover --inspect <run_id>
.\.venv\Scripts\python.exe -m lab.cli recover --resume <run_id>

# === 运行实验 ===
.\.venv\Scripts\python.exe continuous_runner.py --operator rms_norm --max-model-turns 1
.\.venv\Scripts\python.exe run_episode.py --episode campaigns/rms_norm/episode_3

# === 测试 ===
.\.venv\Scripts\python.exe -m pytest lab/tests/ -q
