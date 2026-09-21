# aka-local 复现说明

## 前置条件

### 硬件

- 可通过 SSH 访问的 NVIDIA Tesla V100-PCIE-16GB（或兼容 GPU）。
- V100 服务器上的 CUDA 11.8+ 与 `nvcc`。

### 本地环境

- Windows 10+ 或 Linux；历史验证使用 Windows PowerShell。
- Python 3.11+ 与 OpenSSH client。
- 能访问 V100 SSH 服务的网络。

### Python 包

```text
paramiko>=3.0
scp>=0.14
pyyaml>=6.0
```

安装：`pip install paramiko scp pyyaml`。

可选的 Codex Agent 用于自主 candidate 生成，需要 Codex CLI 和 OpenAI API access。

## 安装

```bash
git clone <repo-url> aka-local
cd aka-local
python -m venv .venv
.venv\Scripts\activate
pip install paramiko scp pyyaml
```

## 配置

将 `config/environments/v100.example.yaml` 复制为被 Git 忽略的 `config/environments/v100.yaml`，填写自己的 host、user 和远程路径。请使用 SSH key、交互式认证或本地被忽略的 secret 机制；不要把密码写入命令、文档、`.env` 或 Git。

评估 shape 配置位于 `config/environments/v100_sm70/evaluation.json`。其中的 `score`、`correctness_tolerance`、`warmup` 和 `iterations` 是机器可读字段，不应随中文化修改。

## 首次运行前的查看

以下命令会检查已配置的环境；只有在你明确计划进行远程评估时才执行：

```bash
python -m lab.cli doctor --env v100
```

列出可用 operator：

```bash
python -m lab.cli operators
```

`run`、`evaluate` 和 `replay` 可能编译或使用远程 evaluator。执行前先核对 target、candidate、环境和成本边界。

## 目录结构

```text
aka-local/
  config/environments/     # 环境配置
  operators/               # 算子定义与 reference
  campaigns/               # 实验 campaign（episode、lineage）
  knowledge/environments/  # 按环境隔离的知识（V100/RTX5060）
  lab/                     # CLI、evaluator、runtime、工具
  docs/                    # 文档
```

## 常见问题

| 现象 | 检查方向 |
|---|---|
| SSH 连接失败 | 检查本地 `v100.yaml` 的 host/user 与认证方式。 |
| 找不到 `nvcc` | 确认 V100 上安装 CUDA 11.8。 |
| 找不到 `eval.sh` | 确认 evaluator 位于配置的远程路径。 |
| Compile 失败 | 确认 `nvcc` 支持 `sm_70`。 |
| Correctness 失败 | 检查 `candidate.cu` 是否符合 contract signature。 |
