# 安装与安全复现

## 本地 Windows 环境

请在版本控制之外创建 Python virtual environment。历史上没有保存完整的 package lock，因此仓库不声称提供精确的 pinned version 复现。先检查你准备使用的工作流所需 import；常见可选依赖包括 PyTorch、PyYAML 和 Paramiko。CUDA 工作流还需要与 target 配置兼容的 NVCC toolchain。

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

只安装所选命令实际需要的依赖。本说明不构成启动 GPU campaign 的授权。

## 非破坏性查看

以下命令只读取已有状态，不会启动 Agent campaign 或 GPU benchmark：

```powershell
py -3 -m json.tool .\five_target_campaign_state.json
py -3 -m json.tool .\real_target_campaign_index.json
.\lab.ps1 status
.\lab.ps1 list-ops
```

使用 `run`、`evaluate` 或 replay 命令前，请先阅读 `lab.ps1` 和 `lab/cli.py --help`。部分历史 target 使用专用脚本；收尾报告已经记录这一复现基础设施限制。

## 可选的远程 V100 配置

将 `config/environments/v100.example.yaml` 复制为 `config/environments/v100.yaml`，再仅填写你自己的 host、user 和远程路径。`v100.yaml` 被 Git 忽略。请使用 SSH key、交互式提示或本地被忽略的 secret 机制；不要在命令行或可能被日志记录的环境文件中放置密码。

示例保留公开的科学约束（Tesla V100、CUDA 11.8 与 `sm_70`），但不含可用 endpoint。本地 Megatron checkout 是外部依赖，本 workbench 不会修改它。
