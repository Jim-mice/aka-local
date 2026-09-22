# GPU 实验层

本目录提供本地 GPU 性能工程、实验记录、Workbench、Supervisor、归档和安全边界。当前本地 RTX5060 + RMSNorm 流程只在明确批准后使用；远程优化不会自动启用。

## 只读查看

```powershell
python -m lab.cli status
python -m lab.cli knowledge --operator rms_norm_train --hardware rtx5060_laptop_sm120
python -m lab.tools.validate_lab
python -m lab.tools.rebuild_index
```

registry manifest、experiment record、frontier 文件和 knowledge card 是事实来源；生成的 index 只是导航缓存。

日常流程和安全边界见 [使用指南](../docs/USER_GUIDE.md)。`python -m lab.ui_server` 当前只是 localhost-only 的只读 scaffold，不会自动启动 GPU probe、Agent turn、evaluator、SSH 或远程优化。
