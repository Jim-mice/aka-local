# GPU Learning Lab

本地 RTX5060 + RMSNorm 的真实执行流程、Workbench 批准、Supervisor、归档与安全边界见 [根目录中文用户手册](../USER_MANUAL_CN.md)。远程优化尚未启用。

Personal evidence-grounded GPU performance engineering library for humans and optimization agents. The `lab/` layer is additive: legacy `aka-local` experiments remain in place.

## Quick start

```powershell
python -m lab.cli status
python -m lab.cli knowledge --operator rms_norm_train --hardware rtx5060_laptop_sm120
python -m lab.tools.validate_lab
python -m lab.tools.rebuild_index
```

Source of truth is registry manifests, experiment records, frontier files, and knowledge cards. Generated indexes and views are navigation caches.

For the end-user workflow, safety boundaries, current execution status, and the
Project → Operator → Campaign → Workbench process, read [USER_GUIDE.md](USER_GUIDE.md).

## Workbench phase

`python -m lab.ui_server` starts the current localhost-only control console. It is intentionally a read-only scaffold in this phase: GPU probes, Agent turns, evaluator execution, SSH, and remote optimization are not started automatically. `lab.ps1` provides the same status/validation entry points for PowerShell.
