# V100 RMSNorm Overnight E2E Campaign

这是 campaign `v100_rmsnorm_e2e_20260922` 的唯一恢复目录。所有长期状态以 `STATE.json`、`JOURNAL.jsonl`、`RUN_INDEX.json`、`REMOTE_JOBS.json` 和远端 job 目录为准，不依赖聊天上下文。

恢复入口：

```powershell
.\.venv\Scripts\python.exe scripts\resume_v100_rmsnorm_overnight.py --remote
```

脚本会交互式读取 SSH 密码；密码不会写入命令行、文件、日志或 JSON。

当前环境限制：工作区可写，但 `.git` 元数据由沙箱只读挂载，因此所要求的 challenge branch 与本地 checkpoint commit 暂时无法创建。该限制记录在 `STATE.json` 和 `JOURNAL.jsonl`，不会伪称已 commit。
