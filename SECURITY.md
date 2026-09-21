# 安全说明

不要提交密码、API key、bearer token、SSH 私钥、`.env` 文件或真实远程配置。本地 V100 配置应放在被 Git 忽略的 `config/environments/v100.yaml`；`.venv/`、profiler 二进制文件和本地 artifact 也均被忽略。

远程访问请使用 SSH key 或交互式认证。如果 secret 已被提交或公开，立即撤销或轮换 credential，并私下通知仓库维护者。仅在后续 commit 中删除不足以消除风险，因为 Git 历史可能仍保留该 secret。
