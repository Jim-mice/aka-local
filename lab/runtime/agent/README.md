# Agent 边界

LOCAL RTX5060 RMSNorm 路径只有在用户批准 Workbench 并在 Tk GUI 中确认 Start 后，才运行一次有界的持久 `CodexAgentSession`。每次实验前，runner 都会写入 knowledge snapshot 和 context snapshot。

Agent 只能修改已批准的 candidate root；Controller 会检查路径差异，并保护 incumbent、reference、knowledge 和 supervisor 路径。Agent 不能写入 `supervisor_decision`，也不能晋升 incumbent。
