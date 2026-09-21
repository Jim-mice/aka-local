"""Exactly-one-turn RMSNorm candidate generation launcher.

This script is intentionally one-shot: no retry, no reviewer, no fallback.
The caller must perform mechanical evaluation after this process exits.
"""

from __future__ import annotations

# AKA_CODEX_PROXY_BEGIN
# Codex native/app-server does not reliably inherit the Windows GUI proxy.
# Keep this project-local: it affects only this Python process and children.
import os as _aka_proxy_os

_aka_proxy_os.environ["HTTP_PROXY"] = "http://127.0.0.1:10808"
_aka_proxy_os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10808"
_aka_proxy_os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
_aka_proxy_os.environ.pop("ALL_PROXY", None)
# AKA_CODEX_PROXY_END


import json
from pathlib import Path

from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox


ROOT = Path(r"<PROJECT_ROOT>")
EPISODE = ROOT / "campaigns" / "rms_norm" / "backward" / "episode_1"
CODEX_BIN = Path(
    r"<LOCAL_USER_HOME>\AppData\Roaming\npm\node_modules\@openai\codex"
    r"\node_modules\@openai\codex-win32-x64\vendor\x86_64-pc-windows-msvc\bin\codex.exe"
)

PROMPT = r"""You are performing exactly one isolated CUDA candidate-generation turn.

Operator: official Atrex-Bench rms_norm reduction operator.
Target: NVIDIA GeForce RTX 5060 Laptop GPU, compute capability 12.0, sm_120.
Runtime: CUDA 13.4, PyTorch 2.14.0+cu130.

Read only the minimum necessary files:
- <PROJECT_ROOT>\campaigns\rms_norm\backward\episode_1\operator_context.md
- <PROJECT_ROOT>\ops\rms_norm\reference.py
- <PROJECT_ROOT>\ops\rms_norm\input.py
- <PROJECT_ROOT>\ops\rms_norm\shapes.json
- <PROJECT_ROOT>\ops\rms_norm\metadata.json

Choose exactly ONE CUDA optimization direction focused on the real RMSNorm
reduction workload, such as one reduction structure, one memory-traffic
choice, or one cooperation strategy. Do not combine unrelated changes.

Implement exactly one candidate in this workspace only:
<PROJECT_ROOT>\campaigns\rms_norm\backward\episode_1

Requirements:
- Preserve official semantics, float32 accumulation, bfloat16 I/O, and all 56 shapes.
- Use a real CUDA candidate; no PyTorch fallback.
- No external Python dependency.
- Do not modify the official reference, evaluator, baseline files, or any file outside this episode workspace.
- Do not run CUDA, Atrex, benchmark, profiler, or remote jobs.
- Stop immediately after candidate source is ready; do not ask for review or another turn.

Keep the explanation short. Write AGENT.md in the episode workspace containing:
hypothesis, evidence, changed mechanism, correctness risk, and any GPU Wiki
record IDs actually used. The evaluator will run after this turn closes.
"""


def main() -> int:
    config = CodexConfig(
        codex_bin=str(CODEX_BIN),
        cwd=str(EPISODE),
        client_name="aka_local_rmsnorm_episode",
        client_title="aka-local RMSNorm episode",
        client_version="0.1.0",
    )
    codex = Codex(config)
    try:
        thread = codex.thread_start(
            model="gpt-5.6-luna",
            cwd=str(EPISODE),
            sandbox=Sandbox.workspace_write,
            approval_mode=ApprovalMode.deny_all,
            ephemeral=True,
        )
        result = thread.run(
            PROMPT,
            model="gpt-5.6-luna",
            effort="low",
            cwd=str(EPISODE),
            sandbox=Sandbox.workspace_write,
            approval_mode=ApprovalMode.deny_all,
        )
        (EPISODE / "agent_response.md").write_text(
            str(getattr(result, "final_response", "")), encoding="utf-8"
        )
        (EPISODE / "agent_metadata.json").write_text(
            json.dumps(
                {
                    "model": "gpt-5.6-luna",
                    "reasoning_effort": "low",
                    "model_calls": 1,
                    "reviewers": 0,
                    "fallback": "none",
                    "turn_id": getattr(result, "id", None),
                    "status": str(getattr(result, "status", None)),
                    "usage": getattr(result, "usage", None),
                },
                default=str,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(str(getattr(result, "final_response", "")))
        return 0
    finally:
        codex.close()


if __name__ == "__main__":
    raise SystemExit(main())
