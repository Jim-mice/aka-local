"""Thin official Codex Python SDK/app-server backend.

This module owns exactly one agent turn. It never compiles, benchmarks, or
decides promotion; those remain local mechanical responsibilities.
"""

# AKA_CODEX_PROXY_BEGIN
# Codex native/app-server does not reliably inherit the Windows GUI proxy.
# Keep this project-local: it affects only this Python process and children.
import os as _aka_proxy_os

_aka_proxy_os.environ["HTTP_PROXY"] = "http://127.0.0.1:10808"
_aka_proxy_os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10808"
_aka_proxy_os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
_aka_proxy_os.environ.pop("ALL_PROXY", None)
# AKA_CODEX_PROXY_END

from pathlib import Path
from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox

CODEX_BIN = Path(r"<PROJECT_ROOT>\runtimes\codex-0.154.0\node_modules\@openai\codex-win32-x64\vendor\x86_64-pc-windows-msvc\bin\codex.exe")

def run_one_turn(prompt: str, workspace: Path):
    client = Codex(CodexConfig(codex_bin=str(CODEX_BIN), cwd=str(workspace), client_name="aka_local_appserver", client_title="aka-local app-server backend", client_version="0.1"))
    try:
        thread = client.thread_start(model="gpt-5.6-luna", cwd=str(workspace), sandbox=Sandbox.workspace_write, approval_mode=ApprovalMode.deny_all, ephemeral=True)
        return thread.run(prompt, model="gpt-5.6-luna", effort="low", cwd=str(workspace), sandbox=Sandbox.workspace_write, approval_mode=ApprovalMode.deny_all)
    finally:
        client.close()
