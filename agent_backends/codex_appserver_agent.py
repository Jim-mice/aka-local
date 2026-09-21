"""Isolated Codex app-server backend skeleton for aka-local.

This module deliberately contains no automatic sampling.  The only method
that can create a thread or turn is an explicit caller action in a future
episode runner.  Mechanical evaluation remains outside this backend.
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


from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai_codex import Codex, CodexConfig


CODEX_BIN = Path(
    r"<LOCAL_USER_HOME>\AppData\Roaming\npm\node_modules\@openai\codex"
    r"\node_modules\@openai\codex-win32-x64\vendor\x86_64-pc-windows-msvc\bin\codex.exe"
)


@dataclass(frozen=True)
class AgentSettings:
    model: str = "gpt-5.6-luna"
    effort: str = "low"
    cwd: str = r"<PROJECT_ROOT>"


class CodexAppServerAgent:
    """Own one independent app-server process for one bounded campaign."""

    def __init__(self, settings: AgentSettings | None = None) -> None:
        self.settings = settings or AgentSettings()
        self._codex: Codex | None = None

    def start(self) -> Codex:
        if self._codex is None:
            config = CodexConfig(
                codex_bin=str(CODEX_BIN),
                cwd=self.settings.cwd,
                client_name="aka_local_codex_appserver",
                client_title="aka-local Codex app-server backend",
                client_version="0.1.0",
            )
            self._codex = Codex(config)
        return self._codex

    def close(self) -> None:
        if self._codex is not None:
            self._codex.close()
            self._codex = None

    def start_thread_for_future_episode(self, *, episode_cwd: str) -> Any:
        """Explicit future sampling boundary; never called by this module.

        The eventual runner must pass model/effort explicitly and must stop
        immediately after the candidate is written.  This method is kept
        separate so importing or starting the backend cannot sample.
        """
        codex = self.start()
        return codex.thread_start(
            model=self.settings.model,
            cwd=episode_cwd,
        )

    @staticmethod
    def result_summary(result: Any) -> dict[str, Any]:
        """Map only fields actually supplied by SDK TurnResult.

        No synthetic Atrex phase markers or tool events are invented here.
        Callers may persist these fields, while detailed item mapping remains
        an explicit integration task after the first bounded episode.
        """
        return {
            "id": getattr(result, "id", None),
            "status": getattr(result, "status", None),
            "error": getattr(result, "error", None),
            "started_at": getattr(result, "started_at", None),
            "completed_at": getattr(result, "completed_at", None),
            "duration_ms": getattr(result, "duration_ms", None),
            "final_response": getattr(result, "final_response", None),
            "items": getattr(result, "items", []),
            "usage": getattr(result, "usage", None),
        }


if __name__ == "__main__":
    raise SystemExit("Import-only backend skeleton; no model sampling is performed.")
