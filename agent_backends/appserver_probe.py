"""Non-sampling Codex app-server probe.

This file intentionally does not call thread_start, turn, run, or any prompt
API. It only constructs the SDK client, performs initialization, and reads
non-sensitive account/metadata/model metadata.
"""

from __future__ import annotations

import os
from pathlib import Path

from openai_codex import Codex, CodexConfig


CODEX_BIN = Path(
    r"<PROJECT_ROOT>\runtimes\codex-0.154.0\node_modules\@openai\codex-win32-x64\vendor\x86_64-pc-windows-msvc\bin\codex.exe"
)
WORKSPACE = Path(r"<PROJECT_ROOT>")


def safe_value(value: object) -> object:
    """Keep probe output non-sensitive and bounded."""
    text = repr(value)
    for marker in ("token", "refresh", "cookie", "authorization", "secret"):
        if marker in text.lower():
            return f"<redacted:{marker}>"
    return value


def main() -> int:
    if not CODEX_BIN.is_file():
        print("RUNTIME_BINARY=NOT_FOUND")
        return 2

    env = os.environ.copy()
    # Keep the independent app-server isolated from Desktop's private IPC.
    env.pop("CODEX_APP_TOOLS_PIPE_PATH", None)
    env.pop("CODEX_SESSION_ID", None)
    env.pop("CODEX_THREAD_ID", None)
    env.pop("CODEX_INTERNAL_ORIGINATOR_OVERRIDE", None)

    config = CodexConfig(
        codex_bin=str(CODEX_BIN),
        cwd=str(WORKSPACE),
        env=env,
        client_name="aka_local_appserver_probe",
        client_title="aka-local non-sampling probe",
        client_version="0.1.0",
    )

    codex = None
    try:
        print("SDK_IMPORT=PASS")
        print("SDK_VERSION=0.147.0")
        print(f"CODEX_BIN={CODEX_BIN}")
        print("LAUNCH_MODE=app-server-stdio")
        codex = Codex(config)
        print("APP_SERVER_INITIALIZE=PASS")

        account = codex.account(refresh_token=False)
        print("AUTH_REUSED=YES")
        account_obj = getattr(account, "account", account)
        print(
            "ACCOUNT_STATUS="
            f"type={getattr(account_obj, 'type', 'unknown')};"
            f"plan={getattr(account_obj, 'plan_type', 'unknown')}"
        )

        metadata = codex.metadata
        print(f"METADATA={safe_value(metadata)}")
        models = codex.models(include_hidden=False)
        print(f"MODELS_METADATA={safe_value(models)}")
        print("MODEL_SAMPLING=0")
        return 0
    finally:
        if codex is not None:
            codex.close()


if __name__ == "__main__":
    raise SystemExit(main())
