"""Bounded read-only smoke test for the current Codex planning transport."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()
    command = [
        "codex.exe", "exec", "--ephemeral", "--sandbox", "read-only",
        "--json", "--skip-git-repo-check", "--cd", str(args.repo), "-",
    ]
    prompt = 'Return exactly:\n{"status":"ok"}\n'
    started = time.time()
    record: dict[str, object] = {
        "command": command,
        "cwd": str(args.repo),
        "prompt": prompt,
        "timeout_seconds": args.timeout,
        "state": "CREATED",
        "stdout": "",
        "stderr": "",
        "returncode": None,
        "timed_out": False,
    }
    try:
        record["state"] = "PROMPT_SENT"
        completed = subprocess.run(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            cwd=str(args.repo),
            timeout=args.timeout,
            check=False,
        )
        record["stdout"] = completed.stdout
        record["stderr"] = completed.stderr
        record["returncode"] = completed.returncode
        record["state"] = "FINAL_RECEIVED" if completed.stdout.strip() else "NO_EVENT"
    except subprocess.TimeoutExpired as exc:
        record["timed_out"] = True
        record["state"] = "WAITING_FOR_FIRST_EVENT"
        record["stdout"] = exc.stdout or ""
        record["stderr"] = exc.stderr or ""
    except Exception as exc:  # pragma: no cover - environment diagnostic
        record["state"] = "LAUNCH_FAILED"
        record["exception"] = repr(exc)
    record["elapsed_seconds"] = time.time() - started
    record["smoke_status"] = "PASS" if record["state"] == "FINAL_RECEIVED" and record["returncode"] == 0 else "BLOCKED"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("state", "returncode", "timed_out", "smoke_status", "elapsed_seconds")}, indent=2))
    return 0 if record["smoke_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
