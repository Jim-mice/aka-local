"""Read-only verifier for the pinned Megatron source-audit artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any


def verify_source_audit(megatron_root: Path, audit: dict[str, Any]) -> tuple[str, ...]:
    failures: list[str] = []
    if not isinstance(audit, dict):
        return ("audit_type",)
    root = Path(megatron_root).resolve()
    if not root.is_dir():
        return ("megatron_root",)
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        head = completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ("git_head",)
    if head != audit.get("target_commit"):
        failures.append("target_commit")
    source_files = audit.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        return tuple(dict.fromkeys([*failures, "source_files"]))
    for relative, expected_digest in source_files.items():
        if not isinstance(relative, str) or not isinstance(expected_digest, str):
            failures.append("source_entry")
            continue
        portable = PurePosixPath(relative)
        if portable.is_absolute() or ".." in portable.parts:
            failures.append(f"unsafe_path:{relative}")
            continue
        path = (root / Path(*portable.parts)).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            failures.append(f"unsafe_path:{relative}")
            continue
        if not path.is_file():
            failures.append(f"missing:{relative}")
            continue
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            failures.append(f"unreadable:{relative}")
            continue
        if digest != expected_digest:
            failures.append(f"digest:{relative}")
    anchors = audit.get("operator_anchors")
    if not isinstance(anchors, dict) or not anchors:
        failures.append("operator_anchors")
    else:
        for operator_id, entries in anchors.items():
            if not isinstance(operator_id, str) or not operator_id or not isinstance(entries, list) or not entries:
                failures.append("operator_anchor_entry")
                continue
            for anchor in entries:
                match = re.fullmatch(r"(.+\.py):(\d+) (.+)", anchor) if isinstance(anchor, str) else None
                if match is None:
                    failures.append(f"anchor_schema:{operator_id}")
                    continue
                relative, line_text, declared_symbol = match.groups()
                if relative not in source_files:
                    failures.append(f"anchor_file:{operator_id}")
                    continue
                path = (root / Path(*PurePosixPath(relative).parts)).resolve()
                try:
                    lines = path.read_text(encoding="utf-8").splitlines()
                except OSError:
                    failures.append(f"anchor_unreadable:{operator_id}")
                    continue
                line_number = int(line_text)
                if line_number < 1 or line_number > len(lines):
                    failures.append(f"anchor_line:{operator_id}")
                    continue
                symbol = declared_symbol.rsplit(".", 1)[-1]
                if symbol not in lines[line_number - 1]:
                    failures.append(f"anchor_symbol:{operator_id}")
    return tuple(dict.fromkeys(failures))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "targets" / "megatron_5be9626" / "source_audit.json",
    )
    args = parser.parse_args()
    try:
        audit = json.loads(args.audit.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"verdict": "SOURCE_AUDIT_FAIL", "failures": [f"audit_file:{exc}"]}, ensure_ascii=False))
        return 1
    failures = verify_source_audit(args.megatron_root, audit)
    print(json.dumps({"verdict": "SOURCE_AUDIT_PASS" if not failures else "SOURCE_AUDIT_FAIL", "failures": failures}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
