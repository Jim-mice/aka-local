"""Read-only verification for the stable Night Shift artifact hash list."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "docs" / "night_shift" / "2026-09-21" / "ARTIFACT_SHA256.json"


def main() -> int:
    try:
        manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"verdict": "NIGHT_ARTIFACT_FAIL", "failures": [f"manifest:{exc}"]}))
        return 1
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict) or not files:
        print(json.dumps({"verdict": "NIGHT_ARTIFACT_FAIL", "failures": ["files"]}))
        return 1
    failures: list[str] = []
    if isinstance(manifest, dict):
        allowed_manifest_fields = {"schema_version", "generated_at", "scope", "files"}
        unknown_fields = sorted(set(manifest) - allowed_manifest_fields)
        if unknown_fields:
            failures.append(f"unknown_manifest_fields:{','.join(unknown_fields)}")
        if manifest.get("schema_version") != 1:
            failures.append("schema_version")
        for required in ("generated_at", "scope"):
            if not isinstance(manifest.get(required), str) or not manifest[required].strip():
                failures.append(f"manifest_field:{required}")
    for relative, expected in files.items():
        portable = PurePosixPath(relative) if isinstance(relative, str) else None
        if portable is None or portable.is_absolute() or ".." in portable.parts:
            failures.append(f"unsafe_path:{relative}")
            continue
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            failures.append(f"invalid_digest:{relative}")
            continue
        path = (PROJECT_ROOT / Path(*portable.parts)).resolve()
        try:
            path.relative_to(PROJECT_ROOT)
        except ValueError:
            failures.append(f"unsafe_path:{relative}")
            continue
        if not path.is_file():
            failures.append(f"missing:{relative}")
            continue
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            failures.append(f"unreadable:{relative}")
            continue
        if actual != expected:
            failures.append(f"digest:{relative}")
    print(json.dumps({"verdict": "NIGHT_ARTIFACT_PASS" if not failures else "NIGHT_ARTIFACT_FAIL", "failures": failures}))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
