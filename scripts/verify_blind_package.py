"""Fail-closed integrity check for public blind inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FORBIDDEN = ("episode 28", "retain x", "second global read", "second global load", "delta =", "deterministic candidate", "mechanism family")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()
    errors = []
    public = [p for p in args.root.rglob("*") if p.is_file() and "results" not in p.parts and "evaluator_only" not in p.parts and p.name != "package_hashes.json"]
    for path in public:
        text = path.read_text(encoding="utf-8").lower()
        for phrase in FORBIDDEN:
            if phrase in text:
                errors.append({"file": str(path), "reason": "ANSWER_LEAK", "phrase": phrase})
    cases = {"rmsnorm", "gdn", "swiglu"}
    for case in cases:
        for name in ("case.json", "facts.json", "contract.json"):
            if not (args.root / case / name).exists():
                errors.append({"file": str(args.root / case / name), "reason": "MISSING"})
    hashes = {str(p.relative_to(args.root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(public)}
    args.manifest.write_text(json.dumps({"algorithm": "sha256", "files": hashes, "status": "PASS" if not errors else "FAIL", "errors": errors}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS" if not errors else "FAIL", "file_count": len(public), "errors": errors}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
