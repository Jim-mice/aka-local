"""Fail-closed integrity checker for the final v3 held-out case."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PUBLIC_FILES = {"case.json", "facts.json", "contract.json"}
LEAK_TOKENS = (
    "v3_evaluator_only", "expected_mechanism_family", "acceptable_equivalents", "full_hit_requirements",
    "heldout_a_", "heldout_b_", "case_a_run", "case_b_run", "fp-repeated-read-lifetime",
    "fp-producer-consumer-boundary", "measurement-low-e2e-ceiling", "episode 28",
    "retain x", "fuse kernels", "second global load", "v1", "v2",
)


def all_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def schema_is_closed(value: object) -> bool:
    if isinstance(value, dict):
        if value.get("type") == "object" and value.get("additionalProperties") is not False:
            return False
        return all(schema_is_closed(item) for item in value.values())
    if isinstance(value, list):
        return all(schema_is_closed(item) for item in value)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()
    public = args.root / "public"
    evaluator = args.root / "evaluator_only"
    errors: list[dict[str, str]] = []
    expected = {"STOP_RULE.md", "BLIND_PROMPT.md", "planning_result.schema.json", "public", "evaluator_only"}
    actual = {path.name for path in args.root.iterdir()}
    for name in sorted(expected - actual):
        errors.append({"reason": "MISSING_PACKAGE_ENTRY", "detail": name})
    public_case = public / "case_final"
    if not public_case.exists():
        errors.append({"reason": "MISSING_PUBLIC_CASE", "detail": str(public_case)})
    else:
        for name in PUBLIC_FILES:
            if not (public_case / name).exists():
                errors.append({"reason": "MISSING_PUBLIC_FILE", "detail": name})
        extra = {path.name for path in public_case.iterdir()} - PUBLIC_FILES
        for name in sorted(extra):
            errors.append({"reason": "UNAPPROVED_PUBLIC_FILE", "detail": name})
    schema_path = args.root / "planning_result.schema.json"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        if not schema_is_closed(schema):
            errors.append({"reason": "OPEN_OBJECT_SCHEMA", "detail": str(schema_path)})
    except Exception as exc:
        errors.append({"reason": "SCHEMA_INVALID", "detail": repr(exc)})
    public_text = "\n".join(path.read_text(encoding="utf-8").lower() for path in all_files(public) if path.suffix in {".json", ".md"})
    for token in LEAK_TOKENS:
        if token.lower() in public_text:
            errors.append({"reason": "PUBLIC_ANSWER_OR_HISTORY_LEAK", "detail": token})
    answer_path = evaluator / "answer_key.json"
    try:
        answer = json.loads(answer_path.read_text(encoding="utf-8"))
        sentinel = str(answer["public_sentinel"])
        if sentinel.lower() in public_text:
            errors.append({"reason": "EVALUATOR_SENTINEL_LEAK", "detail": sentinel})
    except Exception as exc:
        errors.append({"reason": "ANSWER_KEY_INVALID", "detail": repr(exc)})
    files = all_files(args.root)
    hashes = {str(path.relative_to(args.root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files if path != args.manifest}
    payload = {"algorithm": "sha256", "status": "PASS" if not errors else "FAIL", "files": hashes, "errors": errors}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "file_count": len(hashes), "errors": errors}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
