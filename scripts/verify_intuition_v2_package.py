"""Fail-closed integrity checker for the adversarial held-out package."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PUBLIC_REQUIRED = {"case.json", "facts.json", "contract.json"}
LEAK_TOKENS = (
    "episode 28", "fp-repeated-read-lifetime", "fp-producer-consumer-boundary",
    "measurement-low-e2e-ceiling", "expected_families", "known_traps",
    "answer_key_text", "deterministic candidate", "retain x", "fuse kernels",
    "second global load", "delta =",
)


def files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()
    public = args.root / "public"
    evaluator = args.root / "evaluator_only"
    errors: list[dict[str, str]] = []
    expected_cases = {"case_a", "case_b", "case_c", "case_d", "case_e"}
    actual_cases = {p.name for p in public.iterdir() if p.is_dir()} if public.exists() else set()
    if actual_cases != expected_cases:
        errors.append({"reason": "PUBLIC_CASE_SET_MISMATCH", "detail": repr(sorted(actual_cases))})
    for case in sorted(expected_cases):
        case_dir = public / case
        for name in PUBLIC_REQUIRED:
            if not (case_dir / name).exists():
                errors.append({"reason": "MISSING_PUBLIC_FILE", "detail": str(case_dir / name)})
        if case_dir.exists():
            extra = {p.name for p in case_dir.iterdir()} - PUBLIC_REQUIRED
            for name in sorted(extra):
                errors.append({"reason": "UNAPPROVED_PUBLIC_FILE", "detail": str(case_dir / name)})
    public_text = "\n".join(p.read_text(encoding="utf-8").lower() for p in files(public) if p.suffix in {".json", ".md"})
    for token in LEAK_TOKENS:
        if token.lower() in public_text:
            errors.append({"reason": "ANSWER_OR_TEMPLATE_LEAK", "detail": token})
    eval_files = files(evaluator)
    if not eval_files:
        errors.append({"reason": "MISSING_EVALUATOR_METADATA", "detail": str(evaluator)})
    eval_data = json.loads((evaluator / "answer_keys.json").read_text(encoding="utf-8")) if (evaluator / "answer_keys.json").exists() else {}
    for case_id, metadata in eval_data.items():
        sentinel = metadata.get("public_sentinel")
        if not sentinel or sentinel.lower() in public_text:
            errors.append({"reason": "EVALUATOR_SENTINEL_LEAK", "detail": case_id})
    all_files = files(args.root)
    hashes = {str(p.relative_to(args.root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in all_files if p != args.manifest}
    payload = {"algorithm": "sha256", "status": "PASS" if not errors else "FAIL", "files": hashes, "errors": errors}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "file_count": len(hashes), "errors": errors}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
