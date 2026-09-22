"""Validate an externally produced blind planning JSON without calling a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = ("observation", "mechanism", "transformation", "preconditions", "expected_effect", "risks", "unknowns", "required_evidence", "evidence_refs")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=Path, required=True)
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    facts = json.loads((args.case / "facts.json").read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    valid_ids = {item["id"] for item in facts.get("facts", [])}
    accepted, rejected = [], []
    if result.get("case_id") != facts.get("case_id"):
        rejected.append({"hypothesis": None, "reason": "CASE_ID_MISMATCH"})
    for index, hypothesis in enumerate(result.get("hypotheses", [])):
        missing = [key for key in REQUIRED if key not in hypothesis]
        bad_refs = [ref for ref in hypothesis.get("evidence_refs", []) if ref not in valid_ids]
        if missing:
            rejected.append({"hypothesis": index, "reason": "SCHEMA_MISSING_FIELDS", "fields": missing})
        elif bad_refs:
            rejected.append({"hypothesis": index, "reason": "REJECT_EVIDENCE_HALLUCINATION", "unknown_evidence_refs": bad_refs})
        else:
            accepted.append(hypothesis)
    payload = {"case_id": facts.get("case_id"), "accepted": accepted, "rejected": rejected, "evidence_guard": "PASS", "status": "PASS" if not rejected else "PARTIAL"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "accepted": len(accepted), "rejected": len(rejected)}, indent=2))
    return 0 if payload["status"] in ("PASS", "PARTIAL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
