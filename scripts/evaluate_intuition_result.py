"""Apply an auditable rubric after blind result collection; no model call."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DIMENSIONS = ("EVIDENCE_GROUNDED", "MECHANISM_DEPTH", "STRUCTURAL_CHANGE", "TRIVIALITY", "UNKNOWN_DISCIPLINE", "IMPLEMENTABILITY", "NOVELTY", "SYSTEM_AWARENESS")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collected", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    collected = json.loads(args.collected.read_text(encoding="utf-8"))
    reviews = []
    for index, item in enumerate(collected.get("accepted", [])):
        reviews.append({"hypothesis_index": index, "rubric": {dimension: "NOT_REVIEWED" for dimension in DIMENSIONS}, "review_note": "Human semantic review required; no automatic total score."})
    payload = {"case_id": collected.get("case_id"), "reviews": reviews, "rejected": collected.get("rejected", []), "status": "PENDING_HUMAN_REVIEW"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "review_count": len(reviews)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
