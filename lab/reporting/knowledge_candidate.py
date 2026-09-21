
"""Generate knowledge candidates from experiment reports.

These are *candidates* only — they require human review before
being written to formal knowledge (lab/knowledge/).
"""
from __future__ import annotations

from pathlib import Path

from ..core.report import ExperimentReport
from ..core.evidence_store import EvidenceStore


def generate_knowledge_candidate(report: ExperimentReport) -> dict | None:
    """Generate a knowledge candidate from a single report."""
    kc = report.knowledge_candidate
    if not kc or not kc.get("claim"):
        return None
    return kc


def generate_knowledge_candidates(
    reports: list[ExperimentReport],
) -> list[dict]:
    """Generate knowledge candidates from a list of reports.

    Only returns candidates with confidence 'high' or 'medium'.
    """
    candidates = []
    for report in reports:
        kc = generate_knowledge_candidate(report)
        if kc and kc.get("confidence") in ("high", "medium"):
            candidates.append(kc)
    return candidates


def aggregate_knowledge_candidates(
    campaign_dir: Path,
) -> list[dict]:
    """Aggregate knowledge candidates from all episodes in a campaign."""
    store = EvidenceStore(campaign_dir)
    summaries = store.hypothesis_summaries()
    
    candidates = []
    for s in summaries:
        if not s.claim:
            continue
        verdict = s.verdict
        confidence = "high" if verdict == "SUPPORTED" and s.total_count >= 2 else (
            "medium" if verdict == "SUPPORTED" else "low"
        )
        candidates.append({
            "claim": s.claim,
            "mechanism": s.mechanism,
            "verdict": verdict,
            "confidence": confidence,
            "evidence_count": s.total_count,
            "best_speedup": s.best_speedup,
            "experiment_ids": s.experiment_ids,
        })
    
    return sorted(candidates, key=lambda c: c.get("evidence_count", 0), reverse=True)
