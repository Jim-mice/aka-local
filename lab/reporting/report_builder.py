
"""Build ExperimentReport objects from journal and evidence store."""
from __future__ import annotations

from pathlib import Path

from ..core.report import ExperimentReport
from ..core.journal import read_experiments
from ..core.evidence_store import EvidenceStore


def build_experiment_report(
    episode_dir: Path,
    experiment_id: str,
    *,
    campaign_dir: Path | None = None,
) -> ExperimentReport | None:
    """Build a single ExperimentReport from an experiment record."""
    records = read_experiments(episode_dir / "journal.jsonl")
    for rec in records:
        if rec.get("experiment_id") == experiment_id:
            return ExperimentReport.from_experiment(rec)
    return None


def build_all_reports(
    episode_dir: Path,
    *,
    campaign_dir: Path | None = None,
) -> list[ExperimentReport]:
    """Build ExperimentReport objects for all experiments in an episode."""
    records = read_experiments(episode_dir / "journal.jsonl")
    reports = []
    for rec in records:
        try:
            report = ExperimentReport.from_experiment(rec)
            reports.append(report)
        except Exception:
            pass
    return reports


def build_campaign_reports(
    campaign_dir: Path,
) -> list[ExperimentReport]:
    """Build reports for all experiments across all episodes in a campaign."""
    reports = []
    campaign_dir = Path(campaign_dir)
    for path in sorted(campaign_dir.iterdir()):
        if path.is_dir() and path.name.startswith("e"):
            journal = path / "journal.jsonl"
            if journal.is_file():
                reports.extend(build_all_reports(path, campaign_dir=campaign_dir))
    return reports
