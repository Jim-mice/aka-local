
"""Render ExperimentReport as human-readable Markdown."""
from __future__ import annotations

from pathlib import Path

from ..core.report import ExperimentReport


def render_markdown(report: ExperimentReport) -> str:
    """Generate a Markdown report from an ExperimentReport."""

    hyp = report.hypothesis or {}
    ev = report.evidence or {}
    cand = report.candidate or {}
    perf = ev.get("performance") or {}
    correctness = ev.get("correctness") or {}
    compile_info = ev.get("compile") or {}
    kc = report.knowledge_candidate or {}

    lines = [
        f"# Experiment {report.experiment_id}",
        "",
    ]

    # Task
    lines.extend([
        "## Task",
        "",
        f"- **Operator:** {report.operator}",
        f"- **GPU:** {report.hardware}",
        f"- **Campaign:** {report.campaign_id}",
        f"- **Episode:** {report.episode_id}",
        "",
    ])

    # Hypothesis
    if hyp.get("claim"):
        lines.extend([
            "## Hypothesis",
            "",
            f"- **Claim:** {hyp.get('claim', '')}",
            f"- **Mechanism:** {hyp.get('mechanism', '')}",
            f"- **Expected effect:** {hyp.get('expected_effect', '')}",
            f"- **Verdict:** {hyp.get('verdict', 'INCONCLUSIVE')}",
            "",
        ])

    # Candidate
    lines.extend([
        "## Candidate",
        "",
        f"- **ID:** {cand.get('id', '')[:24]}",
        f"- **Parent:** {cand.get('parent', '')[:24]}",
        f"- **Hypothesis:** {cand.get('hypothesis_id', '')[:16]}",
        "",
    ])

    # Changes
    if report.changes:
        lines.append("### Changes")
        lines.append("")
        for f in report.changes:
            lines.append(f"- {f}")
        lines.append("")

    # Evidence
    lines.extend([
        "## Evidence",
        "",
        f"- **Compile:** {'PASS' if compile_info.get('pass') else 'FAIL'}",
        f"- **Correctness:** {'PASS' if correctness.get('pass') else 'FAIL'}",
    ])

    speedup = perf.get("speedup")
    if speedup is not None:
        pct = (float(speedup) - 1.0) * 100
        lines.append(f"- **Speedup:** {speedup:.4f}x ({pct:+.1f}%)")

    geo = perf.get("geometric_mean")
    if geo is not None:
        lines.append(f"- **Geometric mean:** {geo:.4f}x")

    # Typed evidence
    typed = ev.get("typed") or {}
    if typed.get("type"):
        lines.append(f"- **Evidence type:** {typed.get('type')}")
        lines.append(f"- **Evidence verdict:** {typed.get('verdict', '')}")

    lines.append("")

    # Diagnostics
    if report.diagnostics:
        lines.append("## Diagnostics")
        lines.append("")
        for d in report.diagnostics:
            cat = d.get("category", "UNKNOWN")
            sev = d.get("severity", "INFO")
            msg = d.get("message", "")
            lines.append(f"- **[{cat}]** ({sev}) {msg}")
        lines.append("")

    # Decision
    lines.extend([
        "## Decision",
        "",
        f"- **Agent decision:** {report.decision}",
        f"- **Supervisor action:** {report.supervisor_action}",
        "",
    ])

    # Knowledge candidate
    if kc.get("claim"):
        conf = kc.get("confidence", "low")
        lines.extend([
            "## Knowledge Candidate",
            "",
            f"- **Claim:** {kc.get('claim', '')}",
            f"- **Mechanism:** {kc.get('mechanism', '')}",
            f"- **Confidence:** {conf}",
            f"- **Verdict:** {kc.get('verdict', 'INCONCLUSIVE')}",
        ])
        if kc.get("speedup") is not None:
            lines.append(f"- **Speedup:** {kc['speedup']:.4f}x")
        lines.append("")
        lines.append("> This is a knowledge *candidate* only. Human review required before promotion to formal knowledge.")
        lines.append("")

    lines.extend([
        "---",
        f"*Report generated: {report.timestamp}*",
        f"*Report ID: {report.report_id}*",
        "",
    ])

    return "\n".join(lines)


def save_markdown(report: ExperimentReport, output_dir: Path) -> Path:
    """Save an ExperimentReport as a Markdown file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{report.experiment_id}.md"
    path.write_text(render_markdown(report), encoding="utf-8")
    return path
