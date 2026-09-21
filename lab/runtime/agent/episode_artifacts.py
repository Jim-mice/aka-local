"""Write the human-readable outputs of a completed local episode.

The canonical technical record remains ``memory/vNNN.json``.  This module
derives compact archive and knowledge-candidate views from it; it never
rewrites the incumbent or alters evaluator measurements.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ...core.persistence import atomic_json


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _metrics(record: dict) -> dict:
    return {
        "compile": record.get("compile"),
        "correctness": record.get("correctness"),
        "development_benchmark": record.get("development_benchmark"),
        "authoritative_abba": record.get("authoritative_abba"),
        "robustness": record.get("robustness"),
    }


def _report(record: dict, canonical: dict) -> str:
    plan = record.get("plan", {})
    return "\n".join((
        f"# {record['experiment_id']}", "",
        "## Identity", "",
        f"- Campaign: `{record.get('campaign_id')}`",
        f"- Episode: `{record.get('episode_id')}`",
        f"- Working directory: `{record.get('working_directory')}`",
        f"- Candidate hash: `{record.get('candidate_hash')}`", "",
        "## Hypothesis", "", json.dumps(plan.get("hypothesis", {}), ensure_ascii=False, indent=2), "",
        "## Change", "", str(plan.get("planned_change", "UNKNOWN")), "",
        "## Correctness", "", json.dumps(record.get("correctness", {}), ensure_ascii=False, indent=2), "",
        "## Performance", "", json.dumps(record.get("development_benchmark", {}), ensure_ascii=False, indent=2), "",
        "## Supervisor result", "", str(canonical.get("supervisor_decision", record.get("decision", "UNKNOWN"))), "",
        "## Next directions", "", "\n".join(f"- {x}" for x in canonical.get("next_directions", [])) or "- UNKNOWN", "",
    ))


def _zh_report(record: dict, canonical: dict) -> str:
    plan = record.get("plan", {})
    return "\n".join((
        f"# {record['experiment_id']}（中文归档）", "",
        "## 实验身份", "",
        f"- Campaign：`{record.get('campaign_id')}`",
        f"- Episode：`{record.get('episode_id')}`",
        f"- Agent 工作目录：`{record.get('working_directory')}`",
        f"- Candidate hash：`{record.get('candidate_hash')}`", "",
        "## 假设", "", json.dumps(plan.get("hypothesis", {}), ensure_ascii=False, indent=2), "",
        "## 改动", "", str(plan.get("planned_change", "UNKNOWN")), "",
        "## 正确性", "", json.dumps(record.get("correctness", {}), ensure_ascii=False, indent=2), "",
        "## 性能", "", json.dumps(record.get("development_benchmark", {}), ensure_ascii=False, indent=2), "",
        "## Supervisor 结果", "", str(canonical.get("supervisor_decision", record.get("decision", "UNKNOWN"))), "",
        "## 后续方向", "", "\n".join(f"- {x}" for x in canonical.get("next_directions", [])) or "- UNKNOWN", "",
        "说明：中文版本只改变表述；数字、ID、路径、代码符号和 scope 保持 canonical technical record 不变。", "",
    ))


def archive_episode(lab_root: Path, canonical_path: Path, canonical: dict) -> list[Path]:
    """Materialize archive folders for all real records in a gated episode."""
    written: list[Path] = []
    for record in canonical.get("experiments", []):
        experiment_id = record["experiment_id"]
        out = lab_root / "experiment_archive" / experiment_id
        out.mkdir(parents=True, exist_ok=True)
        original = _report(record, canonical)
        zh = _zh_report(record, canonical)
        (out / "report_original.md").write_text(original, encoding="utf-8")
        (out / "report_zh-CN.md").write_text(zh, encoding="utf-8")
        (out / "analysis_original.md").write_text(
            "# Public structured analysis\n\n" + json.dumps({k: record.get("plan", {}).get(k) for k in ("analysis_summary", "bottleneck", "evidence_summary", "expected_risk")}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (out / "analysis_zh-CN.md").write_text(
            "# 公开结构化分析\n\n" + json.dumps({k: record.get("plan", {}).get(k) for k in ("analysis_summary", "bottleneck", "evidence_summary", "expected_risk")}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        atomic_json(out / "metrics.json", _metrics(record))
        atomic_json(out / "provenance.json", {
            "canonical_memory": str(canonical_path), "canonical_memory_hash": _sha(canonical_path),
            "experiment_id": experiment_id, "candidate_hash": record.get("candidate_hash"),
            "candidate_commit": record.get("candidate_commit"), "working_directory": record.get("working_directory"),
            "environment_fingerprint": record.get("environment_fingerprint"),
        })
        atomic_json(out / "knowledge_used.json", record.get("knowledge", {}))
        atomic_json(out / "knowledge_produced.json", canonical.get("knowledge_candidates", []))
        atomic_json(out / "next_directions.json", canonical.get("next_directions", []))
        (out / "diff.patch").write_text(record.get("diff_patch", "# No reliable parent/candidate diff was captured.\n"), encoding="utf-8")
        atomic_json(out / "source_manifest.json", {"candidate_root": record.get("working_directory"), "candidate_hash": record.get("candidate_hash"), "changed_files": record.get("changed_files", [])})
        written.append(out)
    return written


def write_knowledge_candidates(lab_root: Path, canonical_path: Path, canonical: dict) -> list[Path]:
    """Derive non-promoted, scoped knowledge candidates from evidence only."""
    destination = lab_root / "knowledge" / "pending"
    destination.mkdir(parents=True, exist_ok=True)
    output: list[Path] = []
    for record in canonical.get("experiments", []):
        decision = record.get("decision", "")
        if decision not in {"REJECT_COMPILE", "REJECT_CORRECTNESS", "REJECT_AND_CONTINUE", "REJECT_PERFORMANCE", "READY_FOR_GATE"}:
            continue
        kind = "HYPOTHESIS" if decision == "READY_FOR_GATE" else "OBSERVATION"
        plan = record.get("plan", {})
        item = {
            "schema_version": 1,
            "id": f"candidate-{record['experiment_id']}",
            "title": plan.get("analysis_summary", record["experiment_id"]),
            "type": kind,
            "statement": plan.get("hypothesis", {}).get("claim", "UNKNOWN"),
            "scope": {"hardware": ["rtx5060_laptop_sm120"], "architecture": ["sm_120"], "backend": ["cuda_cpp"], "operator": ["rms_norm_train"], "dtype": [], "shape_regime": [], "training_or_inference": ["training"]},
            "evidence_for": [record["experiment_id"]], "evidence_against": [],
            "confidence": "SINGLE_EXPERIMENT", "status": "PENDING_HUMAN_REVIEW",
            "provenance": {"canonical_memory": str(canonical_path), "canonical_memory_hash": _sha(canonical_path), "evidence_level": 2},
            "created_at": _now(),
        }
        path = destination / f"{item['id']}.json"
        atomic_json(path, item)
        # Localization is representation only.  Retain canonical numbers/IDs
        # by reusing the same technical claim where no human translation exists.
        loc = lab_root / "i18n" / "zh-CN" / "knowledge" / f"{item['id']}.json"
        atomic_json(loc, {"source_record_id": item["id"], "source_hash": _sha(path), "language": "zh-CN", "translation_status": "CURRENT", "translator": "runtime-template", "schema_version": 1, "title_zh_CN": item["title"], "claim_zh_CN": item["statement"]})
        output.append(path)
    return output


def archive_framework_capability_gap(lab_root: Path, campaign_dir: Path, episode_dir: Path) -> Path:
    """Archive a tooling limitation without converting it into GPU knowledge.

    This is intentionally separate from ``archive_episode``: e0005 never
    reached a supervisor gate and therefore has no canonical performance
    decision.  Its record is scoped to the runner capability that was missing.
    """
    from ...core.journal import read_experiments
    episode_dir = Path(episode_dir); records = read_experiments(episode_dir / "journal.jsonl")
    out = Path(lab_root) / "experiment_archive" / f"{episode_dir.name}-framework-diagnostic-capability-gap"
    out.mkdir(parents=True, exist_ok=True)
    report = "\n".join((
        f"# {episode_dir.name}: Diagnostic capability gap", "",
        "## Scope", "", "- Classification: `FRAMEWORK_TOOLING_OBSERVATION`", "- This archive is not a GPU, kernel, or performance conclusion.",
        f"- Campaign: `{Path(campaign_dir).name}`", f"- Episode: `{episode_dir.name}`", f"- Recorded standard-loop attempts: `{len(records)}`", "",
        "## Factual outcome", "", "The active directive requested repeated per-shape statistics, cross-batch stability, regime grouping, static evidence, and representative profiling before candidate modification. At the time of this episode the runner supported only the fixed optimization path `edit -> compile -> correctness -> development ABBA`. It did not expose a first-class diagnostic action space.", "",
        "## Candidate status", "", "- Candidate remained identical to the incumbent according to the episode records.", "- No anti-strategy, hardware fact, or RMSNorm performance claim is produced by this archive.", "",
        "## Resolution", "", "Diagnostic experiment routing and controller-owned actions were added after this episode. A future user-approved episode may collect the requested evidence before proposing an optimization change.", "",
    ))
    zh = "\n".join((
        f"# {episode_dir.name}：诊断能力缺口", "",
        "## 适用范围", "", "- 分类：`FRAMEWORK_TOOLING_OBSERVATION`", "- 本归档不是 GPU、kernel 或性能结论。",
        f"- Campaign：`{Path(campaign_dir).name}`", f"- Episode：`{episode_dir.name}`", f"- 已记录的固定优化流水线尝试：`{len(records)}`", "",
        "## 事实结果", "", "该 Episode 的 directive 要求在修改 candidate 前获得 repeated per-shape statistics、cross-batch stability、regime grouping、static evidence 和 representative profiling。当时 Runner 只支持固定优化路径：`edit -> compile -> correctness -> development ABBA`，没有一等的 DIAGNOSTIC action space。", "",
        "## Candidate 状态", "", "- 依据 episode record，candidate 保持与 incumbent 相同。", "- 本归档不产生 ANTI_STRATEGY、HARDWARE_FACT 或 RMSNorm 性能结论。", "",
        "## 后续处理", "", "该 Episode 之后已补上诊断 experiment routing 和 controller-owned diagnostic actions。未来只有用户批准的新 Episode 才能先收集所请求的证据，再提出优化改动。", "",
    ))
    (out / "report_original.md").write_text(report, encoding="utf-8")
    (out / "report_zh-CN.md").write_text(zh, encoding="utf-8")
    (out / "analysis_original.md").write_text("# Public structured analysis\n\nRunner action space lacked DIAGNOSTIC experiments; no kernel conclusion was justified.\n", encoding="utf-8")
    (out / "analysis_zh-CN.md").write_text("# 公开结构化分析\n\nRunner 动作空间缺少 DIAGNOSTIC experiments，因此不能据此产生 kernel 结论。\n", encoding="utf-8")
    atomic_json(out / "metrics.json", {"test_only": False, "framework_observation": True, "experiment_records": [record.get("experiment_id") for record in records], "performance_conclusion": None})
    atomic_json(out / "provenance.json", {"episode_path": str(episode_dir), "journal": str(episode_dir / "journal.jsonl"), "candidate_path": str(episode_dir / "candidate"), "campaign": str(campaign_dir)})
    atomic_json(out / "knowledge_used.json", {"knowledge_produced": [], "reason": "framework tooling observation only"})
    atomic_json(out / "knowledge_produced.json", [])
    atomic_json(out / "next_directions.json", ["Use a user-approved diagnostic episode to characterize noise and select representative profile shapes before editing candidate code."])
    (out / "diff.patch").write_text("# No candidate diff: e0005 candidate remained identical to incumbent.\n", encoding="utf-8")
    atomic_json(out / "source_manifest.json", {"candidate_root": str(episode_dir / "candidate"), "records": [record.get("experiment_id") for record in records]})
    return out
