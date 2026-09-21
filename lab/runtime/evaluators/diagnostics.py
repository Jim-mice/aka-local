"""Machine-readable, campaign-scoped diagnostic evidence helpers.

These helpers deliberately contain no GPU-specific performance conclusion.
They turn an existing RMSNorm repeated A/B/B/A result into durable evidence
that an Agent may reason about, while retaining all raw measurements.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


def _stats(values: list[float]) -> dict:
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"n": 0, "mean": None, "median": None, "std": None, "cv": None,
                "min": None, "max": None, "p25": None, "p75": None, "p95": None}
    ordered = sorted(values)
    def quantile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"n": len(values), "mean": mean, "median": statistics.median(values), "std": std,
            "cv": (std / mean) if mean else None, "min": min(values), "max": max(values),
            "p25": quantile(.25), "p75": quantile(.75), "p95": quantile(.95)}


def _latency_bucket(value: float, thresholds: tuple[float, float, float]) -> str:
    first, second, third = thresholds
    if value <= first:
        return "very-small"
    if value <= second:
        return "small"
    if value <= third:
        return "medium"
    return "large"


def summarize_repeated_per_shape(raw: dict, *, identical_implementation: bool, policy: dict | None = None) -> dict:
    """Summarize the established ``rmsnorm_repeated.py`` JSON output.

    Regime cutoffs are data-derived from incumbent per-shape mean latency.
    Stability labels are explicitly scoped to this campaign diagnostic policy;
    they are never promoted to a cross-GPU hardware rule.
    """
    policy = policy or {}
    shapes = raw.get("shapes") or {}
    baseline_means = []
    for value in shapes.values():
        batches = value.get("batches") or []
        baseline_means.append(statistics.mean([row.get("incumbent_mean_ms", 0.0) for row in batches]) if batches else 0.0)
    ordered = sorted(value for value in baseline_means if value > 0)
    def q(fraction: float) -> float:
        if not ordered:
            return 0.0
        return ordered[round((len(ordered) - 1) * fraction)]
    thresholds = (q(.25), q(.50), q(.75))
    stability = policy.get("stability") or {}
    max_cv = float(stability.get("max_cv", .05))
    min_consistency = float(stability.get("min_direction_consistency", .80))
    min_effect = float(stability.get("min_effect_size", .01))
    rows, regime_rows = [], {}
    for shape_id, value in sorted(shapes.items(), key=lambda item: int(item[0])):
        batches = value.get("batches") or []
        incumbent = [row.get("incumbent_mean_ms") for row in batches]
        candidate = [row.get("candidate_mean_ms") for row in batches]
        speedups = [row.get("speedup") for row in batches]
        inc_stats, cand_stats, speed_stats = _stats(incumbent), _stats(candidate), _stats(speedups)
        direction = sum(1 for value in speedups if value and value > 1.0) / len(speedups) if speedups else 0.0
        effect = abs((speed_stats["mean"] or 1.0) - 1.0)
        if identical_implementation:
            classification = "IDENTICAL_IMPLEMENTATION_NOISE_CHARACTERIZATION"
        elif speed_stats["cv"] is not None and speed_stats["cv"] > max_cv:
            classification = "noise_dominated"
        elif effect < min_effect:
            classification = "noise_dominated"
        elif direction >= min_consistency or direction <= (1.0 - min_consistency):
            classification = "stable_slow" if (speed_stats["mean"] or 1.0) < 1.0 else "stable_fast"
        else:
            classification = "unstable"
        input_data = value.get("input") or {}
        latency = inc_stats["mean"] or 0.0
        regime = _latency_bucket(latency, thresholds)
        row = {"shape_id": str(shape_id), "token_count": input_data.get("token_count"),
               "hidden_size": input_data.get("hidden_size"), "latency_regime": regime,
               "incumbent": inc_stats, "candidate": cand_stats, "speedup": speed_stats,
               "cross_batch_direction_consistency": direction, "classification": classification}
        rows.append(row)
        regime_rows.setdefault(regime, []).append(row)
    regimes = {}
    for regime, values in regime_rows.items():
        regimes[regime] = {"n_shapes": len(values),
            "incumbent_latency": _stats([value["incumbent"]["mean"] for value in values]),
            "candidate_latency": _stats([value["candidate"]["mean"] for value in values]),
            "speedup": _stats([value["speedup"]["mean"] for value in values]),
            "cross_batch_stability": _stats([value["speedup"]["cv"] for value in values if value["speedup"]["cv"] is not None])}
    return {"kind": "IDENTICAL_IMPLEMENTATION_NOISE_CHARACTERIZATION" if identical_implementation else "REPEATED_PER_SHAPE_BENCHMARK",
            "protocol": raw.get("protocol"), "parameters": {"batches": raw.get("batches"), "warmup": raw.get("warmup"), "repeats": raw.get("repeats")},
            "latency_regime_method": {"method": "incumbent_latency_quartiles", "thresholds_ms": thresholds},
            "policy": {"max_cv": max_cv, "min_direction_consistency": min_consistency, "min_effect_size": min_effect,
                       "scope": "campaign diagnostic policy; not a portable GPU rule"},
            "shapes": rows, "regimes": regimes, "aggregate": raw.get("aggregate") or {}}


def write_repeated_artifacts(destination: Path, raw: dict, summary: dict) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    raw_path, summary_path = destination / "raw.json", destination / "summary.json"
    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    per_shape = destination / "per_shape.csv"
    with per_shape.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["shape_id", "token_count", "hidden_size", "latency_regime", "incumbent_mean_ms", "candidate_mean_ms", "speedup_mean", "speedup_median", "speedup_std", "speedup_cv", "direction_consistency", "classification"])
        writer.writeheader()
        for row in summary.get("shapes", []):
            writer.writerow({"shape_id": row["shape_id"], "token_count": row.get("token_count"), "hidden_size": row.get("hidden_size"), "latency_regime": row["latency_regime"], "incumbent_mean_ms": row["incumbent"]["mean"], "candidate_mean_ms": row["candidate"]["mean"], "speedup_mean": row["speedup"]["mean"], "speedup_median": row["speedup"]["median"], "speedup_std": row["speedup"]["std"], "speedup_cv": row["speedup"]["cv"], "direction_consistency": row["cross_batch_direction_consistency"], "classification": row["classification"]})
    regime = destination / "regime_summary.csv"
    with regime.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["regime", "n_shapes", "incumbent_mean_ms", "candidate_mean_ms", "speedup_mean", "speedup_cv"])
        writer.writeheader()
        for name, row in summary.get("regimes", {}).items():
            writer.writerow({"regime": name, "n_shapes": row["n_shapes"], "incumbent_mean_ms": row["incumbent_latency"]["mean"], "candidate_mean_ms": row["candidate_latency"]["mean"], "speedup_mean": row["speedup"]["mean"], "speedup_cv": row["cross_batch_stability"]["mean"]})
    return {"raw": str(raw_path), "summary": str(summary_path), "per_shape_csv": str(per_shape), "regime_summary_csv": str(regime)}
