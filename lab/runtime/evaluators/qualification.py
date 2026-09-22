"""Offline, explicit qualification statistics for repeated measurements."""
from __future__ import annotations

import math
import statistics
from typing import Any


DEFAULT_QUALIFICATION = {
    "repeats": 5,
    "statistic": "median",
    "max_cv": 0.10,
    "bimodality_policy": {"min_cluster_size": 2, "min_normalized_gap": 0.50},
    "order_policy": "shape_round_robin",
}


def qualification_config(evaluation: dict[str, Any]) -> dict[str, Any]:
    config = dict(DEFAULT_QUALIFICATION)
    supplied = evaluation.get("qualification") or {}
    config.update({key: value for key, value in supplied.items() if key != "bimodality_policy"})
    policy = dict(DEFAULT_QUALIFICATION["bimodality_policy"])
    policy.update(supplied.get("bimodality_policy") or {})
    config["bimodality_policy"] = policy
    return config


def ordered_shapes(shapes: list[str], repeat_index: int, config: dict[str, Any]) -> list[str]:
    """Return the declared per-repeat order without changing shape membership."""
    ordered = list(shapes)
    if config.get("order_policy") == "shape_round_robin" and ordered:
        offset = repeat_index % len(ordered)
        return ordered[offset:] + ordered[:offset]
    return ordered


def summarize_samples(samples: list[float], config: dict[str, Any]) -> dict[str, Any]:
    values = sorted(float(value) for value in samples)
    if not values:
        return {"n": 0, "stable": False, "bimodal": False, "reason": "no samples"}
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    median = statistics.median(values)
    cv = stdev / mean if mean else math.inf
    gaps = [right - left for left, right in zip(values, values[1:])]
    largest_gap = max(gaps, default=0.0)
    split_index = gaps.index(largest_gap) + 1 if gaps else 0
    left_count, right_count = split_index, len(values) - split_index
    denominator = median if median else 1.0
    normalized_gap = largest_gap / denominator
    policy = config["bimodality_policy"]
    bimodal = (
        len(values) >= 2 * int(policy["min_cluster_size"])
        and left_count >= int(policy["min_cluster_size"])
        and right_count >= int(policy["min_cluster_size"])
        and normalized_gap >= float(policy["min_normalized_gap"])
    )
    stable = cv <= float(config["max_cv"]) and not bimodal
    return {
        "n": len(values), "samples": values, "min": values[0], "max": values[-1],
        "mean": mean, "median": median, "stdev": stdev, "cv": cv,
        "largest_normalized_gap": normalized_gap, "split_index": split_index,
        "bimodal": bimodal, "stable": stable,
        "reason": "bimodal" if bimodal else "cv_exceeded" if not stable else "stable",
    }


def qualify_runs(repeated_results: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Summarize N complete evaluations; no remote side effects occur here."""
    per_shape: dict[str, list[float]] = {}
    aggregate_scores: list[float] = []
    provenance: list[dict[str, Any]] = []
    for result in repeated_results:
        if result.get("aggregate_score") is not None:
            aggregate_scores.append(float(result["aggregate_score"]))
        provenance.extend(result.get("runs") or [])
        for shape in result.get("shapes") or []:
            latency = shape.get("latency_us")
            if latency is not None:
                per_shape.setdefault(str(shape["shape"]), []).append(float(latency))
    per_shape_summary = {shape: summarize_samples(samples, config) for shape, samples in sorted(per_shape.items())}
    aggregate = summarize_samples(aggregate_scores, config)
    stable = bool(aggregate.get("stable")) and bool(per_shape_summary) and all(item["stable"] for item in per_shape_summary.values())
    return {
        "status": "QUALIFIED_ACCEPT" if stable else "PROVISIONAL_UNSTABLE",
        "configuration": config,
        "aggregate": aggregate,
        "per_shape": per_shape_summary,
        "run_provenance": provenance,
    }
