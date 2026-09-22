"""Offline-capable L1 integration judge.

The caller owns execution. This judge only validates structured evidence and
therefore cannot grant an Agent authority to run or promote a candidate.
"""
from __future__ import annotations

from collections.abc import Callable
from math import isfinite
import re
from typing import Any

from .oj_models import IntegrationVerdict, JudgeResult


REQUIRED_CHECKS = (
    "replacement_invoked",
    "no_silent_fallback",
    "forward_correct",
    "backward_correct",
    "shape_compatible",
    "distributed_invariants",
)


class IntegrationOJ:
    def __init__(self, runner: Callable[[dict[str, Any]], dict[str, Any]]):
        self._runner = runner

    def evaluate(self, request: dict[str, Any]) -> JudgeResult:
        raw_result = self._runner(dict(request))
        raw = raw_result if isinstance(raw_result, dict) else {}
        raw_checks = raw.get("checks") if isinstance(raw.get("checks"), dict) else {}
        checks = {name: raw_checks.get(name) for name in REQUIRED_CHECKS}
        missing = [name for name, value in checks.items() if value is None]
        failed = [name for name, value in checks.items() if value is not None and value is not True]
        marker_expected = request.get("expected_replacement_marker")
        marker_actual = raw.get("replacement_marker")
        checks["replacement_marker_matches"] = isinstance(marker_expected, str) and bool(marker_expected) and marker_actual == marker_expected
        if not checks["replacement_marker_matches"]:
            failed.append("replacement_marker_matches")
        expected_collectives = request.get("expected_collective_trace")
        if expected_collectives is not None:
            checks["collective_trace_matches"] = raw.get("collective_trace") == expected_collectives
            if not checks["collective_trace_matches"]:
                failed.append("collective_trace_matches")
        expected_world_size = request.get("expected_world_size")
        if expected_world_size is not None:
            world_size_valid = isinstance(expected_world_size, int) and not isinstance(expected_world_size, bool) and expected_world_size > 0
            checks["collective_world_size"] = world_size_valid
            if not world_size_valid:
                failed.append("collective_world_size")
            traces_by_rank = raw.get("collective_traces_by_rank")
            expected_ranks = {str(rank) for rank in range(expected_world_size)} if world_size_valid else set()
            traces_valid = (
                expected_collectives is not None
                and isinstance(traces_by_rank, dict)
                and set(traces_by_rank) == expected_ranks
                and all(traces_by_rank[str(rank)] == expected_collectives for rank in range(expected_world_size))
            )
            checks["collective_traces_all_ranks"] = traces_valid
            if not traces_valid:
                failed.append("collective_traces_all_ranks")
        required_metrics = request.get("required_metrics", ())
        if not isinstance(required_metrics, (list, tuple)) or not required_metrics or any(not isinstance(metric, str) or not metric for metric in required_metrics):
            checks["required_metrics_schema"] = False
            failed.append("required_metrics_schema")
            required_metrics = ()
        metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}
        for metric in required_metrics:
            name = f"metric:{metric}"
            value = metrics.get(metric)
            checks[name] = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and isfinite(float(value))
                and value >= 0
            )
            if not checks[name]:
                failed.append(name)
        fallback_count = raw.get("fallback_count")
        checks["fallback_count_zero"] = isinstance(fallback_count, int) and not isinstance(fallback_count, bool) and fallback_count == 0
        if not checks["fallback_count_zero"]:
            failed.append("fallback_count_zero")
        provenance = raw.get("provenance")
        artifact_refs = provenance.get("artifact_refs") if isinstance(provenance, dict) else None
        checks["runner_provenance"] = (
            isinstance(provenance, dict)
            and isinstance(provenance.get("runner_id"), str)
            and bool(provenance["runner_id"])
            and isinstance(artifact_refs, (list, tuple))
            and bool(artifact_refs)
            and all(isinstance(item, str) and bool(item) for item in artifact_refs)
        )
        if not checks["runner_provenance"]:
            failed.append("runner_provenance")
        for identity in ("contract_hash", "candidate_hash"):
            expected = request.get(f"expected_{identity}")
            actual = provenance.get(identity) if isinstance(provenance, dict) else None
            checks[f"{identity}_matches"] = (
                isinstance(expected, str)
                and re.fullmatch(r"[0-9a-f]{64}", expected) is not None
                and actual == expected
            )
            if not checks[f"{identity}_matches"]:
                failed.append(f"{identity}_matches")
        passed = not missing and not failed and all(value is True for value in checks.values())
        reasons = tuple([*(f"missing:{item}" for item in missing), *(f"failed:{item}" for item in failed)])
        return JudgeResult(
            level="L1_INTEGRATION",
            verdict=(IntegrationVerdict.INTEGRATION_PASS if passed else IntegrationVerdict.INTEGRATION_FAIL).value,
            checks=checks,
            raw_metrics=dict(metrics),
            evidence={"replacement_marker": marker_actual, "collective_trace": raw.get("collective_trace"), "collective_traces_by_rank": raw.get("collective_traces_by_rank"), "runner_provenance": raw.get("provenance")},
            reasons=reasons,
        )
