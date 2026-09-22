"""Configurable L2 end-to-end policy with raw-metric preservation."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from math import inf, isfinite
from pathlib import Path
from typing import Any

from .oj_models import JudgeResult, SystemVerdict


@dataclass(frozen=True)
class EndToEndPolicy:
    min_iteration_speedup: float = 1.0
    min_throughput_gain: float = 0.0
    max_memory_delta_bytes: int | None = None
    max_abs_loss_delta: float | None = None
    max_abs_convergence_proxy_delta: float | None = None
    require_gradient_check: bool = True
    require_repeated_qualification: bool = True
    min_qualification_runs: int = 2
    required_metrics: tuple[str, ...] = (
        "baseline_iteration_ms", "candidate_iteration_ms",
        "baseline_samples_per_sec", "candidate_samples_per_sec",
        "baseline_tokens_per_sec", "candidate_tokens_per_sec",
        "baseline_peak_memory_bytes", "candidate_peak_memory_bytes",
        "convergence_proxy_delta", "loss_delta", "gradient_check",
    )

    def __post_init__(self) -> None:
        if not _is_finite_number(self.min_iteration_speedup, positive=True):
            raise ValueError("min_iteration_speedup must be finite and positive")
        if not _is_finite_number(self.min_throughput_gain):
            raise ValueError("min_throughput_gain must be finite")
        if self.max_memory_delta_bytes is not None and not _is_finite_number(self.max_memory_delta_bytes):
            raise ValueError("max_memory_delta_bytes must be finite")
        if self.max_abs_loss_delta is not None and not _is_finite_number(self.max_abs_loss_delta, nonnegative=True):
            raise ValueError("max_abs_loss_delta must be finite and nonnegative")
        if self.max_abs_convergence_proxy_delta is not None and not _is_finite_number(self.max_abs_convergence_proxy_delta, nonnegative=True):
            raise ValueError("max_abs_convergence_proxy_delta must be finite and nonnegative")
        if isinstance(self.min_qualification_runs, bool) or not isinstance(self.min_qualification_runs, int) or self.min_qualification_runs < 1:
            raise ValueError("min_qualification_runs must be a positive integer")
        if not isinstance(self.require_gradient_check, bool) or not isinstance(self.require_repeated_qualification, bool):
            raise ValueError("require_* policy fields must be booleans")
        if not isinstance(self.required_metrics, tuple) or not self.required_metrics or any(not isinstance(item, str) or not item for item in self.required_metrics):
            raise ValueError("required_metrics must be a non-empty tuple of names")
        if len(set(self.required_metrics)) != len(self.required_metrics):
            raise ValueError("required_metrics cannot contain duplicates")

    @classmethod
    def from_json(cls, path: Path) -> "EndToEndPolicy":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EndToEndPolicy":
        if not isinstance(raw, dict):
            raise TypeError("end-to-end policy must be a dictionary")
        if raw.get("schema_version", 1) != 1:
            raise ValueError("unsupported end-to-end policy schema_version")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(raw) - allowed - {"schema_version", "note"}
        if unknown:
            raise ValueError(f"unknown end-to-end policy fields: {sorted(unknown)}")
        raw = dict(raw)
        if "required_metrics" in raw:
            if not isinstance(raw["required_metrics"], (list, tuple)):
                raise ValueError("required_metrics must be a list or tuple")
            raw["required_metrics"] = tuple(raw["required_metrics"])
        return cls(**{key: raw[key] for key in allowed if key in raw})


@dataclass(frozen=True)
class QualificationEvidence:
    protocol_hash: str
    baseline_run_ids: tuple[str, ...]
    candidate_run_ids: tuple[str, ...]
    stable: bool
    comparable: bool
    artifact_refs: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "QualificationEvidence":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown qualification fields: {sorted(unknown)}")

        def sequence(value: Any) -> tuple[Any, ...]:
            return tuple(value) if isinstance(value, (list, tuple)) else ()

        return cls(
            protocol_hash=raw.get("protocol_hash", ""),
            baseline_run_ids=sequence(raw.get("baseline_run_ids")),
            candidate_run_ids=sequence(raw.get("candidate_run_ids")),
            stable=raw.get("stable"),
            comparable=raw.get("comparable"),
            artifact_refs=sequence(raw.get("artifact_refs")),
        )

    def checks(self, minimum_runs: int) -> dict[str, bool]:
        baseline_ids_valid = isinstance(self.baseline_run_ids, tuple) and bool(self.baseline_run_ids) and all(isinstance(item, str) and bool(item) for item in self.baseline_run_ids)
        candidate_ids_valid = isinstance(self.candidate_run_ids, tuple) and bool(self.candidate_run_ids) and all(isinstance(item, str) and bool(item) for item in self.candidate_run_ids)
        unique_ids = baseline_ids_valid and candidate_ids_valid and len(set(self.baseline_run_ids + self.candidate_run_ids)) == len(self.baseline_run_ids) + len(self.candidate_run_ids)
        artifacts_valid = isinstance(self.artifact_refs, tuple) and bool(self.artifact_refs) and all(isinstance(item, str) and bool(item) for item in self.artifact_refs)
        return {
            "qualification_protocol": isinstance(self.protocol_hash, str) and re.fullmatch(r"[0-9a-f]{64}", self.protocol_hash) is not None,
            "qualification_baseline_runs": baseline_ids_valid and len(self.baseline_run_ids) >= minimum_runs,
            "qualification_candidate_runs": candidate_ids_valid and len(self.candidate_run_ids) >= minimum_runs,
            "qualification_unique_runs": unique_ids,
            "qualification_stable": self.stable is True,
            "qualification_comparable": self.comparable is True,
            "qualification_artifacts": artifacts_valid,
        }


def _is_finite_number(value: Any, *, positive: bool = False, nonnegative: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        return False
    if positive:
        return value > 0
    if nonnegative:
        return value >= 0
    return True


def _pair_valid(left: Any, right: Any, *, positive: bool = False, nonnegative: bool = False) -> bool | None:
    if left is None or right is None:
        return None
    return _is_finite_number(left, positive=positive, nonnegative=nonnegative) and _is_finite_number(right, positive=positive, nonnegative=nonnegative)


def amdahl_upper_bound(operator_fraction_of_step: float) -> float:
    if not _is_finite_number(operator_fraction_of_step, nonnegative=True) or operator_fraction_of_step > 1.0:
        raise ValueError("operator_fraction_of_step must be in [0, 1]")
    return inf if operator_fraction_of_step == 1.0 else 1.0 / (1.0 - operator_fraction_of_step)


class EndToEndOJ:
    def __init__(self, policy: EndToEndPolicy | None = None):
        self.policy = policy or EndToEndPolicy()

    def evaluate(self, metrics: dict[str, Any], *, qualification: QualificationEvidence | dict[str, Any] | None = None) -> JudgeResult:
        input_error = None
        if not isinstance(metrics, dict):
            raw = {}
            input_error = "metrics must be a dictionary"
        else:
            raw = dict(metrics)
            try:
                json.dumps(raw, allow_nan=False)
            except (TypeError, ValueError) as exc:
                input_error = f"metrics must be strict-JSON serializable: {exc}"
                raw = {}
        baseline_ms = raw.get("baseline_iteration_ms")
        candidate_ms = raw.get("candidate_iteration_ms")
        baseline_tps = raw.get("baseline_tokens_per_sec")
        candidate_tps = raw.get("candidate_tokens_per_sec")
        baseline_sps = raw.get("baseline_samples_per_sec")
        candidate_sps = raw.get("candidate_samples_per_sec")
        valid_iteration = _pair_valid(baseline_ms, candidate_ms, positive=True)
        valid_tps = _pair_valid(baseline_tps, candidate_tps, positive=True)
        valid_sps = _pair_valid(baseline_sps, candidate_sps, positive=True)
        iteration_speedup = baseline_ms / candidate_ms if valid_iteration is True else None
        throughput_gain = ((candidate_tps / baseline_tps) - 1.0) if valid_tps is True else None
        samples_per_sec_gain = ((candidate_sps / baseline_sps) - 1.0) if valid_sps is True else None
        supplied_memory_delta = raw.get("memory_delta_bytes")
        valid_memory_pair = _pair_valid(raw.get("baseline_peak_memory_bytes"), raw.get("candidate_peak_memory_bytes"), nonnegative=True)
        computed_memory_delta = (
            raw["candidate_peak_memory_bytes"] - raw["baseline_peak_memory_bytes"]
            if valid_memory_pair is True else None
        )
        supplied_memory_delta_valid = supplied_memory_delta is None or _is_finite_number(supplied_memory_delta)
        memory_delta_consistent = (
            None if supplied_memory_delta is None
            else False if not supplied_memory_delta_valid or computed_memory_delta is None
            else float(supplied_memory_delta) == float(computed_memory_delta)
        )
        memory_delta = computed_memory_delta
        loss_delta = raw.get("loss_delta")
        valid_loss = None if loss_delta is None else _is_finite_number(loss_delta)
        convergence_delta = raw.get("convergence_proxy_delta")
        valid_convergence = None if convergence_delta is None else _is_finite_number(convergence_delta)
        derived = {
            "iteration_speedup": iteration_speedup,
            "throughput_gain": throughput_gain,
            "samples_per_sec_gain": samples_per_sec_gain,
            "memory_delta_bytes": memory_delta,
        }
        fraction = raw.get("operator_fraction_of_step")
        valid_fraction = fraction is None or (_is_finite_number(fraction, nonnegative=True) and fraction <= 1.0)
        if fraction is not None and valid_fraction:
            ceiling = amdahl_upper_bound(fraction)
            derived["max_possible_e2e_speedup"] = None if ceiling == inf else ceiling
            derived["max_possible_e2e_speedup_unbounded"] = ceiling == inf

        checks: dict[str, bool | None] = {
            "input_json": input_error is None,
            "iteration_inputs": valid_iteration,
            "iteration_speedup": None if valid_iteration is None else (False if valid_iteration is False else iteration_speedup >= self.policy.min_iteration_speedup),
            "throughput_inputs": valid_tps,
            "throughput_gain": None if valid_tps is None else (False if valid_tps is False else throughput_gain >= self.policy.min_throughput_gain),
            "memory": True if self.policy.max_memory_delta_bytes is None else (None if memory_delta is None else memory_delta <= self.policy.max_memory_delta_bytes),
            "loss": True if self.policy.max_abs_loss_delta is None else (None if valid_loss is None else (False if valid_loss is False else abs(float(loss_delta)) <= self.policy.max_abs_loss_delta)),
            "convergence_proxy": True if self.policy.max_abs_convergence_proxy_delta is None else (None if valid_convergence is None else (False if valid_convergence is False else abs(float(convergence_delta)) <= self.policy.max_abs_convergence_proxy_delta)),
            "samples_per_sec": valid_sps,
            "peak_memory": valid_memory_pair,
            "loss_finite": valid_loss,
            "convergence_proxy_finite": valid_convergence,
            "gradient": True if not self.policy.require_gradient_check else (None if raw.get("gradient_check") is None else raw.get("gradient_check") is True),
        }
        if fraction is not None:
            checks["operator_fraction_of_step"] = valid_fraction
        if supplied_memory_delta is not None:
            checks["memory_delta_input"] = supplied_memory_delta_valid
            checks["memory_delta_consistent"] = memory_delta_consistent
        qualification_input = qualification
        qualification_error = None
        try:
            if isinstance(qualification, dict):
                qualification = QualificationEvidence.from_dict(qualification)
            if qualification is not None and not isinstance(qualification, QualificationEvidence):
                raise TypeError("qualification must be QualificationEvidence, a dict, or None")
        except (TypeError, ValueError) as exc:
            qualification_error = str(exc)
            qualification = None
            checks["qualification_schema"] = False
        if self.policy.require_repeated_qualification:
            if qualification is None:
                checks["qualification_evidence"] = None
            else:
                checks.update(qualification.checks(self.policy.min_qualification_runs))
        checks.update({f"metric:{name}": (True if raw.get(name) is not None else None) for name in self.policy.required_metrics})
        failed = [name for name, value in checks.items() if value is False]
        missing = [name for name, value in checks.items() if value is None]
        qualification_confirmed = not self.policy.require_repeated_qualification or (
            qualification is not None and all(qualification.checks(self.policy.min_qualification_runs).values())
        )
        if failed:
            verdict = SystemVerdict.SYSTEM_REJECT
        elif missing or (self.policy.require_repeated_qualification and not qualification_confirmed):
            verdict = SystemVerdict.SYSTEM_PROVISIONAL
        else:
            verdict = SystemVerdict.SYSTEM_QUALIFIED_ACCEPT
        reasons = tuple([*(f"missing:{item}" for item in missing), *(f"failed:{item}" for item in failed)])
        if self.policy.require_repeated_qualification and not qualification_confirmed and not failed:
            reasons += ("repeated_qualification_required",)
        return JudgeResult(
            level="L2_END_TO_END",
            verdict=verdict.value,
            checks=checks,
            raw_metrics={"input": raw, "derived": derived},
            evidence={"policy": asdict(self.policy), "qualified": qualification_confirmed, "qualification": qualification_input if isinstance(qualification_input, dict) else (None if qualification_input is None else asdict(qualification_input) if isinstance(qualification_input, QualificationEvidence) else repr(qualification_input)), "qualification_error": qualification_error, "input_error": input_error},
            reasons=reasons,
        )
