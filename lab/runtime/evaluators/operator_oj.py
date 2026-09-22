"""L0 operator judge for compile, contract, correctness, and stable latency."""
from __future__ import annotations

from collections.abc import Callable
from math import isfinite
import re
from typing import Any

from .oj_models import JudgeResult, OperatorVerdict


class OperatorOJ:
    def __init__(self, runner: Callable[[dict[str, Any]], dict[str, Any]]):
        self._runner = runner

    def evaluate(self, request: dict[str, Any]) -> JudgeResult:
        raw_result = self._runner(dict(request))
        raw = raw_result if isinstance(raw_result, dict) else {}
        latency = raw.get("latency_ms")
        samples = raw.get("samples_ms")
        latency_valid = isinstance(latency, (int, float)) and not isinstance(latency, bool) and isfinite(float(latency)) and latency > 0
        samples_valid = isinstance(samples, (list, tuple)) and bool(samples) and all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(float(value)) and value > 0
            for value in samples
        )
        checks = {
            "compile": raw.get("compile_pass"),
            "contract": raw.get("contract_pass"),
            "correctness": raw.get("correctness_pass"),
            "shapes": raw.get("shapes_pass"),
            "stability": raw.get("stability_pass"),
            "latency_valid": latency_valid,
            "samples_valid": samples_valid,
            "contract_hash": isinstance(raw.get("contract_hash"), str) and re.fullmatch(r"[0-9a-f]{64}", raw["contract_hash"]) is not None,
            "candidate_hash": isinstance(raw.get("candidate_hash"), str) and re.fullmatch(r"[0-9a-f]{64}", raw["candidate_hash"]) is not None,
        }
        passed = all(value is True for value in checks.values())
        failed = tuple(f"failed_or_missing:{name}" for name, value in checks.items() if value is not True)
        return JudgeResult(
            level="L0_OPERATOR",
            verdict=(OperatorVerdict.OPERATOR_PASS if passed else OperatorVerdict.OPERATOR_FAIL).value,
            checks=checks,
            raw_metrics={"latency_ms": latency, "samples_ms": samples},
            evidence={"contract_hash": raw.get("contract_hash"), "candidate_hash": raw.get("candidate_hash")},
            reasons=failed,
        )
