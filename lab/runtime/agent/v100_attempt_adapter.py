"""P1 boundary between the generic attempt controller and V100 evaluation.

The adapter deliberately has no SSH construction.  Its ``measure`` callable
is supplied by the deterministic evaluator owner; an Agent never receives it.
For a remote helper whose one job currently performs compile, correctness and
benchmark together, the adapter caches that one result and exposes the three
controller stages without creating duplicate jobs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..evaluators.contract_validation import load_contract_bundle, validate_candidate_source


class V100AttemptEvaluatorAdapter:
    def __init__(self, *, project_root: Path, operator: str, measure: Callable[[Path], dict[str, Any]],
                 qualify: Callable[[Path, dict[str, Any]], dict[str, Any]], profile: Callable[[Path], dict[str, Any] | None]):
        self.project_root = Path(project_root)
        self.operator = operator
        self.measure = measure
        self._qualify = qualify
        self._profile = profile
        self._cached: dict[str, dict[str, Any]] = {}

    def validate_contract(self, candidate: Path) -> dict[str, Any]:
        contract, _evaluation, fingerprint = load_contract_bundle(self.project_root, self.operator)
        failures = validate_candidate_source(Path(candidate), contract)
        return {"pass": not failures, "status": "PASS" if not failures else "REJECT_CONTRACT",
                "semantic_contract_sha256": contract.semantic_contract_sha256,
                "evaluation_fingerprint": fingerprint, "failures": failures}

    def _result(self, candidate: Path) -> dict[str, Any]:
        key = str(Path(candidate).resolve())
        if key not in self._cached:
            self._cached[key] = self.measure(Path(candidate))
        return self._cached[key]

    def compile(self, candidate: Path) -> dict[str, Any]:
        result = self._result(candidate)
        return {"pass": bool(result.get("compile_pass")), "diagnostics": result.get("compile_diagnostics") or [], "run_refs": result.get("runs") or []}

    def correctness(self, candidate: Path) -> dict[str, Any]:
        result = self._result(candidate)
        return {"pass": bool(result.get("correctness_pass")), "failed_shapes": result.get("failed_shapes") or [],
                "max_error": result.get("max_error"), "tolerance": result.get("tolerance")}

    def benchmark(self, candidate: Path) -> dict[str, Any]:
        result = self._result(candidate)
        return {"aggregate_score": result.get("aggregate_score"), "shapes": result.get("shapes") or [],
                "stability": result.get("qualification"), "runs": result.get("runs") or []}

    def profile(self, candidate: Path) -> dict[str, Any] | None:
        return self._profile(Path(candidate))

    def qualify(self, candidate: Path, benchmark: dict[str, Any]) -> dict[str, Any]:
        return self._qualify(Path(candidate), benchmark)
