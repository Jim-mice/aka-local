"""D-side SwiGLU L1 injection and evidence harness.

The harness never edits Megatron.  It patches the already imported MLP module
for one process, records the replacement invocation, and restores the symbol
when the context exits.  It is intentionally usable with a tiny fake module so
the wiring can be tested without torch or a GPU.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Callable


def _flatten(value: Any) -> list[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        result: list[float] = []
        for item in value:
            result.extend(_flatten(item))
        return result
    if hasattr(value, "detach"):
        return _flatten(value.detach().cpu().reshape(-1).tolist())
    raise TypeError(f"unsupported numeric value: {type(value).__name__}")


def _shape(value: Any) -> tuple[int, ...]:
    raw = getattr(value, "shape", None)
    if raw is not None:
        return tuple(int(x) for x in raw)
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        return (len(value),) + _shape(value[0])
    return ()


def _metadata(value: Any) -> dict[str, Any]:
    shape = _shape(value)
    stride = getattr(value, "stride", None)
    if callable(stride):
        stride = tuple(int(x) for x in stride())
    if stride is None and shape:
        running = 1
        calculated: list[int] = []
        for size in reversed(shape):
            calculated.append(running)
            running *= size
        stride = tuple(reversed(calculated))
    return {
        "shape": list(shape),
        "dtype": str(getattr(value, "dtype", "python-float")),
        "device": str(getattr(value, "device", "cpu")),
        "stride": list(stride or ()),
        "requires_grad": bool(getattr(value, "requires_grad", False)),
    }


def _error(reference: Any, candidate: Any) -> dict[str, Any]:
    left, right = _flatten(reference), _flatten(candidate)
    if len(left) != len(right):
        return {"shape_equal": False, "max_abs_error": None, "max_rel_error": None, "finite": False}
    abs_errors = [abs(a - b) for a, b in zip(left, right)]
    rel_errors = [abs(a - b) / max(abs(a), 1e-12) for a, b in zip(left, right)]
    return {
        "shape_equal": _shape(reference) == _shape(candidate),
        "max_abs_error": max(abs_errors, default=0.0),
        "max_rel_error": max(rel_errors, default=0.0),
        "finite": all(abs(x) != float("inf") and x == x for x in left + right),
    }


@dataclass
class SwiGLUReplacement:
    candidate: Callable[..., Any]
    replacement_id: str = "aka-local-swiglu-reference-v1"
    invocation_count: int = 0
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, input_tensor: Any, bias: Any = None, *args: Any, **kwargs: Any) -> Any:
        self.invocation_count += 1
        output = self.candidate(input_tensor, bias, *args, **kwargs)
        self.calls.append({"input": _metadata(input_tensor), "output": _metadata(output), "replacement_id": self.replacement_id})
        return output


@dataclass
class SwiGLUIntegrationResult:
    source_commit: str | None
    integration_contract_hash: str
    candidate_hash: str
    captured_shape: dict[str, Any] | None
    forward_correctness: dict[str, Any]
    backward_correctness: dict[str, Any]
    replacement_invocations: int
    baseline_target_invocations: int
    fallback_detected: bool | None
    runtime_environment: dict[str, Any]
    timing_if_available: dict[str, Any] | None
    status: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SwiGLUL1Harness:
    def __init__(self, target_module: Any, *, target_symbol: str = "bias_swiglu_impl"):
        self.target_module = target_module
        self.target_symbol = target_symbol
        self._original: Any = None
        self.replacement: SwiGLUReplacement | None = None

    def install(self, candidate: Callable[..., Any]) -> SwiGLUReplacement:
        if not hasattr(self.target_module, self.target_symbol):
            raise AttributeError(f"target symbol missing: {self.target_symbol}")
        self._original = getattr(self.target_module, self.target_symbol)
        self.replacement = SwiGLUReplacement(candidate)
        setattr(self.target_module, self.target_symbol, self.replacement)
        return self.replacement

    def restore(self) -> None:
        if self._original is not None:
            setattr(self.target_module, self.target_symbol, self._original)
            self._original = None

    def __enter__(self) -> "SwiGLUL1Harness":
        return self

    def __exit__(self, *_: Any) -> None:
        self.restore()

    def run_checks(
        self,
        baseline: Callable[[], Any],
        candidate: Callable[[], Any],
        *,
        baseline_input: Any,
        candidate_input: Any,
        baseline_grad: Any | None = None,
        candidate_grad: Any | None = None,
    ) -> dict[str, Any]:
        baseline_output = baseline()
        candidate_output = candidate()
        forward = _error(baseline_output, candidate_output)
        if baseline_grad is None or candidate_grad is None:
            backward = {"status": "BACKWARD_RUNTIME_BLOCKED", "finite": None}
        else:
            backward = {"status": "CHECKED", **_error(baseline_grad, candidate_grad)}
        invocation_count = self.replacement.invocation_count if self.replacement else 0
        return {
            "captured_shape": {"input": _metadata(candidate_input), "output": _metadata(candidate_output)},
            "forward_correctness": forward,
            "backward_correctness": backward,
            "replacement_invocations": invocation_count,
            "baseline_target_invocations": 1,
            "fallback_detected": invocation_count == 0,
        }


def build_swiglu_oj_evidence(result: SwiGLUIntegrationResult) -> dict[str, Any]:
    """Translate real/fake harness evidence into the existing L1 OJ input."""
    forward = result.forward_correctness
    backward = result.backward_correctness
    gradient_checks = []
    if isinstance(backward.get("input"), dict):
        gradient_checks.append(backward["input"])
    if isinstance(backward.get("parameters"), dict):
        gradient_checks.extend(item for item in backward["parameters"].values() if isinstance(item, dict))
    if not gradient_checks and {"finite", "max_abs_error", "max_rel_error"}.issubset(backward):
        gradient_checks.append(backward)
    max_abs_grad_error = max((float(item.get("max_abs_error", float("inf"))) for item in gradient_checks), default=float("inf"))
    max_rel_grad_error = max((float(item.get("max_rel_error", float("inf"))) for item in gradient_checks), default=float("inf"))
    finite_gradients = bool(gradient_checks) and all(item.get("finite") is True for item in gradient_checks)
    checks = {
        "replacement_invoked": result.replacement_invocations > 0,
        "no_silent_fallback": result.fallback_detected is False,
        "forward_correct": forward.get("shape_equal") is True and forward.get("finite") is True and forward.get("max_rel_error", 1.0) <= 1e-5,
        "backward_correct": backward.get("status") == "CHECKED" and finite_gradients and max_rel_grad_error <= 1e-5,
        "shape_compatible": result.captured_shape is not None and result.captured_shape.get("input", {}).get("shape") is not None,
        "distributed_invariants": result.runtime_environment.get("distributed_invariants", False),
    }
    return {
        "checks": checks,
        "replacement_marker": "aka-local-swiglu-reference-v1",
        "fallback_count": 1 if result.fallback_detected else 0,
        "provenance": {
            "runner_id": "aka-local-swiglu-l1-harness",
            "artifact_refs": ["swiglu_integration_result.json"],
            "contract_hash": result.integration_contract_hash,
            "candidate_hash": result.candidate_hash,
        },
        "metrics": result.timing_if_available or {},
        "gradient_metrics": {"max_abs_grad_error": max_abs_grad_error, "max_rel_grad_error": max_rel_grad_error, "finite": finite_gradients},
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_reasoner_snapshot(captured_shape: dict[str, Any], knowledge_path: Path | None = None) -> dict[str, Any]:
    """Convert captured runtime metadata into the existing reasoner path."""
    from lab.runtime.reasoning.hypothesis_planner import HypothesisPlanner
    from lab.runtime.reasoning.mechanism_memory import MechanismStore
    from lab.runtime.reasoning.performance_model import PerformanceFacts

    input_meta = captured_shape.get("input", {})
    shape_values = input_meta.get("shape") if isinstance(input_meta, dict) else None
    if not isinstance(shape_values, list) or not shape_values or any(not isinstance(x, int) or x <= 0 for x in shape_values):
        raise ValueError("captured input shape must contain positive integer dimensions")
    facts = PerformanceFacts(
        operator="swiglu",
        shape={f"dim_{index}": value for index, value in enumerate(shape_values)},
        dtype=str(input_meta.get("dtype", "unknown")),
        dataflow=("fc1", "swiglu", "fc2"),
        tensor_lifetimes={"fc1_output": "until_swiglu", "swiglu_output": "until_fc2"},
        global_memory_reads=("fc1_output",),
        global_memory_writes=("swiglu_output",),
        producer_consumer_boundaries=("fc1->swiglu", "swiglu->fc2"),
        unknown_fields=("profile_evidence",),
    )
    mechanisms = MechanismStore(knowledge_path).query("swiglu", operator="swiglu") if knowledge_path else []
    opportunities = HypothesisPlanner().rank(facts, mechanisms)
    return {"performance_facts": facts.to_dict(), "top_opportunities": [item.to_dict() for item in opportunities]}


def write_result(path: Path, result: SwiGLUIntegrationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
