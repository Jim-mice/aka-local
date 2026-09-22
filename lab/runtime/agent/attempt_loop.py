"""Bounded, controller-owned implementation attempts for one hypothesis.

This is an extension point for :class:`LongHorizonRunner`, not an evaluator
and not an Agent.  The controller owns contract checks, evaluation,
qualification, lineage and persistence.  An Agent only receives structured
feedback and decides whether to repair or abandon its current hypothesis.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


FAILURE_CLASSES = {
    "IMPLEMENTATION_COMPILE_ERROR", "IMPLEMENTATION_CORRECTNESS_ERROR",
    "PERFORMANCE_REJECT", "FRAMEWORK_ERROR", "CONTRACT_ERROR",
    "ENVIRONMENT_ERROR", "QUALIFICATION_UNSTABLE", "ACCEPTED",
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_attempt_budget(path: Path | None = None) -> dict[str, Any]:
    default = {
        "schema_version": 1,
        "max_attempts_per_hypothesis": 5,
        "max_compile_repairs": 2,
        "max_correctness_repairs": 2,
        "max_performance_refinements": 2,
        "profile_after_correctness": True,
    }
    if path is not None and Path(path).is_file():
        supplied = json.loads(Path(path).read_text(encoding="utf-8"))
        default.update(supplied)
    return default


def profile_feedback(profile: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize optional profile facts; absent data is never fabricated."""
    profile = profile or {}
    status = profile.get("evidence_status")
    if status not in {"MEASURED", "UNAVAILABLE", "UNKNOWN"}:
        status = "MEASURED" if profile else "UNAVAILABLE"
    return {
        "kind": "profile_evidence",
        "registers_per_thread": profile.get("registers_per_thread"),
        "spills": profile.get("spills"),
        "occupancy": profile.get("occupancy"),
        "dram_bytes": profile.get("dram_bytes"),
        "kernel_time": profile.get("kernel_time"),
        "evidence_status": status,
        "raw_ref": profile.get("raw_ref"),
    }


class LineageStore:
    """Append-only lineage; it never rewrites episode artifacts."""
    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


class StructuredKnowledgeSink:
    """Write evidence records only when they are not framework failures."""
    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, record: dict[str, Any]) -> bool:
        if record.get("failure_class") in {"FRAMEWORK_ERROR", "CONTRACT_ERROR", "ENVIRONMENT_ERROR"}:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return True


@dataclass
class AttemptResult:
    status: str
    hypothesis_id: str
    attempts: list[dict[str, Any]]
    agent_thread_id: str | None
    budget: dict[str, Any]


class HypothesisAttemptController:
    """Run multiple implementation attempts in one already-started session.

    ``agent_step`` receives `(session, feedback_or_none, attempt_id)` and must
    return `{"action": "REPAIR"|"ABANDON_HYPOTHESIS"|"REQUEST_QUALIFICATION",
    "candidate_path": Path-or-string}`.  It may edit only the caller-owned
    workbench; this class does not grant remote or promotion authority.
    """
    def __init__(self, *, session: Any, evaluator: Any, lineage: LineageStore,
                 knowledge: StructuredKnowledgeSink, budget: dict[str, Any]):
        self.session = session
        self.evaluator = evaluator
        self.lineage = lineage
        self.knowledge = knowledge
        self.budget = dict(budget)

    def _thread_id(self) -> str | None:
        status = self.session.status() if hasattr(self.session, "status") else {}
        return status.get("thread_id") if isinstance(status, dict) else None

    @staticmethod
    def _compile_feedback(attempt_id: str, candidate_hash: str, result: dict[str, Any]) -> dict[str, Any]:
        diagnostics = result.get("diagnostics") or []
        return {"kind": "compile_error", "attempt_id": attempt_id,
                "candidate_hash": candidate_hash, "diagnostics": diagnostics}

    @staticmethod
    def _correctness_feedback(attempt_id: str, candidate_hash: str, result: dict[str, Any]) -> dict[str, Any]:
        return {"kind": "correctness_error", "attempt_id": attempt_id,
                "candidate_hash": candidate_hash,
                "failed_shapes": result.get("failed_shapes") or result.get("shapes") or [],
                "max_error": result.get("max_error"), "tolerance": result.get("tolerance")}

    @staticmethod
    def _benchmark_feedback(attempt_id: str, candidate_hash: str, benchmark: dict[str, Any], incumbent: float | None) -> dict[str, Any]:
        return {"kind": "benchmark_result", "attempt_id": attempt_id,
                "candidate_hash": candidate_hash, "per_shape": benchmark.get("per_shape") or benchmark.get("shapes") or [],
                "aggregate": benchmark.get("aggregate") or benchmark.get("aggregate_score"),
                "incumbent": incumbent, "stability": benchmark.get("stability")}

    def _record(self, *, hypothesis_id: str, attempt_id: str, episode: int | str,
                parent_hash: str | None, candidate_hash: str, status: str,
                failure_class: str, thread_id: str | None, result_ref: str | None,
                details: dict[str, Any]) -> dict[str, Any]:
        if failure_class not in FAILURE_CLASSES:
            raise ValueError(f"unknown failure class: {failure_class}")
        # These are deliberately explicit even when unknown.  A missing
        # profiler or an unobserved data-lifetime fact must not be converted
        # into an Agent guess in later knowledge retrieval.
        record = {
            "schema_version": 1, "hypothesis_id": hypothesis_id,
            "attempt_id": attempt_id, "episode": episode,
            "parent_candidate_hash": parent_hash, "candidate_hash": candidate_hash,
            "status": status, "failure_class": failure_class,
            "created_at": utcnow(), "result_ref": result_ref,
            "agent_thread_id": thread_id,
            "mechanism": details.pop("mechanism", None),
            "data_lifetime": details.pop("data_lifetime", None),
            "memory_passes": details.pop("memory_passes", None),
            "vectorization": details.pop("vectorization", None),
            "block_mapping": details.pop("block_mapping", None),
            "reduction_strategy": details.pop("reduction_strategy", None),
            "register_strategy": details.pop("register_strategy", None),
            "shared_strategy": details.pop("shared_strategy", None),
            "known_failure_mode": details.pop("known_failure_mode", None),
            "failed_shapes": details.get("correctness", {}).get("failed_shapes") or [],
            "compile_diagnostics": details.get("compile", {}).get("diagnostics") or [],
            "correctness_error": details.get("correctness"),
            "benchmark_distribution": details.get("benchmark"),
            "profile_evidence": details.get("profile_evidence") or profile_feedback(None),
            "evidence_level": details.get("evidence_level", "NONE"),
            **details,
        }
        self.lineage.append(record)
        self.knowledge.append(record)
        return record

    def run(self, *, hypothesis_id: str, episode: int | str,
            initial_candidate: Path, agent_step: Callable[[Any, dict[str, Any] | None, str], dict[str, Any]],
            incumbent_score: float | None = None, result_ref_prefix: str = "") -> AttemptResult:
        # The deterministic contract gate is intentionally before session start
        # and before an Agent can spend repair budget.
        gate = self.evaluator.validate_contract(Path(initial_candidate))
        if not gate.get("pass"):
            candidate_hash = file_sha256(Path(initial_candidate))
            rec = self._record(hypothesis_id=hypothesis_id, attempt_id=f"{hypothesis_id}-a0",
                episode=episode, parent_hash=None, candidate_hash=candidate_hash,
                status="REJECT_CONTRACT", failure_class="CONTRACT_ERROR", thread_id=None,
                result_ref=result_ref_prefix, details={"contract_validation": gate})
            return AttemptResult("REJECT_CONTRACT", hypothesis_id, [rec], None, self.budget)

        self.session.start()
        thread_id = self._thread_id()
        attempts: list[dict[str, Any]] = []
        feedback: dict[str, Any] | None = None
        parent_hash: str | None = None
        compile_repairs = correctness_repairs = performance_refinements = 0
        current = Path(initial_candidate)
        try:
            for number in range(1, int(self.budget["max_attempts_per_hypothesis"]) + 1):
                attempt_id = f"{hypothesis_id}-a{number}"
                step = agent_step(self.session, feedback, attempt_id) or {}
                action = step.get("action", "REPAIR")
                if action == "ABANDON_HYPOTHESIS":
                    return AttemptResult("ABANDON_HYPOTHESIS", hypothesis_id, attempts, thread_id, self.budget)
                current = Path(step.get("candidate_path", current))
                candidate_hash = file_sha256(current)
                try:
                    compile_result = self.evaluator.compile(current)
                except Exception as exc:
                    rec = self._record(hypothesis_id=hypothesis_id, attempt_id=attempt_id, episode=episode,
                        parent_hash=parent_hash, candidate_hash=candidate_hash, status="FRAMEWORK_ERROR",
                        failure_class="FRAMEWORK_ERROR", thread_id=thread_id, result_ref=f"{result_ref_prefix}{attempt_id}",
                        details={"framework_error": f"{type(exc).__name__}: {exc}", "evidence_level": "NONE"})
                    attempts.append(rec)
                    return AttemptResult("FRAMEWORK_ERROR", hypothesis_id, attempts, thread_id, self.budget)
                if not compile_result.get("pass"):
                    feedback = self._compile_feedback(attempt_id, candidate_hash, compile_result)
                    rec = self._record(hypothesis_id=hypothesis_id, attempt_id=attempt_id, episode=episode,
                        parent_hash=parent_hash, candidate_hash=candidate_hash, status="REJECT_COMPILE",
                        failure_class="IMPLEMENTATION_COMPILE_ERROR", thread_id=thread_id,
                        result_ref=f"{result_ref_prefix}{attempt_id}", details={"compile": compile_result, "feedback": feedback})
                    attempts.append(rec); parent_hash = candidate_hash; compile_repairs += 1
                    if compile_repairs > int(self.budget["max_compile_repairs"]):
                        return AttemptResult("ATTEMPT_BUDGET_EXHAUSTED", hypothesis_id, attempts, thread_id, self.budget)
                    continue
                try:
                    correctness = self.evaluator.correctness(current)
                except Exception as exc:
                    rec = self._record(hypothesis_id=hypothesis_id, attempt_id=attempt_id, episode=episode,
                        parent_hash=parent_hash, candidate_hash=candidate_hash, status="FRAMEWORK_ERROR",
                        failure_class="FRAMEWORK_ERROR", thread_id=thread_id, result_ref=f"{result_ref_prefix}{attempt_id}",
                        details={"compile": compile_result, "framework_error": f"{type(exc).__name__}: {exc}", "evidence_level": "NONE"})
                    attempts.append(rec)
                    return AttemptResult("FRAMEWORK_ERROR", hypothesis_id, attempts, thread_id, self.budget)
                if not correctness.get("pass"):
                    feedback = self._correctness_feedback(attempt_id, candidate_hash, correctness)
                    rec = self._record(hypothesis_id=hypothesis_id, attempt_id=attempt_id, episode=episode,
                        parent_hash=parent_hash, candidate_hash=candidate_hash, status="REJECT_CORRECTNESS",
                        failure_class="IMPLEMENTATION_CORRECTNESS_ERROR", thread_id=thread_id,
                        result_ref=f"{result_ref_prefix}{attempt_id}", details={"compile": compile_result, "correctness": correctness, "feedback": feedback})
                    attempts.append(rec); parent_hash = candidate_hash; correctness_repairs += 1
                    if correctness_repairs > int(self.budget["max_correctness_repairs"]):
                        return AttemptResult("ATTEMPT_BUDGET_EXHAUSTED", hypothesis_id, attempts, thread_id, self.budget)
                    continue
                try:
                    benchmark = self.evaluator.benchmark(current)
                    raw_profile = self.evaluator.profile(current) if self.budget.get("profile_after_correctness") else None
                    qualification = self.evaluator.qualify(current, benchmark)
                except Exception as exc:
                    rec = self._record(hypothesis_id=hypothesis_id, attempt_id=attempt_id, episode=episode,
                        parent_hash=parent_hash, candidate_hash=candidate_hash, status="FRAMEWORK_ERROR",
                        failure_class="FRAMEWORK_ERROR", thread_id=thread_id, result_ref=f"{result_ref_prefix}{attempt_id}",
                        details={"compile": compile_result, "correctness": correctness,
                                 "framework_error": f"{type(exc).__name__}: {exc}", "evidence_level": "NONE"})
                    attempts.append(rec)
                    return AttemptResult("FRAMEWORK_ERROR", hypothesis_id, attempts, thread_id, self.budget)
                profile = profile_feedback(raw_profile)
                qualified = qualification.get("status") == "QUALIFIED_ACCEPT"
                aggregate = benchmark.get("aggregate") or benchmark.get("aggregate_score")
                above_incumbent = incumbent_score is None or (aggregate is not None and float(aggregate) > float(incumbent_score))
                if qualified and above_incumbent:
                    feedback = self._benchmark_feedback(attempt_id, candidate_hash, benchmark, incumbent_score)
                    rec = self._record(hypothesis_id=hypothesis_id, attempt_id=attempt_id, episode=episode,
                        parent_hash=parent_hash, candidate_hash=candidate_hash, status="QUALIFIED_ACCEPT",
                        failure_class="ACCEPTED", thread_id=thread_id, result_ref=f"{result_ref_prefix}{attempt_id}",
                        details={"compile": compile_result, "correctness": correctness, "benchmark": benchmark,
                                 "qualification": qualification, "profile_evidence": profile, "feedback": feedback,
                                 "evidence_level": "QUALIFIED"})
                    attempts.append(rec)
                    return AttemptResult("QUALIFIED_ACCEPT", hypothesis_id, attempts, thread_id, self.budget)
                failure = "QUALIFICATION_UNSTABLE" if not qualified else "PERFORMANCE_REJECT"
                feedback = self._benchmark_feedback(attempt_id, candidate_hash, benchmark, incumbent_score)
                rec = self._record(hypothesis_id=hypothesis_id, attempt_id=attempt_id, episode=episode,
                    parent_hash=parent_hash, candidate_hash=candidate_hash, status="PROVISIONAL_REJECTED",
                    failure_class=failure, thread_id=thread_id, result_ref=f"{result_ref_prefix}{attempt_id}",
                    details={"compile": compile_result, "correctness": correctness, "benchmark": benchmark,
                             "qualification": qualification, "profile_evidence": profile, "feedback": feedback,
                             "evidence_level": "UNQUALIFIED"})
                attempts.append(rec); parent_hash = candidate_hash; performance_refinements += 1
                if performance_refinements > int(self.budget["max_performance_refinements"]):
                    return AttemptResult("ATTEMPT_BUDGET_EXHAUSTED", hypothesis_id, attempts, thread_id, self.budget)
            return AttemptResult("ATTEMPT_BUDGET_EXHAUSTED", hypothesis_id, attempts, thread_id, self.budget)
        finally:
            self.session.close()
