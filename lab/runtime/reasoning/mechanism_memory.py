"""Mechanism records stored through the existing structured knowledge sink."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from lab.runtime.agent.attempt_loop import StructuredKnowledgeSink


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


class EvidenceStatus(str, Enum):
    CAUSAL_PROVEN = "CAUSAL_PROVEN"
    CAUSAL_UNPROVEN = "CAUSAL_UNPROVEN"
    OBSERVED = "OBSERVED"
    HYPOTHETICAL = "HYPOTHETICAL"


@dataclass(frozen=True)
class MechanismRecord:
    mechanism_id: str
    pattern: str
    symptoms: tuple[str, ...]
    mechanism: str
    transformation: str
    preconditions: tuple[str, ...]
    expected_effects: tuple[str, ...]
    risks: tuple[str, ...]
    counter_evidence: tuple[str, ...] = ()
    measured_evidence: tuple[dict[str, Any], ...] = ()
    applicability: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    evidence_status: EvidenceStatus = EvidenceStatus.HYPOTHETICAL

    def __post_init__(self) -> None:
        for name in ("mechanism_id", "pattern", "mechanism", "transformation"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("symptoms", "preconditions", "expected_effects", "risks", "counter_evidence"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or any(not isinstance(item, str) or not item.strip() for item in value):
                raise ValueError(f"{name} must be a tuple of non-empty strings")
        if not isinstance(self.measured_evidence, tuple) or any(not isinstance(item, dict) for item in self.measured_evidence):
            raise ValueError("measured_evidence must be a tuple of dictionaries")
        if not isinstance(self.applicability, dict) or not isinstance(self.provenance, dict):
            raise ValueError("applicability and provenance must be dictionaries")
        if not isinstance(self.evidence_status, EvidenceStatus):
            raise ValueError("evidence_status must be an EvidenceStatus")
        serializable = asdict(self)
        serializable["evidence_status"] = self.evidence_status.value
        try:
            json.dumps(serializable, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("mechanism evidence must be strict-JSON serializable") from exc

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["record_kind"] = "MECHANISM"
        result["evidence_status"] = self.evidence_status.value
        return result


class MechanismStore(StructuredKnowledgeSink):
    def add(self, record: MechanismRecord) -> bool:
        return self.append(record.to_dict())

    def query(self, observation: str, *, operator: str | None = None) -> list[MechanismRecord]:
        if not self.path.is_file():
            return []
        terms = {term for term in observation.lower().split() if len(term) > 2}
        found: list[tuple[int, MechanismRecord]] = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(
                    line,
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_reject_duplicate_keys,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid mechanism JSONL at line {line_number}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"invalid mechanism record at line {line_number}")
            if raw.get("record_kind") != "MECHANISM":
                continue
            try:
                record = MechanismRecord(
                    mechanism_id=raw["mechanism_id"], pattern=raw["pattern"], symptoms=tuple(raw.get("symptoms", [])),
                    mechanism=raw["mechanism"], transformation=raw["transformation"], preconditions=tuple(raw.get("preconditions", [])),
                    expected_effects=tuple(raw.get("expected_effects", [])), risks=tuple(raw.get("risks", [])),
                    counter_evidence=tuple(raw.get("counter_evidence", [])), measured_evidence=tuple(raw.get("measured_evidence", [])),
                    applicability=dict(raw.get("applicability", {})), provenance=dict(raw.get("provenance", {})),
                    evidence_status=EvidenceStatus(raw.get("evidence_status", "HYPOTHETICAL")))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid mechanism record at line {line_number}") from exc
            if operator and record.applicability.get("operator") not in {None, operator}:
                continue
            haystack = " ".join([record.pattern, record.mechanism, *record.symptoms]).lower()
            score = sum(term in haystack for term in terms)
            if score:
                found.append((score, record))
        return [record for _, record in sorted(found, key=lambda item: (-item[0], item[1].mechanism_id))]


EPISODE28_MECHANISM = MechanismRecord(
    mechanism_id="immutable_reduction_operand_residency",
    pattern="repeated immutable tensor consumption across reduction",
    symptoms=("same input read before and after reduction", "redundant global memory pass"),
    mechanism="redundant HBM traversal",
    transformation="extend x lifetime across reduction",
    preconditions=("per-thread live set acceptable", "values can remain live across synchronization"),
    expected_effects=("remove one global read pass", "reduce HBM traffic"),
    risks=("register pressure", "spill", "occupancy loss"),
    measured_evidence=(
        {"candidate": "v23", "latency_us_approx": 24.4},
        {"candidate": "v28_fast_mode", "latency_us_approx": 9.0},
    ),
    applicability={"operator": "rms_norm", "architecture": "sm70"},
    provenance={"source": "docs/audits/human_agent_episode28_audit.md", "confounders": ["vectorization", "specialization", "unrolling"]},
    evidence_status=EvidenceStatus.CAUSAL_UNPROVEN,
)
