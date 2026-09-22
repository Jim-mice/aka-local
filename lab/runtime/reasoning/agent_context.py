"""Deterministic reasoner payload added to the existing Agent context."""
from __future__ import annotations

from typing import Any
import hashlib
import json

from .mechanism_memory import MechanismRecord
from .performance_model import Opportunity, PerformanceFacts
from .hypothesis_planner import PlanningResult


AUTHORITY_BOUNDARY = {
    "planner": "proposes mechanism-level hypotheses only",
    "agent": "implements one controller-approved hypothesis",
    "attempt_loop": "repairs implementation only",
    "oj": "evaluates mechanical evidence",
    "promotion": "controller-owned and not delegated to Agent",
}

IMPLEMENTATION_INSTRUCTION = "Implement only an approved transformation; do not infer acceptance or fabricate missing evidence."


def build_performance_context(
    facts: PerformanceFacts,
    opportunities: list[Opportunity],
    mechanisms: list[MechanismRecord],
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
        raise ValueError("top_k must be nonnegative")
    selected = opportunities[:top_k]
    relevant_ids = {item.mechanism_id for item in selected}
    relevant = [item for item in mechanisms if item.mechanism_id in relevant_ids]
    return {
        "schema_version": 1,
        "authority": dict(AUTHORITY_BOUNDARY),
        "performance_facts": facts.to_dict(),
        "top_opportunities": [item.to_dict() for item in selected],
        "relevant_mechanisms": [item.to_dict() for item in relevant],
        "implementation_instruction": IMPLEMENTATION_INSTRUCTION,
    }


def validate_performance_context(performance_context: dict[str, Any]) -> None:
    if not isinstance(performance_context, dict):
        raise TypeError("performance_context must be a dictionary")
    if performance_context.get("schema_version") != 1:
        raise ValueError("unsupported performance_context schema_version")
    if performance_context.get("authority") != AUTHORITY_BOUNDARY:
        raise ValueError("performance_context authority boundary mismatch")
    required = {"schema_version", "authority", "performance_facts", "top_opportunities", "relevant_mechanisms", "implementation_instruction"}
    if not required.issubset(performance_context):
        raise ValueError("performance_context is missing required fields")
    if set(performance_context) != required:
        raise ValueError("performance_context contains unknown fields")
    if not isinstance(performance_context["performance_facts"], dict):
        raise ValueError("performance_facts must be a dictionary")
    if not isinstance(performance_context["top_opportunities"], list) or not all(isinstance(item, dict) for item in performance_context["top_opportunities"]):
        raise ValueError("top_opportunities must be a list of dictionaries")
    if not isinstance(performance_context["relevant_mechanisms"], list) or not all(isinstance(item, dict) for item in performance_context["relevant_mechanisms"]):
        raise ValueError("relevant_mechanisms must be a list of dictionaries")
    opportunity_ids = [item.get("mechanism_id") for item in performance_context["top_opportunities"]]
    mechanism_ids = [item.get("mechanism_id") for item in performance_context["relevant_mechanisms"]]
    if any(not isinstance(item, str) or not item for item in opportunity_ids + mechanism_ids):
        raise ValueError("performance_context mechanism IDs must be non-empty strings")
    if len(set(opportunity_ids)) != len(opportunity_ids) or len(set(mechanism_ids)) != len(mechanism_ids):
        raise ValueError("performance_context mechanism IDs must be unique")
    if not set(opportunity_ids).issubset(mechanism_ids):
        raise ValueError("top opportunities must bind to relevant mechanisms")
    if performance_context["implementation_instruction"] != IMPLEMENTATION_INSTRUCTION:
        raise ValueError("performance_context implementation instruction mismatch")
    try:
        json.dumps(performance_context, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("performance_context must be JSON serializable") from exc


def augment_authoritative_context(context: dict[str, Any], performance_context: dict[str, Any]) -> dict[str, Any]:
    """Attach reasoner output and re-hash the controller-owned snapshot."""
    validate_performance_context(performance_context)
    result = dict(context)
    result["performance_reasoning"] = dict(performance_context)
    result.pop("context_hash", None)
    result["context_hash"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return result


def build_planning_context(
    facts: PerformanceFacts,
    plan: PlanningResult,
    mechanisms: list[MechanismRecord],
) -> dict[str, Any]:
    """Build a planning-only payload; it grants no implementation authority."""
    retrieved = [item.to_dict() for item in mechanisms]
    return {
        "schema_version": 2,
        "mode": "PLANNING_ONLY",
        "authority": dict(AUTHORITY_BOUNDARY),
        "performance_facts": facts.to_dict(),
        "retrieved_mechanisms": retrieved,
        "generated_hypotheses": [item.to_dict() for item in plan.ranked],
        "rejected_hypotheses": list(plan.rejected),
        "evidence_status": {
            "validated": len(plan.validated),
            "rejected": len(plan.rejected),
            "all_generation_is_advisory": True,
        },
        "risks": [risk for item in plan.ranked for risk in item.risks],
        "ablation_plans": list(plan.ablation_plans),
        "implementation_instruction": "Planning only: do not create or edit an optimization candidate.",
    }


def validate_planning_context(planning_context: dict[str, Any]) -> None:
    if not isinstance(planning_context, dict) or planning_context.get("schema_version") != 2:
        raise ValueError("unsupported planning_context schema_version")
    required = {"schema_version", "mode", "authority", "performance_facts", "retrieved_mechanisms", "generated_hypotheses", "rejected_hypotheses", "evidence_status", "risks", "ablation_plans", "implementation_instruction"}
    if set(planning_context) != required:
        raise ValueError("planning_context contains unknown or missing fields")
    if planning_context["mode"] != "PLANNING_ONLY" or planning_context["authority"] != AUTHORITY_BOUNDARY:
        raise ValueError("planning-only authority boundary mismatch")
    if planning_context["implementation_instruction"] != "Planning only: do not create or edit an optimization candidate.":
        raise ValueError("planning-only instruction mismatch")
    for name in ("retrieved_mechanisms", "generated_hypotheses", "rejected_hypotheses", "risks", "ablation_plans"):
        if not isinstance(planning_context[name], list):
            raise ValueError(f"{name} must be a list")
    if not isinstance(planning_context["evidence_status"], dict) or planning_context["evidence_status"].get("all_generation_is_advisory") is not True:
        raise ValueError("planning evidence authority mismatch")
    try:
        json.dumps(planning_context, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("planning_context must be JSON serializable") from exc


def augment_planning_context(context: dict[str, Any], planning_context: dict[str, Any]) -> dict[str, Any]:
    validate_planning_context(planning_context)
    result = dict(context)
    result["performance_planning"] = dict(planning_context)
    result.pop("context_hash", None)
    result["context_hash"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return result
