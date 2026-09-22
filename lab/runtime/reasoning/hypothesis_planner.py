"""Evidence-aware transformation ranking; it does not generate CUDA."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Callable, Iterable

from .mechanism_memory import EvidenceStatus, MechanismRecord
from .performance_model import Magnitude, Opportunity, PerformanceFacts


def _magnitude(score: float) -> Magnitude:
    if score >= 7:
        return Magnitude.LARGE
    if score >= 4:
        return Magnitude.MEDIUM
    if score > 0:
        return Magnitude.SMALL
    return Magnitude.UNKNOWN


@dataclass(frozen=True)
class CounterfactualAnswers:
    data_read_twice: tuple[str, ...]
    unnecessary_materializations: tuple[str, ...]
    values_that_could_remain_live: tuple[str, ...]
    removable_synchronizations: tuple[str, ...]
    fixed_dimensions_treated_dynamically: tuple[str, ...]
    fusible_boundaries: tuple[str, ...]
    idle_resources: tuple[str, ...]
    infinite_operator_e2e_ceiling: float | None

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        if self.infinite_operator_e2e_ceiling is not None and not isfinite(self.infinite_operator_e2e_ceiling):
            result["infinite_operator_e2e_ceiling"] = "UNBOUNDED"
        return result


class HypothesisPlanner:
    def counterfactuals(self, facts: PerformanceFacts) -> CounterfactualAnswers:
        reads = tuple(sorted({name for name in facts.global_memory_reads if facts.global_memory_reads.count(name) > 1}))
        materialized = tuple(name for name in facts.global_memory_writes if name in facts.global_memory_reads)
        live = tuple(name for name in reads if name not in facts.register_residency and name not in facts.shared_residency)
        dynamic_fixed = tuple(name for name in facts.dynamic_dimensions if name in facts.fixed_dimensions)
        idle = tuple(str(x) for x in facts.profile_evidence.get("idle_resources", []))
        fraction = facts.e2e_profile.get("operator_fraction_of_step")
        if fraction is None:
            ceiling = None
        elif isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not isfinite(float(fraction)) or not 0.0 <= float(fraction) <= 1.0:
            raise ValueError("operator_fraction_of_step must be a finite number in [0, 1]")
        else:
            ceiling = float("inf") if float(fraction) == 1.0 else 1.0 / (1.0 - float(fraction))
        return CounterfactualAnswers(
            data_read_twice=reads,
            unnecessary_materializations=materialized,
            values_that_could_remain_live=live,
            removable_synchronizations=facts.synchronizations,
            fixed_dimensions_treated_dynamically=dynamic_fixed,
            fusible_boundaries=facts.producer_consumer_boundaries,
            idle_resources=idle,
            infinite_operator_e2e_ceiling=ceiling,
        )

    def rank(self, facts: PerformanceFacts, mechanisms: Iterable[MechanismRecord]) -> list[Opportunity]:
        cf = self.counterfactuals(facts)
        candidates: list[Opportunity] = []
        for record in mechanisms:
            applicability = record.applicability.get("operator")
            if applicability not in {None, facts.operator}:
                continue
            score = 1.0
            factors: dict[str, float | int | bool | str | None] = {"base": 1.0}
            if cf.infinite_operator_e2e_ceiling is not None:
                factors["infinite_operator_e2e_ceiling"] = (
                    cf.infinite_operator_e2e_ceiling
                    if isfinite(cf.infinite_operator_e2e_ceiling)
                    else "UNBOUNDED"
                )
            mechanism_text = (record.mechanism + " " + record.transformation).lower()
            if cf.data_read_twice and any(word in mechanism_text for word in ("memory", "hbm", "pass", "lifetime")):
                score += 4.0
                factors["repeated_read_match"] = 4.0
            if cf.unnecessary_materializations and any(word in mechanism_text for word in ("material", "fuse", "write")):
                score += 3.0
                factors["materialization_match"] = 3.0
            if facts.kernel_launches and facts.kernel_launches > 1 and "fuse" in mechanism_text:
                launch_score = min(3.0, facts.kernel_launches - 1)
                score += launch_score
                factors["launches_removable"] = facts.kernel_launches - 1
                factors["launch_score"] = launch_score
            if facts.synchronizations and any(word in mechanism_text for word in ("sync", "reduction", "fuse", "lifetime")):
                sync_score = min(2.0, 0.5 * len(facts.synchronizations))
                score += sync_score
                factors["synchronizations_exposed"] = len(facts.synchronizations)
                factors["synchronization_score"] = sync_score
            if facts.known_reuse and any(word in mechanism_text for word in ("reuse", "resident", "lifetime")):
                score += 2.0
                factors["reuse_score"] = 2.0
            scoped_estimates = facts.profile_evidence.get("mechanism_estimates")
            if isinstance(scoped_estimates, dict):
                mechanism_estimate = scoped_estimates.get(record.mechanism_id, {})
            else:
                mechanism_estimate = facts.profile_evidence
            estimated_bytes_saved = mechanism_estimate.get("estimated_bytes_saved")
            if isinstance(estimated_bytes_saved, (int, float)) and not isinstance(estimated_bytes_saved, bool) and isfinite(float(estimated_bytes_saved)) and estimated_bytes_saved > 0:
                byte_score = 2.0 if facts.estimated_bytes and estimated_bytes_saved >= 0.25 * facts.estimated_bytes else 1.0
                score += byte_score
                factors["estimated_bytes_saved"] = estimated_bytes_saved
                factors["bytes_score"] = byte_score
            feasibility = mechanism_estimate.get("working_set_feasible")
            if feasibility is True:
                score += 1.0
                factors["working_set_feasibility"] = "SUPPORTED"
                factors["feasibility_score"] = 1.0
            elif feasibility is False:
                score -= 3.0
                factors["working_set_feasibility"] = "CONTRADICTED"
                factors["feasibility_score"] = -3.0
            supported = set(facts.profile_evidence.get("supports_mechanism_ids", ()))
            contradicted = set(facts.profile_evidence.get("contradicts_mechanism_ids", ()))
            if record.mechanism_id in supported:
                score += 2.0
                factors["profile_evidence_score"] = 2.0
            elif record.mechanism_id in contradicted:
                score -= 4.0
                factors["profile_evidence_score"] = -4.0
            if record.evidence_status in {EvidenceStatus.CAUSAL_PROVEN, EvidenceStatus.OBSERVED}:
                score += 1.0
                factors["prior_evidence_score"] = 1.0
            risk_penalty = min(2.0, 0.35 * len(record.risks))
            score -= risk_penalty
            factors["risk_penalty"] = -risk_penalty
            candidates.append(Opportunity(
                observation=record.pattern,
                mechanism=record.mechanism,
                transformation=record.transformation,
                preconditions=record.preconditions,
                expected_effect=record.expected_effects,
                risks=record.risks,
                estimated_magnitude=_magnitude(score),
                confidence=record.evidence_status.value,
                required_evidence=("L0 correctness and latency", "L1 replacement marker and backward", "L2 end-to-end profile"),
                score=round(score, 3),
                mechanism_id=record.mechanism_id,
                ranking_factors=factors,
            ))
        return sorted(candidates, key=lambda item: (-item.score, item.transformation, item.mechanism_id or ""))


@dataclass(frozen=True)
class HypothesisCandidate:
    """A bounded proposal; it has no promotion or implementation authority."""

    hypothesis_id: str
    origin: str
    category: str
    observation: str
    mechanism: str
    transformation: str
    preconditions: tuple[str, ...]
    expected_effect: tuple[str, ...]
    estimated_magnitude: Magnitude
    risks: tuple[str, ...]
    required_evidence: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    unknowns: tuple[str, ...]
    novelty: float
    score: float = 0.0
    asserted_facts: tuple[dict[str, Any], ...] = ()
    mechanism_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.origin not in {"RETRIEVED", "FIRST_PRINCIPLES", "COMPOSED", "EXPLORATORY"}:
            raise ValueError("unsupported hypothesis origin")
        if self.category not in {"exploit", "adjacent", "exploratory", "measurement"}:
            raise ValueError("unsupported hypothesis category")
        if not self.hypothesis_id or not self.observation or not self.mechanism or not self.transformation:
            raise ValueError("hypothesis identity and causal fields are required")
        if isinstance(self.novelty, bool) or not isinstance(self.novelty, (int, float)) or not 0.0 <= float(self.novelty) <= 1.0:
            raise ValueError("novelty must be in [0, 1]")
        for name in ("preconditions", "expected_effect", "risks", "required_evidence", "evidence_refs", "unknowns", "mechanism_ids"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or any(not isinstance(item, str) or not item for item in value):
                raise ValueError(f"{name} must contain non-empty strings")
        if not isinstance(self.asserted_facts, tuple) or any(not isinstance(item, dict) for item in self.asserted_facts):
            raise ValueError("asserted_facts must contain dictionaries")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["estimated_magnitude"] = self.estimated_magnitude.value
        return result


@dataclass(frozen=True)
class NoveltyPolicy:
    """Configurable budget; ratios are guidance, not an acceptance rule."""

    exploit_ratio: float = 0.50
    adjacent_ratio: float = 0.30
    exploratory_ratio: float = 0.20
    max_candidates: int = 12

    def __post_init__(self) -> None:
        ratios = (self.exploit_ratio, self.adjacent_ratio, self.exploratory_ratio)
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) or item < 0 for item in ratios):
            raise ValueError("novelty ratios must be nonnegative")
        if sum(ratios) <= 0 or self.max_candidates <= 0:
            raise ValueError("novelty policy must have a positive budget")


@dataclass(frozen=True)
class PlanningResult:
    raw_generation: tuple[HypothesisCandidate, ...]
    validated: tuple[HypothesisCandidate, ...]
    rejected: tuple[dict[str, Any], ...]
    ranked: tuple[HypothesisCandidate, ...]
    ablation_plans: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_generation": [item.to_dict() for item in self.raw_generation],
            "validated": [item.to_dict() for item in self.validated],
            "rejected": list(self.rejected),
            "ranked": [item.to_dict() for item in self.ranked],
            "ablation_plans": list(self.ablation_plans),
        }


def _field_value(facts: PerformanceFacts, path: str) -> tuple[bool, Any]:
    if not path.startswith("performance_facts."):
        return False, None
    value: Any = facts.to_dict()
    for part in path.split(".")[1:]:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return False, None
    return True, value


def validate_hypothesis_evidence(candidate: HypothesisCandidate, facts: PerformanceFacts) -> dict[str, Any]:
    """Reject unsupported measured claims; UNKNOWN cannot silently become fact."""
    known_refs = []
    for reference in candidate.evidence_refs:
        exists, value = _field_value(facts, reference)
        if not exists:
            return {"valid": False, "reason": "EVIDENCE_REFERENCE_NOT_FOUND", "reference": reference}
        known_refs.append({"reference": reference, "value": value})
    unknown_fields = set(facts.unknown_fields)
    for claim in candidate.asserted_facts:
        field = claim.get("field")
        if not isinstance(field, str) or not field:
            return {"valid": False, "reason": "MALFORMED_ASSERTED_FACT"}
        short_name = field.removeprefix("performance_facts.")
        if short_name in unknown_fields or field in unknown_fields:
            if claim.get("evidence_status") == "MEASURED" or claim.get("value") is not None:
                return {"valid": False, "reason": "REJECT_EVIDENCE_HALLUCINATION", "field": field}
        exists, actual = _field_value(facts, field)
        if not exists:
            return {"valid": False, "reason": "EVIDENCE_REFERENCE_NOT_FOUND", "field": field}
        if claim.get("evidence_status") == "MEASURED" and actual is None:
            return {"valid": False, "reason": "REJECT_EVIDENCE_HALLUCINATION", "field": field}
        if "value" in claim and claim.get("value") is not None and actual != claim.get("value"):
            return {"valid": False, "reason": "EVIDENCE_VALUE_MISMATCH", "field": field}
    return {"valid": True, "references": known_refs}


class GenerativeHypothesisPlanner(HypothesisPlanner):
    """Retrieval plus deterministic first-principles generation.

    An optional ``llm_generator`` may propose JSON-compatible candidates through
    an existing CodexAgentSession. Its output is advisory only and passes the
    same validation, deduplication, ranking, and budget controls.
    """

    def _first_principles(self, facts: PerformanceFacts, cf: CounterfactualAnswers) -> list[HypothesisCandidate]:
        result: list[HypothesisCandidate] = []
        reads = tuple(sorted({name for name in facts.global_memory_reads if facts.global_memory_reads.count(name) > 1}))
        if reads:
            result.append(HypothesisCandidate(
                hypothesis_id="fp-repeated-read-lifetime",
                origin="FIRST_PRINCIPLES", category="exploratory",
                observation="A value is consumed more than once across the recorded dataflow",
                mechanism="redundant memory traversal may be caused by a value not remaining live between uses",
                transformation="extend the value lifetime across the reuse boundary and remove the redundant read",
                preconditions=("the value is immutable between uses", "the live set fits the selected residency"),
                expected_effect=("fewer global memory reads", "lower memory traffic if the second read is eliminated"),
                estimated_magnitude=Magnitude.UNKNOWN,
                risks=("register pressure", "spill", "occupancy loss"),
                required_evidence=("L0 correctness", "boundary timing", "memory traffic measurement"),
                evidence_refs=("performance_facts.global_memory_reads", "performance_facts.tensor_lifetimes"),
                unknowns=("whether the value is immutable", "register/shared residency cost"),
                novelty=1.0,
            ))
        materialized = tuple(name for name in facts.global_memory_writes if name in facts.global_memory_reads)
        if materialized or facts.producer_consumer_boundaries:
            result.append(HypothesisCandidate(
                hypothesis_id="fp-producer-consumer-boundary",
                origin="FIRST_PRINCIPLES", category="exploratory",
                observation="A producer-consumer boundary is recorded around an intermediate tensor",
                mechanism="the intermediate may be materialized only to cross an artificial kernel boundary",
                transformation="fuse the compatible producer and consumer or remove the intermediate materialization",
                preconditions=("producer and consumer have compatible layouts", "fusion does not exceed working-set limits"),
                expected_effect=("remove an intermediate write/read round trip", "reduce launch overhead if a boundary disappears"),
                estimated_magnitude=Magnitude.UNKNOWN,
                risks=("register/shared pressure", "reduced scheduling flexibility", "numerical drift"),
                required_evidence=("L0 correctness", "CUDA Event boundary timing", "kernel launch or memory traffic measurement"),
                evidence_refs=("performance_facts.producer_consumer_boundaries", "performance_facts.global_memory_writes"),
                unknowns=("actual kernel count", "actual intermediate traffic", "layout compatibility"),
                novelty=1.0,
            ))
        if facts.kernel_launches is None or "kernel_launch_count" in facts.unknown_fields:
            result.append(HypothesisCandidate(
                hypothesis_id="measurement-kernel-boundary-cost",
                origin="EXPLORATORY", category="measurement",
                observation="Kernel launch count and boundary-level profile evidence are unknown",
                mechanism="the dominant cost cannot be distinguished between launch, memory, and compute without a boundary measurement",
                transformation="measure kernel launches, boundary latency, and memory traffic before selecting an optimization",
                preconditions=("a repeatable local workload exists",),
                expected_effect=("reduce uncertainty in bottleneck classification",),
                estimated_magnitude=Magnitude.UNKNOWN,
                risks=("measurement overhead",),
                required_evidence=("CUDA Event timing", "kernel launch trace", "memory traffic evidence"),
                evidence_refs=("performance_facts.unknown_fields", "performance_facts.profile_evidence"),
                unknowns=("kernel launch count", "register pressure", "memory traffic"),
                novelty=1.0,
            ))
        fraction = facts.e2e_profile.get("operator_fraction_of_step")
        if isinstance(fraction, (int, float)) and not isinstance(fraction, bool) and fraction < 0.05:
            result.append(HypothesisCandidate(
                hypothesis_id="measurement-low-e2e-ceiling",
                origin="FIRST_PRINCIPLES", category="measurement",
                observation="The operator occupies only a small fraction of the recorded step",
                mechanism="micro-optimization payoff is bounded by the system-level operator fraction",
                transformation="measure the operator and step again before allocating implementation search budget",
                preconditions=("the fraction is measured for the target workload",),
                expected_effect=("avoid low-value optimization search",),
                estimated_magnitude=Magnitude.UNKNOWN,
                risks=("premature deprioritization if workload changes",),
                required_evidence=("representative workload profile",),
                evidence_refs=("performance_facts.e2e_profile",),
                unknowns=("workload representativeness",),
                novelty=0.9,
            ))
        return result

    @staticmethod
    def _from_llm(raw: Iterable[dict[str, Any]]) -> list[HypothesisCandidate]:
        result = []
        for item in raw:
            result.append(HypothesisCandidate(
                hypothesis_id=str(item["hypothesis_id"]), origin=str(item.get("origin", "EXPLORATORY")),
                category=str(item.get("category", "exploratory")), observation=str(item["observation"]),
                mechanism=str(item["mechanism"]), transformation=str(item["transformation"]),
                preconditions=tuple(item.get("preconditions", [])), expected_effect=tuple(item.get("expected_effect", [])),
                estimated_magnitude=Magnitude(str(item.get("estimated_magnitude", "UNKNOWN"))),
                risks=tuple(item.get("risks", [])), required_evidence=tuple(item.get("required_evidence", [])),
                evidence_refs=tuple(item.get("evidence_refs", [])), unknowns=tuple(item.get("unknowns", [])),
                novelty=float(item.get("novelty", 1.0)), score=float(item.get("score", 0.0)),
                asserted_facts=tuple(item.get("asserted_facts", [])), mechanism_ids=tuple(item.get("mechanism_ids", [])),
            ))
        return result

    def plan(
        self, facts: PerformanceFacts, mechanisms: Iterable[MechanismRecord] = (), *,
        novelty_policy: NoveltyPolicy | None = None,
        llm_generator: Callable[[dict[str, Any]], Iterable[dict[str, Any]]] | None = None,
    ) -> PlanningResult:
        policy = novelty_policy or NoveltyPolicy()
        mechanisms = tuple(mechanisms)
        retrieved = []
        for opportunity in self.rank(facts, mechanisms):
            retrieved.append(HypothesisCandidate(
                hypothesis_id=f"retrieved-{opportunity.mechanism_id}", origin="RETRIEVED", category="exploit",
                observation=opportunity.observation, mechanism=opportunity.mechanism,
                transformation=opportunity.transformation, preconditions=opportunity.preconditions,
                expected_effect=opportunity.expected_effect, estimated_magnitude=opportunity.estimated_magnitude,
                risks=opportunity.risks, required_evidence=opportunity.required_evidence,
                evidence_refs=("performance_facts.profile_evidence",), unknowns=(), novelty=0.0,
                score=opportunity.score, mechanism_ids=(opportunity.mechanism_id,) if opportunity.mechanism_id else (),
            ))
        cf = self.counterfactuals(facts)
        generated = self._first_principles(facts, cf)
        if llm_generator is not None:
            prompt_input = {"performance_facts": facts.to_dict(), "counterfactuals": cf.to_dict(), "matched_mechanisms": [item.to_dict() for item in mechanisms], "planning_only": True}
            generated.extend(self._from_llm(llm_generator(prompt_input)))
        if retrieved and generated:
            base = retrieved[0]
            adjacent = generated[0]
            generated.append(HypothesisCandidate(
                hypothesis_id=f"composed-{base.hypothesis_id}-{adjacent.hypothesis_id}",
                origin="COMPOSED", category="adjacent",
                observation=f"{base.observation}; {adjacent.observation}",
                mechanism=f"{base.mechanism} interacts with {adjacent.mechanism}",
                transformation=f"{base.transformation} + {adjacent.transformation}",
                preconditions=tuple(dict.fromkeys(base.preconditions + adjacent.preconditions)),
                expected_effect=tuple(dict.fromkeys(base.expected_effect + adjacent.expected_effect)),
                estimated_magnitude=Magnitude.UNKNOWN,
                risks=tuple(dict.fromkeys(base.risks + adjacent.risks + ("interaction effect",))),
                required_evidence=tuple(dict.fromkeys(base.required_evidence + adjacent.required_evidence)),
                evidence_refs=tuple(dict.fromkeys(base.evidence_refs + adjacent.evidence_refs)),
                unknowns=tuple(dict.fromkeys(base.unknowns + adjacent.unknowns + ("causal contribution of each transformation",))),
                novelty=0.7, mechanism_ids=tuple(dict.fromkeys(base.mechanism_ids)),
            ))
        raw = retrieved + generated
        validated: list[HypothesisCandidate] = []
        rejected: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for candidate in raw:
            evidence = validate_hypothesis_evidence(candidate, facts)
            key = (candidate.observation, candidate.transformation)
            if not evidence["valid"]:
                rejected.append({"hypothesis_id": candidate.hypothesis_id, **evidence})
            elif key in seen:
                rejected.append({"hypothesis_id": candidate.hypothesis_id, "reason": "DUPLICATE_HYPOTHESIS"})
            else:
                seen.add(key)
                validated.append(candidate)
        ranked = []
        ceiling = cf.infinite_operator_e2e_ceiling
        for candidate in validated:
            score = candidate.score + candidate.novelty
            if candidate.category == "measurement":
                score += 0.5
            if ceiling is not None and ceiling <= 1.05:
                score -= 2.0
            score -= min(2.0, 0.25 * len(candidate.risks))
            ranked.append(HypothesisCandidate(**{**candidate.to_dict(), "estimated_magnitude": candidate.estimated_magnitude, "score": round(score, 3), "preconditions": tuple(candidate.preconditions), "expected_effect": tuple(candidate.expected_effect), "risks": tuple(candidate.risks), "required_evidence": tuple(candidate.required_evidence), "evidence_refs": tuple(candidate.evidence_refs), "unknowns": tuple(candidate.unknowns), "mechanism_ids": tuple(candidate.mechanism_ids), "asserted_facts": tuple(candidate.asserted_facts)}))
        ranked.sort(key=lambda item: (-item.score, item.category, item.hypothesis_id))
        ranked = ranked[:policy.max_candidates]
        ablations = []
        for candidate in ranked:
            parts = tuple(part.strip() for part in candidate.transformation.split(" + ") if part.strip())
            if len(parts) > 1:
                ablations.append({"hypothesis_id": candidate.hypothesis_id, "stages": ["baseline", *parts, "full"], "interaction_risk": "test each transformation independently before the combined change"})
        return PlanningResult(tuple(raw), tuple(validated), tuple(rejected), tuple(ranked), tuple(ablations))
