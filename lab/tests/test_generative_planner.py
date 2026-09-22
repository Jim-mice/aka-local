import unittest

from lab.runtime.reasoning.agent_context import build_planning_context, validate_planning_context, augment_planning_context
from lab.runtime.reasoning.hypothesis_planner import (
    GenerativeHypothesisPlanner,
    HypothesisCandidate,
    Magnitude,
    NoveltyPolicy,
    validate_hypothesis_evidence,
)
from lab.runtime.reasoning.mechanism_memory import EPISODE28_MECHANISM
from lab.runtime.reasoning.performance_model import PerformanceFacts


def facts(**overrides):
    data = dict(
        operator="test_operator", shape={"rows": 8, "width": 16}, dtype="float32",
        dataflow=("producer", "consumer"), tensor_lifetimes={"x": "between_uses"},
        global_memory_reads=("x", "x"), global_memory_writes=("intermediate",),
        producer_consumer_boundaries=("producer->consumer",),
        profile_evidence={"idle_resources": [], "supports_mechanism_ids": [], "contradicts_mechanism_ids": []},
        unknown_fields=("kernel_launch_count", "memory_traffic"),
        e2e_profile={"operator_fraction_of_step": 0.2},
    )
    data.update(overrides)
    return PerformanceFacts(**data)


class GenerativePlannerTests(unittest.TestCase):
    def test_repeated_immutable_observation_generates_without_memory(self):
        result = GenerativeHypothesisPlanner().plan(facts())
        self.assertTrue(result.ranked)
        self.assertTrue(any(item.origin in {"FIRST_PRINCIPLES", "EXPLORATORY"} for item in result.ranked))
        self.assertTrue(all(item.evidence_refs for item in result.ranked))
        self.assertTrue(any(item.estimated_magnitude is Magnitude.UNKNOWN for item in result.ranked))

    def test_producer_consumer_boundary_generates_structured_candidate(self):
        result = GenerativeHypothesisPlanner().plan(facts(global_memory_reads=("a",), global_memory_writes=("b",)))
        self.assertTrue(any(item.transformation and item.preconditions and item.required_evidence for item in result.ranked))

    def test_low_e2e_fraction_penalizes_priority(self):
        high = GenerativeHypothesisPlanner().plan(facts(e2e_profile={"operator_fraction_of_step": 0.2}))
        low = GenerativeHypothesisPlanner().plan(facts(e2e_profile={"operator_fraction_of_step": 0.002}))
        self.assertLess(max(item.score for item in low.ranked), max(item.score for item in high.ranked))

    def test_unknown_profile_creates_measurement_candidate(self):
        result = GenerativeHypothesisPlanner().plan(facts())
        measurement = [item for item in result.ranked if item.category == "measurement"]
        self.assertTrue(measurement)
        self.assertTrue(all("kernel" in " ".join(item.unknowns).lower() or "traffic" in " ".join(item.unknowns).lower() for item in measurement))

    def test_retrieval_and_composition_coexist_without_duplicate(self):
        rms = facts(operator="rms_norm", global_memory_reads=("x", "x"))
        result = GenerativeHypothesisPlanner().plan(rms, [EPISODE28_MECHANISM])
        self.assertTrue(any(item.origin == "RETRIEVED" for item in result.ranked))
        self.assertTrue(any(item.origin == "COMPOSED" for item in result.ranked))
        keys = [(item.observation, item.transformation) for item in result.ranked]
        self.assertEqual(len(keys), len(set(keys)))

    def test_episode_28_blind_reconstruction_shape(self):
        blind = facts(operator="rms_norm", global_memory_reads=("x", "x"), global_memory_writes=(), producer_consumer_boundaries=(), unknown_fields=())
        result = GenerativeHypothesisPlanner().plan(blind)
        candidates = [item for item in result.ranked if item.origin in {"FIRST_PRINCIPLES", "EXPLORATORY"}]
        self.assertTrue(candidates)
        self.assertTrue(any(len(item.evidence_refs) >= 1 and len(item.risks) >= 1 for item in candidates))

    def test_hallucinated_unknown_measured_claim_is_rejected(self):
        candidate = HypothesisCandidate(
            hypothesis_id="hallucination", origin="EXPLORATORY", category="exploratory",
            observation="unknown profile", mechanism="unproven mechanism", transformation="measure first",
            preconditions=("repeatable run",), expected_effect=("reduce uncertainty",), estimated_magnitude=Magnitude.UNKNOWN,
            risks=("measurement cost",), required_evidence=("profile",), evidence_refs=("performance_facts.unknown_fields",),
            unknowns=("kernel_launch_count",), novelty=1.0,
            asserted_facts=({"field": "kernel_launch_count", "value": 3, "evidence_status": "MEASURED"},),
        )
        verdict = validate_hypothesis_evidence(candidate, facts())
        self.assertFalse(verdict["valid"])
        self.assertEqual("REJECT_EVIDENCE_HALLUCINATION", verdict["reason"])

    def test_planning_only_context_wiring(self):
        source_facts = facts()
        result = GenerativeHypothesisPlanner().plan(source_facts)
        payload = build_planning_context(source_facts, result, [])
        validate_planning_context(payload)
        context = augment_planning_context({"context_hash": "old", "snapshot": "local"}, payload)
        self.assertEqual("PLANNING_ONLY", context["performance_planning"]["mode"])
        self.assertTrue(context["context_hash"] != "old")

    def test_configurable_novelty_budget_is_not_a_constant(self):
        policy = NoveltyPolicy(exploit_ratio=0.2, adjacent_ratio=0.2, exploratory_ratio=0.6, max_candidates=2)
        result = GenerativeHypothesisPlanner().plan(facts(), novelty_policy=policy)
        self.assertLessEqual(len(result.ranked), 2)

    def test_advisory_llm_candidates_use_same_supervisor(self):
        def advisory(_payload):
            return [{
                "hypothesis_id": "llm-advisory", "origin": "EXPLORATORY", "category": "exploratory",
                "observation": "the recorded boundary may hide an avoidable materialization",
                "mechanism": "producer-consumer materialization cost is unmeasured",
                "transformation": "measure the boundary before changing implementation",
                "preconditions": ["repeatable workload"], "expected_effect": ["reduce uncertainty"],
                "estimated_magnitude": "UNKNOWN", "risks": ["measurement cost"],
                "required_evidence": ["boundary timing"],
                "evidence_refs": ["performance_facts.producer_consumer_boundaries"],
                "unknowns": ["kernel count"], "novelty": 1.0,
            }]
        result = GenerativeHypothesisPlanner().plan(facts(), llm_generator=advisory)
        self.assertTrue(any(item.hypothesis_id == "llm-advisory" for item in result.validated))


if __name__ == "__main__":
    unittest.main()
