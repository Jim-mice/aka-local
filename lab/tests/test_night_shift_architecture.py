from __future__ import annotations

import hashlib
import copy
import json
import sys
import types
import unittest
from dataclasses import replace
from pathlib import Path

from lab.runtime.blueprints import DENSE_ATTENTION_BLUEPRINT, SWIGLU_BLUEPRINT, MockBlueprintHarness, strict_json_loads, validate_blueprint_artifact_manifest, validate_blueprint_result_bundle, validate_integration_contract
from lab.runtime.evaluators.end_to_end_oj import EndToEndOJ, EndToEndPolicy, QualificationEvidence, amdahl_upper_bound
from lab.runtime.evaluators.integration_oj import IntegrationOJ
from lab.runtime.evaluators.oj_models import JudgeResult, PromotionState, PromotionStatus, legacy_standalone_promotion
from lab.runtime.evaluators.operator_oj import OperatorOJ
from lab.runtime.reasoning.ablation_planner import ExperimentPlanner
from lab.runtime.reasoning.agent_context import augment_authoritative_context, build_performance_context
from lab.runtime.reasoning.hypothesis_planner import HypothesisPlanner
from lab.runtime.reasoning.mechanism_memory import EPISODE28_MECHANISM, EvidenceStatus, MechanismRecord, MechanismStore
from lab.runtime.reasoning.performance_model import Magnitude, PerformanceFacts


ROOT = Path(__file__).resolve().parents[2]


class FiveOperatorSchemaTests(unittest.TestCase):
    def test_all_five_contracts_have_required_sections(self):
        directory = ROOT / "targets" / "megatron_5be9626" / "integration_contracts"
        schema = json.loads((directory / "schema.json").read_text(encoding="utf-8"))
        contracts = [json.loads(path.read_text(encoding="utf-8")) for path in directory.glob("*.json") if path.name != "schema.json"]
        self.assertEqual(5, len(contracts))
        self.assertEqual(5, len({item["operator_id"] for item in contracts}))
        for contract in contracts:
            self.assertEqual((), validate_integration_contract(schema, contract, target_commit="5be9626709af2722333bf54797c954c09edeada3"))
            self.assertTrue(set(schema["required"]).issubset(contract))
            self.assertEqual("5be9626709af2722333bf54797c954c09edeada3", contract["megatron"]["commit"])
            self.assertIn("fallback_detection", contract["replacement_boundary"])
            self.assertTrue({"convergence_proxy_delta", "loss_delta", "gradient_check"}.issubset(contract["performance"]["e2e_metrics"]))
            for section, required_fields in schema["section_required"].items():
                self.assertTrue(set(required_fields).issubset(contract[section]), f"{contract['operator_id']}:{section}")

        malformed = copy.deepcopy(contracts[0])
        malformed["megatron"]["commit"] = "unverified"
        malformed["correctness"]["gradient_checks"] = []
        malformed["typo_section"] = {}
        failures = validate_integration_contract(schema, malformed, target_commit="5be9626709af2722333bf54797c954c09edeada3")
        self.assertIn("megatron.commit", failures)
        self.assertIn("correctness.gradient_checks", failures)
        self.assertIn("unknown:typo_section", failures)

    def test_source_specific_contract_distinctions_are_explicit(self):
        directory = ROOT / "targets" / "megatron_5be9626" / "integration_contracts"
        residual = json.loads((directory / "residual_add_rmsnorm.json").read_text(encoding="utf-8"))
        self.assertIn("TENorm", residual["megatron"]["classes"])
        self.assertNotIn("build_norm", residual["megatron"]["functions"])
        self.assertIn("not proof", residual["replacement_boundary"]["replace"])
        ce = json.loads((directory / "vocab_parallel_cross_entropy.json").read_text(encoding="utf-8"))
        self.assertIn("MAX,SUM,SUM", ce["distributed_context"]["tensor_parallel"])
        self.assertIn("MAX,SUM", ce["distributed_context"]["tensor_parallel"])
        swiglu = json.loads((directory / "swiglu.json").read_text(encoding="utf-8"))
        self.assertIn("per_token_scale dispatch", swiglu["replacement_boundary"]["must_remain_untouched"])

    def test_source_audit_covers_every_contract_file_and_operator(self):
        directory = ROOT / "targets" / "megatron_5be9626" / "integration_contracts"
        contracts = [json.loads(path.read_text(encoding="utf-8")) for path in directory.glob("*.json") if path.name != "schema.json"]
        audit = json.loads((directory.parent / "source_audit.json").read_text(encoding="utf-8"))
        self.assertEqual("5be9626709af2722333bf54797c954c09edeada3", audit["target_commit"])
        expected_files = {source for contract in contracts for source in contract["megatron"]["source_files"]}
        self.assertEqual(expected_files, set(audit["source_files"]))
        self.assertTrue(all(len(digest) == 64 and set(digest) <= set("0123456789abcdef") for digest in audit["source_files"].values()))
        self.assertEqual({contract["operator_id"] for contract in contracts}, set(audit["operator_anchors"]))
        self.assertTrue(all(audit["operator_anchors"][contract["operator_id"]] for contract in contracts))

    def test_import_guard_rejects_preloaded_foreign_module_without_fixture_io(self):
        from scripts.run_lab import assert_lab_is_local
        probe = types.ModuleType("lab.foreign_night_shift_probe")
        probe.__file__ = r"C:\foreign-checkout\lab\probe.py"
        sys.modules[probe.__name__] = probe
        try:
            with self.assertRaisesRegex(RuntimeError, "REFUSING_TO_RUN"):
                assert_lab_is_local()
        finally:
            sys.modules.pop(probe.__name__, None)


class ThreeLevelOJTests(unittest.TestCase):
    contract_hash = "a" * 64
    candidate_hash = "b" * 64
    qualification = QualificationEvidence(
        protocol_hash="b" * 64,
        baseline_run_ids=("baseline-1", "baseline-2"),
        candidate_run_ids=("candidate-1", "candidate-2"),
        stable=True,
        comparable=True,
        artifact_refs=("l2/raw_metrics.json",),
    )

    @staticmethod
    def passing_runner(_request):
        return {"checks": {name: True for name in ("replacement_invoked", "no_silent_fallback", "forward_correct", "backward_correct", "shape_compatible", "distributed_invariants")}, "replacement_marker": "swiglu-v1", "fallback_count": 0, "provenance": {"runner_id": "mock-l1", "artifact_refs": ["l1/raw.json"], "contract_hash": ThreeLevelOJTests.contract_hash, "candidate_hash": ThreeLevelOJTests.candidate_hash}, "metrics": {"module_ms": 2.0}}

    def test_integration_requires_real_replacement_marker(self):
        result = IntegrationOJ(self.passing_runner).evaluate({"expected_replacement_marker": "swiglu-v1", "required_metrics": ["module_ms"], "expected_contract_hash": self.contract_hash, "expected_candidate_hash": self.candidate_hash})
        self.assertEqual("INTEGRATION_PASS", result.verdict)
        wrong_identity = IntegrationOJ(self.passing_runner).evaluate({"expected_replacement_marker": "swiglu-v1", "required_metrics": ["module_ms"], "expected_contract_hash": self.contract_hash, "expected_candidate_hash": "c" * 64})
        self.assertEqual("INTEGRATION_FAIL", wrong_identity.verdict)
        self.assertIn("failed:candidate_hash_matches", wrong_identity.reasons)
        bad = IntegrationOJ(lambda _: {"checks": {name: True for name in ("replacement_invoked", "no_silent_fallback", "forward_correct", "backward_correct", "shape_compatible", "distributed_invariants")}, "replacement_marker": "fallback"}).evaluate({"expected_replacement_marker": "swiglu-v1"})
        self.assertEqual("INTEGRATION_FAIL", bad.verdict)
        undeclared = IntegrationOJ(self.passing_runner).evaluate({})
        self.assertEqual("INTEGRATION_FAIL", undeclared.verdict)
        self.assertIn("failed:replacement_marker_matches", undeclared.reasons)
        self.assertIn("failed:required_metrics_schema", undeclared.reasons)

    def test_integration_checks_collective_order_and_required_metrics(self):
        expected = ["MAX:logits_max", "SUM:predicted_logits", "SUM:sum_exp_logits"]
        runner = lambda _: {"checks": {name: True for name in ("replacement_invoked", "no_silent_fallback", "forward_correct", "backward_correct", "shape_compatible", "distributed_invariants")}, "replacement_marker": "ce-v1", "fallback_count": 0, "provenance": {"runner_id": "mock-l1", "artifact_refs": ["l1/raw.json"], "contract_hash": self.contract_hash, "candidate_hash": self.candidate_hash}, "collective_trace": expected, "collective_traces_by_rank": {"0": expected, "1": expected}, "metrics": {"module_ms": 3.0, "peak_memory_bytes": 1024}}
        request = {"expected_replacement_marker": "ce-v1", "expected_collective_trace": expected, "expected_world_size": 2, "required_metrics": ["module_ms", "peak_memory_bytes"], "expected_contract_hash": self.contract_hash, "expected_candidate_hash": self.candidate_hash}
        result = IntegrationOJ(runner).evaluate(request)
        self.assertEqual("INTEGRATION_PASS", result.verdict)
        reordered = IntegrationOJ(lambda request: {**runner(request), "collective_trace": list(reversed(expected))}).evaluate({"expected_replacement_marker": "ce-v1", "expected_collective_trace": expected})
        self.assertEqual("INTEGRATION_FAIL", reordered.verdict)
        invalid_metric = IntegrationOJ(lambda request: {**runner(request), "metrics": {"module_ms": float("nan")}}).evaluate({"expected_replacement_marker": "ce-v1", "required_metrics": ["module_ms"]})
        self.assertEqual("INTEGRATION_FAIL", invalid_metric.verdict)
        rank_mismatch = IntegrationOJ(lambda inner: {**runner(inner), "collective_traces_by_rank": {"0": expected, "1": list(reversed(expected))}}).evaluate(request)
        self.assertEqual("INTEGRATION_FAIL", rank_mismatch.verdict)
        self.assertIn("failed:collective_traces_all_ranks", rank_mismatch.reasons)
        moe_trace = ["ALL_TO_ALL:dispatch", "ALL_TO_ALL:combine"]
        moe_runner = lambda _: {**self.passing_runner({}), "replacement_marker": "moe-v1", "collective_trace": moe_trace, "collective_traces_by_rank": {str(rank): moe_trace for rank in range(4)}}
        moe_result = IntegrationOJ(moe_runner).evaluate({"expected_replacement_marker": "moe-v1", "expected_collective_trace": moe_trace, "expected_world_size": 4, "required_metrics": ["module_ms"], "expected_contract_hash": self.contract_hash, "expected_candidate_hash": self.candidate_hash})
        self.assertEqual("INTEGRATION_PASS", moe_result.verdict)

    def test_operator_oj_requires_all_gates_and_latency(self):
        hashes = {"contract_hash": "a" * 64, "candidate_hash": "b" * 64}
        result = OperatorOJ(lambda _: {"compile_pass": True, "contract_pass": True, "correctness_pass": True, "shapes_pass": True, "stability_pass": True, "latency_ms": 1.5, "samples_ms": [1.5, 1.49], **hashes}).evaluate({})
        self.assertEqual("OPERATOR_PASS", result.verdict)
        missing_latency = OperatorOJ(lambda _: {"compile_pass": True, "contract_pass": True, "correctness_pass": True, "shapes_pass": True, "stability_pass": True, **hashes}).evaluate({})
        self.assertEqual("OPERATOR_FAIL", missing_latency.verdict)
        nonfinite_latency = OperatorOJ(lambda _: {"compile_pass": True, "contract_pass": True, "correctness_pass": True, "shapes_pass": True, "stability_pass": True, "latency_ms": float("nan"), "samples_ms": [1.0], **hashes}).evaluate({})
        self.assertEqual("OPERATOR_FAIL", nonfinite_latency.verdict)
        missing_hashes = OperatorOJ(lambda _: {"compile_pass": True, "contract_pass": True, "correctness_pass": True, "shapes_pass": True, "stability_pass": True, "latency_ms": 1.0, "samples_ms": [1.0]}).evaluate({})
        self.assertEqual("OPERATOR_FAIL", missing_hashes.verdict)

    def test_e2e_policy_preserves_raw_metrics_and_qualification(self):
        loaded = EndToEndPolicy.from_json(ROOT / "config" / "policies" / "end_to_end_promotion.json")
        policy = EndToEndPolicy(**{**loaded.__dict__, "max_abs_loss_delta": 0.01})
        metrics = {"baseline_iteration_ms": 100, "candidate_iteration_ms": 90, "baseline_tokens_per_sec": 1000, "candidate_tokens_per_sec": 1100, "baseline_samples_per_sec": 10, "candidate_samples_per_sec": 11, "baseline_peak_memory_bytes": 10000, "candidate_peak_memory_bytes": 9500, "convergence_proxy_delta": 0.002, "loss_delta": 0.001, "gradient_check": True}
        provisional = EndToEndOJ(policy).evaluate(metrics)
        self.assertEqual("SYSTEM_PROVISIONAL", provisional.verdict)
        accepted = EndToEndOJ(policy).evaluate(metrics, qualification=self.qualification)
        self.assertEqual("SYSTEM_QUALIFIED_ACCEPT", accepted.verdict)
        self.assertEqual(metrics, accepted.raw_metrics["input"])
        self.assertAlmostEqual(0.1, accepted.raw_metrics["derived"]["samples_per_sec_gain"])
        incomplete = EndToEndOJ(policy).evaluate({"baseline_iteration_ms": 100, "candidate_iteration_ms": 90, "baseline_tokens_per_sec": 1000, "candidate_tokens_per_sec": 1100, "gradient_check": True}, qualification=self.qualification)
        self.assertEqual("SYSTEM_PROVISIONAL", incomplete.verdict)

    def test_e2e_policy_rejects_malformed_or_nonfinite_evidence(self):
        policy = EndToEndPolicy(max_abs_loss_delta=0.01, max_abs_convergence_proxy_delta=0.01)
        valid = {"baseline_iteration_ms": 100, "candidate_iteration_ms": 90, "baseline_tokens_per_sec": 1000, "candidate_tokens_per_sec": 1100, "baseline_samples_per_sec": 10, "candidate_samples_per_sec": 11, "baseline_peak_memory_bytes": 10000, "candidate_peak_memory_bytes": 9500, "convergence_proxy_delta": 0.002, "loss_delta": 0.001, "gradient_check": True}
        malformed_gradient = EndToEndOJ(policy).evaluate({**valid, "gradient_check": "true"}, qualification=self.qualification)
        self.assertEqual("SYSTEM_REJECT", malformed_gradient.verdict)
        nonfinite_rate = EndToEndOJ(policy).evaluate({**valid, "candidate_tokens_per_sec": float("nan")}, qualification=self.qualification)
        self.assertEqual("SYSTEM_REJECT", nonfinite_rate.verdict)
        malformed_loss = EndToEndOJ(policy).evaluate({**valid, "loss_delta": "unknown"}, qualification=self.qualification)
        self.assertEqual("SYSTEM_REJECT", malformed_loss.verdict)
        malformed_convergence = EndToEndOJ(policy).evaluate({**valid, "convergence_proxy_delta": float("inf")}, qualification=self.qualification)
        self.assertEqual("SYSTEM_REJECT", malformed_convergence.verdict)
        inconsistent_memory = EndToEndOJ(policy).evaluate({**valid, "memory_delta_bytes": -999999}, qualification=self.qualification)
        self.assertEqual("SYSTEM_REJECT", inconsistent_memory.verdict)
        self.assertIs(inconsistent_memory.checks["memory_delta_consistent"], False)
        malformed_memory = EndToEndOJ(policy).evaluate({**valid, "memory_delta_bytes": "lower"}, qualification=self.qualification)
        self.assertEqual("SYSTEM_REJECT", malformed_memory.verdict)
        self.assertIs(malformed_memory.checks["memory_delta_input"], False)
        bad_fraction = EndToEndOJ(policy).evaluate({**valid, "operator_fraction_of_step": 1.1}, qualification=self.qualification)
        self.assertEqual("SYSTEM_REJECT", bad_fraction.verdict)
        unbounded = EndToEndOJ(policy).evaluate({**valid, "operator_fraction_of_step": 1.0}, qualification=self.qualification)
        self.assertEqual("SYSTEM_QUALIFIED_ACCEPT", unbounded.verdict)
        self.assertIsNone(unbounded.raw_metrics["derived"]["max_possible_e2e_speedup"])
        self.assertIs(unbounded.raw_metrics["derived"]["max_possible_e2e_speedup_unbounded"], True)
        json.dumps(unbounded.to_dict(), allow_nan=False)
        nonstandard_extra = EndToEndOJ(policy).evaluate({**valid, "unused": float("nan")}, qualification=self.qualification)
        self.assertEqual("SYSTEM_REJECT", nonstandard_extra.verdict)
        self.assertIs(nonstandard_extra.checks["input_json"], False)
        malformed_qualification = QualificationEvidence("", ("same",), ("same",), True, True)
        rejected_qualification = EndToEndOJ(policy).evaluate(valid, qualification=malformed_qualification)
        self.assertEqual("SYSTEM_REJECT", rejected_qualification.verdict)
        unknown_qualification = EndToEndOJ(policy).evaluate(valid, qualification={"protocol_hash": "b" * 64, "invented": True})
        self.assertEqual("SYSTEM_REJECT", unknown_qualification.verdict)
        self.assertIn("unknown qualification fields", unknown_qualification.evidence["qualification_error"])
        with self.assertRaisesRegex(ValueError, "unknown.*field"):
            EndToEndPolicy.from_dict({"schema_version": 1, "min_iteration_speeedup": 1.1})
        with self.assertRaisesRegex(ValueError, "booleans"):
            EndToEndPolicy(require_gradient_check="yes")  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "dictionary"):
            EndToEndPolicy.from_dict([])  # type: ignore[arg-type]

    def test_promotion_axes_are_separate(self):
        state = PromotionState()
        state.set_kernel(PromotionStatus.ACCEPTED)
        self.assertEqual("NOT_EVALUATED", state.to_dict()["SystemPromotion"])
        with self.assertRaises(ValueError):
            state.set_system(PromotionStatus.ACCEPTED)
        with self.assertRaises(ValueError):
            state.set_system(PromotionStatus.PROVISIONAL)
        state.set_integration(PromotionStatus.ACCEPTED)
        state.set_system(PromotionStatus.ACCEPTED)
        state.set_kernel(PromotionStatus.REJECTED)
        self.assertEqual({"KernelPromotion": "REJECTED", "IntegrationPromotion": "NOT_EVALUATED", "SystemPromotion": "NOT_EVALUATED"}, state.to_dict())
        fresh = PromotionState()
        with self.assertRaises(ValueError):
            fresh.set_integration(PromotionStatus.REJECTED)
        applied = PromotionState()
        applied.apply_result(JudgeResult("L0_OPERATOR", "OPERATOR_PASS", {"compile": True}))
        applied.apply_result(JudgeResult("L1_INTEGRATION", "INTEGRATION_PASS", {"replacement": True}))
        applied.apply_result(JudgeResult("L2_END_TO_END", "SYSTEM_PROVISIONAL", {"qualification": None}))
        self.assertEqual({"KernelPromotion": "ACCEPTED", "IntegrationPromotion": "ACCEPTED", "SystemPromotion": "PROVISIONAL"}, applied.to_dict())
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            applied.apply_result(JudgeResult("L0_OPERATOR", "SYSTEM_REJECT", {"failed": True}))
        legacy = legacy_standalone_promotion("QUALIFIED_ACCEPT")
        self.assertEqual({"KernelPromotion": "ACCEPTED", "IntegrationPromotion": "NOT_EVALUATED", "SystemPromotion": "NOT_EVALUATED"}, legacy.to_dict())
        with self.assertRaisesRegex(ValueError, "unknown legacy"):
            legacy_standalone_promotion("SYSTEM_QUALIFIED_ACCEPT")

    def test_amdahl_ceiling(self):
        self.assertAlmostEqual(1.25, amdahl_upper_bound(0.2))


class ReasonerTests(unittest.TestCase):
    def facts(self):
        return PerformanceFacts(
            operator="rms_norm", shape={"rows": 256, "hidden": 4096}, dtype="FP16",
            dataflow=("read x", "reduce", "read x", "write y"),
            global_memory_reads=("x", "x", "weight"), global_memory_writes=("partial", "y"),
            register_residency=("partial",), estimated_bytes=5_000_000, estimated_flops=2_000_000,
            reductions=("sum squares",), synchronizations=("block reduction",), kernel_launches=2,
            producer_consumer_boundaries=("reduction -> normalize",), known_reuse=("x",),
            fixed_dimensions={"hidden": 4096}, dynamic_dimensions=("hidden",),
            parallel_mapping={"row": "block"}, profile_evidence={"idle_resources": ["tensor cores"], "estimated_bytes_saved": 2_000_000, "working_set_feasible": True, "supports_mechanism_ids": ["immutable_reduction_operand_residency"]},
            unknown_fields=("registers_per_thread",), e2e_profile={"operator_fraction_of_step": 0.1},
        )

    def test_performance_facts_and_counterfactuals(self):
        answers = HypothesisPlanner().counterfactuals(self.facts())
        self.assertEqual(("x",), answers.data_read_twice)
        self.assertIn("hidden", answers.fixed_dimensions_treated_dynamically)
        self.assertAlmostEqual(1.0 / 0.9, answers.infinite_operator_e2e_ceiling)
        with self.assertRaisesRegex(ValueError, "estimated_bytes"):
            PerformanceFacts(operator="swiglu", shape={"tokens": 8}, dtype="BF16", estimated_bytes=float("nan"))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "kernel_launches"):
            PerformanceFacts(operator="swiglu", shape={"tokens": 8}, dtype="BF16", kernel_launches=True)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "shape"):
            PerformanceFacts(operator="swiglu", shape={"tokens": 0}, dtype="BF16")
        with self.assertRaisesRegex(ValueError, "shape"):
            PerformanceFacts(operator="swiglu", shape={}, dtype="BF16")
        with self.assertRaisesRegex(ValueError, "tensor_lifetimes"):
            PerformanceFacts(operator="swiglu", shape={"tokens": 8}, dtype="BF16", tensor_lifetimes={"x": 1})  # type: ignore[dict-item]
        with self.assertRaisesRegex(ValueError, "parallel_mapping"):
            PerformanceFacts(operator="swiglu", shape={"tokens": 8}, dtype="BF16", parallel_mapping={"token": ""})
        with self.assertRaisesRegex(ValueError, "supports_mechanism_ids"):
            PerformanceFacts(operator="swiglu", shape={"tokens": 8}, dtype="BF16", profile_evidence={"supports_mechanism_ids": "not-a-list"})

    def test_mechanism_status_is_causally_honest(self):
        self.assertEqual(EvidenceStatus.CAUSAL_UNPROVEN, EPISODE28_MECHANISM.evidence_status)
        self.assertIn("vectorization", EPISODE28_MECHANISM.provenance["confounders"])
        found = MechanismStore(ROOT / "knowledge" / "mechanisms.jsonl").query("redundant memory pass", operator="rms_norm")
        self.assertEqual("immutable_reduction_operand_residency", found[0].mechanism_id)
        with self.assertRaisesRegex(ValueError, "mechanism_id"):
            MechanismRecord(
                mechanism_id="", pattern="pattern", symptoms=(), mechanism="mechanism",
                transformation="transform", preconditions=(), expected_effects=(), risks=(),
            )
        with self.assertRaisesRegex(ValueError, "EvidenceStatus"):
            MechanismRecord(
                mechanism_id="id", pattern="pattern", symptoms=(), mechanism="mechanism",
                transformation="transform", preconditions=(), expected_effects=(), risks=(),
                evidence_status="CAUSAL_PROVEN",  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "strict-JSON"):
            MechanismRecord(
                mechanism_id="id", pattern="pattern", symptoms=(), mechanism="mechanism",
                transformation="transform", preconditions=(), expected_effects=(), risks=(),
                measured_evidence=({"latency_us": float("nan")},),
            )

    def test_ranking_and_agent_context_keep_authority(self):
        from lab.runtime.agent.codex_session import CodexAgentSession

        ranked = HypothesisPlanner().rank(self.facts(), [EPISODE28_MECHANISM])
        self.assertTrue(ranked)
        self.assertIn(ranked[0].estimated_magnitude, set(Magnitude))
        self.assertEqual(2_000_000, ranked[0].ranking_factors["estimated_bytes_saved"])
        self.assertEqual("SUPPORTED", ranked[0].ranking_factors["working_set_feasibility"])
        self.assertEqual(2.0, ranked[0].ranking_factors["profile_evidence_score"])
        self.assertAlmostEqual(1.0 / 0.9, ranked[0].ranking_factors["infinite_operator_e2e_ceiling"])
        alternative = MechanismRecord(
            mechanism_id="vector_width_change", pattern="scalar aligned loads", symptoms=("low transaction width",),
            mechanism="narrow loads", transformation="vectorize aligned loads", preconditions=("alignment",),
            expected_effects=("fewer transactions",), risks=("tail handling",),
        )
        scoped_facts = replace(self.facts(), profile_evidence={
            "mechanism_estimates": {
                EPISODE28_MECHANISM.mechanism_id: {"estimated_bytes_saved": 2_000_000, "working_set_feasible": True},
                alternative.mechanism_id: {"working_set_feasible": False},
            },
        })
        scoped = {item.mechanism_id: item for item in HypothesisPlanner().rank(scoped_facts, [EPISODE28_MECHANISM, alternative])}
        self.assertEqual(2_000_000, scoped[EPISODE28_MECHANISM.mechanism_id].ranking_factors["estimated_bytes_saved"])
        self.assertNotIn("estimated_bytes_saved", scoped[alternative.mechanism_id].ranking_factors)
        self.assertEqual("CONTRADICTED", scoped[alternative.mechanism_id].ranking_factors["working_set_feasibility"])
        unbounded_facts = replace(self.facts(), e2e_profile={"operator_fraction_of_step": 1.0})
        unbounded_ranked = HypothesisPlanner().rank(unbounded_facts, [EPISODE28_MECHANISM])
        self.assertEqual("UNBOUNDED", unbounded_ranked[0].ranking_factors["infinite_operator_e2e_ceiling"])
        build_performance_context(unbounded_facts, unbounded_ranked, [EPISODE28_MECHANISM])
        self.assertEqual("UNBOUNDED", HypothesisPlanner().counterfactuals(unbounded_facts).to_dict()["infinite_operator_e2e_ceiling"])
        payload = build_performance_context(self.facts(), ranked, [EPISODE28_MECHANISM])
        self.assertEqual("controller-owned and not delegated to Agent", payload["authority"]["promotion"])
        augmented = augment_authoritative_context({"campaign": "unit", "context_hash": "old"}, payload)
        self.assertNotEqual("old", augmented["context_hash"])
        self.assertIn("performance_reasoning", augmented)
        prepared = CodexAgentSession.prepare_context({"campaign": "unit", "context_hash": "old"}, payload)
        self.assertEqual(augmented, prepared)
        authority_injection = copy.deepcopy(payload)
        authority_injection["authority"]["promotion"] = "agent-owned"
        with self.assertRaisesRegex(ValueError, "authority boundary"):
            augment_authoritative_context({"campaign": "unit"}, authority_injection)
        instruction_injection = copy.deepcopy(payload)
        instruction_injection["implementation_instruction"] = "Agent decides promotion"
        with self.assertRaisesRegex(ValueError, "instruction mismatch"):
            augment_authoritative_context({"campaign": "unit"}, instruction_injection)
        unknown_field = copy.deepcopy(payload)
        unknown_field["agent_override"] = True
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            augment_authoritative_context({"campaign": "unit"}, unknown_field)
        unbound_opportunity = copy.deepcopy(payload)
        unbound_opportunity["relevant_mechanisms"] = []
        with self.assertRaisesRegex(ValueError, "bind"):
            augment_authoritative_context({"campaign": "unit"}, unbound_opportunity)
        nonfinite_context = copy.deepcopy(payload)
        nonfinite_context["performance_facts"]["profile_evidence"]["unscoped_value"] = float("nan")
        with self.assertRaisesRegex(TypeError, "JSON serializable"):
            augment_authoritative_context({"campaign": "unit"}, nonfinite_context)
        with self.assertRaisesRegex(ValueError, "top_k"):
            build_performance_context(self.facts(), ranked, [EPISODE28_MECHANISM], top_k=-1)
        with self.assertRaisesRegex(TypeError, "serializable"):
            augment_authoritative_context({"bad": object()}, payload)


class BlueprintAndAblationTests(unittest.TestCase):
    @staticmethod
    def artifact_manifest(blueprint):
        return {
            "operator_id": blueprint.operator_id,
            "target_commit": blueprint.target_commit,
            "candidate_hash": "b" * 64,
            "artifacts": {role: {"path": path, "sha256": "a" * 64} for role, path in blueprint.artifact_layout.items()},
            "promotion_state": {name: "NOT_EVALUATED" for name in blueprint.promotion_states},
            "replacement_marker": f"{blueprint.operator_id}-v1",
            "fallback_count": 0,
        }

    def test_swiglu_mock_harness(self):
        runner = ThreeLevelOJTests.passing_runner
        harness = MockBlueprintHarness(runner)
        self.assertEqual("INTEGRATION_PASS", harness.run_l1({"expected_replacement_marker": "swiglu-v1", "required_metrics": ["module_ms"], "expected_contract_hash": ThreeLevelOJTests.contract_hash, "expected_candidate_hash": ThreeLevelOJTests.candidate_hash}).verdict)
        self.assertEqual("swiglu", SWIGLU_BLUEPRINT.operator_id)
        self.assertEqual("EXTERNAL_INTEGRATION_REQUIRED", SWIGLU_BLUEPRINT.external_status)
        template = json.loads((ROOT / "config" / "protocols" / "swiglu_end_to_end_template.json").read_text(encoding="utf-8"))
        policy = EndToEndPolicy.from_json(ROOT / "config" / "policies" / "end_to_end_promotion.json")
        self.assertEqual(SWIGLU_BLUEPRINT.target_commit, template["target_commit"])
        self.assertEqual("EXTERNAL_INTEGRATION_REQUIRED", template["status"])
        self.assertIsNone(template["training_command"])
        self.assertEqual(list(policy.required_metrics), template["measurement"]["required_metrics"])
        self.assertEqual(policy.min_qualification_runs, template["measurement"]["repeats"])
        self.assertEqual([], template["qualification"]["artifact_refs"])
        self.assertIsNone(template["replacement_evidence"]["contract_hash"])
        self.assertIsNone(template["replacement_evidence"]["candidate_hash"])
        empty_metrics = {name: None for name in policy.required_metrics}
        self.assertNotEqual("SYSTEM_QUALIFIED_ACCEPT", EndToEndOJ(policy).evaluate(empty_metrics, qualification=template["qualification"]).verdict)

    def test_dense_attention_stretch_blueprint(self):
        self.assertEqual("dense_fused_attention", DENSE_ATTENTION_BLUEPRINT.operator_id)
        self.assertIn("fallback", DENSE_ATTENTION_BLUEPRINT.fallback_detection)
        self.assertIn("samples_per_sec", DENSE_ATTENTION_BLUEPRINT.metrics)
        template = json.loads((ROOT / "config" / "protocols" / "dense_fused_attention_end_to_end_template.json").read_text(encoding="utf-8"))
        policy = EndToEndPolicy.from_json(ROOT / "config" / "policies" / "end_to_end_promotion.json")
        self.assertEqual(DENSE_ATTENTION_BLUEPRINT.target_commit, template["target_commit"])
        self.assertIsNone(template["training_command"])
        self.assertEqual(list(policy.required_metrics), template["measurement"]["required_metrics"])

    def test_blueprint_artifact_manifest_is_fail_closed(self):
        manifest = self.artifact_manifest(SWIGLU_BLUEPRINT)
        self.assertEqual((), validate_blueprint_artifact_manifest(SWIGLU_BLUEPRINT, manifest))
        self.assertEqual(("manifest_type",), validate_blueprint_artifact_manifest(SWIGLU_BLUEPRINT, []))  # type: ignore[arg-type]
        manifest["fallback_count"] = 1
        manifest["artifacts"]["l2"]["sha256"] = "invented"
        manifest["artifacts"]["unexpected"] = {"path": "extra.json", "sha256": "a" * 64}
        manifest["promotion_state"]["SystemPromotion"] = "ACCEPTED"
        failures = validate_blueprint_artifact_manifest(SWIGLU_BLUEPRINT, manifest)
        self.assertIn("fallback_count", failures)
        self.assertIn("artifact_sha256:l2", failures)
        self.assertIn("artifact_roles", failures)
        self.assertIn("promotion_state_order", failures)

    def test_bundle_json_decoder_rejects_nonstandard_and_ambiguous_input(self):
        self.assertEqual({"value": 1}, strict_json_loads(b'{"value": 1}'))
        for payload in ('{"value": NaN}', '{"value": Infinity}', '{"value": 1, "value": 2}'):
            with self.assertRaises(ValueError):
                strict_json_loads(payload)
        with self.assertRaises(TypeError):
            strict_json_loads({"value": 1})  # type: ignore[arg-type]

    def test_blueprint_result_bundle_binds_digests_verdicts_and_promotions(self):
        bundle_root = ROOT / "lab" / "tests" / "fixtures" / "night_shift_bundle"
        manifest = self.artifact_manifest(SWIGLU_BLUEPRINT)
        manifest["promotion_state"] = {name: "ACCEPTED" for name in SWIGLU_BLUEPRINT.promotion_states}
        for role, relative_path in SWIGLU_BLUEPRINT.artifact_layout.items():
            payload = (bundle_root / relative_path).read_bytes()
            manifest["artifacts"][role]["sha256"] = hashlib.sha256(payload).hexdigest()

        self.assertEqual((), validate_blueprint_result_bundle(SWIGLU_BLUEPRINT, bundle_root, manifest))

        manifest["candidate_hash"] = "c" * 64
        identity_failures = validate_blueprint_result_bundle(SWIGLU_BLUEPRINT, bundle_root, manifest)
        self.assertIn("candidate_identity:l0", identity_failures)
        self.assertIn("candidate_identity:l1", identity_failures)
        self.assertIn("provenance_artifact", identity_failures)
        manifest["candidate_hash"] = "b" * 64

        invalid_document = bundle_root / "l2" / "invalid_result.json"
        for role, expected_failure in (("contract", "contract_artifact_identity"), ("shapes", "shape_manifest"), ("provenance", "provenance_artifact")):
            original = dict(manifest["artifacts"][role])
            manifest["artifacts"][role] = {"path": "l2/invalid_result.json", "sha256": hashlib.sha256(invalid_document.read_bytes()).hexdigest()}
            self.assertIn(expected_failure, validate_blueprint_result_bundle(SWIGLU_BLUEPRINT, bundle_root, manifest))
            manifest["artifacts"][role] = original

        provenance_digest = manifest["artifacts"]["provenance"]["sha256"]
        manifest["artifacts"]["provenance"]["sha256"] = "0" * 64
        self.assertIn("qualification_protocol_binding:l2", validate_blueprint_result_bundle(SWIGLU_BLUEPRINT, bundle_root, manifest))
        manifest["artifacts"]["provenance"]["sha256"] = provenance_digest

        manifest["promotion_state"]["SystemPromotion"] = "PROVISIONAL"
        self.assertIn("promotion_result_mismatch:l2", validate_blueprint_result_bundle(SWIGLU_BLUEPRINT, bundle_root, manifest))
        manifest["promotion_state"]["SystemPromotion"] = "ACCEPTED"

        manifest["artifacts"]["l2"]["sha256"] = "0" * 64
        self.assertIn("artifact_digest_mismatch:l2", validate_blueprint_result_bundle(SWIGLU_BLUEPRINT, bundle_root, manifest))

        invalid_l2 = bundle_root / "l2" / "invalid_result.json"
        manifest["artifacts"]["l2"] = {"path": "l2/invalid_result.json", "sha256": hashlib.sha256(invalid_l2.read_bytes()).hexdigest()}
        failures = validate_blueprint_result_bundle(SWIGLU_BLUEPRINT, bundle_root, manifest)
        self.assertIn("qualified_result_provenance:l2", failures)
        self.assertIn("result_checks_missing:l2", failures)
        self.assertIn("result_policy_metrics:l2", failures)

        inconsistent = bundle_root / "l2" / "inconsistent_provisional.json"
        manifest["artifacts"]["l2"] = {"path": "l2/inconsistent_provisional.json", "sha256": hashlib.sha256(inconsistent.read_bytes()).hexdigest()}
        failures = validate_blueprint_result_bundle(SWIGLU_BLUEPRINT, bundle_root, manifest)
        self.assertIn("provisional_result_checks:l2", failures)
        self.assertIn("nonaccepted_result_reasons:l2", failures)

        valid_l2 = bundle_root / SWIGLU_BLUEPRINT.artifact_layout["l2"]
        manifest["artifacts"]["l2"] = {"path": SWIGLU_BLUEPRINT.artifact_layout["l2"], "sha256": hashlib.sha256(valid_l2.read_bytes()).hexdigest()}
        invalid_raw = bundle_root / "l2" / "invalid_raw_metrics.json"
        manifest["artifacts"]["l2_raw"] = {"path": "l2/invalid_raw_metrics.json", "sha256": hashlib.sha256(invalid_raw.read_bytes()).hexdigest()}
        self.assertIn("qualification_run_binding:l2", validate_blueprint_result_bundle(SWIGLU_BLUEPRINT, bundle_root, manifest))

    def test_checked_in_bundle_manifest_is_valid(self):
        bundle_root = ROOT / "lab" / "tests" / "fixtures" / "night_shift_bundle"
        manifest = json.loads((bundle_root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual((), validate_blueprint_result_bundle(SWIGLU_BLUEPRINT, bundle_root, manifest))

    def test_ablation_plan_has_main_and_marginal_effects(self):
        plan = ExperimentPlanner().plan(["register_cache", "float4", "unroll", "specialization"])
        enabled = {item.enabled for item in plan}
        self.assertIn(("register_cache",), enabled)
        self.assertIn(("register_cache", "float4", "unroll", "specialization"), enabled)
        self.assertIn(("float4", "unroll", "specialization"), enabled)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            ExperimentPlanner().plan(["register_cache", ""])


if __name__ == "__main__":
    unittest.main()
