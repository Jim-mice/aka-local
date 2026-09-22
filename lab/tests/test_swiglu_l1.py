import types
import unittest

from lab.runtime.evaluators.integration_oj import IntegrationOJ
from lab.runtime.integrations.swiglu_l1 import SwiGLUL1Harness, SwiGLUIntegrationResult, build_reasoner_snapshot, build_swiglu_oj_evidence


def swiglu(values, bias):
    result = []
    for row in values:
        shifted = [x + b for x, b in zip(row, bias)]
        half = len(shifted) // 2
        result.append([(a / (1.0 + abs(a))) * b for a, b in zip(shifted[:half], shifted[half:])])
    return result


class FakeTensor:
    shape = (2, 4)
    dtype = "float32"
    device = "cpu"
    requires_grad = True

    def __init__(self, values):
        self.values = values

    def stride(self):
        return (4, 1)


class SwiGLUL1Tests(unittest.TestCase):
    def test_hook_replacement_capture_and_restore(self):
        calls = {"baseline": 0}

        def baseline_impl(values, bias, *_args, **_kwargs):
            calls["baseline"] += 1
            return swiglu(values, bias)

        module = types.SimpleNamespace(bias_swiglu_impl=baseline_impl)
        input_tensor = FakeTensor([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0]])
        bias = [0.1, 0.2, 0.3, 0.4]
        baseline_output = module.bias_swiglu_impl(input_tensor.values, bias)
        with SwiGLUL1Harness(module) as harness:
            replacement = harness.install(swiglu)
            candidate_output = module.bias_swiglu_impl(input_tensor.values, bias)
            checks = harness.run_checks(
                lambda: baseline_output,
                lambda: candidate_output,
                baseline_input=input_tensor,
                candidate_input=input_tensor,
                baseline_grad=[[0.0] * 4],
                candidate_grad=[[0.0] * 4],
            )
            self.assertEqual(1, replacement.invocation_count)
            self.assertEqual([2, 2], checks["captured_shape"]["output"]["shape"])
            self.assertEqual(0.0, checks["forward_correctness"]["max_abs_error"])
            self.assertFalse(checks["fallback_detected"])
        self.assertIs(module.bias_swiglu_impl, baseline_impl)
        self.assertEqual(1, calls["baseline"])

    def test_l1_evidence_fails_when_backward_or_replacement_missing(self):
        result = SwiGLUIntegrationResult(
            source_commit="5be9626709af2722333bf54797c954c09edeada3",
            integration_contract_hash="a" * 64,
            candidate_hash="b" * 64,
            captured_shape=None,
            forward_correctness={"status": "RUNTIME_BLOCKED"},
            backward_correctness={"status": "BACKWARD_RUNTIME_BLOCKED"},
            replacement_invocations=0,
            baseline_target_invocations=0,
            fallback_detected=True,
            runtime_environment={"distributed_invariants": False},
            timing_if_available=None,
            status="INTEGRATION_BLOCKED",
        )
        evidence = build_swiglu_oj_evidence(result)
        judged = IntegrationOJ(lambda _: evidence).evaluate({
            "expected_replacement_marker": "aka-local-swiglu-reference-v1",
            "required_metrics": ["module_ms"],
            "expected_contract_hash": "a" * 64,
            "expected_candidate_hash": "b" * 64,
        })
        self.assertEqual("INTEGRATION_FAIL", judged.verdict)
        self.assertIn("failed:replacement_invoked", judged.reasons)

    def test_complete_fake_evidence_passes_existing_l1_oj(self):
        result = SwiGLUIntegrationResult(
            source_commit="5be9626709af2722333bf54797c954c09edeada3",
            integration_contract_hash="a" * 64,
            candidate_hash="b" * 64,
            captured_shape={"input": {"shape": [2, 4]}, "output": {"shape": [2, 2]}},
            forward_correctness={"shape_equal": True, "finite": True, "max_rel_error": 0.0},
            backward_correctness={"status": "CHECKED", "shape_equal": True, "finite": True, "max_abs_error": 0.0, "max_rel_error": 0.0},
            replacement_invocations=1,
            baseline_target_invocations=1,
            fallback_detected=False,
            runtime_environment={"distributed_invariants": True},
            timing_if_available={"module_ms": 1.0},
            status="INTEGRATION_PASS",
        )
        judged = IntegrationOJ(lambda _: build_swiglu_oj_evidence(result)).evaluate({
            "expected_replacement_marker": "aka-local-swiglu-reference-v1",
            "required_metrics": ["module_ms"],
            "expected_contract_hash": "a" * 64,
            "expected_candidate_hash": "b" * 64,
        })
        self.assertEqual("INTEGRATION_PASS", judged.verdict)

    def test_real_shape_metadata_enters_existing_reasoner(self):
        snapshot = build_reasoner_snapshot({"input": {"shape": [8, 2, 16], "dtype": "torch.float32"}})
        self.assertEqual("swiglu", snapshot["performance_facts"]["operator"])
        self.assertEqual({"dim_0": 8, "dim_1": 2, "dim_2": 16}, snapshot["performance_facts"]["shape"])
        self.assertIsInstance(snapshot["top_opportunities"], list)


if __name__ == "__main__":
    unittest.main()
