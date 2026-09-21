import unittest
from lab.core.decision import Decision, compute_confidence

class DecisionModelTests(unittest.TestCase):
    def test_decision_serialize_roundtrip(self):
        d = Decision(action="PROMOTE", reason="test", confidence=0.85,
            evidence_summary={"speedup": 1.05}, hypothesis_verdict="SUPPORTED",
            experiment_id="e0001-x001")
        dd = d.to_dict()
        self.assertEqual(dd["action"], "PROMOTE")
        self.assertEqual(dd["confidence"], 0.85)
        self.assertEqual(dd["evidence_summary"]["speedup"], 1.05)
        # Roundtrip
        d2 = Decision.from_dict(dd)
        self.assertEqual(d2.action, d.action)
        self.assertEqual(d2.confidence, d.confidence)
        self.assertEqual(d2.experiment_id, d.experiment_id)

    def test_promote_decision_structure(self):
        d = Decision(action="PROMOTE", reason="Clear improvement",
            confidence=0.90, hypothesis_verdict="SUPPORTED")
        self.assertEqual(d.action, "PROMOTE")
        self.assertGreater(d.confidence, 0.8)
        self.assertEqual(d.hypothesis_verdict, "SUPPORTED")
        self.assertTrue(len(d.decision_id) > 0)
        self.assertTrue(len(d.timestamp) > 0)

    def test_reject_correctness_structure(self):
        d = Decision(action="REJECT_CORRECTNESS", reason="Tests failed",
            hypothesis_verdict="REFUTED")
        self.assertEqual(d.action, "REJECT_CORRECTNESS")
        self.assertEqual(d.hypothesis_verdict, "REFUTED")
        self.assertLess(d.confidence, 0.5)

    def test_robustness_required_structure(self):
        d = Decision(action="ROBUSTNESS_REQUIRED",
            reason="Weak gain, need robustness",
            confidence=0.70, hypothesis_verdict="SUPPORTED")
        self.assertEqual(d.action, "ROBUSTNESS_REQUIRED")
        self.assertEqual(d.hypothesis_verdict, "SUPPORTED")

    def test_inconclusive_structure(self):
        d = Decision(action="INCONCLUSIVE", reason="No benchmark data")
        self.assertEqual(d.action, "INCONCLUSIVE")
        self.assertEqual(d.hypothesis_verdict, "INCONCLUSIVE")

    def test_string_equality_backward_compat(self):
        d = Decision(action="PROMOTE")
        self.assertTrue(d == "PROMOTE")
        self.assertFalse(d == "REJECT_COMPILE")
        # Also test via hash-based containers
        actions = {d}
        self.assertIn("PROMOTE", {x.action for x in actions})

    def test_compute_confidence_full(self):
        c = compute_confidence(compile_pass=True, correctness_pass=True,
            benchmark_available=True, robustness_pass=True, speedup=1.05)
        self.assertAlmostEqual(c, 1.0)

    def test_compute_confidence_no_benchmark(self):
        c = compute_confidence(compile_pass=True, correctness_pass=True)
        self.assertAlmostEqual(c, 0.55)

    def test_compute_confidence_compile_fail(self):
        c = compute_confidence(compile_pass=False)
        self.assertAlmostEqual(c, 0.0)

    def test_compute_confidence_reject_performance(self):
        c = compute_confidence(compile_pass=True, correctness_pass=True,
            benchmark_available=True, speedup=0.95)
        self.assertAlmostEqual(c, 0.90)

    def test_decision_save_and_load(self):
        import tempfile, json
        from pathlib import Path
        d = Decision(action="PROMOTE", reason="test save",
            experiment_id="e0001-x001", confidence=0.95)
        tmp = Path(tempfile.mkdtemp(prefix="aka-decision-"))
        try:
            saved = d.save(tmp / "decision.json")
            self.assertTrue(saved.exists())
            loaded = json.loads(saved.read_text(encoding="utf-8"))
            self.assertEqual(loaded["action"], "PROMOTE")
            self.assertEqual(loaded["experiment_id"], "e0001-x001")
        finally:
            import shutil; shutil.rmtree(tmp)
