import unittest
from lab.runtime.supervisor.controller_policy import decide, finalize_after_robustness
from lab.core.decision import Decision

class SupervisorPolicyTests(unittest.TestCase):
    def base(self, speedup=1.0):
        return {"compile":{"pass":True},"correctness":{"pass":True},"authoritative_abba":{"pass":True,"metrics":{"arithmetic_mean_speedup":speedup}}}

    def test_compile_and_correctness_reject(self):
        d = decide({"compile":{"pass":False}})
        self.assertIsInstance(d, Decision)
        self.assertEqual(d.action, "REJECT_COMPILE")
        self.assertEqual(d, "REJECT_COMPILE")  # backward compat

        x = self.base(); x["correctness"] = {"pass": False}
        d = decide(x)
        self.assertEqual(d.action, "REJECT_CORRECTNESS")

    def test_performance_weak_and_clear(self):
        d = decide(self.base(1.0))
        self.assertEqual(d.action, "REJECT_PERFORMANCE")
        self.assertLess(d.confidence, 1.0)

        d = decide(self.base(1.01))
        self.assertEqual(d.action, "ROBUSTNESS_REQUIRED")

        d = decide(self.base(1.03))
        self.assertEqual(d.action, "PROMOTE")
        self.assertEqual(d.hypothesis_verdict, "SUPPORTED")

    def test_robustness_finalization(self):
        d = finalize_after_robustness({"pass":True,"metrics":{"aggregate":{"arithmetic_mean_speedup":1.01}}})
        self.assertIsInstance(d, Decision)
        self.assertEqual(d.action, "PROMOTE")

        d = finalize_after_robustness({"pass":False})
        self.assertEqual(d.action, "REJECT_ROBUSTNESS")

    def test_decision_serialize(self):
        d = Decision(action="PROMOTE", reason="test", confidence=0.85,
            evidence_summary={"speedup": 1.05}, hypothesis_verdict="SUPPORTED",
            experiment_id="e0001-x001")
        dd = d.to_dict()
        self.assertEqual(dd["action"], "PROMOTE")
        self.assertEqual(dd["confidence"], 0.85)

        d2 = Decision.from_dict(dd)
        self.assertEqual(d2.action, d.action)
        self.assertEqual(d2.confidence, d.confidence)

    def test_old_decision_compatibility(self):
        # String equality still works
        d = Decision(action="PROMOTE")
        self.assertTrue(d == "PROMOTE")
        self.assertFalse(d == "REJECT_COMPILE")
        # Action access still works
        self.assertEqual(d.action, "PROMOTE")

    def test_inconclusive_decision(self):
        d = decide({"compile":{"pass":True},"correctness":{"pass":True},"authoritative_abba":{"metrics":{}}})
        self.assertEqual(d.action, "INCONCLUSIVE")
        self.assertEqual(d.hypothesis_verdict, "INCONCLUSIVE")
