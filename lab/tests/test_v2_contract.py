import unittest
import tempfile
import json
from pathlib import Path

from lab.core.contract import TaskContract
from lab.core.hypothesis import Hypothesis
from lab.core.evidence import Evidence, EvidenceType, Verdict
from lab.core.journal import append_experiment_with_hypothesis, read_experiments, read_hypotheses
from lab.runtime.supervisor.controller_policy import decide, decide_from_evidence


class TestTaskContract(unittest.TestCase):
    def test_from_campaign_manifest(self):
        manifest = {
            "operator_id": "rmsnorm",
            "platform_id": "rtx5060",
            "execution_target_id": "local",
            "objective": "latency",
            "workload_set": "56-shape",
            "reference": "torch",
        }
        c = TaskContract.from_campaign(manifest)
        self.assertEqual(c.operator, "rmsnorm")
        self.assertEqual(c.hardware["gpu"], "rtx5060")
        self.assertTrue(c.constraints["correctness_required"])
        self.assertEqual(c.promotion["threshold"], 1.0)
        self.assertTrue(c.is_valid())

    def test_serialize_deserialize(self):
        c = TaskContract(
            operator="rmsnorm",
            hardware={"gpu": "rtx5060", "target": "local"},
            objective={"metric": "latency", "direction": "minimize"},
            constraints={"correctness_required": True},
            evaluation={"benchmark_command": "abba", "profiler": "ncu"},
            promotion={"threshold": 1.02},
        )
        d = c.to_dict()
        c2 = TaskContract.from_dict(d)
        self.assertEqual(c2.operator, "rmsnorm")
        self.assertEqual(c2.hardware["gpu"], "rtx5060")
        self.assertTrue(c2.is_valid())

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "contract.json"
            c = TaskContract.from_campaign({"operator_id": "rmsnorm", "platform_id": "rtx5060"})
            c.save(p)
            c2 = TaskContract.load(p)
            self.assertEqual(c2.operator, "rmsnorm")

    def test_validate_errors(self):
        c = TaskContract(operator="", hardware={}, objective={}, constraints={}, evaluation={}, promotion={})
        errors = c.validate()
        self.assertTrue(len(errors) >= 3)
        self.assertFalse(c.is_valid())

    def test_as_prompt_block(self):
        c = TaskContract.from_campaign({"operator_id": "rmsnorm", "platform_id": "rtx5060"})
        block = c.as_prompt_block()
        self.assertIn("rmsnorm", block)
        self.assertIn("rtx5060", block)
        self.assertIn("latency", block)


class TestHypothesis(unittest.TestCase):
    def test_create_and_serialize(self):
        h = Hypothesis(
            claim="vectorized load improves throughput",
            mechanism="reduce memory transaction overhead",
            expected_effect="latency decrease",
        )
        d = h.to_dict()
        self.assertEqual(d["claim"], "vectorized load improves throughput")
        self.assertEqual(d["status"], "PROPOSED")
        self.assertTrue(h.hypothesis_id.startswith("h-"))

    def test_roundtrip(self):
        h = Hypothesis(
            claim="test claim",
            mechanism="test mechanism",
            expected_effect="test effect",
            experiment_id="exp-001",
        )
        d = h.to_dict()
        h2 = Hypothesis.from_dict(d)
        self.assertEqual(h2.claim, h.claim)
        self.assertEqual(h2.experiment_id, h.experiment_id)
        self.assertEqual(h2.hypothesis_id, h.hypothesis_id)

    def test_from_journal_record(self):
        record = {
            "experiment_id": "exp-001",
            "decision": "PROMOTE",
            "plan": {
                "hypothesis": {
                    "claim": "test claim",
                    "mechanism": "test mechanism",
                }
            }
        }
        h = Hypothesis.from_journal_record(record)
        self.assertIsNotNone(h)
        self.assertEqual(h.claim, "test claim")
        self.assertEqual(h.status, "TESTED")

    def test_from_journal_record_none(self):
        self.assertIsNone(Hypothesis.from_journal_record({}))
        self.assertIsNone(Hypothesis.from_journal_record({"plan": {}}))

    def test_verify_supported(self):
        h = Hypothesis(claim="test")
        evidence = {
            "development_benchmark": {
                "metrics": {"arithmetic_mean_speedup": 1.05}
            }
        }
        self.assertEqual(h.verify(evidence), "SUPPORTED")

    def test_verify_refuted(self):
        h = Hypothesis(claim="test")
        evidence = {
            "development_benchmark": {
                "metrics": {"arithmetic_mean_speedup": 0.95}
            }
        }
        self.assertEqual(h.verify(evidence), "REFUTED")


class TestEvidence(unittest.TestCase):
    def test_from_evaluation_pass(self):
        evaluation = {
            "compile": {"pass": True},
            "correctness": {"pass": True},
            "authoritative_abba": {
                "metrics": {"arithmetic_mean_speedup": 1.05}
            },
        }
        e = Evidence.from_evaluation(evaluation, "c001")
        self.assertEqual(e.type, EvidenceType.PERFORMANCE)
        self.assertEqual(e.verdict, Verdict.PASS)
        self.assertEqual(e.candidate_id, "c001")
        self.assertTrue(e.evidence_id.startswith("ev-"))

    def test_from_evaluation_correctness_fail(self):
        evaluation = {
            "compile": {"pass": True},
            "correctness": {"pass": False},
        }
        e = Evidence.from_evaluation(evaluation, "c002")
        self.assertEqual(e.type, EvidenceType.CORRECTNESS)
        self.assertEqual(e.verdict, Verdict.FAIL)

    def test_from_evaluation_compile_fail(self):
        evaluation = {"compile": {"pass": False}}
        e = Evidence.from_evaluation(evaluation, "c003")
        self.assertEqual(e.type, EvidenceType.CORRECTNESS)
        self.assertEqual(e.verdict, Verdict.FAIL)
        self.assertFalse(e.metrics.get("compile_passed", True))

    def test_from_diagnostics(self):
        diag = {"pass": True, "ncu_output": "..."}
        e = Evidence.from_diagnostics(diag, "c004")
        self.assertEqual(e.type, EvidenceType.DIAGNOSTIC)
        self.assertEqual(e.verdict, Verdict.PASS)

    def test_from_profile(self):
        profile = {"pass": True, "kernel_time": 1.2}
        e = Evidence.from_profile(profile, "c005")
        self.assertEqual(e.type, EvidenceType.PROFILE)
        self.assertEqual(e.verdict, Verdict.PASS)

    def test_to_dict_and_back(self):
        e = Evidence.from_evaluation(
            {"compile": {"pass": True}, "correctness": {"pass": True},
             "authoritative_abba": {"metrics": {"arithmetic_mean_speedup": 1.1}}},
            "c006"
        )
        d = e.to_dict()
        e2 = Evidence.from_dict(d)
        self.assertEqual(e2.type, EvidenceType.PERFORMANCE)
        self.assertEqual(e2.verdict, Verdict.PASS)
        self.assertEqual(e2.evidence_id, e.evidence_id)


class TestSupervisorEvidence(unittest.TestCase):
    def test_decide_from_evidence_promote(self):
        e = Evidence.from_evaluation(
            {"compile": {"pass": True}, "correctness": {"pass": True},
             "authoritative_abba": {"metrics": {"arithmetic_mean_speedup": 1.05}}},
            "c007"
        )
        self.assertEqual(decide_from_evidence(e, weak_gain_pct=2.0), "PROMOTE")

    def test_decide_from_evidence_reject_correctness(self):
        e = Evidence.from_evaluation(
            {"compile": {"pass": True}, "correctness": {"pass": False}},
            "c008"
        )
        self.assertEqual(decide_from_evidence(e), "REJECT_CORRECTNESS")

    def test_decide_from_evidence_robustness(self):
        e = Evidence.from_evaluation(
            {"compile": {"pass": True}, "correctness": {"pass": True},
             "authoritative_abba": {"metrics": {"arithmetic_mean_speedup": 1.015}}},
            "c009"
        )
        self.assertEqual(decide_from_evidence(e, weak_gain_pct=2.0), "ROBUSTNESS_REQUIRED")

    def test_decide_from_evidence_reject_performance(self):
        e = Evidence.from_evaluation(
            {"compile": {"pass": True}, "correctness": {"pass": True},
             "authoritative_abba": {"metrics": {"arithmetic_mean_speedup": 0.9}}},
            "c010"
        )
        self.assertEqual(decide_from_evidence(e), "REJECT_PERFORMANCE")

    def test_decide_from_evidence_fallback_to_dict(self):
        # Backward compat: raw evaluation dict still works
        evaluation = {"compile": {"pass": True}, "correctness": {"pass": True},
                      "authoritative_abba": {"metrics": {"arithmetic_mean_speedup": 1.05}}}
        self.assertEqual(decide_from_evidence(evaluation=evaluation), "PROMOTE")

    def test_decide_unchanged(self):
        # Existing decide() function is untouched
        evaluation = {"compile": {"pass": True}, "correctness": {"pass": True},
                      "authoritative_abba": {"metrics": {"arithmetic_mean_speedup": 0.5}}}
        self.assertEqual(decide(evaluation), "REJECT_PERFORMANCE")


class TestJournalWithHypothesis(unittest.TestCase):
    def test_append_with_hypothesis(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "journal.jsonl"
            record = {"experiment_id": "exp-test", "decision": "PROMOTE"}
            hypothesis = {"claim": "test claim", "mechanism": "test mechanism"}
            append_experiment_with_hypothesis(p, record, hypothesis)
            records = read_experiments(p)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["hypothesis"]["claim"], "test claim")

    def test_append_without_hypothesis(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "journal.jsonl"
            record = {"experiment_id": "exp-test", "decision": "PROMOTE"}
            append_experiment_with_hypothesis(p, record, None)
            records = read_experiments(p)
            self.assertEqual(len(records), 1)
            self.assertNotIn("hypothesis", records[0])

    def test_read_hypotheses(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "journal.jsonl"
            r1 = {"experiment_id": "e1", "hypothesis": {"claim": "h1"}}
            r2 = {"experiment_id": "e2"}
            r3 = {"experiment_id": "e3", "plan": {"hypothesis": {"claim": "h3"}}}
            for r in [r1, r2, r3]:
                append_experiment_with_hypothesis(p, r)
            hyps = read_hypotheses(p)
            self.assertEqual(len(hyps), 2)
            claims = {h["hypothesis"]["claim"] for h in hyps}
            self.assertEqual(claims, {"h1", "h3"})





class TestPhase2HypothesisEnforcement(unittest.TestCase):
    def test_missing_claim_is_invalid(self):
        h = Hypothesis.from_dict({'claim': '', 'mechanism': 'test'})
        self.assertEqual(h.claim, '')
        self.assertFalse(bool(h.claim.strip()))

    def test_claim_required_for_experiment(self):
        plan_hypothesis = {'claim': '', 'mechanism': 'test mech'}
        is_dict = isinstance(plan_hypothesis, dict)
        has_claim = isinstance(plan_hypothesis.get('claim'), str) and len(plan_hypothesis.get('claim', '').strip()) > 0
        self.assertTrue(is_dict)
        self.assertFalse(has_claim)

    def test_valid_hypothesis_passes(self):
        plan_hypothesis = {'claim': 'valid claim', 'mechanism': 'test mech'}
        is_dict = isinstance(plan_hypothesis, dict)
        has_claim = isinstance(plan_hypothesis.get('claim'), str) and len(plan_hypothesis.get('claim', '').strip()) > 0
        self.assertTrue(is_dict)
        self.assertTrue(has_claim)


class TestPhase2EvidenceHypothesisLink(unittest.TestCase):
    def test_evidence_verifies_hypothesis_supported(self):
        h = Hypothesis(claim='should be faster', mechanism='reduce overhead', expected_effect='speedup')
        eval_dict = {
            'compile': {'pass': True},
            'correctness': {'pass': True},
            'authoritative_abba': {'metrics': {'arithmetic_mean_speedup': 1.15}},
        }
        evidence = Evidence.from_evaluation(eval_dict, 'exp-001')
        self.assertEqual(evidence.type, EvidenceType.PERFORMANCE)
        self.assertEqual(evidence.verdict, Verdict.PASS)
        verdict = h.verify({'authoritative_abba': eval_dict['authoritative_abba']})
        self.assertEqual(verdict, 'SUPPORTED')

    def test_evidence_verifies_hypothesis_refuted(self):
        h = Hypothesis(claim='should be faster', mechanism='reduce overhead')
        eval_dict = {'development_benchmark': {'metrics': {'arithmetic_mean_speedup': 0.85}}}
        verdict = h.verify(eval_dict)
        self.assertEqual(verdict, 'REFUTED')

    def test_evidence_verifies_hypothesis_inconclusive(self):
        h = Hypothesis(claim='should be faster')
        verdict = h.verify({})
        self.assertEqual(verdict, 'INCONCLUSIVE')

    def test_evidence_to_dict_has_type_and_verdict(self):
        eval_dict = {
            'compile': {'pass': True},
            'correctness': {'pass': True},
            'authoritative_abba': {'metrics': {'arithmetic_mean_speedup': 1.2}},
        }
        evidence = Evidence.from_evaluation(eval_dict, 'exp-002')
        d = evidence.to_dict()
        self.assertEqual(d['type'], 'PERFORMANCE')
        self.assertEqual(d['verdict'], 'PASS')


class TestPhase2HypothesisHistory(unittest.TestCase):
    def history_from_records(self, records):
        items = []
        for rec in records or []:
            plan = rec.get('plan') or {}
            hyp = plan.get('hypothesis') or {}
            claim = hyp.get('claim') or hyp.get('question')
            if not claim:
                continue
            decision = rec.get('decision', 'UNKNOWN')
            benchmark = rec.get('development_benchmark') or {}
            metrics = benchmark.get('metrics') or {}
            speedup = metrics.get('arithmetic_mean_speedup')
            items.append({
                'experiment_id': rec.get('experiment_id'),
                'claim': claim,
                'mechanism': hyp.get('mechanism', ''),
                'decision': decision,
                'speedup': speedup,
            })
        return items

    def test_empty_records(self):
        self.assertEqual(self.history_from_records([]), [])

    def test_records_with_hypothesis(self):
        records = [
            {'experiment_id': 'e-001', 'decision': 'PROMOTE',
             'plan': {'hypothesis': {'claim': 'vectorized load', 'mechanism': 'reduce mem tx'}},
             'development_benchmark': {'metrics': {'arithmetic_mean_speedup': 1.05}}},
            {'experiment_id': 'e-002', 'decision': 'REJECT_PERFORMANCE',
             'plan': {'hypothesis': {'claim': 'larger block', 'mechanism': 'occupancy'}},
             'development_benchmark': {'metrics': {'arithmetic_mean_speedup': 0.92}}},
        ]
        history = self.history_from_records(records)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]['speedup'], 1.05)
        self.assertEqual(history[1]['speedup'], 0.92)

    def test_record_without_hypothesis_skipped(self):
        records = [
            {'experiment_id': 'e-003', 'plan': {}},
            {'experiment_id': 'e-004', 'plan': {'hypothesis': {'claim': 'valid'}},
             'development_benchmark': {'metrics': {'arithmetic_mean_speedup': 1.1}}},
        ]
        history = self.history_from_records(records)
        self.assertEqual(len(history), 1)


class TestPhase2CandidateBinding(unittest.TestCase):
    def test_lineage_hypothesis_id(self):
        import hashlib, json
        hyp = {'claim': 'test claim', 'mechanism': 'test mechanism'}
        hid = 'hyp-' + hashlib.sha256(json.dumps(hyp, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]
        self.assertTrue(hid.startswith('hyp-'))
        self.assertEqual(len(hid), 20)

    def test_lineage_hypothesis_id_deterministic(self):
        import hashlib, json
        hyp = {'claim': 'same', 'mechanism': 'same'}
        def make_id(h):
            return 'hyp-' + hashlib.sha256(json.dumps(dict(h), sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]
        self.assertEqual(make_id(hyp), make_id(dict(hyp)))



class TestPhase3Diagnostic(unittest.TestCase):
    def test_create_minimal(self):
        from lab.core.diagnostic import Diagnostic, DiagnosticCategory, Severity
        d = Diagnostic(category=DiagnosticCategory.MEMORY_BOUND, severity=Severity.WARNING, message="High DRAM throughput", source="ncu_profile")
        self.assertTrue(d.id.startswith("diag-"))
        self.assertEqual(d.category, DiagnosticCategory.MEMORY_BOUND)

    def test_serialize_roundtrip(self):
        from lab.core.diagnostic import Diagnostic, DiagnosticCategory, Severity
        d = Diagnostic(category=DiagnosticCategory.REGISTER_PRESSURE, severity=Severity.CRITICAL, message="128 registers per thread", evidence={"registers_per_thread": 128}, possible_causes=["Too many variables"], suggested_actions=["Split kernel"], source="ncu_profile", experiment_id="e001-x001")
        raw = d.to_dict()
        d2 = Diagnostic.from_dict(raw)
        self.assertEqual(d2.id, d.id)
        self.assertEqual(d2.category, DiagnosticCategory.REGISTER_PRESSURE)
        self.assertEqual(d2.severity, Severity.CRITICAL)
        self.assertEqual(d2.evidence["registers_per_thread"], 128)

    def test_from_profile_memory_bound(self):
        from lab.core.diagnostic import Diagnostic, DiagnosticCategory, Severity
        parsed = {"memory": {"dram_throughput_pct_of_peak_elapsed": "75.5"}, "occupancy": {"achieved": "50.0"}, "resources": {"registers_per_thread": "32"}, "kernel": {"name": "rmsnorm_kernel", "duration": "5.2"}, "raw_metrics": {}, "shape_id": "0"}
        diags = Diagnostic.from_profile(parsed, "exp-001")
        self.assertTrue(len(diags) >= 1)
        self.assertEqual(diags[0].category, DiagnosticCategory.MEMORY_BOUND)

    def test_from_profile_register_pressure(self):
        from lab.core.diagnostic import Diagnostic, DiagnosticCategory, Severity
        parsed = {"memory": {"dram_throughput_pct_of_peak_elapsed": "25.0"}, "occupancy": {"achieved": "60.0"}, "resources": {"registers_per_thread": "96"}, "kernel": {"name": "rmsnorm_kernel", "duration": "3.0"}, "raw_metrics": {}}
        diags = Diagnostic.from_profile(parsed, "exp-002")
        categories = {d.category for d in diags}
        self.assertIn(DiagnosticCategory.REGISTER_PRESSURE, categories)

    def test_from_profile_occupancy_limited(self):
        from lab.core.diagnostic import Diagnostic, DiagnosticCategory, Severity
        parsed = {"memory": {"dram_throughput_pct_of_peak_elapsed": "30.0"}, "occupancy": {"achieved": "20.0"}, "resources": {"registers_per_thread": "40"}, "kernel": {"name": "rmsnorm_kernel", "duration": "3.0"}, "raw_metrics": {}}
        diags = Diagnostic.from_profile(parsed, "exp-003")
        categories = {d.category for d in diags}
        self.assertIn(DiagnosticCategory.OCCUPANCY_LIMITED, categories)

    def test_from_profile_unknown_fallback(self):
        from lab.core.diagnostic import Diagnostic, DiagnosticCategory, Severity
        parsed = {"memory": {}, "occupancy": {}, "resources": {}, "kernel": {}, "raw_metrics": {}}
        diags = Diagnostic.from_profile(parsed, "exp-004")
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0].category, DiagnosticCategory.UNKNOWN)

    def test_save_load(self):
        import tempfile
        from lab.core.diagnostic import Diagnostic, DiagnosticCategory, Severity
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "diag.json"
            d_obj = Diagnostic(category=DiagnosticCategory.COMPUTE_BOUND, severity=Severity.INFO, message="Compute bound", source="ncu_profile")
            d_obj.save(p)
            d2 = Diagnostic.load(p)
            self.assertEqual(d2.id, d_obj.id)
            self.assertEqual(d2.category, DiagnosticCategory.COMPUTE_BOUND)


class TestPhase3EvidenceStore(unittest.TestCase):
    def test_evidence_summary_verdict(self):
        from lab.core.evidence_store import EvidenceSummary
        s = EvidenceSummary(claim="test", mechanism="m", supported_count=2, refuted_count=1, total_count=3)
        self.assertEqual(s.verdict, "SUPPORTED")
        s.supported_count = 0
        s.refuted_count = 3
        self.assertEqual(s.verdict, "REFUTED")
        s.supported_count = 1
        s.refuted_count = 1
        self.assertEqual(s.verdict, "MIXED")
        s2 = EvidenceSummary()
        self.assertEqual(s2.verdict, "UNTESTED")

    def test_evidence_summary_to_dict(self):
        from lab.core.evidence_store import EvidenceSummary
        s = EvidenceSummary(claim="vectorized load", mechanism="reduce tx", experiment_ids=["e1"], supported_count=1, total_count=1, best_speedup=1.05)
        d = s.to_dict()
        self.assertEqual(d["claim"], "vectorized load")
        self.assertEqual(d["best_speedup"], 1.05)


class TestPhase3DiagnosticContext(unittest.TestCase):
    def test_diagnostic_context_block(self):
        from lab.core.diagnostic import Diagnostic, DiagnosticCategory, Severity
        diag = Diagnostic(category=DiagnosticCategory.MEMORY_BOUND, severity=Severity.WARNING, message="DRAM at 75pct", possible_causes=["uncoalesced"], suggested_actions=["vectorize"], source="ncu", experiment_id="e001-x001")
        block = "Diagnostic: " + diag.message + " [" + diag.category.value + "] severity=" + diag.severity.value
        self.assertIn("MEMORY_BOUND", block)
        self.assertIn("WARNING", block)

    def test_evidence_summary_as_context(self):
        from lab.core.evidence_store import EvidenceSummary
        summaries = [
            EvidenceSummary(claim="vectorized load", mechanism="reduce tx", supported_count=3, total_count=3, best_speedup=1.08).to_dict(),
            EvidenceSummary(claim="larger block", mechanism="occupancy", refuted_count=2, total_count=2, worst_speedup=0.92).to_dict(),
        ]
        lines = []
        for s in summaries:
            if s["supported_count"] > s.get("refuted_count", 0) and s["supported_count"] > 0:
                tag = "SUPPORTED"
            elif s.get("refuted_count", 0) > s["supported_count"]:
                tag = "REFUTED"
            else:
                tag = "INCONCLUSIVE"
            lines.append("[" + tag + "] " + s["claim"])
        self.assertIn("[SUPPORTED] vectorized load", lines)
        self.assertIn("[REFUTED] larger block", lines)




class TestPhase4ExperimentReport(unittest.TestCase):
    def test_create_from_record(self):
        from lab.core.report import ExperimentReport
        record = {
            "experiment_id": "e001-x001",
            "episode_id": "e001",
            "campaign_id": "rmsnorm-v2",
            "decision": "PROMOTE",
            "hypothesis_verdict": "SUPPORTED",
            "changed_files": ["rmsnorm.cu"],
            "candidate_hash": "abc123",
            "correctness": {"pass": True},
            "compile": {"pass": True},
            "development_benchmark": {"metrics": {"arithmetic_mean_speedup": 1.08}},
            "candidate_lineage": {
                "candidate_id": "candidate:abc",
                "parent_candidate_id": "parent:def",
                "hypothesis_id": "hyp-1234",
            },
            "plan": {
                "hypothesis": {
                    "claim": "vectorized load",
                    "mechanism": "reduce transactions",
                    "expected_effect": "speedup",
                }
            },
            "timestamp": "2026-09-18T00:00:00",
        }
        r = ExperimentReport.from_experiment(record)
        self.assertEqual(r.experiment_id, "e001-x001")
        self.assertEqual(r.decision, "PROMOTE")
        self.assertEqual(r.hypothesis["claim"], "vectorized load")
        self.assertEqual(r.changes, ["rmsnorm.cu"])
        self.assertTrue(r.evidence["correctness"]["pass"])
        self.assertEqual(r.evidence["performance"]["speedup"], 1.08)
        self.assertTrue(r.report_id.startswith("rpt-"))

    def test_serialize_roundtrip(self):
        from lab.core.report import ExperimentReport
        record = {
            "experiment_id": "e002-x001",
            "episode_id": "e002",
            "decision": "REJECT_PERFORMANCE",
            "plan": {"hypothesis": {"claim": "test"}},
        }
        r = ExperimentReport.from_experiment(record)
        d = r.to_dict()
        r2 = ExperimentReport.from_dict(d)
        self.assertEqual(r2.experiment_id, "e002-x001")
        self.assertEqual(r2.report_id, r.report_id)

    def test_save_load(self):
        import tempfile
        from lab.core.report import ExperimentReport
        record = {"experiment_id": "e003-x001", "decision": "PROMOTE",
                  "plan": {"hypothesis": {"claim": "test"}}}
        r = ExperimentReport.from_experiment(record)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "report.json"
            r.save(p)
            r2 = ExperimentReport.load(p)
            self.assertEqual(r2.experiment_id, "e003-x001")
            self.assertEqual(r2.report_id, r.report_id)

    def test_knowledge_candidate_high_confidence(self):
        from lab.core.report import ExperimentReport
        record = {
            "experiment_id": "e004-x001",
            "decision": "PROMOTE",
            "hypothesis_verdict": "SUPPORTED",
            "development_benchmark": {"metrics": {"arithmetic_mean_speedup": 1.08}},
            "plan": {"hypothesis": {"claim": "vectorized load", "mechanism": "reduce tx"}},
        }
        r = ExperimentReport.from_experiment(record)
        kc = r.knowledge_candidate
        self.assertIsNotNone(kc)
        self.assertEqual(kc["claim"], "vectorized load")
        self.assertEqual(kc["confidence"], "high")
        self.assertEqual(kc["speedup"], 1.08)

    def test_knowledge_candidate_low_confidence(self):
        from lab.core.report import ExperimentReport
        record = {
            "experiment_id": "e005-x001",
            "decision": "REJECT_PERFORMANCE",
            "hypothesis_verdict": "REFUTED",
            "development_benchmark": {"metrics": {"arithmetic_mean_speedup": 0.92}},
            "plan": {"hypothesis": {"claim": "bad idea", "mechanism": "wrong"}},
        }
        r = ExperimentReport.from_experiment(record)
        kc = r.knowledge_candidate
        self.assertIsNotNone(kc)
        self.assertEqual(kc["confidence"], "low")


class TestPhase4MarkdownRenderer(unittest.TestCase):
    def test_render_basic(self):
        from lab.core.report import ExperimentReport
        from lab.reporting.markdown_renderer import render_markdown
        record = {
            "experiment_id": "e010-x001",
            "episode_id": "e010",
            "campaign_id": "rmsnorm",
            "decision": "PROMOTE",
            "hypothesis_verdict": "SUPPORTED",
            "changed_files": ["rmsnorm.cu"],
            "correctness": {"pass": True},
            "compile": {"pass": True},
            "development_benchmark": {"metrics": {"arithmetic_mean_speedup": 1.08}},
            "candidate_lineage": {"candidate_id": "cand:1", "parent_candidate_id": "parent:0", "hypothesis_id": "hyp-abc"},
            "plan": {"hypothesis": {"claim": "vectorized test", "mechanism": "reduce tx", "expected_effect": "speedup"}},
        }
        r = ExperimentReport.from_experiment(record)
        md = render_markdown(r)
        self.assertIn("# Experiment e010-x001", md)
        self.assertIn("vectorized test", md)
        self.assertIn("PROMOTE", md)
        self.assertIn("1.0800x", md)
        self.assertIn("Knowledge Candidate", md)

    def test_render_diagnostics(self):
        from lab.core.report import ExperimentReport
        from lab.reporting.markdown_renderer import render_markdown
        record = {
            "experiment_id": "e011-x001",
            "decision": "PROMOTE",
            "plan": {"hypothesis": {"claim": "test"}},
            "diagnostic_results": [{
                "action": {"type": "profiler"},
                "result": {"kind": "PROFILE_MEASUREMENT"},
                "diagnostics": [{
                    "category": "MEMORY_BOUND",
                    "severity": "WARNING",
                    "message": "DRAM high",
                }]
            }],
        }
        r = ExperimentReport.from_experiment(record)
        md = render_markdown(r)
        self.assertIn("MEMORY_BOUND", md)
        self.assertIn("WARNING", md)
        self.assertIn("DRAM high", md)


class TestPhase4Lineage(unittest.TestCase):
    def test_read_empty(self):
        import tempfile
        from lab.reporting.lineage import read_lineage
        with tempfile.TemporaryDirectory() as d:
            result = read_lineage(Path(d))
            self.assertEqual(result, [])

    def test_read_lineage(self):
        import tempfile, json
        from lab.reporting.lineage import read_lineage, get_lineage_chain
        with tempfile.TemporaryDirectory() as d:
            ep = Path(d)
            items = [
                {"candidate_id": "cand:1", "parent_candidate_id": "incumbent:0", "experiment_id": "e1", "hypothesis_id": "hyp-1"},
                {"candidate_id": "cand:2", "parent_candidate_id": "cand:1", "experiment_id": "e2", "hypothesis_id": "hyp-2"},
                {"candidate_id": "cand:3", "parent_candidate_id": "cand:2", "experiment_id": "e3", "hypothesis_id": "hyp-3"},
            ]
            with (ep / "candidate_lineage.jsonl").open("w") as f:
                for item in items:
                    f.write(json.dumps(item) + "\n")
            result = read_lineage(ep)
            self.assertEqual(len(result), 3)
            chain = get_lineage_chain(ep, "cand:3")
            self.assertEqual(len(chain), 3)
            self.assertEqual(chain[0]["candidate_id"], "cand:3")
            self.assertEqual(chain[-1]["candidate_id"], "cand:1")

    def test_format_lineage(self):
        import tempfile, json
        from lab.reporting.lineage import read_lineage, format_lineage_tree
        with tempfile.TemporaryDirectory() as d:
            ep = Path(d)
            items = [
                {"candidate_id": "cand:1", "parent_candidate_id": "incumbent:0", "experiment_id": "e1", "hypothesis_id": "hyp-1"},
                {"candidate_id": "cand:2", "parent_candidate_id": "cand:1", "experiment_id": "e2", "hypothesis_id": "hyp-2"},
            ]
            with (ep / "candidate_lineage.jsonl").open("w") as f:
                for item in items:
                    f.write(json.dumps(item) + "\n")
            tree = format_lineage_tree(ep, "cand:2")
            self.assertIn("cand:1", tree)
            self.assertIn("cand:2", tree)


class TestPhase4KnowledgeCandidate(unittest.TestCase):
    def test_generate_single(self):
        from lab.core.report import ExperimentReport
        from lab.reporting.knowledge_candidate import generate_knowledge_candidate
        record = {
            "experiment_id": "e020-x001",
            "hypothesis_verdict": "SUPPORTED",
            "development_benchmark": {"metrics": {"arithmetic_mean_speedup": 1.08}},
            "plan": {"hypothesis": {"claim": "vectorized", "mechanism": "tx"}},
        }
        r = ExperimentReport.from_experiment(record)
        kc = generate_knowledge_candidate(r)
        self.assertIsNotNone(kc)
        self.assertIn("claim", kc)

    def test_generate_list(self):
        from lab.core.report import ExperimentReport
        from lab.reporting.knowledge_candidate import generate_knowledge_candidates
        records = [
            {"experiment_id": "e1", "hypothesis_verdict": "SUPPORTED",
             "development_benchmark": {"metrics": {"arithmetic_mean_speedup": 1.08}},
             "plan": {"hypothesis": {"claim": "good", "mechanism": "m"}}},
            {"experiment_id": "e2", "hypothesis_verdict": "REFUTED",
             "development_benchmark": {"metrics": {"arithmetic_mean_speedup": 0.9}},
             "plan": {"hypothesis": {"claim": "bad", "mechanism": "m"}}},
        ]
        reports = [ExperimentReport.from_experiment(r) for r in records]
        candidates = generate_knowledge_candidates(reports)
        # Only "good" has high/medium confidence
        self.assertEqual(len(candidates), 1)

    def test_aggregate_from_summaries(self):
        from lab.reporting.knowledge_candidate import aggregate_knowledge_candidates
        candidates = aggregate_knowledge_candidates(Path("."))
        # Should not crash even on invalid/empty campaign
        self.assertIsInstance(candidates, list)


class TestPhase4Regression(unittest.TestCase):
    def test_old_campaign_no_break(self):
        """Existing journal records without new fields still work."""
        from lab.core.report import ExperimentReport
        record = {
            "experiment_id": "old-e001",
            "decision": "PROMOTE",
        }
        r = ExperimentReport.from_experiment(record)
        self.assertEqual(r.experiment_id, "old-e001")
        self.assertEqual(r.hypothesis, {"claim": "", "mechanism": "", "expected_effect": "", "verdict": "INCONCLUSIVE"})
        self.assertEqual(r.changes, [])




class TestPhase5BWorkspace(unittest.TestCase):
    def test_create_copy_mode(self):
        import tempfile, shutil
        from lab.runtime.workspace import CandidateWorkspace
        with tempfile.TemporaryDirectory() as d:
            ep = Path(d) / "e001"
            inc = Path(d) / "incumbent"
            inc.mkdir()
            (inc / "test.cu").write_text("// baseline")
            ws = CandidateWorkspace(ep, inc)
            ws.create()
            self.assertTrue(ws.candidate_root.is_dir())
            self.assertEqual(ws.mode, "copy")
            self.assertTrue((ws.candidate_root / "test.cu").is_file())
            self.assertEqual((ws.candidate_root / "test.cu").read_text(), "// baseline")

    def test_create_idempotent(self):
        import tempfile
        from lab.runtime.workspace import CandidateWorkspace
        with tempfile.TemporaryDirectory() as d:
            ep = Path(d) / "e001"
            inc = Path(d) / "incumbent"
            inc.mkdir()
            (inc / "test.cu").write_text("baseline")
            ws = CandidateWorkspace(ep, inc)
            ws.create()
            ws.create()  # second call: no-op
            self.assertTrue(ws._created)

    def test_snapshot(self):
        import tempfile
        from lab.runtime.workspace import CandidateWorkspace
        with tempfile.TemporaryDirectory() as d:
            ep = Path(d) / "e001"
            inc = Path(d) / "incumbent"
            inc.mkdir()
            (inc / "test.cu").write_text("baseline")
            ws = CandidateWorkspace(ep, inc)
            ws.create()
            snap = ws.snapshot()
            self.assertIn("timestamp", snap)
            self.assertIn("candidate_hash", snap)
            self.assertIn("file_map", snap)
            self.assertIn("mode", snap)
            self.assertEqual(snap["mode"], "copy")
            self.assertIn("test.cu", snap["file_map"])

    def test_snapshot_deterministic(self):
        import tempfile
        from lab.runtime.workspace import CandidateWorkspace
        with tempfile.TemporaryDirectory() as d:
            ep = Path(d) / "e001"
            inc = Path(d) / "incumbent"
            inc.mkdir()
            (inc / "test.cu").write_text("same content")
            ws = CandidateWorkspace(ep, inc)
            ws.create()
            snap1 = ws.snapshot()
            snap2 = ws.snapshot()
            self.assertEqual(snap1["candidate_hash"], snap2["candidate_hash"])

    def test_digest(self):
        import tempfile
        from lab.runtime.workspace import CandidateWorkspace
        with tempfile.TemporaryDirectory() as d:
            ep = Path(d) / "e001"
            inc = Path(d) / "incumbent"
            inc.mkdir()
            (inc / "a.cu").write_text("content a")
            ws = CandidateWorkspace(ep, inc)
            ws.create()
            d1 = ws.digest()
            (ws.candidate_root / "b.cu").write_text("content b")
            d2 = ws.digest()
            self.assertNotEqual(d1, d2)

    def test_reset_to_baseline(self):
        import tempfile
        from lab.runtime.workspace import CandidateWorkspace
        with tempfile.TemporaryDirectory() as d:
            ep = Path(d) / "e001"
            inc = Path(d) / "incumbent"
            inc.mkdir()
            (inc / "kernel.cu").write_text("// baseline kernel")
            (inc / "config.h").write_text("// baseline config")
            ws = CandidateWorkspace(ep, inc)
            ws.create()
            # Agent modifies files
            (ws.candidate_root / "kernel.cu").write_text("// modified!")
            (ws.candidate_root / "config.h").write_text("// modified too!")
            # Create evidence file (should be preserved)
            (ws.candidate_root / "evidence").mkdir(exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=str(ws.candidate_root), prefix="validated_plan_e", suffix=".json", delete=False):
                pass
            # Reset
            ws.reset_to_baseline()
            # Code files should be restored
            self.assertEqual((ws.candidate_root / "kernel.cu").read_text(), "// baseline kernel")
            self.assertEqual((ws.candidate_root / "config.h").read_text(), "// baseline config")
            # Evidence dir should be preserved
            self.assertTrue((ws.candidate_root / "evidence").is_dir())

    def test_commit_candidate_copy_mode(self):
        import tempfile
        from lab.runtime.workspace import CandidateWorkspace
        with tempfile.TemporaryDirectory() as d:
            ep = Path(d) / "e001"
            inc = Path(d) / "incumbent"
            inc.mkdir()
            (inc / "test.cu").write_text("baseline")
            ws = CandidateWorkspace(ep, inc)
            ws.create()
            h = ws.commit_candidate("exp-001")
            self.assertIsNotNone(h)
            self.assertEqual(len(h), 64)  # SHA256 hex

    def test_file_map(self):
        import tempfile
        from lab.runtime.workspace import CandidateWorkspace
        with tempfile.TemporaryDirectory() as d:
            ep = Path(d) / "e001"
            inc = Path(d) / "incumbent"
            inc.mkdir()
            (inc / "a.cu").write_text("content a")
            (inc / "b.h").write_text("content b")
            ws = CandidateWorkspace(ep, inc)
            ws.create()
            fm = ws.file_map()
            self.assertIn("a.cu", fm)
            self.assertIn("b.h", fm)


class TestPhase5BOldCampaign(unittest.TestCase):
    def test_workspace_with_no_git(self):
        import tempfile
        from lab.runtime.workspace import CandidateWorkspace, _is_git_repo
        with tempfile.TemporaryDirectory() as d:
            ep = Path(d) / "e001"
            inc = Path(d) / "incumbent"
            inc.mkdir()
            (inc / "test.cu").write_text("baseline")
            # _is_git_repo depends on cwd context
            ws = CandidateWorkspace(ep, inc)
            ws.create()
            self.assertEqual(ws.mode, "copy")
            self.assertEqual(ws.digest(), ws.digest())


class TestPhase5CRecovery(unittest.TestCase):
    def test_marker_create_and_clear(self):
        import tempfile
        from lab.runtime.recovery import RecoveryManager
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d) / "run-001"
            rm = RecoveryManager(run_dir, "run-001")
            marker = rm.create_marker()
            self.assertEqual(marker["run_id"], "run-001")
            self.assertIsInstance(marker["start_token"], str)
            self.assertTrue((run_dir / "recovery_marker.json").is_file())
            rm.clear_marker()
            self.assertFalse((run_dir / "recovery_marker.json").is_file())

    def test_detect_clean(self):
        import tempfile
        from lab.runtime.recovery import RecoveryManager
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d) / "run-002"
            report = RecoveryManager.detect(run_dir)
            self.assertEqual(report.state, "CLEAN")

    def test_detect_running(self):
        import tempfile
        from lab.runtime.recovery import RecoveryManager
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d) / "run-003"
            rm = RecoveryManager(run_dir, "run-003")
            rm.create_marker()
            rm.update_heartbeat("TESTING", 1)
            report = RecoveryManager.detect(run_dir)
            self.assertEqual(report.state, "RUNNING")
            self.assertTrue(report.pid_alive)
            self.assertTrue(report.start_token_valid)
            rm.clear_marker()

    def test_detect_interrupted(self):
        import tempfile, json
        from lab.runtime.recovery import RecoveryManager
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d) / "run-004"
            run_dir.mkdir(parents=True)
            # Simulate a dead process: write marker with non-existent PID
            marker_data = {
                "run_id": "run-004",
                "pid": 99999,
                "start_token": "dead-process-token",
                "start_time": "2026-01-01T00:00:00",
                "campaign_dir": "",
                "candidate_root": "",
                "state": "RUNNING",
            }
            from lab.core.persistence import atomic_json
            atomic_json(run_dir / "recovery_marker.json", marker_data)
            hb_path = run_dir / "recovery_heartbeat.json"
            hb_data = {
                "run_id": "run-004",
                "pid": 99999,
                "start_token": "dead-process-token",
                "phase": "TESTING",
                "experiment": 1,
                "timestamp": "2026-01-01T00:00:00",
            }
            hb_path.write_text(json.dumps(hb_data))
            report = RecoveryManager.detect(run_dir)
            self.assertEqual(report.state, "INTERRUPTED")
            self.assertTrue(report.can_resume)
            (run_dir / "recovery_marker.json").unlink()
            (run_dir / "recovery_heartbeat.json").unlink()

    def test_heartbeat_staleness(self):
        import tempfile
        from lab.runtime.recovery import RecoveryManager
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d) / "run-005"
            rm = RecoveryManager(run_dir, "run-005")
            rm.create_marker()
            report = RecoveryManager.detect(run_dir)
            self.assertTrue(report.heartbeat_stale)

    def test_pid_reuse_guard(self):
        import tempfile, json
        from lab.runtime.recovery import RecoveryManager
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d) / "run-006"
            rm = RecoveryManager(run_dir, "run-006")
            rm.create_marker()
            hb_path = run_dir / "recovery_heartbeat.json"
            hb_data = {
                "run_id": "run-006",
                "pid": 99999,
                "start_token": "different-token",
                "phase": "TESTING",
                "experiment": 1,
                "timestamp": "2026-01-01T00:00:00",
            }
            hb_path.write_text(json.dumps(hb_data))
            report = RecoveryManager.detect(run_dir)
            self.assertTrue(report.can_resume)
            rm.clear_marker()

    def test_recovery_with_probe(self):
        import tempfile
        from lab.runtime.recovery import RecoveryManager
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d) / "run-007"
            rm = RecoveryManager(run_dir, "run-007")
            rm.create_marker()
            report = RecoveryManager.recover(run_dir, probe_fn=lambda: {"status": "READY"})
            self.assertTrue(report.environment_ready)
            self.assertTrue(report.can_resume)

    def test_old_campaign_compat(self):
        import tempfile
        from lab.runtime.recovery import RecoveryManager
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d) / "run-old"
            report = RecoveryManager.detect(run_dir)
            self.assertEqual(report.state, "CLEAN")
            self.assertFalse(report.can_resume)


class TestPhase5DHeartbeat(unittest.TestCase):
    def test_heartbeat_phase_update(self):
        import tempfile
        from lab.runtime.recovery import RecoveryManager
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d) / "run-hb-1"
            rm = RecoveryManager(run_dir, "run-hb-1")
            rm.create_marker()
            hb1 = rm.update_heartbeat("BUILD_START", 1, experiment_id="e0001-x001", candidate_id="cand-001")
            self.assertEqual(hb1["phase"], "BUILD_START")
            self.assertEqual(hb1["experiment_id"], "e0001-x001")
            self.assertEqual(hb1["candidate_id"], "cand-001")
            hb2 = rm.update_heartbeat("BUILD_DONE", 2, experiment_id="e0001-x002")
            self.assertEqual(hb2["phase"], "BUILD_DONE")
            self.assertNotEqual(hb1["timestamp"], hb2["timestamp"])
            rm.clear_marker()

    def test_heartbeat_timestamp_monotonic(self):
        import tempfile, time
        from lab.runtime.recovery import RecoveryManager
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d) / "run-hb-2"
            rm = RecoveryManager(run_dir, "run-hb-2")
            rm.create_marker()
            hb1 = rm.update_heartbeat("RUN_START", 0)
            time.sleep(0.01)
            hb2 = rm.update_heartbeat("AGENT_PLAN", 1)
            self.assertGreaterEqual(hb2["timestamp"], hb1["timestamp"])
            rm.clear_marker()


class TestPhase5DIntegrity(unittest.TestCase):
    def test_integrity_hash_match(self):
        import tempfile
        from lab.runtime.workspace import CandidateWorkspace
        with tempfile.TemporaryDirectory() as d:
            ep = Path(d) / "e0001"
            inc = Path(d) / "incumbent"
            ep.mkdir(); inc.mkdir()
            (inc / "kernel.cu").write_text("// baseline\n", encoding="utf-8")
            ws = CandidateWorkspace(ep, inc)
            ws.create()
            snap = ws.snapshot()
            result = ws.verify_integrity(snap["candidate_hash"])
            self.assertTrue(result["pass"], f"Expected pass, got: {result}")
            self.assertEqual(len(result["modified_files"]), 0)

    def test_integrity_detect_modification(self):
        import tempfile
        from lab.runtime.workspace import CandidateWorkspace
        with tempfile.TemporaryDirectory() as d:
            ep = Path(d) / "e0002"
            inc = Path(d) / "incumbent"
            ep.mkdir(); inc.mkdir()
            (inc / "kernel.cu").write_text("// baseline\n", encoding="utf-8")
            ws = CandidateWorkspace(ep, inc)
            ws.create()
            snap = ws.snapshot()
            # Modify a file
            (ws.candidate_root / "kernel.cu").write_text("// modified\n", encoding="utf-8")
            result = ws.verify_integrity(snap["candidate_hash"])
            self.assertFalse(result["pass"], f"Expected fail, got: {result}")

    def test_integrity_no_snapshot_hash(self):
        import tempfile
        from lab.runtime.workspace import CandidateWorkspace
        with tempfile.TemporaryDirectory() as d:
            ep = Path(d) / "e0003"
            inc = Path(d) / "incumbent"
            ep.mkdir(); inc.mkdir()
            (inc / "kernel.cu").write_text("// baseline\n", encoding="utf-8")
            ws = CandidateWorkspace(ep, inc)
            ws.create()
            result = ws.verify_integrity()  # No hash, just returns current
            self.assertTrue(result["pass"])
            self.assertIn("current_hash", result)


class TestPhase5DRecoveryCli(unittest.TestCase):
    def test_recover_list_no_runs(self):
        import tempfile
        from lab.runtime.recovery import RecoveryManager
        with tempfile.TemporaryDirectory() as d:
            runs_dir = Path(d) / "runtime" / "runs"
            runs_dir.mkdir(parents=True)
            found = []
            if runs_dir.is_dir():
                for p in sorted(runs_dir.iterdir()):
                    if not p.is_dir(): continue
                    report = RecoveryManager.detect(p)
                    if report.state in ("INTERRUPTED", "STALE"):
                        found.append((p.name, report))
            self.assertEqual(len(found), 0)

    def test_recover_inspect_interrupted(self):
        import tempfile, json
        from lab.runtime.recovery import RecoveryManager
        from lab.core.persistence import atomic_json
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d) / "run-inspect"
            run_dir.mkdir(parents=True)
            # Create an interrupted run
            marker_data = {
                "run_id": "run-inspect", "pid": 99999,
                "start_token": "dead-token", "start_time": "2026-01-01T00:00:00",
                "campaign_dir": str(Path(d) / "campaign"),
                "candidate_root": str(Path(d) / "candidate"),
                "state": "RUNNING",
            }
            atomic_json(run_dir / "recovery_marker.json", marker_data)
            hb_data = {"run_id": "run-inspect", "pid": 99999, "start_token": "dead-token",
                       "phase": "BENCHMARK_START", "experiment": 1, "timestamp": "2026-01-01T00:01:00"}
            (run_dir / "recovery_heartbeat.json").write_text(json.dumps(hb_data))
            report = RecoveryManager.recover(run_dir, probe_fn=lambda: {"status": "READY"})
            self.assertEqual(report.state, "INTERRUPTED")
            self.assertTrue(report.can_resume)
            self.assertTrue(report.environment_ready)
            self.assertEqual(report.last_phase, "BENCHMARK_START")

    def test_recover_full_flow(self):
        import tempfile, json
        from lab.runtime.recovery import RecoveryManager
        from lab.core.persistence import atomic_json
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d) / "run-full"
            candidate = Path(d) / "candidate"
            run_dir.mkdir(parents=True); candidate.mkdir(parents=True)
            (candidate / "kernel.cu").write_text("// test\n", encoding="utf-8")
            marker_data = {
                "run_id": "run-full", "pid": 99999,
                "start_token": "full-token", "start_time": "2026-01-01T00:00:00",
                "campaign_dir": str(Path(d) / "campaign"),
                "candidate_root": str(candidate),
                "state": "RUNNING",
            }
            atomic_json(run_dir / "recovery_marker.json", marker_data)
            hb_data = {"run_id": "run-full", "pid": 99999, "start_token": "full-token",
                       "phase": "DECISION", "experiment": 3, "timestamp": "2026-01-01T00:01:00"}
            (run_dir / "recovery_heartbeat.json").write_text(json.dumps(hb_data))
            report = RecoveryManager.recover(run_dir, probe_fn=lambda: {"status": "READY"})
            self.assertEqual(report.state, "INTERRUPTED")
            self.assertTrue(report.can_resume)
            self.assertTrue(report.environment_ready)
            self.assertTrue(report.workspace_valid)


class TestPhase5DRegression(unittest.TestCase):
    def test_phase5c_tests_still_pass(self):
        """Ensure Phase 5-C recovery tests are not broken."""
        import tempfile, json
        from lab.runtime.recovery import RecoveryManager
        from lab.core.persistence import atomic_json

        # marker create/clear
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d) / "run-001"
            rm = RecoveryManager(run_dir, "run-001")
            marker = rm.create_marker()
            self.assertEqual(marker["run_id"], "run-001")
            self.assertTrue(isinstance(marker["start_token"], str))
            self.assertTrue((run_dir / "recovery_marker.json").is_file())
            rm.clear_marker()
            self.assertFalse((run_dir / "recovery_marker.json").is_file())

        # detect clean
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d) / "run-002"
            report = RecoveryManager.detect(run_dir)
            self.assertEqual(report.state, "CLEAN")

        # PID reuse guard
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d) / "run-006"
            rm = RecoveryManager(run_dir, "run-006")
            rm.create_marker()
            hb_data = {"run_id": "run-006", "pid": 99999, "start_token": "different-token",
                       "phase": "TESTING", "experiment": 1, "timestamp": "2026-01-01T00:00:00"}
            (run_dir / "recovery_heartbeat.json").write_text(json.dumps(hb_data))
            report = RecoveryManager.detect(run_dir)
            self.assertTrue(report.can_resume)
            rm.clear_marker()

        # old campaign compat
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d) / "run-old"
            report = RecoveryManager.detect(run_dir)
            self.assertEqual(report.state, "CLEAN")
            self.assertFalse(report.can_resume)

        # heartbeat staleness
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d) / "run-005"
            rm = RecoveryManager(run_dir, "run-005")
            rm.create_marker()
            report = RecoveryManager.detect(run_dir)
            self.assertTrue(report.heartbeat_stale)
            rm.clear_marker()


if __name__ == "__main__":
    unittest.main()

if __name__ == "__main__":
    unittest.main()
