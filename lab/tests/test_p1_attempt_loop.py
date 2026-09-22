"""Offline P1 tests: no SSH, CUDA, Agent service, or historical artifacts."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lab.runtime.agent.attempt_loop import (
    HypothesisAttemptController,
    LineageStore,
    StructuredKnowledgeSink,
    file_sha256,
    load_attempt_budget,
    profile_feedback,
)
from lab.runtime.agent.v100_attempt_adapter import V100AttemptEvaluatorAdapter


class FakeSession:
    def __init__(self):
        self.starts = 0; self.closed = 0; self.turn_threads = []
        self.thread_id = None
    def start(self):
        self.starts += 1; self.thread_id = f"thread-{self.starts}"
    def close(self): self.closed += 1
    def status(self): return {"thread_id": self.thread_id}
    def run_experiment_turn(self, _prompt, _context=None):
        self.turn_threads.append(self.thread_id)
        return {"final_response": "{}"}


class FakeEvaluator:
    def __init__(self, *, compile_results=None, correctness_results=None, qualification="QUALIFIED_ACCEPT", profile=None, contract=True, raises_at=None):
        self.compile_results = list(compile_results or [{"pass": True}])
        self.correctness_results = list(correctness_results or [{"pass": True}])
        self.qualification = qualification; self.profile_value = profile; self.contract = contract; self.raises_at = raises_at
        self.calls = []
    def validate_contract(self, _candidate): return {"pass": self.contract, "failures": [] if self.contract else [{"field": "marker"}]}
    def compile(self, _candidate):
        self.calls.append("compile")
        if self.raises_at == "compile": raise RuntimeError("framework dispatch failed")
        return self.compile_results.pop(0) if self.compile_results else {"pass": True}
    def correctness(self, _candidate):
        self.calls.append("correctness")
        if self.raises_at == "correctness": raise RuntimeError("framework dispatch failed")
        return self.correctness_results.pop(0) if self.correctness_results else {"pass": True}
    def benchmark(self, _candidate):
        self.calls.append("benchmark")
        if self.raises_at == "benchmark": raise RuntimeError("framework dispatch failed")
        return {"aggregate_score": 2.0, "shapes": [{"shape": "1,4", "latency_us": 9.0}]}
    def profile(self, _candidate): return self.profile_value
    def qualify(self, _candidate, _benchmark): return {"status": self.qualification}


def budget(**overrides):
    value = load_attempt_budget()
    value.update(overrides)
    return value


class AttemptLoopTests(unittest.TestCase):
    def build(self, root, evaluator, session=None):
        session = session or FakeSession()
        return (HypothesisAttemptController(session=session, evaluator=evaluator,
            lineage=LineageStore(root / "campaigns" / "unit" / "lineage.jsonl"),
            knowledge=StructuredKnowledgeSink(root / "knowledge" / "unit.jsonl"), budget=budget()), session)

    @staticmethod
    def agent(candidate, seen, action="REPAIR"):
        def step(session, feedback, attempt_id):
            session.run_experiment_turn("repair", {"attempt_id": attempt_id})
            seen.append((attempt_id, feedback, session.status()["thread_id"]))
            candidate.write_text(candidate.read_text(encoding="utf-8") + f"// {attempt_id}\n", encoding="utf-8")
            return {"action": action, "candidate_path": candidate}
        return step

    def test_compile_error_repairs_in_same_session_then_passes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); candidate = root / "candidate.cu"; candidate.write_text("x\n", encoding="utf-8")
            controller, session = self.build(root, FakeEvaluator(compile_results=[{"pass": False, "diagnostics": [{"line": 3, "message": "bad"}]}, {"pass": True}]))
            seen = []; outcome = controller.run(hypothesis_id="h1", episode=1, initial_candidate=candidate, agent_step=self.agent(candidate, seen), incumbent_score=1.0)
            self.assertEqual(outcome.status, "QUALIFIED_ACCEPT")
            self.assertEqual([row[1]["kind"] if row[1] else None for row in seen], [None, "compile_error"])
            self.assertEqual({row[2] for row in seen}, {"thread-1"})
            self.assertEqual([row["attempt_id"] for row in outcome.attempts], ["h1-a1", "h1-a2"])

    def test_correctness_error_repairs_then_passes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); candidate = root / "candidate.cu"; candidate.write_text("x\n", encoding="utf-8")
            controller, _ = self.build(root, FakeEvaluator(correctness_results=[{"pass": False, "failed_shapes": ["1,4"], "max_error": 1.0, "tolerance": 0.1}, {"pass": True}]))
            seen = []; outcome = controller.run(hypothesis_id="h2", episode=1, initial_candidate=candidate, agent_step=self.agent(candidate, seen))
            self.assertEqual(outcome.status, "QUALIFIED_ACCEPT")
            self.assertEqual(seen[1][1]["kind"], "correctness_error")

    def test_two_compile_repairs_remain_one_hypothesis(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); candidate = root / "candidate.cu"; candidate.write_text("x\n", encoding="utf-8")
            controller, session = self.build(root, FakeEvaluator(compile_results=[{"pass": False}, {"pass": False}, {"pass": True}]))
            outcome = controller.run(hypothesis_id="h3", episode=1, initial_candidate=candidate, agent_step=self.agent(candidate, []))
            self.assertEqual(outcome.status, "QUALIFIED_ACCEPT")
            self.assertEqual({row["hypothesis_id"] for row in outcome.attempts}, {"h3"})
            self.assertEqual(session.starts, 1)

    def test_attempt_budget_and_explicit_abandon(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); candidate = root / "candidate.cu"; candidate.write_text("x\n", encoding="utf-8")
            controller, _ = self.build(root, FakeEvaluator(compile_results=[{"pass": False}]))
            controller.budget["max_compile_repairs"] = 0
            exhausted = controller.run(hypothesis_id="h4", episode=1, initial_candidate=candidate, agent_step=self.agent(candidate, []))
            self.assertEqual(exhausted.status, "ATTEMPT_BUDGET_EXHAUSTED")
            controller, _ = self.build(root, FakeEvaluator())
            abandoned = controller.run(hypothesis_id="h5", episode=1, initial_candidate=candidate, agent_step=self.agent(candidate, [], "ABANDON_HYPOTHESIS"))
            self.assertEqual(abandoned.status, "ABANDON_HYPOTHESIS")

    def test_lineage_parent_and_rejections_are_preserved(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); candidate = root / "candidate.cu"; candidate.write_text("x\n", encoding="utf-8")
            controller, _ = self.build(root, FakeEvaluator(compile_results=[{"pass": False}, {"pass": True}], qualification="QUALIFIED_ACCEPT"))
            outcome = controller.run(hypothesis_id="h6", episode=9, initial_candidate=candidate, agent_step=self.agent(candidate, []))
            self.assertEqual(outcome.attempts[1]["parent_candidate_hash"], outcome.attempts[0]["candidate_hash"])
            rows = [json.loads(line) for line in (root / "campaigns" / "unit" / "lineage.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[-1]["failure_class"], "ACCEPTED")
            for field in ("mechanism", "data_lifetime", "memory_passes", "vectorization", "register_strategy", "compile_diagnostics", "benchmark_distribution", "profile_evidence", "evidence_level"):
                self.assertIn(field, rows[-1])

    def test_framework_error_does_not_write_strategy_knowledge(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); candidate = root / "candidate.cu"; candidate.write_text("x\n", encoding="utf-8")
            controller, _ = self.build(root, FakeEvaluator(raises_at="compile"))
            outcome = controller.run(hypothesis_id="h7", episode=1, initial_candidate=candidate, agent_step=self.agent(candidate, []))
            self.assertEqual(outcome.status, "FRAMEWORK_ERROR")
            self.assertEqual(outcome.attempts[0]["failure_class"], "FRAMEWORK_ERROR")
            self.assertFalse((root / "knowledge" / "unit.jsonl").exists())

    def test_profile_unavailable_and_qualification_cannot_be_bypassed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); candidate = root / "candidate.cu"; candidate.write_text("x\n", encoding="utf-8")
            controller, _ = self.build(root, FakeEvaluator(qualification="PROVISIONAL_UNSTABLE", profile=None))
            controller.budget["max_performance_refinements"] = 0
            outcome = controller.run(hypothesis_id="h8", episode=1, initial_candidate=candidate, agent_step=self.agent(candidate, [], "REQUEST_QUALIFICATION"))
            self.assertNotEqual(outcome.status, "QUALIFIED_ACCEPT")
            self.assertEqual(outcome.attempts[0]["failure_class"], "QUALIFICATION_UNSTABLE")
            self.assertEqual(outcome.attempts[0]["profile_evidence"]["evidence_status"], "UNAVAILABLE")

    def test_contract_failure_prevents_session_and_new_hypothesis_gets_new_thread(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); candidate = root / "candidate.cu"; candidate.write_text("x\n", encoding="utf-8")
            bad_session = FakeSession(); bad, _ = self.build(root, FakeEvaluator(contract=False), bad_session)
            rejected = bad.run(hypothesis_id="h9", episode=1, initial_candidate=candidate, agent_step=self.agent(candidate, []))
            self.assertEqual(rejected.status, "REJECT_CONTRACT"); self.assertEqual(bad_session.starts, 0)
            session = FakeSession(); controller, _ = self.build(root, FakeEvaluator(), session)
            one = controller.run(hypothesis_id="h10", episode=1, initial_candidate=candidate, agent_step=self.agent(candidate, []))
            two = controller.run(hypothesis_id="h11", episode=1, initial_candidate=candidate, agent_step=self.agent(candidate, []))
            self.assertNotEqual(one.agent_thread_id, two.agent_thread_id)

    def test_v100_adapter_caches_one_measurement_and_exposes_stages(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); (root / "operators" / "unit").mkdir(parents=True); (root / "config" / "environments" / "v100_sm70").mkdir(parents=True)
            metadata = {"operator": "unit", "interface": "standalone_cuda", "arch": "sm_70", "dtype": "float32", "semantic_identifier": "u", "required_source_marker": "// AKA_CONTRACT: x", "contract_schema": {"operator": "unit", "entry": "launch", "version": 1, "arguments": [{"name": "x", "type": "float*", "role": "input"}], "semantics": "u"}}
            evaluation = {"shapes": [[1, 4]], "score": "x", "correctness_tolerance": 0.1}
            (root / "operators" / "unit" / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            (root / "config" / "environments" / "v100_sm70" / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")
            candidate = root / "candidate.cu"; candidate.write_text('// AKA_CONTRACT: x\nextern "C" void launch(float* x) {}\n', encoding="utf-8")
            calls = []
            adapter = V100AttemptEvaluatorAdapter(project_root=root, operator="unit", measure=lambda _p: calls.append(1) or {"compile_pass": True, "correctness_pass": True, "aggregate_score": 2.0}, qualify=lambda _p, _b: {"status": "QUALIFIED_ACCEPT"}, profile=lambda _p: None)
            self.assertTrue(adapter.validate_contract(candidate)["pass"])
            self.assertTrue(adapter.compile(candidate)["pass"]); self.assertTrue(adapter.correctness(candidate)["pass"]); adapter.benchmark(candidate)
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
