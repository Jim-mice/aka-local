"""CPU fixtures for first-class diagnostic experiment routing; no CUDA work."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from lab.core.events import EventStore
from lab.runtime.agent.long_horizon import LongHorizonRunner
from lab.runtime.agent.long_horizon import InvalidAgentOutput
from lab.runtime.evaluators.diagnostics import summarize_repeated_per_shape


def plan(kind, actions, decision):
    return {"experiment_kind": kind, "diagnostic_actions": actions, "supporting_evidence_ids": [],
            "analysis_summary":"fixture diagnostic", "bottleneck":"unknown", "evidence_summary":"fixture",
            "hypothesis":{"question":"q","claim":"c","rationale":"r","expected_effect":"e","support_condition":"s","refute_condition":"f"},
            "planned_change":"no code change", "expected_risk":"none", "decision_request":decision,"next_direction":"next"}


class DiagnosticSession:
    model="gpt-5.6-luna";effort="low"
    def __init__(self): self.calls=[];self.index=0
    def start(self): pass
    def close(self): pass
    def send_context(self, context): self.context=context
    def run_experiment_turn(self, prompt, context):
        self.calls.append(prompt)
        if "DIAGNOSTIC experiment" in prompt:
            decision="blocked" if self.index==2 else "reject_and_continue"
            payload={"decision_request":decision,"result_summary":"diagnostic evidence returned","next_direction":"next diagnostic"}
        else:
            self.index+=1
            actions=[{"type":"inspect_shape_metadata"}] if self.index==1 else [{"type":"profiler","shape_ids":[1],"questions":["occupancy"]}]
            payload=plan("DIAGNOSTIC",actions,"reject_and_continue")
        return type("Reply",(),{"final_response":"```json\n"+json.dumps(payload)+"\n```"})()


class DiagnosticEvaluator:
    def __init__(self): self.compile_calls=0;self.actions=[]
    def prepare(self): return {"pass":True}
    def compile(self): self.compile_calls+=1;return {"pass":True}
    def check_correctness(self): raise AssertionError("DIAGNOSTIC must not call correctness")
    def development_benchmark(self): raise AssertionError("DIAGNOSTIC must not call dev ABBA")
    def run_diagnostic_action(self, action, progress_callback=None):
        self.actions.append(action)
        if progress_callback:progress_callback({"batch":1,"batches":1,"test_only":True})
        return {"phase":action["type"],"pass":True,"evidence_id":"fixture-"+action["type"],"artifacts":{"summary":"fixture://"+action["type"]}}


class EmptyDiagnosticSession:
    def __init__(self): self.calls=[]
    def start(self): pass
    def close(self): pass
    def send_context(self, context): self.context=context
    def run_experiment_turn(self,prompt,context):
        self.calls.append(prompt)
        payload=plan("DIAGNOSTIC",[],"reject_and_continue")
        return type("Reply",(),{"final_response":"```json\n"+json.dumps(payload)+"\n```"})()


class DiagnosticRoutingTests(unittest.TestCase):
    def test_profiler_plan_schema_normalizes_shape_selection_and_capability_discovery(self):
        selected=LongHorizonRunner._execution_action({"type":"profiler","requirements":{"shape_selection":[{"shape_id":"37"},{"shape_id":15}]}})
        self.assertEqual(selected,{"type":"profiler","shape_ids":["37","15"]})
        capability=LongHorizonRunner._execution_action({"type":"profiler","requirements":{"discover":["whether_ncu_is_available"]}})
        self.assertEqual(capability,{"type":"profiler","capability_only":True})

    def test_framework_retry_reuses_newest_prior_validated_plan(self):
        with tempfile.TemporaryDirectory(prefix="aka-plan-retry-") as temp:
            root=Path(temp);episode=root/"e0001";candidate=episode/"candidate";incumbent=root/"incumbent";candidate.mkdir(parents=True);incumbent.mkdir()
            runner=LongHorizonRunner(campaign_dir=root,candidate_root=candidate,incumbent=incumbent,session=object(),evaluator=object(),events=EventStore(root/"events.jsonl"),lab_root=root)
            plan_payload={"plan":plan("DIAGNOSTIC",[{"type":"static_evidence"}],"reject_and_continue")}
            (episode/"validated_plan_e001.json").write_text(json.dumps(plan_payload),encoding="utf-8")
            self.assertEqual(runner._load_validated_plan(1,{},2),plan_payload["plan"])
    def test_diagnostic_actions_do_not_edit_or_run_standard_pipeline(self):
        temp=Path(tempfile.mkdtemp(prefix="aka-diagnostic-"))
        try:
            lab=temp/"lab";campaign=lab/"campaigns"/"fixture";candidate=campaign/"episodes"/"e0001"/"candidate";incumbent=temp/"incumbent"
            candidate.mkdir(parents=True);incumbent.mkdir();(candidate/"candidate.py").write_text("x=1\n",encoding="utf-8");(incumbent/"candidate.py").write_text("x=1\n",encoding="utf-8")
            (campaign/"memory").mkdir();(campaign/"campaign.json").write_text(json.dumps({"campaign_id":"fixture"}),encoding="utf-8");(campaign/"frontier.json").write_text("{}",encoding="utf-8");(lab/"knowledge").mkdir()
            evaluator=DiagnosticEvaluator();events=EventStore(lab/"runtime"/"events.jsonl");runner=LongHorizonRunner(campaign_dir=campaign,candidate_root=candidate,incumbent=incumbent,session=DiagnosticSession(),evaluator=evaluator,events=events,budget={"max_experiments":2},workbench={"workbench_id":"fixture"},lab_root=lab)
            import lab.runtime.agent.long_horizon as module
            old=module.probe_local;module.probe_local=lambda:{"status":"READY","environment_fingerprint":"fixture"}
            try: result=runner.run()
            finally: module.probe_local=old
            self.assertIn(result["state"],{"WAITING_FOR_HUMAN","BUDGET_EXHAUSTED"})
            self.assertEqual(evaluator.compile_calls,0)
            self.assertEqual([item["type"] for item in evaluator.actions],["inspect_shape_metadata","profiler"])
            records=[json.loads(line) for line in (candidate.parent/"journal.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([record["experiment_kind"] for record in records],["DIAGNOSTIC","DIAGNOSTIC"])
            self.assertTrue(all(not record.get("changed_files") for record in records))
            event_types=[event["type"] for event in events.replay()]
            self.assertIn("DIAGNOSTIC_PLAN_READY",event_types)
            self.assertIn("DIAGNOSTIC_ACTION_STARTED",event_types)
            self.assertIn("DIAGNOSTIC_PROGRESS",event_types)
            self.assertIn("DIAGNOSTIC_EVIDENCE_WRITTEN",event_types)
            self.assertNotIn("COMPILE_STARTED",event_types)
        finally: shutil.rmtree(temp)

    def test_regime_summary_is_data_driven_and_marks_identical_noise(self):
        raw={"protocol":"fixture","batches":2,"warmup":1,"repeats":2,"shapes":{"1":{"input":{"token_count":1,"hidden_size":64},"batches":[{"incumbent_mean_ms":1.0,"candidate_mean_ms":1.0,"speedup":1.0},{"incumbent_mean_ms":1.1,"candidate_mean_ms":1.1,"speedup":1.0}]},"2":{"input":{"token_count":8,"hidden_size":128},"batches":[{"incumbent_mean_ms":5.0,"candidate_mean_ms":5.0,"speedup":1.0},{"incumbent_mean_ms":5.2,"candidate_mean_ms":5.2,"speedup":1.0}]}},"aggregate":{}}
        summary=summarize_repeated_per_shape(raw,identical_implementation=True)
        self.assertEqual(summary["kind"],"IDENTICAL_IMPLEMENTATION_NOISE_CHARACTERIZATION")
        self.assertEqual(len(summary["shapes"]),2)
        self.assertTrue(all(item["classification"]=="IDENTICAL_IMPLEMENTATION_NOISE_CHARACTERIZATION" for item in summary["shapes"]))

    def test_empty_diagnostic_plan_gets_one_repair_then_fails_cleanly(self):
        with tempfile.TemporaryDirectory(prefix="aka-empty-diag-") as temp:
            root=Path(temp);candidate=root/"candidate";incumbent=root/"incumbent";candidate.mkdir();incumbent.mkdir()
            session=EmptyDiagnosticSession();runner=LongHorizonRunner(campaign_dir=root,candidate_root=candidate,incumbent=incumbent,session=session,evaluator=DiagnosticEvaluator(),events=EventStore(root/"events.jsonl"),lab_root=root)
            with self.assertRaisesRegex(InvalidAgentOutput,"INVALID_DIAGNOSTIC_PLAN"):
                runner._plan({})
            self.assertEqual(len(session.calls),2)
            self.assertIn("did not provide executable diagnostic_actions",session.calls[1])

    def test_empty_diagnostic_plan_enters_waiting_for_human_without_dispatch(self):
        with tempfile.TemporaryDirectory(prefix="aka-empty-diag-run-") as temp:
            root=Path(temp);lab=root/"lab";campaign=lab/"campaigns"/"fixture";candidate=campaign/"episodes"/"e0001"/"candidate";incumbent=root/"incumbent"
            candidate.mkdir(parents=True);incumbent.mkdir();(candidate/"candidate.py").write_text("x=1\n",encoding="utf-8");(incumbent/"candidate.py").write_text("x=1\n",encoding="utf-8");(campaign/"memory").mkdir();(campaign/"campaign.json").write_text(json.dumps({"campaign_id":"fixture"}),encoding="utf-8");(campaign/"frontier.json").write_text("{}",encoding="utf-8");(lab/"knowledge").mkdir()
            events=EventStore(lab/"runtime"/"events.jsonl");runner=LongHorizonRunner(campaign_dir=campaign,candidate_root=candidate,incumbent=incumbent,session=EmptyDiagnosticSession(),evaluator=DiagnosticEvaluator(),events=events,budget={"max_experiments":1},workbench={"workbench_id":"fixture"},lab_root=lab)
            import lab.runtime.agent.long_horizon as module
            old=module.probe_local;module.probe_local=lambda:{"status":"READY","environment_fingerprint":"fixture"}
            try: result=runner.run()
            finally: module.probe_local=old
            self.assertEqual(result["state"],"WAITING_FOR_HUMAN");self.assertEqual(result["reason"],"INVALID_DIAGNOSTIC_PLAN")
            types=[item["type"] for item in events.replay()];self.assertNotIn("DIAGNOSTIC_ACTION_STARTED",types);self.assertNotIn("COMPILE_STARTED",types)
