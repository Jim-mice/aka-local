"""CPU/mock coverage for the bounded episode plumbing; no CUDA workload."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from lab.core.events import EventStore
from lab.runtime.agent.long_horizon import LongHorizonRunner, digest
from lab.runtime.path_policy import CandidatePathPolicy


class ScriptedSession:
    model="gpt-5.6-luna"; effort="low"
    def __init__(self, root): self.root=Path(root);self.experiment=0;self.turn_contexts=[]
    def start(self): return {}
    def close(self): pass
    def send_context(self, context): self.context=context
    def run_experiment_turn(self,prompt,context):
        self.turn_contexts.append(context)
        assert context.get("context_hash"),"formal model call lacked context hash"
        assert context.get("workbench") is not None,"formal model call lacked workbench"
        assert context.get("knowledge") is not None,"formal model call lacked knowledge"
        assert context.get("incumbent"),"formal model call lacked incumbent"
        assert context.get("current_episode") is not None,"formal model call lacked active episode"
        assert context.get("recent_canonical_experiments") is not None,"formal model call lacked canonical history section"
        return self.ask(prompt)
    def ask(self, prompt):
        if "Implement this plan" not in prompt and "Mechanical evidence" not in prompt:
            self.experiment+=1;index=self.experiment
            return type("R",(),{"final_response":"```json\n"+json.dumps({"analysis_summary":f"fixture {index}","bottleneck":"fixture","evidence_summary":"fixture","hypothesis":{"question":"q","claim":"c","rationale":"r","expected_effect":"e","support_condition":"s","refute_condition":"f"},"planned_change":"fixture edit","expected_risk":"none","decision_request":"ready_for_gate","next_direction":"next"})+"\n```"})()
        index=self.experiment
        if "Implement this plan" in prompt:
            (self.root/f"fixture_{index}.txt").write_text(str(index),encoding="utf-8")
            return type("R",(),{"final_response":"```json\n"+json.dumps({"changed_files":[f"fixture_{index}.txt"],"diff_summary":"fixture","build_expectation":"fixture"})+"\n```"})()
        decision="ready_for_gate" if index==3 else "reject_and_continue"
        return type("R",(),{"final_response":"```json\n"+json.dumps({"decision_request":decision,"result_summary":"fixture","next_direction":"next"})+"\n```"})()


class FakeEvaluator:
    def __init__(self): self.n=0
    def prepare(self): return {"pass":True}
    def compile(self): self.n+=1; return {"pass":self.n!=1,"metrics":{}}
    def check_correctness(self): return {"pass":self.n!=2,"metrics":{"tests":56}}
    def development_benchmark(self): return {"pass":True,"metrics":{"arithmetic_mean_speedup":1.05}}
    def authoritative_abba(self): return {"pass":True,"metrics":{"arithmetic_mean_speedup":1.05}}
    def robustness(self): return {"pass":True,"metrics":{"aggregate":{"arithmetic_mean_speedup":1.05}}}


class ExecutionPipelineTests(unittest.TestCase):
    def test_derived_pycache_does_not_change_candidate_identity(self):
        temp=Path(tempfile.mkdtemp(prefix="aka-pycache-identity-"))
        try:
            root=temp/"candidate";root.mkdir();(root/"candidate.py").write_text("x=1\n",encoding="utf-8")
            before_digest=digest(root);before_snapshot=CandidatePathPolicy.snapshot(root)
            pycache=root/"__pycache__";pycache.mkdir();(pycache/"candidate.pyc").write_bytes(b"derived bytes")
            self.assertEqual(digest(root),before_digest)
            self.assertEqual(CandidatePathPolicy.snapshot(root),before_snapshot)
        finally: shutil.rmtree(temp)
    def test_three_experiment_fixture_writes_canonical_archive_and_candidates(self):
        temp=Path(tempfile.mkdtemp(prefix="aka-lab-execution-"))
        try:
            lab=temp/"lab"; campaign=lab/"campaigns"/"fixture"; candidate=campaign/"episodes"/"e0001"/"candidate"; incumbent=temp/"incumbent"
            candidate.mkdir(parents=True);incumbent.mkdir();(candidate/"candidate.py").write_text("x=1\n",encoding="utf-8");(incumbent/"candidate.py").write_text("x=1\n",encoding="utf-8")
            (campaign/"memory").mkdir();(campaign/"campaign.json").write_text(json.dumps({"campaign_id":"fixture"}),encoding="utf-8");(campaign/"frontier.json").write_text("{}",encoding="utf-8");(lab/"knowledge").mkdir(parents=True)
            session=ScriptedSession(candidate);events=EventStore(lab/"runtime"/"events.jsonl");runner=LongHorizonRunner(campaign_dir=campaign,candidate_root=candidate,incumbent=incumbent,session=session,evaluator=FakeEvaluator(),events=events,budget={"max_experiments":3,"max_consecutive_failures":3},workbench={"workbench_id":"fixture"},lab_root=lab)
            # Avoid a hardware dependency in this CPU fixture.
            import lab.runtime.agent.long_horizon as module
            original=module.probe_local;module.probe_local=lambda:{"status":"READY","environment_fingerprint":"fixture"}
            try: result=runner.run()
            finally: module.probe_local=original
            self.assertEqual(result["state"],"PROMOTE")
            self.assertEqual(len(result["experiments"]),3)
            self.assertTrue((campaign/"memory"/"v000.json").exists())
            self.assertTrue(list((campaign/"incumbent").glob("*/candidate.py")))
            self.assertTrue((lab/"experiment_archive"/"e0001-x003"/"report_zh-CN.md").exists())
            lineage_path=campaign/"episodes"/"e0001"/"candidate_lineage.jsonl";self.assertTrue(lineage_path.exists())
            lineage=[json.loads(line) for line in lineage_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(lineage),3);self.assertEqual(len({item["candidate_id"] for item in lineage}),3)
            self.assertTrue(all(item["parent_candidate_id"] and item["incumbent_id"] and item["source_hash"] and item["hypothesis_id"] for item in lineage))
            self.assertEqual(lineage[1]["parent_candidate_id"],lineage[0]["candidate_id"]);self.assertEqual(lineage[2]["parent_candidate_id"],lineage[1]["candidate_id"])
            journal_records=[json.loads(line) for line in (campaign/"episodes"/"e0001"/"journal.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(record.get("candidate_lineage",{}).get("candidate_id") for record in journal_records))
            self.assertTrue(list((lab/"knowledge"/"pending").glob("*.json")))
            types=[x["type"] for x in events.replay()]
            self.assertLess(types.index("KNOWLEDGE_REFRESHED"),types.index("HYPOTHESIS_RECORDED"))
            self.assertTrue(session.turn_contexts)
            self.assertTrue(all(x.get("knowledge") is not None and x.get("incumbent") for x in session.turn_contexts))
        finally: shutil.rmtree(temp)
