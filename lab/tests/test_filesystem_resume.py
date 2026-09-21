"""Filesystem-only resume planning; never starts Luna or a GPU evaluator."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from lab.core.controller import LabController
from lab.core.events import EventStore
from lab.runtime.agent.long_horizon import LongHorizonRunner


class _DryResumeController(LabController):
    def __init__(self, root):
        super().__init__(root); self.started=[]
    def probe_local(self):
        if self.workbench is None:self.configure_local("rms_norm_train","cuda_cpp","fixture")
        return {"status":"READY","environment_fingerprint":"fixture-fingerprint"}
    def start_optimization(self, **kwargs):
        self.started.append(kwargs)
        return {"started":True,"dry":True,**kwargs}


class FilesystemResumeTests(unittest.TestCase):
    def test_filesystem_recovery_reuses_e0004_and_preserves_pending_directive(self):
        root=Path(tempfile.mkdtemp(prefix="aka-filesystem-resume-"))
        try:
            lab=root/"lab";campaign=lab/"campaigns"/"fixture";episode=campaign/"episodes"/"e0004";candidate=episode/"candidate";incumbent=root/"ops"/"rms_norm_v2"
            candidate.mkdir(parents=True);incumbent.mkdir(parents=True);(candidate/"candidate.py").write_text("x=1\n",encoding="utf-8");(incumbent/"candidate.py").write_text("x=1\n",encoding="utf-8");(campaign/"memory").mkdir();(lab/"knowledge").mkdir(parents=True)
            (campaign/"campaign.json").write_text(json.dumps({"campaign_id":"fixture","operator_id":"rms_norm_train","platform_id":"rtx5060_laptop_sm120","backend_id":"cuda_cpp","incumbent":{"source":"ops/rms_norm_v2","read_only":True}}),encoding="utf-8")
            (episode/"episode.json").write_text(json.dumps({"episode_id":"e0004","workbench_id":"fixture-wb"}),encoding="utf-8")
            (episode/"live.json").write_text(json.dumps({"state":"WAITING_FOR_HUMAN","reason":"AGENT_BLOCKED","environment_fingerprint":"fixture-fingerprint"}),encoding="utf-8")
            (episode/"journal.jsonl").write_text("\n".join(json.dumps({"experiment_id":f"e0004-x00{i}","decision":"BLOCKED"}) for i in (1,2))+"\n",encoding="utf-8")
            events=EventStore(lab/"runtime"/"events.jsonl");directive=events.append("HUMAN_DIRECTIVE",{"text":"formal x003 directive"})
            draft={"campaign_id":"fixture","operator_id":"rms_norm_train","platform_id":"rtx5060_laptop_sm120","backend_id":"cuda_cpp","working_directory":str(candidate),"workbench_id":"fixture-wb"}
            ctrl=_DryResumeController(root);plan=ctrl.filesystem_recovery_plan(draft)
            self.assertEqual(plan["episode_id"],"e0004");self.assertEqual(plan["completed_experiments"],2);self.assertEqual(plan["next_experiment_id"],"e0004-x003");self.assertEqual([x["directive_id"] for x in plan["active_directives"]],[directive["event_id"]])
            result=ctrl.resume_optimization(draft)
            self.assertTrue(result["dry"]);self.assertEqual(ctrl.started[0]["candidate_root"],str(candidate));self.assertEqual(ctrl.started[0]["run_id"],None);self.assertFalse((campaign/"episodes"/"e0005").exists())
            types=[event["type"] for event in ctrl.events.replay()];self.assertIn("RECOVERY_APPROVED",types);self.assertIn("ENVIRONMENT_REVALIDATED",types)
            self.assertFalse(any(event["type"]=="HUMAN_DIRECTIVE_STATUS" and (event.get("payload") or {}).get("target_event_id")==directive["event_id"] for event in ctrl.events.replay()))
        finally:shutil.rmtree(root)

    def test_same_process_resume_uses_existing_config(self):
        root=Path(tempfile.mkdtemp(prefix="aka-same-process-resume-"))
        try:
            ctrl=_DryResumeController(root);ctrl.run_config={"campaign_id":"fixture","candidate_root":"C:/existing","incumbent":"C:/incumbent","budget":{"max_experiments":5}}
            result=ctrl.resume_optimization()
            self.assertTrue(result["dry"]);self.assertEqual(ctrl.started[0]["candidate_root"],"C:/existing")
        finally:shutil.rmtree(root)

    def test_filesystem_resume_blocks_environment_mismatch_before_start(self):
        root=Path(tempfile.mkdtemp(prefix="aka-filesystem-mismatch-"))
        try:
            lab=root/"lab";campaign=lab/"campaigns"/"fixture";episode=campaign/"episodes"/"e0004";candidate=episode/"candidate";incumbent=root/"ops"/"rms_norm_v2"
            candidate.mkdir(parents=True);incumbent.mkdir(parents=True);(candidate/"candidate.py").write_text("x=1\n",encoding="utf-8");(incumbent/"candidate.py").write_text("x=1\n",encoding="utf-8");(campaign/"memory").mkdir();(lab/"knowledge").mkdir(parents=True)
            (campaign/"campaign.json").write_text(json.dumps({"campaign_id":"fixture","operator_id":"rms_norm_train","platform_id":"rtx5060_laptop_sm120","backend_id":"cuda_cpp","incumbent":{"source":"ops/rms_norm_v2"}}),encoding="utf-8")
            (episode/"episode.json").write_text(json.dumps({"episode_id":"e0004"}),encoding="utf-8");(episode/"journal.jsonl").write_text(json.dumps({"experiment_id":"e0004-x001"})+"\n",encoding="utf-8");(episode/"live.json").write_text(json.dumps({"state":"WAITING_FOR_HUMAN","environment_fingerprint":"different"}),encoding="utf-8")
            draft={"campaign_id":"fixture","operator_id":"rms_norm_train","platform_id":"rtx5060_laptop_sm120","backend_id":"cuda_cpp","working_directory":str(candidate)};ctrl=_DryResumeController(root)
            with self.assertRaisesRegex(RuntimeError,"ENVIRONMENT_CHANGED"):ctrl.resume_optimization(draft)
            self.assertEqual(ctrl.started,[])
        finally:shutil.rmtree(root)

    def test_failed_inflight_run_is_reconciled_as_recoverable_x001(self):
        root=Path(tempfile.mkdtemp(prefix="aka-inflight-diagnostic-"))
        try:
            lab=root/"lab";campaign=lab/"campaigns"/"fixture";episode=campaign/"episodes"/"e0006";candidate=episode/"candidate";incumbent=root/"ops"/"rms_norm_v2"
            candidate.mkdir(parents=True);incumbent.mkdir(parents=True);(candidate/"candidate.py").write_text("x=1\n",encoding="utf-8");(incumbent/"candidate.py").write_text("x=1\n",encoding="utf-8");(campaign/"memory").mkdir();(lab/"knowledge").mkdir(parents=True)
            (campaign/"campaign.json").write_text(json.dumps({"campaign_id":"fixture","operator_id":"rms_norm_train","platform_id":"rtx5060_laptop_sm120","backend_id":"cuda_cpp","incumbent":{"source":"ops/rms_norm_v2"}}),encoding="utf-8")
            (episode/"live.json").write_text(json.dumps({"state":"ANALYZING_INCUMBENT","environment_fingerprint":"fixture-fingerprint"}),encoding="utf-8")
            run=lab/"runtime"/"runs"/"failed-run";run.mkdir(parents=True);(run/"run.json").write_text(json.dumps({"run_id":"failed-run","campaign_id":"fixture","episode_id":"e0006"}),encoding="utf-8");(run/"resume_manifest.json").write_text(json.dumps({"run_id":"failed-run","state":"FAILED","last_safe_point":"ANALYZING_INCUMBENT"}),encoding="utf-8")
            ctrl=_DryResumeController(root);draft={"campaign_id":"fixture","operator_id":"rms_norm_train","platform_id":"rtx5060_laptop_sm120","backend_id":"cuda_cpp","working_directory":str(candidate)}
            reconciled=ctrl.reconcile_episode_state(draft);self.assertEqual(reconciled["state"],"STOPPED_RECOVERABLE");self.assertEqual(reconciled["last_safe_point"],"AFTER_AGENT_PLAN_INVALID")
            plan=ctrl.filesystem_recovery_plan(draft);self.assertEqual(plan["next_experiment_id"],"e0006-x001")
        finally:shutil.rmtree(root)

    def test_directive_becomes_consumed_only_after_x003_valid_plan_is_persisted(self):
        root=Path(tempfile.mkdtemp(prefix="aka-directive-consume-"))
        try:
            lab=root/"lab";campaign=lab/"campaigns"/"fixture";episode=campaign/"episodes"/"e0004";candidate=episode/"candidate";incumbent=root/"ops"/"rms_norm_v2"
            candidate.mkdir(parents=True);incumbent.mkdir(parents=True);(candidate/"candidate.py").write_text("x=1\n",encoding="utf-8");(incumbent/"candidate.py").write_text("x=1\n",encoding="utf-8");(campaign/"memory").mkdir();(lab/"knowledge").mkdir(parents=True)
            (campaign/"campaign.json").write_text(json.dumps({"campaign_id":"fixture","operator_id":"rms_norm_train","platform_id":"rtx5060_laptop_sm120","backend_id":"cuda_cpp","incumbent":{"source":"ops/rms_norm_v2"}}),encoding="utf-8")
            (episode/"live.json").write_text(json.dumps({"state":"WAITING_FOR_HUMAN"}),encoding="utf-8");(episode/"journal.jsonl").write_text("\n".join(json.dumps({"experiment_id":f"e0004-x00{i}"}) for i in (1,2))+"\n",encoding="utf-8")
            events=EventStore(lab/"runtime"/"events.jsonl");directive=events.append("HUMAN_DIRECTIVE",{"text":"formal x003 directive"})
            class _PlanSession:
                def run_experiment_turn(self, prompt, context):
                    payload={"experiment_kind":"DIAGNOSTIC","diagnostic_actions":[{"type":"inspect_shape_metadata"}],"supporting_evidence_ids":[],"analysis_summary":"a","bottleneck":"b","evidence_summary":"e","hypothesis":{"question":"q"},"planned_change":"none","expected_risk":"none","decision_request":"reject_and_continue","next_direction":"next"}
                    return type("Reply",(),{"final_response":"```json\n"+json.dumps(payload)+"\n```"})()
            runner=LongHorizonRunner(campaign_dir=campaign,candidate_root=candidate,incumbent=incumbent,session=_PlanSession(),evaluator=None,events=events,workbench={"workbench_id":"fixture"},lab_root=lab)
            context=runner._context(3,{"snapshot_hash":"knowledge-test"})
            self.assertEqual([x["directive_id"] for x in context["active_human_directives"]],[directive["event_id"]]);self.assertEqual(context["consumed_directive_ids"],[]);self.assertTrue((episode/"context_snapshot_e003.json").exists())
            plan=runner._plan(context);runner._persist_validated_plan(3,plan,context);runner._consume_directives_at_plan(context,3)
            self.assertTrue((episode/"validated_plan_e003.json").exists())
            statuses=[x for x in events.replay() if x["type"]=="HUMAN_DIRECTIVE_STATUS"]
            self.assertEqual(statuses[-1]["payload"]["status"],"CONSUMED");self.assertEqual(statuses[-1]["payload"]["phase"],"CONSUMED_AT_PLAN")
        finally:shutil.rmtree(root)
