"""Filesystem-only rehydration regression fixture for e0006-style failures."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lab.core.context_builder import build_authoritative_context
from lab.core.controller import LabController
from lab.core.events import EventStore
from lab.core.persistence import read_json
import lab.gui as gui_module


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_fixture(root: Path):
    lab=root/"lab"; campaign=lab/"campaigns"/"rms_norm_train__rtx5060_sm120__cuda_cpp"
    episode=campaign/"episodes"/"e0006"; candidate=episode/"candidate"; incumbent=root/"ops"/"rms_norm_v2"
    candidate.mkdir(parents=True); incumbent.mkdir(parents=True)
    (candidate/"candidate.py").write_text("value = 2\n",encoding="utf-8")
    (incumbent/"candidate.py").write_text("value = 2\n",encoding="utf-8")
    (campaign/"memory").mkdir(parents=True)
    _write(campaign/"campaign.json",{"campaign_id":campaign.name,"operator_id":"rms_norm_train","platform_id":"rtx5060_laptop_sm120","backend_id":"cuda_cpp","budget":{"max_experiments":5},"incumbent":{"source":"ops/rms_norm_v2","read_only":True}})
    _write(episode/"episode.json",{"episode_id":"e0006","workbench_id":"fixture-workbench","working_directory":str(candidate),"source_directory":str(incumbent),"state":"PREPARED"})
    (episode/"journal.jsonl").write_text(json.dumps({"experiment_id":"e0006-x001","experiment_kind":"DIAGNOSTIC","decision":"BLOCKED","candidate_hash":"identical","changed_files":[]})+"\n",encoding="utf-8")
    _write(episode/"live.json",{"state":"WAITING_FOR_HUMAN","experiments":0,"framework_attempts":1,"environment_fingerprint":"fixture-fingerprint"})
    _write(lab/"experiments"/"legacy"/"record.json",{"id":"legacy-rmsnorm","operators":["rms_norm_train"],"platforms":["rtx5060_laptop_sm120"],"backends":["cuda_cpp"]})
    _write(lab/"knowledge"/"external_references"/"imported.json",{"id":"external-imported","knowledge_kind":"EXTERNAL_REFERENCE","type":"EXTERNAL_REFERENCE","title":"Imported background","statement":"background only","scope":{"architecture":["Blackwell"]},"source":{"authority":"OFFICIAL_VENDOR"}})
    _write(lab/"knowledge_sources"/"import_manifest.json",{"task":"External Knowledge Import","episode_scope":"E0006_UNCHANGED"})
    events=EventStore(lab/"runtime"/"events.jsonl")
    directive=events.append("HUMAN_DIRECTIVE",{"text":"governing e0006 diagnostic directive"})
    events.append("HUMAN_DIRECTIVE_STATUS",{"target_event_id":directive["event_id"],"status":"CONSUMED","phase":"CONSUMED_AT_PLAN","episode_id":"e0006","experiment_id":"e0006-x001"})
    events.append("EXPERIMENT_RECLASSIFIED",{"experiment_id":"e0006-x001","old_status":"BLOCKED","new_status":"INVALID_FRAMEWORK_FAILURE","reason":"DIAGNOSTIC_ACTION_PARAMETER_SCHEMA_BUG","counts_against_experiment_budget":False,"optimization_evidence":False,"gpu_evidence":False,"governing_directive_ids":[directive["event_id"]]})
    run=lab/"runtime"/"runs"/"fixture-run"; _write(run/"run.json",{"run_id":"fixture-run","campaign_id":campaign.name,"episode_id":"e0006","candidate_root":str(candidate),"incumbent":str(incumbent)})
    _write(run/"resume_manifest.json",{"run_id":"fixture-run","state":"WAITING_FOR_HUMAN","last_safe_point":"AFTER_EXPERIMENT","retry_count":1,"resume_safe":True,"environment_fingerprint":"fixture-fingerprint"})
    _write(lab/"runtime"/"workspace_draft.json",{"operator_id":"rms_norm_train","platform_id":"rtx5060_laptop_sm120","backend_id":"cuda_cpp","campaign_id":campaign.name,"workbench_id":"fixture-workbench","working_directory":str(candidate),"workbench_state":"PREPARED"})
    return {"lab":lab,"campaign":campaign,"episode":episode,"candidate":candidate,"incumbent":incumbent,"events":events}


class FrameworkAccountingRehydrationTests(unittest.TestCase):
    def test_overlay_and_stale_prepared_metadata_rehydrate_to_recoverable_retry(self):
        with tempfile.TemporaryDirectory(prefix="aka-framework-rehydrate-") as temp:
            fixture=make_fixture(Path(temp)); episode=fixture["episode"]
            context=build_authoritative_context(fixture["lab"],fixture["campaign"],fixture["candidate"],workbench={"workbench_id":"fixture-workbench"},events=fixture["events"],operator_id="rms_norm_train",platform_id="rtx5060_laptop_sm120",backend_id="cuda_cpp")
            current=context["current_episode"]
            self.assertEqual(current["valid_experiment_count"],0)
            self.assertEqual(current["framework_attempt_count"],1)
            self.assertEqual(current["next_slot"],{"logical_experiment_id":"e0006-x001","attempt":2,"retry":True,"blocked":False})
            self.assertTrue(current["resume_possible"])
            self.assertEqual(current["candidate"]["candidate_hash"],current["candidate"]["incumbent_hash"])
            self.assertEqual(context["active_human_directives"],[])
            self.assertEqual([item["text"] for item in context["governing_directives"]],["governing e0006 diagnostic directive"])

            # A fresh controller object represents a restart: no prior Python
            # memory or run_config may be required for these facts.
            controller=LabController(Path(temp)); draft=read_json(fixture["lab"]/"runtime"/"workspace_draft.json",{}) or {}
            reconciled=controller.reconcile_episode_state(draft)
            self.assertEqual(reconciled["state"],"STOPPED_RECOVERABLE")
            self.assertEqual(reconciled["valid_experiment_count"],0)
            self.assertEqual(reconciled["framework_attempt_count"],1)
            self.assertEqual(reconciled["next_slot"]["logical_experiment_id"],"e0006-x001")
            self.assertEqual(reconciled["next_slot"]["attempt"],2)
            self.assertTrue(reconciled["resume_possible"])
            self.assertEqual(len((episode/"journal.jsonl").read_text(encoding="utf-8").splitlines()),1)
            print("FRAMEWORK_ACCOUNTING_REHYDRATION_PASS")

    def test_external_import_artifacts_are_isolated_from_episode_lifecycle(self):
        with tempfile.TemporaryDirectory(prefix="aka-knowledge-isolation-") as temp:
            fixture=make_fixture(Path(temp)); manifest=fixture["lab"]/"knowledge_sources"/"import_manifest.json"; before=_digest(manifest)
            context=build_authoritative_context(fixture["lab"],fixture["campaign"],fixture["candidate"],events=fixture["events"])
            current=context["current_episode"]
            self.assertEqual(current["valid_experiment_count"],0)
            self.assertEqual(current["framework_attempt_count"],1)
            self.assertEqual(current["next_slot"]["attempt"],2)
            self.assertEqual(_digest(manifest),before)
            self.assertTrue(context["knowledge"]["records"])
            print("KNOWLEDGE_IMPORT_EPISODE_ISOLATION_PASS")

    def test_fresh_gui_restart_renders_recoverable_buttons_without_starting_work(self):
        with tempfile.TemporaryDirectory(prefix="aka-gui-rehydrate-") as temp:
            fixture=make_fixture(Path(temp)); old_root,old_lab=gui_module.ROOT,gui_module.LAB
            gui_module.ROOT=Path(temp); gui_module.LAB=fixture["lab"]
            app=None
            try:
                with patch.object(gui_module.App,"probe_local",lambda self:None),patch.object(gui_module.App,"recovery_scan",lambda self:None):
                    app=gui_module.App()
                app.withdraw();app.overview()
                self.assertEqual(app.episode_state,"STOPPED_RECOVERABLE")
                self.assertEqual(str(app.start_button["state"]),"disabled")
                self.assertEqual(str(app.resume_button["state"]),"normal")
                app.probe={"gpu_name":"fixture GPU","status":"READY","environment_fingerprint":"fixture-fingerprint"}
                app.controller.configure_local("rms_norm_train","cuda_cpp",fixture["campaign"].name);app.refresh()
                self.assertIn("STOPPED_RECOVERABLE",app.banner["text"])
                print("GUI_FILESYSTEM_RECOVERY_PASS")
                print("RECOVERABLE_BUTTON_STATE_PASS")
            finally:
                if app is not None:app.destroy()
                gui_module.ROOT,gui_module.LAB=old_root,old_lab


if __name__=="__main__":
    unittest.main()
