"""CPU-only coverage for active-episode context precedence and reconciliation."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from lab.core.context_builder import build_authoritative_context
from lab.core.events import EventStore


class ActiveEpisodeContextTests(unittest.TestCase):
    def test_journal_precedes_history_and_expired_test_directive_is_filtered(self):
        root = Path(tempfile.mkdtemp(prefix="aka-active-context-"))
        try:
            lab = root / "lab"; campaign = lab / "campaigns" / "rms"; episode = campaign / "episodes" / "e0004"; candidate = episode / "candidate"; incumbent = root / "ops" / "rms_norm_v2"
            candidate.mkdir(parents=True); incumbent.mkdir(parents=True); (candidate / "kernel.cu").write_text("candidate\n", encoding="utf-8"); (incumbent / "kernel.cu").write_text("incumbent\n", encoding="utf-8")
            (campaign / "memory").mkdir(); (lab / "knowledge").mkdir(parents=True)
            (campaign / "campaign.json").write_text(json.dumps({"campaign_id":"rms","operator_id":"rms_norm_train","platform_id":"rtx5060_laptop_sm120","backend_id":"cuda_cpp","incumbent":{"source":"ops/rms_norm_v2","read_only":True}}), encoding="utf-8")
            (episode / "journal.jsonl").write_text("\n".join(json.dumps({"experiment_id":f"e0004-x00{i}","decision":"BLOCKED" if i == 2 else "REJECT_AND_CONTINUE","plan":{"hypothesis":{"claim":f"claim {i}"}},"changed_files":["kernel.cu"]}) for i in (1,2))+"\n", encoding="utf-8")
            (episode / "live.json").write_text(json.dumps({"state":"WAITING_FOR_HUMAN","reason":"AGENT_BLOCKED"}), encoding="utf-8")
            events = EventStore(lab / "runtime" / "events.jsonl")
            directive = events.append("HUMAN_DIRECTIVE", {"text":"test directive: do not start optimization"})
            events.append("HUMAN_DIRECTIVE_STATUS", {"target_event_id":directive["event_id"],"status":"TEST_ONLY_EXPIRED"})
            context = build_authoritative_context(lab, campaign, candidate, workbench={"workbench_id":"fixture"}, events=events)
            self.assertEqual(context["current_episode"]["episode_id"], "e0004")
            self.assertEqual(context["current_episode"]["experiment_count"], 2)
            self.assertEqual([item["experiment_id"] for item in context["current_episode"]["experiments"]], ["e0004-x001", "e0004-x002"])
            self.assertTrue(context["current_episode"]["candidate"]["changed"])
            self.assertEqual(context["active_human_directives"], [])
            self.assertTrue(context["current_episode"]["resume_possible"])
        finally:
            shutil.rmtree(root)
