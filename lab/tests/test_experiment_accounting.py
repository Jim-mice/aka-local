"""Framework failure accounting is test-only and never runs an evaluator."""
from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from lab.core.experiment_accounting import account_experiments
from lab.core.context_builder import build_authoritative_context
from lab.core.events import EventStore


class ExperimentAccountingTests(unittest.TestCase):
    def test_framework_failure_is_excluded_then_retries_same_logical_x001(self):
        records=[{"experiment_id":"e0006-x001","experiment_kind":"DIAGNOSTIC","decision":"BLOCKED"}]
        events=[{"type":"EXPERIMENT_RECLASSIFIED","payload":{"experiment_id":"e0006-x001","new_status":"INVALID_FRAMEWORK_FAILURE","counts_against_experiment_budget":False,"optimization_evidence":False,"gpu_evidence":False}}]
        accounting=account_experiments(records,events)
        self.assertEqual(accounting["valid_experiment_count"],0)
        self.assertEqual(accounting["framework_attempt_count"],1)
        self.assertEqual(accounting["next_slot"],{"logical_experiment_id":"e0006-x001","attempt":2,"retry":True,"blocked":False})

    def test_second_framework_failure_blocks_without_consuming_budget(self):
        records=[{"experiment_id":"e0006-x001","decision":"BLOCKED"},{"experiment_id":"e0006-x001-r1","logical_experiment_id":"e0006-x001","attempt":2,"decision":"INVALID_FRAMEWORK_FAILURE"}]
        events=[{"type":"EXPERIMENT_RECLASSIFIED","payload":{"experiment_id":record["experiment_id"],"counts_against_experiment_budget":False}} for record in records]
        accounting=account_experiments(records,events)
        self.assertEqual(accounting["valid_experiment_count"],0)
        self.assertTrue(accounting["next_slot"]["blocked"])
        self.assertEqual(accounting["next_slot"]["reason"],"FRAMEWORK_RETRY_LIMIT_REACHED")

    def test_valid_retry_advances_to_x002(self):
        records=[{"experiment_id":"e0006-x001","decision":"BLOCKED"},{"experiment_id":"e0006-x001-r1","logical_experiment_id":"e0006-x001","attempt":2,"decision":"REJECT_AND_CONTINUE"}]
        events=[{"type":"EXPERIMENT_RECLASSIFIED","payload":{"experiment_id":"e0006-x001","counts_against_experiment_budget":False}}]
        accounting=account_experiments(records,events)
        self.assertEqual(accounting["valid_experiment_count"],1)
        self.assertEqual(accounting["next_slot"]["logical_experiment_id"],"e0006-x002")

    def test_consumed_governing_directive_is_visible_for_framework_retry(self):
        with tempfile.TemporaryDirectory(prefix="aka-governing-directive-") as temp:
            root=Path(temp);lab=root/"lab";campaign=lab/"campaigns"/"fixture";episode=campaign/"episodes"/"e0006";candidate=episode/"candidate";incumbent=root/"ops"/"rms_norm_v2"
            candidate.mkdir(parents=True);incumbent.mkdir(parents=True);(candidate/"candidate.py").write_text("x=1\n",encoding="utf-8");(incumbent/"candidate.py").write_text("x=1\n",encoding="utf-8");(lab/"knowledge").mkdir(parents=True);(campaign/"memory").mkdir();(campaign/"campaign.json").write_text(json.dumps({"operator_id":"rms_norm_train","platform_id":"rtx5060_laptop_sm120","backend_id":"cuda_cpp","incumbent":{"source":"ops/rms_norm_v2"}}),encoding="utf-8");(episode/"journal.jsonl").write_text(json.dumps({"experiment_id":"e0006-x001","decision":"BLOCKED"})+"\n",encoding="utf-8");(episode/"live.json").write_text(json.dumps({"state":"WAITING_FOR_HUMAN"}),encoding="utf-8")
            events=EventStore(lab/"runtime"/"events.jsonl");directive=events.append("HUMAN_DIRECTIVE",{"text":"govern x001"});events.append("HUMAN_DIRECTIVE_STATUS",{"target_event_id":directive["event_id"],"status":"CONSUMED"});events.append("EXPERIMENT_RECLASSIFIED",{"experiment_id":"e0006-x001","counts_against_experiment_budget":False,"governing_directive_ids":[directive["event_id"]]})
            context=build_authoritative_context(lab,campaign,candidate,events=events)
            self.assertEqual(context["active_human_directives"],[])
            self.assertEqual([item["text"] for item in context["governing_directives"]],["govern x001"])
