"""Offline P0 regression tests.  They do not open SSH or start CUDA."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lab.runtime.evaluators.contract_validation import (
    contract_from_metadata,
    evaluation_fingerprint,
    load_contract_bundle,
    validate_episode_contract,
)
from lab.runtime.evaluators.qualification import ordered_shapes, qualification_config, qualify_runs, summarize_samples


ROOT = Path(__file__).resolve().parents[2]


def metadata() -> dict:
    return {
        "operator": "unit_op", "interface": "standalone_cuda", "arch": "sm_70", "dtype": "float32",
        "semantic_identifier": "unit_semantics", "required_source_marker": "// AKA_CONTRACT: x, y",
        "contract_schema": {"operator": "unit_op", "entry": "launch_kernel", "version": 1,
            "arguments": [{"name": "x", "type": "float*", "role": "input"}, {"name": "y", "type": "float*", "role": "output"}],
            "semantics": "y=x"},
    }


def evaluation() -> dict:
    return {"shapes": [[1, 4], [2, 4]], "score": "geometric_mean_speedup", "warmup": 1, "iterations": 2,
            "compiler": "nvcc", "compiler_flags": "-O2", "correctness_tolerance": 0.001}


class CanonicalContractTests(unittest.TestCase):
    def test_key_order_does_not_change_hash(self):
        left = contract_from_metadata(metadata(), evaluation())
        right = contract_from_metadata(json.loads(json.dumps(metadata(), sort_keys=True)), json.loads(json.dumps(evaluation(), sort_keys=True)))
        self.assertEqual(left.semantic_contract_sha256, right.semantic_contract_sha256)
        self.assertEqual(evaluation_fingerprint(evaluation()), evaluation_fingerprint(json.loads(json.dumps(evaluation(), sort_keys=True))))

    def test_semantic_fields_change_hash(self):
        baseline = contract_from_metadata(metadata(), evaluation()).semantic_contract_sha256
        changed_marker = metadata(); changed_marker["required_source_marker"] = "// AKA_CONTRACT: changed"
        changed_tolerance = evaluation(); changed_tolerance["correctness_tolerance"] = 0.01
        changed_shapes = evaluation(); changed_shapes["shapes"] = [[1, 8]]
        changed_arguments = metadata(); changed_arguments["contract_schema"]["arguments"].reverse()
        for meta, eval_data in ((changed_marker, evaluation()), (metadata(), changed_tolerance), (metadata(), changed_shapes), (changed_arguments, evaluation())):
            self.assertNotEqual(baseline, contract_from_metadata(meta, eval_data).semantic_contract_sha256)

    def test_episode_mismatch_is_rejected_before_evaluator(self):
        contract = contract_from_metadata(metadata(), evaluation())
        with tempfile.TemporaryDirectory() as raw:
            ep = Path(raw)
            (ep / "candidate.cu").write_text('// AKA_CONTRACT: x, y\nextern "C" void launch_kernel(float* x, float* y) {}\n', encoding="utf-8")
            (ep / "hypothesis.json").write_text(json.dumps({"operator": "unit_op", "contract_version": 1,
                "semantic_contract_sha256": "wrong", "interface": {"entry": "launch_kernel", "arguments": ["x", "y"]}}), encoding="utf-8")
            manifest = {"operator": "unit_op", "semantic_contract_sha256": contract.semantic_contract_sha256,
                        "interface_entry": "launch_kernel", "interface_arguments": contract.arguments}
            result = validate_episode_contract(ep, contract, manifest)
            self.assertEqual(result["status"], "REJECT_CONTRACT")
            self.assertTrue(any(row["field"] == "hypothesis.semantic_contract_sha256" for row in result["failures"]))

    def test_campaign_rejects_mismatch_without_calling_evaluator(self):
        from lab.runtime.evaluators import phase8d
        from lab.runtime.evaluators import remote_v100_campaign as campaign
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "operators" / "unit_op").mkdir(parents=True)
            (root / "config" / "environments" / "v100_sm70").mkdir(parents=True)
            (root / "operators" / "unit_op" / "metadata.json").write_text(json.dumps(metadata()), encoding="utf-8")
            (root / "config" / "environments" / "v100_sm70" / "evaluation.json").write_text(json.dumps(evaluation()), encoding="utf-8")
            ep = root / "campaigns" / "unit_op" / "episode_1"; ep.mkdir(parents=True)
            (ep / "candidate.cu").write_text('// AKA_CONTRACT: x, y\nextern "C" void launch_kernel(float* x, float* y) {}\n', encoding="utf-8")
            (ep / "hypothesis.json").write_text(json.dumps({"operator": "unit_op", "contract_version": 1,
                "semantic_contract_sha256": "wrong", "interface": {"entry": "launch_kernel", "arguments": ["x", "y"]}}), encoding="utf-8")
            calls = []
            with patch.object(campaign, "ROOT", root), patch.object(phase8d, "ROOT", root), patch.object(campaign, "evaluate_v100", side_effect=lambda *args, **kwargs: calls.append(args)):
                campaign.run_evaluation_for_episode(ep, 1, "unit_op", ["1,4", "2,4"], with_profile=False)
            self.assertEqual(calls, [])
            self.assertEqual(json.loads((ep / "decision.json").read_text(encoding="utf-8"))["decision"], "REJECT_CONTRACT")

    def test_campaign_does_not_rewrite_legacy_episode_artifacts(self):
        from lab.runtime.evaluators import remote_v100_campaign as campaign
        with tempfile.TemporaryDirectory() as raw:
            ep = Path(raw)
            candidate = ep / "candidate.cu"
            manifest = ep / "episode_manifest.json"
            candidate.write_text("// historical candidate\\n", encoding="utf-8")
            manifest.write_text(json.dumps({"contract_hash": "legacy-shapes-only"}), encoding="utf-8")
            before = {path.name: path.read_bytes() for path in (candidate, manifest)}
            with patch.object(campaign, "evaluate_v100", side_effect=AssertionError("must not run")):
                campaign.run_evaluation_for_episode(ep, 28, "unit_op", ["1,4"], with_profile=False)
            self.assertEqual(before, {path.name: path.read_bytes() for path in (candidate, manifest)})
            self.assertFalse((ep / "result.json").exists())
            self.assertFalse((ep / "decision.json").exists())

    def test_direct_remote_entry_rejects_bad_source_before_ssh(self):
        from lab.runtime.evaluators import remote_v100
        with tempfile.TemporaryDirectory() as raw:
            candidate = Path(raw) / "candidate.cu"
            candidate.write_text("// deliberately missing contract marker\\n", encoding="utf-8")
            with patch.object(remote_v100, "_ssh_client", side_effect=AssertionError("must not connect")):
                result = remote_v100.evaluate_candidate_multi_shape(str(candidate), ["1,4096"], with_profile=False)
            self.assertEqual(result["contract_validation"]["status"], "REJECT_CONTRACT")
            self.assertIn("REJECT_CONTRACT", result["error"])


class QualificationTests(unittest.TestCase):
    def test_bimodal_series_is_not_qualified_by_cv_alone(self):
        config = qualification_config(evaluation())
        summary = summarize_samples([9.0, 9.1, 22.0, 9.0, 22.1], config)
        self.assertTrue(summary["bimodal"])
        self.assertFalse(summary["stable"])

    def test_stable_series_qualifies(self):
        config = qualification_config(evaluation())
        samples = [9.0, 9.1, 9.0, 9.05, 9.1]
        summary = summarize_samples(samples, config)
        self.assertTrue(summary["stable"])
        repeated = [{"aggregate_score": 2.0, "runs": [{"run_id": str(i)}], "shapes": [{"shape": "1,4", "latency_us": value}]} for i, value in enumerate(samples)]
        self.assertEqual(qualify_runs(repeated, config)["status"], "QUALIFIED_ACCEPT")
        self.assertEqual(ordered_shapes(["a", "b", "c"], 1, config), ["b", "c", "a"])


class FakeEvaluator:
    instances: list["FakeEvaluator"] = []

    def __init__(self, candidate_path, shape, operator):
        self.shape, self.operator = shape, operator
        self.run_id = f"run-{len(self.instances)}"
        self.job_id = f"job-{len(self.instances)}"
        self.candidate_hash = "candidate-hash"
        self.instances.append(self)

    def prepare(self): return True
    def compile(self): return True
    def check_correctness(self): return True
    def static_evidence(self): return {"compile": True, "max_error": 0.0, "run_id": self.run_id, "job_id": self.job_id}
    def benchmark(self): return {"latency_us": 9.0, "speedup_vs_torch": 2.0, "speedup_vs_naive": 2.0, "run_id": self.run_id, "job_id": self.job_id, "candidate_hash": self.candidate_hash, "shape": self.shape, "operator": self.operator}
    def cleanup(self): return None
    def get_error(self): return "fake error"


class EvaluatorProvenanceTests(unittest.TestCase):
    def test_four_shapes_create_four_measured_jobs_and_share_primary_provenance(self):
        from lab.runtime.evaluators.remote_v100 import evaluate_candidate_multi_shape
        FakeEvaluator.instances = []
        result = evaluate_candidate_multi_shape("candidate.cu", ["1,4096", "4,4096", "8,4096", "32,4096"], with_profile=False, evaluator_factory=FakeEvaluator)
        self.assertEqual(len(FakeEvaluator.instances), 4)
        self.assertEqual(sum(item.shape == "1,4096" for item in FakeEvaluator.instances), 1)
        self.assertEqual(result["runs"][0]["run_id"], result["shapes"][0]["run_id"])
        self.assertEqual(result["runs"][0]["run_id"], result["evidence"]["run_id"])


class LauncherIsolationTests(unittest.TestCase):
    def test_foreign_pythonpath_cannot_merge_lab_namespace(self):
        with tempfile.TemporaryDirectory() as raw:
            foreign = Path(raw) / "foreign" / "lab"
            foreign.mkdir(parents=True)
            (foreign / "__init__.py").write_text("ORIGIN='foreign'\n", encoding="utf-8")
            env = dict(os.environ)
            env["PYTHONPATH"] = str(foreign.parent)
            command = [sys.executable, str(ROOT / "scripts" / "run_lab.py"), "--module", "lab.cli", "status"]
            completed = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertNotIn("foreign", completed.stdout + completed.stderr)

    def test_preloaded_foreign_lab_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            foreign = Path(raw) / "foreign" / "lab"
            foreign.mkdir(parents=True)
            (foreign / "__init__.py").write_text("ORIGIN='foreign'\n", encoding="utf-8")
            code = ("import sys, runpy; sys.path.insert(0, r'%s'); import lab; "
                    "sys.argv=['run_lab.py','--module','lab.cli','status']; runpy.run_path(r'%s', run_name='__main__')") % (foreign.parent, ROOT / "scripts" / "run_lab.py")
            completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("REFUSING_TO_RUN", completed.stderr + completed.stdout)


if __name__ == "__main__":
    unittest.main()
