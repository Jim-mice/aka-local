"""No-GPU tests for NCU profile request validation and evidence parsing."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lab.runtime.evaluators.local import RTX5060LocalEvaluator
from lab.runtime.evaluators.ncu_profile import BASIC_DIAGNOSTIC_METRICS, parsed_profile


class NcuProfileAdapterTests(unittest.TestCase):
    def test_parser_keeps_only_measured_values_and_reports_missing(self):
        worker={"shape_id":12,"token_count":1024,"hidden_size":4096,"expected_kernel_symbol":"rmsnorm_row_kernel"}
        csv='"Kernel Name","Metric Name","Metric Value"\n"void rmsnorm_row_kernel(...) ","gpu__time_duration.avg","1234"\n"void rmsnorm_row_kernel(...) ","dram__throughput.avg.pct_of_peak_sustained_elapsed","42.0"\n'
        parsed=parsed_profile(worker,csv,static_evidence={"registers_per_thread":30,"shared_memory_per_block":1056})
        self.assertEqual(parsed["shape_id"],12)
        self.assertEqual(parsed["resources"]["registers_per_thread"],30)
        self.assertEqual(parsed["kernel"]["duration"],"1234")
        self.assertIn("sm__warps_active.avg.pct_of_peak_sustained_active",parsed["capability_missing"])
        self.assertEqual(BASIC_DIAGNOSTIC_METRICS[0],"gpu__time_duration.avg")

    def test_profile_request_is_limited_to_six_shapes(self):
        with tempfile.TemporaryDirectory(prefix="aka-profile-") as temp:
            root=Path(temp);candidate=root/"candidate";incumbent=root/"incumbent";candidate.mkdir();incumbent.mkdir();(candidate/"candidate.py").write_text("",encoding="utf-8");(incumbent/"candidate.py").write_text("",encoding="utf-8")
            adapter=RTX5060LocalEvaluator(root,candidate=candidate,incumbent=incumbent,evidence_dir=root/"evidence",dry_run=True)
            adapter.profiler_capabilities=lambda:{"pass":True,"status":"AVAILABLE"}
            result=adapter.profile(shape_ids=[1,2,3,4,5,6,7])
            self.assertFalse(result["pass"])
            self.assertEqual(result["reason"],"PROFILE_REQUEST_TOO_LARGE")

    def test_kernel_not_found_is_not_promoted_to_performance_failure(self):
        parsed=parsed_profile({"shape_id":1,"expected_kernel_symbol":"rmsnorm_row_kernel"},'"Metric Name","Metric Value"\n"gpu__time_duration.avg","100"\n')
        self.assertEqual(parsed["kind"],"PROFILE_MEASUREMENT")
        self.assertIn("launch/API overhead requires Nsight Systems; kernel duration is not end-to-end launch overhead",parsed["capability_missing"])
