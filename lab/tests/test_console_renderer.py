"""Regression coverage for immutable legacy Agent Console event payloads."""
from __future__ import annotations

import unittest

from lab.gui import _console_summary, _diagnostic_action_types


class DiagnosticActionRendererTests(unittest.TestCase):
    def test_legacy_string_actions_render_without_exception(self):
        payload={"actions":["repeated_per_shape_benchmark","static_evidence","compare_regimes"]}
        summary=_console_summary("DIAGNOSTIC_STARTED",payload)
        self.assertEqual(summary,"诊断实验开始：repeated_per_shape_benchmark, static_evidence, compare_regimes")

    def test_new_object_actions_render_without_exception(self):
        payload={"actions":[{"type":"repeated_per_shape_benchmark","batches":5},{"type":"static_evidence"}]}
        summary=_console_summary("DIAGNOSTIC_STARTED",payload)
        self.assertEqual(summary,"诊断实验开始：repeated_per_shape_benchmark, static_evidence")

    def test_mixed_legacy_actions_have_safe_unknown_fallback(self):
        actions=["static_evidence",{"type":"profiler","shape_ids":[8]},None,123]
        self.assertEqual(_diagnostic_action_types(actions),["static_evidence","profiler","UNKNOWN","UNKNOWN"])
        self.assertEqual(_console_summary("DIAGNOSTIC_STARTED",{"actions":actions}),"诊断实验开始：static_evidence, profiler, UNKNOWN, UNKNOWN")

