"""CPU-only tests for the Agent Console presentation adapters."""
import unittest

from lab.gui import _console_category, _console_summary
from lab.ui_markdown import markdown_blocks


class ConsolePresentationTests(unittest.TestCase):
    def test_event_categories_are_human_facing(self):
        self.assertEqual(_console_category("KNOWLEDGE_REFRESHED", {})[0], "KNOWLEDGE")
        self.assertEqual(_console_category("HYPOTHESIS_RECORDED", {})[0], "ANALYSIS")
        self.assertEqual(_console_category("CANDIDATE_UPDATED", {})[0], "CODE")
        self.assertEqual(_console_category("HUMAN_DIRECTIVE", {})[0], "HUMAN")
        self.assertIn("知识已刷新", _console_summary("KNOWLEDGE_REFRESHED", {"knowledge_ids":["k1"]}))

    def test_diagnostic_events_are_distinct_and_human_facing(self):
        category, label = _console_category("DIAGNOSTIC_ACTION_RESULT", {"action":"repeated_per_shape_benchmark", "result":{"pass":True}})
        self.assertEqual(category, "DIAGNOSTIC")
        self.assertEqual(label, "诊断 / Evidence")
        self.assertIn("PASS", _console_summary("DIAGNOSTIC_ACTION_RESULT", {"action":"repeated_per_shape_benchmark", "result":{"pass":True}}))

    def test_markdown_blocks_keep_chinese_and_code_as_text(self):
        blocks = markdown_blocks("# 标题\n\n**粗体** 和 `warp reduction`\n\n- 项目\n\n```cpp\n__global__ void foo() {}\n```")
        self.assertIn(("h1", "标题"), blocks)
        self.assertIn(("bullet", "项目"), blocks)
        self.assertIn(("code", "__global__ void foo() {}"), blocks)
