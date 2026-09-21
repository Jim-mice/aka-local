import json
import tempfile
import unittest
from pathlib import Path

from lab.core.external_retrieval import retrieve_knowledge
from lab.knowledge.importers.external import markdown_sections


def write_card(root, name, value):
    path=Path(root)/"knowledge"/f"{name}.json";path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding="utf-8")


class ExternalKnowledgeTests(unittest.TestCase):
    def test_semantic_markdown_sections_keep_heading_context(self):
        parts=markdown_sections("# Performance\ntext\n## Occupancy\nbody", "guide.md")
        self.assertEqual(parts[1]["heading_path"],["Performance","Occupancy"])
        self.assertIn("body",parts[1]["content"])

    def test_empirical_precedes_external_and_architecture_scope_orders_guides(self):
        with tempfile.TemporaryDirectory() as temp:
            empirical={"id":"empirical","type":"OBSERVATION","statement":"occupancy measured locally","scope":{"operators":["rms_norm_train"],"platforms":["rtx5060_laptop_sm120"],"backends":["cuda_cpp"]},"evidence_for":["e1"]}
            blackwell={"id":"blackwell","type":"EXTERNAL_REFERENCE","knowledge_kind":"EXTERNAL_REFERENCE","title":"Blackwell occupancy","statement":"occupancy background","scope":{"architecture":["Blackwell"],"backend":["CUDA"]},"source":{"authority":"OFFICIAL_VENDOR","source_id":"nvidia_blackwell_tuning"}}
            volta={"id":"volta","type":"EXTERNAL_REFERENCE","knowledge_kind":"EXTERNAL_REFERENCE","title":"Volta occupancy","statement":"occupancy background","scope":{"architecture":["Volta"],"backend":["CUDA"]},"source":{"authority":"OFFICIAL_VENDOR","source_id":"nvidia_volta_tuning"}}
            write_card(temp,"empirical",empirical);write_card(temp,"blackwell",blackwell);write_card(temp,"volta",volta)
            rtx=retrieve_knowledge(temp,query="occupancy",operator_id="rms_norm_train",platform_id="rtx5060_laptop_sm120",backend_id="cuda_cpp")
            self.assertEqual([x["id"] for x in rtx][:2],["empirical","blackwell"])
            v100=retrieve_knowledge(temp,query="occupancy",platform_id="v100_sm70",backend_id="cuda_cpp")
            self.assertEqual(v100[0]["id"],"empirical")
            self.assertEqual(v100[1]["id"],"volta")

    def test_external_reference_never_changes_empirical_type(self):
        external={"id":"reference","type":"EXTERNAL_REFERENCE","knowledge_kind":"EXTERNAL_REFERENCE","statement":"register pressure background","source":{"authority":"OFFICIAL_VENDOR"}}
        self.assertEqual(external["type"],"EXTERNAL_REFERENCE")
        self.assertNotIn(external["type"],{"SUPPORTED_RULE","ANTI_STRATEGY","HARDWARE_FACT"})


if __name__=="__main__":
    unittest.main()
