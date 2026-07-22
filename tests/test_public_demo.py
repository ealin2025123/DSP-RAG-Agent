import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ["LYY_OFFLINE"] = "1"
os.environ.pop("LYY_KB_PATH", None)
sys.path.insert(0, str(ROOT / "src"))

from lyy_rag_agent import DSPRAGAgent  # noqa: E402


class PublicDemoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = DSPRAGAgent()

    def test_public_clone_uses_demo_corpus(self):
        self.assertEqual(self.agent.knowledge_mode, "demo")
        self.assertEqual(self.agent.knowledge_path.name, "documents.jsonl")

    def test_demo_retrieval_works_without_api_keys(self):
        response = self.agent.ask("Line item 有哪些演示类型？", "public-demo-test")
        context = "\n".join(item.chunk.text for item in response.retrieved)
        self.assertIn("Display", context)
        self.assertIn("Online Video", context)
        self.assertTrue(response.answer.strip())


if __name__ == "__main__":
    unittest.main()

