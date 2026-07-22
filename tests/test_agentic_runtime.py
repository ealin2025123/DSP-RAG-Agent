import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ["LYY_OFFLINE"] = "1"
os.environ.pop("LYY_KB_PATH", None)
sys.path.insert(0, str(ROOT / "src"))

from lyy_rag_agent.agentic_runtime import AgenticDSPRAGAgent  # noqa: E402


class AgenticRuntimeTest(unittest.TestCase):
    def test_state_graph_returns_grounded_answer(self):
        agent = AgenticDSPRAGAgent()
        response = agent.invoke("Line item 有哪些演示类型？", "agentic-demo", "fast")
        self.assertTrue(response.answer.strip())
        self.assertEqual(response.trace[-1]["orchestration"], "LangGraph StateGraph")
        self.assertGreaterEqual(response.trace[-1]["retrieval_attempts"], 1)

    def test_insufficient_retrieval_rewrites_with_bounded_loop(self):
        agent = AgenticDSPRAGAgent()
        response = agent.invoke("完全不存在的字段XYZABC是什么意思？", "agentic-rewrite", "fast")
        nodes = [item.get("node") for item in response.trace]
        self.assertIn("retrieval_grade", nodes)
        self.assertIn("query_rewrite", nodes)
        self.assertLessEqual(response.trace[-1]["retrieval_attempts"], 2)

    def test_freshness_query_can_use_tavily_tool(self):
        agent = AgenticDSPRAGAgent()

        class FakeClient:
            available = True

        class FakeTool:
            @staticmethod
            def invoke(args):
                return [{
                    "title": "Official demo update",
                    "url": "https://advertising.amazon.com/demo",
                    "content": "A fictional current-policy result used only by the unit test.",
                    "score": 0.9,
                }]

        agent.tavily_client = FakeClient()
        agent.tavily_tool = FakeTool()
        response = agent.invoke("最新官方政策是什么？", "agentic-tavily", "deep", allow_web=True)
        nodes = [item.get("node") for item in response.trace]
        self.assertIn("tavily_search", nodes)
        self.assertTrue(response.trace[-1]["web_used"])
        self.assertTrue(any(item.chunk.chunk_id.startswith("web-tavily") for item in response.retrieved))


if __name__ == "__main__":
    unittest.main()

