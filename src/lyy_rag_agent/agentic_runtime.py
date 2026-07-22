"""LangGraph-based Agentic RAG with retrieval and review feedback loops."""

import json
import re
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .core import AgentResponse, KnowledgeChunk, RetrievalResult
from .langchain_runtime import LangChainDSPRAGAgent
from .orchestrator import PresentationAgent
from .providers import ProviderError
from .retrieval import tokenize
from .tavily import TavilySearchClient, create_tavily_tool


class AgenticState(TypedDict, total=False):
    question: str
    original_query: str
    query: str
    session_id: str
    mode: str
    allow_web: bool
    request_started: float
    route: object
    results: list
    context: str
    answer: str
    reviewed: str
    review_passed: bool
    review_issues: list
    review_feedback: str
    retrieval_sufficient: bool
    retrieval_reason: str
    needs_web: bool
    suggested_query: str
    retrieval_attempts: int
    generation_attempts: int
    web_used: bool
    trace: list
    response: AgentResponse


class AgenticDSPRAGAgent(LangChainDSPRAGAgent):
    """Agentic RAG that can rewrite, retry, search the web, and regenerate."""

    FRESHNESS_TERMS = ("最新", "当前", "现在", "近期", "更新", "官方政策", "实时", "今年")

    def __init__(self, settings_path=None):
        super().__init__(settings_path)
        self.agentic_config = self.base.settings.get("agentic", {})
        self.tavily_client = TavilySearchClient(self.base.settings.get("tavily", {}))
        self.tavily_tool = create_tavily_tool(self.tavily_client)
        self.graph = self._build_graph()

    @staticmethod
    def _append_trace(state, item):
        return list(state.get("trace", [])) + [item]

    def _build_graph(self):
        builder = StateGraph(AgenticState)
        builder.add_node("security", self._security_node)
        builder.add_node("route", self._route_node)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("grade_retrieval", self._grade_retrieval_node)
        builder.add_node("rewrite_query", self._rewrite_query_node)
        builder.add_node("tavily_search", self._tavily_node)
        builder.add_node("generate", self._generate_node)
        builder.add_node("review", self._review_node)
        builder.add_node("persist", self._persist_node)
        builder.add_edge(START, "security")
        builder.add_edge("security", "route")
        builder.add_edge("route", "retrieve")
        builder.add_edge("retrieve", "grade_retrieval")
        builder.add_conditional_edges("grade_retrieval", self._after_grade, {
            "rewrite": "rewrite_query",
            "web": "tavily_search",
            "generate": "generate",
        })
        builder.add_edge("rewrite_query", "retrieve")
        builder.add_edge("tavily_search", "generate")
        builder.add_edge("generate", "review")
        builder.add_conditional_edges("review", self._after_review, {
            "regenerate": "generate",
            "persist": "persist",
        })
        builder.add_edge("persist", END)
        return builder.compile()

    def _security_node(self, state):
        updated = self._security(state)
        return {
            "original_query": updated["query"],
            "query": updated["query"],
            "trace": updated["trace"],
            "retrieval_attempts": 0,
            "generation_attempts": 0,
            "web_used": False,
        }

    def _route_node(self, state):
        updated = self._route(state)
        return {"route": updated["route"], "trace": updated["trace"]}

    def _retrieve_node(self, state):
        updated = self._retrieve(state)
        return {
            "results": updated["results"],
            "context": updated["context"],
            "trace": updated["trace"],
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        }

    @staticmethod
    def _extract_json(text):
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("JSON object not found")
        return json.loads(cleaned[start:end + 1])

    def _heuristic_grade(self, state):
        results = state.get("results", [])
        if not results:
            return False, "no_retrieval_results"
        query_terms = set(tokenize(state["query"]))
        evidence_terms = set(tokenize("\n".join(item.chunk.text for item in results[:4])))
        overlap = len(query_terms & evidence_terms) / float(len(query_terms) or 1)
        return overlap >= 0.12, "token_overlap={:.3f}".format(overlap)

    def _grade_retrieval_node(self, state):
        started = time.perf_counter()
        sufficient, reason = self._heuristic_grade(state)
        suggested = ""
        freshness = any(term in state["original_query"] for term in self.FRESHNESS_TERMS)
        needs_web = freshness
        grader = self.base.providers.get("qwen_fast")
        use_llm = self.agentic_config.get("use_llm_grader", True)
        if use_llm and grader is not None and grader.available and state.get("mode") != "fast":
            try:
                raw = grader.complete([
                    {"role": "system", "content": (
                        "你是RAG检索质量评估器。判断证据能否直接回答问题；不要回答问题。"
                        "如果问题要求最新、当前或官方更新，needs_web应为true。只输出JSON："
                        '{"sufficient":true,"reason":"...","rewrite_query":"...","needs_web":false}'
                    )},
                    {"role": "user", "content": "问题：\n{}\n\n证据：\n{}".format(
                        state["query"], state.get("context", "")
                    )},
                ])
                grade = self._extract_json(raw)
                sufficient = bool(grade.get("sufficient"))
                reason = str(grade.get("reason") or reason)
                suggested = str(grade.get("rewrite_query") or "").strip()
                needs_web = bool(grade.get("needs_web")) or freshness
            except (ProviderError, ValueError, KeyError, json.JSONDecodeError) as exc:
                reason += "; grader_fallback={}".format(exc)
        trace = self._append_trace(state, {
            "node": "retrieval_grade",
            "sufficient": sufficient,
            "reason": reason,
            "needs_web": needs_web,
            "attempt": state.get("retrieval_attempts", 0),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        })
        return {
            "retrieval_sufficient": sufficient,
            "retrieval_reason": reason,
            "needs_web": needs_web,
            "suggested_query": suggested,
            "trace": trace,
        }

    def _after_grade(self, state):
        can_web = bool(
            state.get("allow_web") and self.tavily_client.available
            and state.get("mode") != "fast" and not state.get("web_used")
        )
        if state.get("needs_web") and can_web:
            return "web"
        if state.get("retrieval_sufficient"):
            return "generate"
        max_attempts = self.agentic_config.get("max_retrieval_attempts", 2)
        if state.get("retrieval_attempts", 0) < max_attempts:
            return "rewrite"
        if can_web:
            return "web"
        return "generate"

    def _rewrite_query_node(self, state):
        started = time.perf_counter()
        rewritten = state.get("suggested_query", "").strip()
        provider = self.base.providers.get("qwen_fast")
        if not rewritten and provider is not None and provider.available:
            try:
                rewritten = provider.complete([
                    {"role": "system", "content": (
                        "将用户问题改写为更适合DSP知识库检索的一句话。保留原意和关键字段，"
                        "不要回答问题，只输出改写后的问题。"
                    )},
                    {"role": "user", "content": state["query"]},
                ]).strip()
            except ProviderError:
                rewritten = ""
        if not rewritten or rewritten == state["query"]:
            rewritten = state["original_query"] + " DSP 配置 操作规范 异常排查"
        rewritten, findings = self.base.security.sanitize(rewritten)
        return {
            "query": rewritten,
            "trace": self._append_trace(state, {
                "node": "query_rewrite", "query": rewritten, "findings": findings,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            }),
        }

    def _tavily_node(self, state):
        started = time.perf_counter()
        try:
            rows = self.tavily_tool.invoke({"query": state["query"]})
            web_results = []
            for index, row in enumerate(rows, 1):
                text = "标题：{}\n摘要：{}\n来源：{}".format(
                    row.get("title", ""), row.get("content", ""), row.get("url", "")
                )
                chunk = KnowledgeChunk(
                    "web-tavily-{:03d}".format(index), text,
                    {
                        "title": row.get("title", "Web result"),
                        "heading": "Tavily web search",
                        "source_kind": "web",
                        "url": row.get("url", ""),
                    },
                )
                web_results.append(RetrievalResult(chunk, row.get("score", 0.0), ["tavily"]))
            combined = list(state.get("results", [])) + web_results
            context = self.base._context(combined)
            error = ""
        except Exception as exc:
            combined = list(state.get("results", []))
            context = state.get("context", "")
            error = str(exc)
            web_results = []
        return {
            "results": combined,
            "context": context,
            "web_used": bool(web_results),
            "trace": self._append_trace(state, {
                "node": "tavily_search", "count": len(web_results), "error": error,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            }),
        }

    def _generate_node(self, state):
        updated = self._generate(state)
        return {
            "answer": updated["answer"],
            "trace": updated.get("trace", state.get("trace", [])),
            "route": updated.get("route", state["route"]),
            "generation_attempts": state.get("generation_attempts", 0) + 1,
            "review_feedback": "",
        }

    def _review_node(self, state):
        updated = self._review(state)
        issues = []
        for item in reversed(updated["trace"]):
            if item.get("node") == "review":
                issues = item.get("issues", [])
                break
        return {
            "reviewed": updated["reviewed"],
            "review_passed": updated["review_passed"],
            "review_issues": issues,
            "review_feedback": "; ".join(str(item) for item in issues),
            "trace": updated["trace"],
        }

    def _after_review(self, state):
        max_attempts = self.agentic_config.get("max_generation_attempts", 2)
        if not state.get("review_passed") and state.get("generation_attempts", 0) < max_attempts:
            return "regenerate"
        return "persist"

    def _persist_node(self, state):
        started = time.perf_counter()
        answer = PresentationAgent.render(state.get("reviewed") or state.get("answer", ""))
        self.base.memory.add(state["session_id"], "user", state["original_query"])
        self.base.memory.add(state["session_id"], "assistant", answer)
        trace = self._append_trace(state, {
            "node": "presentation", "internal_citations_hidden": True,
            "orchestration": "LangGraph StateGraph",
            "retrieval_attempts": state.get("retrieval_attempts", 0),
            "generation_attempts": state.get("generation_attempts", 0),
            "web_used": state.get("web_used", False),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "total_ms": round((time.perf_counter() - state["request_started"]) * 1000, 1),
        })
        response = AgentResponse(
            answer=answer,
            route=state["route"],
            citations=[item.chunk.chunk_id for item in state.get("results", [])],
            retrieved=state.get("results", []),
            provider=state["route"].provider,
            review_passed=state.get("review_passed", False),
            trace=trace,
        )
        return {"response": response, "trace": trace}

    def invoke(self, question, session_id="agentic-default", mode="auto", allow_web=False):
        initial = {
            "question": question,
            "session_id": session_id,
            "mode": mode,
            "allow_web": bool(allow_web),
            "request_started": time.perf_counter(),
        }
        result = self.graph.invoke(
            initial,
            config={
                "recursion_limit": self.agentic_config.get("recursion_limit", 20),
                "metadata": {
                    "session_id": session_id,
                    "response_mode": mode,
                    "allow_web": bool(allow_web),
                },
                "run_name": "DSP_Agentic_RAG",
            },
        )
        return result["response"]

    def ask(self, question, session_id="agentic-default", mode="auto", allow_web=False):
        return self.invoke(question, session_id, mode, allow_web)

