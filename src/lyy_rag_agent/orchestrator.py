import json
import os
import re
from pathlib import Path

from .config import load_settings, project_root
from .core import AgentResponse, RouteDecision
from .embeddings import DenseEmbeddingIndex
from .memory import SQLiteMemory
from .providers import OfflineGroundedProvider, OpenAICompatibleProvider, ProviderError
from .retrieval import HybridRetriever, KnowledgeRepository, tokenize
from .rerank import DashScopeReranker
from .security import SecurityAgent


class RouterAgent:
    COMPLEX_TERMS = ("为什么", "排查", "异常", "没有曝光", "无曝光", "转化低", "点击低", "怎么优化", "策略", "归因")
    INTENTS = {
        "troubleshooting": ("排查", "异常", "报错", "失败", "没有曝光", "无曝光", "低转化", "点击低", "审核"),
        "policy": ("尺寸", "logo", "cta", "文件大小", "边框", "素材规范", "文案"),
        "operation": ("怎么配置", "如何创建", "上传", "命名", "order", "line item", "操作"),
        "strategy": ("策略", "漏斗", "受众", "人群", "预算", "优化", "amc"),
        "definition": ("是什么", "含义", "缩写", "指标", "定义"),
    }

    def __init__(self, settings, providers):
        self.settings = settings
        self.providers = providers

    def route(self, query):
        lowered = query.lower()
        intent = "definition"
        line_item_type_question = "line item" in lowered and any(
            term in lowered for term in ("有哪些", "什么类型", "哪些类型", "类型")
        )
        if not line_item_type_question:
            for candidate, terms in self.INTENTS.items():
                if any(term in lowered for term in terms):
                    intent = candidate
                    break
        complexity = "complex" if any(term in lowered for term in self.COMPLEX_TERMS) else "simple"
        provider = self.settings["routing"]["default_provider"]
        use_complex = self.settings["routing"].get(
            "enable_complex_provider",
            self.settings["routing"].get("enable_deepseek_for_complex", False),
        )
        if complexity == "complex" and use_complex:
            provider = self.settings["routing"]["complex_provider"]
        if not self.providers[provider].available:
            fallback = self.settings["routing"]["default_provider"]
            provider = fallback if self.providers[fallback].available else "offline"
        return RouteDecision(intent, complexity, provider, tokenize(query)[:12])


class ReviewerAgent:
    CITATION = re.compile(r"\[(?:知识块ID:\s*)?([a-z0-9_-]+-\d{3})\]", re.I)

    def __init__(self, prompt, provider=None):
        self.prompt = prompt
        self.provider = provider

    @staticmethod
    def _parse_json(text):
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("reviewer did not return JSON")
        return json.loads(cleaned[start:end + 1])

    def review(self, answer, results, context):
        valid = {result.chunk.chunk_id for result in results}
        cited = self.CITATION.findall(answer)
        invalid = [item for item in cited if item not in valid]
        if invalid:
            answer += "\n\n核验提示：回答包含无法对应当前证据的引用，需要人工复核。"
            return answer, False, ["invalid_citation"]

        issues = []
        passed = bool(cited) or not results
        if self.provider is not None and self.provider.available and results:
            try:
                raw = self.provider.complete([
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": "候选回答：\n{}\n\n知识库证据：\n{}".format(answer, context)},
                ])
                review = self._parse_json(raw)
                issues = review.get("issues", []) or []
                if not review.get("passed", False):
                    revised = (review.get("revised_answer") or "").strip()
                    if revised:
                        answer = revised
                passed = bool(review.get("passed", False) or review.get("revised_answer"))
            except (ProviderError, ValueError, KeyError, json.JSONDecodeError) as exc:
                issues.append("reviewer_error: {}".format(exc))
                passed = False

        cited = self.CITATION.findall(answer)
        invalid = [item for item in cited if item not in valid]
        if invalid:
            issues.append("invalid_citation_after_review")
            passed = False
        if results and not cited:
            answer += "\n\n参考知识块：" + "、".join("[{}]".format(r.chunk.chunk_id) for r in results[:4])
            issues.append("missing_inline_citations")
            passed = False
        return answer, passed, issues


class PresentationAgent:
    """Remove internal evidence identifiers from user-facing responses."""

    CITATION = ReviewerAgent.CITATION

    @classmethod
    def render(cls, answer):
        lines = []
        for line in answer.splitlines():
            if line.strip().startswith("参考知识块："):
                continue
            cleaned = cls.CITATION.sub("", line)
            cleaned = re.sub(r"`\s*`", "", cleaned)
            cleaned = re.sub(r"[ \t]+([，。；、：,.!?])", r"\1", cleaned)
            cleaned = re.sub(r" {2,}", " ", cleaned).rstrip()
            lines.append(cleaned)
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


class DSPRAGAgent:
    def __init__(self, settings_path=None):
        self.root = project_root()
        self.settings = load_settings(settings_path)
        configured_path = os.getenv("LYY_KB_PATH", "").strip()
        private_path = self.root / "data" / "private_knowledge_base" / "exports" / "documents.jsonl"
        demo_path = self.root / "data" / "demo" / "documents.jsonl"
        if configured_path:
            repository_path = Path(configured_path).expanduser().resolve()
        else:
            repository_path = private_path if private_path.exists() else demo_path
        self.knowledge_path = repository_path
        self.knowledge_mode = "private" if repository_path != demo_path else "demo"
        repository = KnowledgeRepository(repository_path)
        self.embedding_index = DenseEmbeddingIndex(
            repository.chunks, self.settings["embedding"], self.root / "runtime" / "embeddings.db"
        )
        self.reranker = DashScopeReranker(self.settings["rerank"])
        self.retriever = HybridRetriever(repository, self.settings, self.embedding_index, self.reranker)
        self.security = SecurityAgent(self.settings.get("security", {}).get("custom_sensitive_terms", []))
        self.memory = SQLiteMemory(self.root / "runtime" / "chat_history.db")
        self.providers = {
            "qwen_fast": OpenAICompatibleProvider("qwen_fast", self.settings["models"]["qwen_fast"]),
            "qwen": OpenAICompatibleProvider("qwen", self.settings["models"]["qwen"]),
            "deepseek": OpenAICompatibleProvider("deepseek", self.settings["models"]["deepseek"]),
            "offline": OfflineGroundedProvider(),
        }
        self.router = RouterAgent(self.settings, self.providers)
        self.prompts = {
            "answer": (self.root / "prompts" / "answer.txt").read_text(encoding="utf-8"),
            "troubleshooting": (self.root / "prompts" / "troubleshooting.txt").read_text(encoding="utf-8"),
            "reviewer": (self.root / "prompts" / "reviewer.txt").read_text(encoding="utf-8"),
        }
        review_cfg = self.settings.get("review", {})
        review_provider = self.providers.get(review_cfg.get("provider")) if review_cfg.get("enabled") else None
        self.reviewer = ReviewerAgent(self.prompts["reviewer"], review_provider)

    def _context(self, results):
        return "\n\n---\n\n".join(
            "[知识块ID: {}]\n标题: {}\n章节: {}\n内容:\n{}".format(
                r.chunk.chunk_id, r.chunk.metadata.get("title", ""),
                r.chunk.metadata.get("heading", ""), r.chunk.text,
            ) for r in results
        )

    def ask(self, query, session_id="default"):
        trace = []
        safe_query, findings = self.security.sanitize(query.strip())
        trace.append({"node": "security", "findings": findings})
        route = self.router.route(safe_query)
        trace.append({"node": "router", "intent": route.intent, "complexity": route.complexity, "provider": route.provider})
        results = self.retriever.search(safe_query)
        trace.append({
            "node": "retrieval", "count": len(results),
            "ids": [r.chunk.chunk_id for r in results],
            "diagnostics": self.retriever.last_diagnostics,
        })
        history = self.memory.get(session_id, self.settings["history"]["max_messages"])
        provider = self.providers[route.provider]

        if route.provider == "offline":
            answer = provider.complete_from_results(safe_query, results)
        else:
            prompt = self.prompts["troubleshooting"] if route.intent in ("troubleshooting", "strategy") and route.complexity == "complex" else self.prompts["answer"]
            messages = [{"role": "system", "content": prompt}] + history
            messages.append({"role": "user", "content": "用户问题：\n{}\n\n知识库证据：\n{}".format(safe_query, self._context(results))})
            try:
                answer = provider.complete(messages)
            except ProviderError as exc:
                trace.append({"node": "provider_fallback", "error": str(exc)})
                route.provider = "offline"
                answer = self.providers["offline"].complete_from_results(safe_query, results)

        safe_answer, output_findings = self.security.sanitize(answer)
        trace.append({"node": "output_security", "findings": output_findings})
        reviewed_answer, review_passed, review_issues = self.reviewer.review(safe_answer, results, self._context(results))
        trace.append({"node": "review", "passed": review_passed, "issues": review_issues})
        display_answer = PresentationAgent.render(reviewed_answer)
        trace.append({"node": "presentation", "internal_citations_hidden": True})
        self.memory.add(session_id, "user", safe_query)
        self.memory.add(session_id, "assistant", display_answer)
        citations = [result.chunk.chunk_id for result in results]
        return AgentResponse(display_answer, route, citations, results, route.provider, review_passed, trace)
