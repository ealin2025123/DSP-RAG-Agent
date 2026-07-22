"""LangChain 1.x orchestration for the existing DSP RAG components."""

import os
import time

from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from .core import AgentResponse
from .orchestrator import DSPRAGAgent, PresentationAgent
from .providers import OfflineGroundedProvider, ProviderError


def _offline():
    return os.getenv("LYY_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")


class LangChainChatProvider:
    """Expose a LangChain ChatOpenAI model through the project's provider API."""

    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.api_key = os.getenv(config["api_key_env"], "")
        self.models = []
        self.last_model = None
        if self.api_key:
            model_kwargs = {}
            if "enable_thinking" in config:
                model_kwargs["extra_body"] = {"enable_thinking": config["enable_thinking"]}
            model_names = [config["model"]] + config.get("fallback_models", [])
            for model_name in model_names:
                self.models.append((model_name, ChatOpenAI(
                    model=model_name,
                    api_key=self.api_key,
                    base_url=config["base_url"],
                    temperature=config.get("temperature", 0.1),
                    max_tokens=config.get("max_tokens", 1200),
                    timeout=90,
                    max_retries=1,
                    **model_kwargs,
                )))

    @property
    def available(self):
        return bool(self.models) and not _offline()

    @staticmethod
    def _content_text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("text"):
                    parts.append(str(block["text"]))
            return "\n".join(parts)
        return str(content)

    def complete(self, messages):
        if not self.available:
            raise ProviderError("{} LangChain model is not configured".format(self.name))
        errors = []
        for model_name, model in self.models:
            try:
                response = model.invoke(messages)
                content = self._content_text(response.content)
                finish_reason = response.response_metadata.get("finish_reason", "unknown")
                if finish_reason == "length":
                    raise ProviderError("output token limit reached")
                if not content.strip():
                    raise ProviderError("empty answer (finish_reason={})".format(finish_reason))
                self.last_model = model_name
                return content
            except Exception as exc:
                errors.append("{}: {}".format(model_name, exc))
        raise ProviderError(
            "{} LangChain models all failed: {}".format(self.name, " | ".join(errors))
        )


class LangChainDSPRAGAgent:
    """Runnable-based RAG chain without requiring direct LangGraph usage."""

    def __init__(self, settings_path=None):
        self.base = DSPRAGAgent(settings_path)
        self.base.providers = {
            "qwen_fast": LangChainChatProvider(
                "qwen_fast", self.base.settings["models"]["qwen_fast"]
            ),
            "qwen": LangChainChatProvider("qwen", self.base.settings["models"]["qwen"]),
            "deepseek": LangChainChatProvider("deepseek", self.base.settings["models"]["deepseek"]),
            "offline": OfflineGroundedProvider(),
        }
        self.base.router.providers = self.base.providers
        review_cfg = self.base.settings.get("review", {})
        self.base.reviewer.provider = (
            self.base.providers.get(review_cfg.get("provider")) if review_cfg.get("enabled") else None
        )
        self.chain = (
            RunnableLambda(self._security).with_config({"run_name": "01_security"})
            | RunnableLambda(self._route).with_config({"run_name": "02_router"})
            | RunnableLambda(self._retrieve).with_config({"run_name": "03_retrieve_and_rerank"})
            | RunnableLambda(self._generate).with_config({"run_name": "04_generate"})
            | RunnableLambda(self._review).with_config({"run_name": "05_review"})
            | RunnableLambda(self._persist).with_config({"run_name": "06_persist"})
        ).with_config({"run_name": "LYY_DSP_RAG_Agent"})

    @staticmethod
    def _copy(state, **updates):
        copied = dict(state)
        copied.update(updates)
        return copied

    def _security(self, state):
        started = time.perf_counter()
        query, findings = self.base.security.sanitize(str(state["question"]).strip())
        if not query:
            raise ValueError("问题不能为空")
        trace = [{
            "node": "security", "findings": findings, "runtime": "langchain-1.x",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }]
        return self._copy(state, query=query, trace=trace)

    def _route(self, state):
        started = time.perf_counter()
        route = self.base.router.route(state["query"])
        mode = state.get("mode", "auto")
        if mode == "fast":
            route.provider = "qwen_fast" if self.base.providers["qwen_fast"].available else "offline"
        elif mode == "deep":
            route.provider = "qwen" if self.base.providers["qwen"].available else "offline"
        trace = state["trace"] + [{
            "node": "router", "intent": route.intent,
            "complexity": route.complexity, "provider": route.provider, "mode": mode,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }]
        return self._copy(state, route=route, trace=trace)

    def _retrieve(self, state):
        started = time.perf_counter()
        fast = state.get("mode") == "fast"
        results = self.base.retriever.search(
            state["query"], use_dense=not fast, use_rerank=not fast
        )
        trace = state["trace"] + [{
            "node": "retrieval", "count": len(results),
            "ids": [item.chunk.chunk_id for item in results],
            "diagnostics": self.base.retriever.last_diagnostics,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }]
        return self._copy(state, results=results, context=self.base._context(results), trace=trace)

    def _generate(self, state):
        started = time.perf_counter()
        route = state["route"]
        provider = self.base.providers[route.provider]
        if route.provider == "offline":
            answer = provider.complete_from_results(state["query"], state["results"])
        else:
            prompt = self.base.prompts["answer"]
            if route.intent in ("troubleshooting", "strategy") and route.complexity == "complex":
                prompt = self.base.prompts["troubleshooting"]
            history = self.base.memory.get(
                state["session_id"], self.base.settings["history"]["max_messages"]
            )
            messages = [{"role": "system", "content": prompt}] + history
            messages.append({
                "role": "user",
                "content": "用户问题：\n{}\n\n知识库证据：\n{}".format(state["query"], state["context"]),
            })
            try:
                answer = provider.complete(messages)
                if not answer or not answer.strip():
                    raise ProviderError("provider returned an empty answer")
            except ProviderError as exc:
                route.provider = "offline"
                answer = self.base.providers["offline"].complete_from_results(
                    state["query"], state["results"]
                )
                trace = state["trace"] + [{
                    "node": "provider_fallback", "error": str(exc),
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                }]
                return self._copy(state, answer=answer, trace=trace)
        trace = state["trace"] + [{
            "node": "generation", "model": getattr(provider, "last_model", route.provider),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }]
        return self._copy(state, answer=answer, trace=trace)

    def _review(self, state):
        started = time.perf_counter()
        safe_answer, findings = self.base.security.sanitize(state["answer"])
        skip_review = state.get("mode") == "fast" or (
            state.get("mode", "auto") == "auto" and state["route"].complexity == "simple"
        )
        if skip_review:
            reviewed, passed, issues = safe_answer, True, ["skipped_for_speed"]
        else:
            reviewed, passed, issues = self.base.reviewer.review(
                safe_answer, state["results"], state["context"]
            )
        trace = state["trace"] + [
            {"node": "output_security", "findings": findings},
            {
                "node": "review", "passed": passed, "issues": issues,
                "skipped": skip_review,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        ]
        return self._copy(state, reviewed=reviewed, review_passed=passed, trace=trace)

    def _persist(self, state):
        started = time.perf_counter()
        answer = PresentationAgent.render(state["reviewed"])
        self.base.memory.add(state["session_id"], "user", state["query"])
        self.base.memory.add(state["session_id"], "assistant", answer)
        trace = state["trace"] + [{
            "node": "presentation", "internal_citations_hidden": True,
            "orchestration": "LangChain RunnableSequence",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "total_ms": round((time.perf_counter() - state["request_started"]) * 1000, 1),
        }]
        return AgentResponse(
            answer=answer,
            route=state["route"],
            citations=[item.chunk.chunk_id for item in state["results"]],
            retrieved=state["results"],
            provider=state["route"].provider,
            review_passed=state["review_passed"],
            trace=trace,
        )

    def invoke(self, question, session_id="langchain-default", mode="auto"):
        return self.chain.invoke(
            {
                "question": question, "session_id": session_id,
                "mode": mode, "request_started": time.perf_counter(),
            },
            config={"metadata": {"session_id": session_id, "response_mode": mode}},
        )

    def ask(self, question, session_id="langchain-default", mode="auto"):
        return self.invoke(question, session_id, mode)
