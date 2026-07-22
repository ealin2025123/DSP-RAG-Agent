import json
import os
import urllib.error
import urllib.request


class ProviderError(RuntimeError):
    pass


class OpenAICompatibleProvider:
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.api_key = os.getenv(config["api_key_env"], "")
        self.last_model = None

    @property
    def available(self):
        offline = os.getenv("LYY_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")
        return bool(self.api_key) and not offline

    def complete(self, messages):
        if not self.available:
            raise ProviderError("{} API key is not configured".format(self.name))
        models = [self.config["model"]] + self.config.get("fallback_models", [])
        errors = []
        for model in models:
            try:
                content = self._complete_with_model(messages, model)
                self.last_model = model
                return content
            except ProviderError as exc:
                errors.append("{}: {}".format(model, exc))
        raise ProviderError("{} models all failed: {}".format(self.name, " | ".join(errors)))

    def _complete_with_model(self, messages, model):
        url = self.config["base_url"].rstrip("/") + "/chat/completions"
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "temperature": self.config.get("temperature", 0.1),
            "max_tokens": self.config.get("max_tokens", 1200),
            **({"enable_thinking": self.config["enable_thinking"]}
               if "enable_thinking" in self.config else {}),
        }).encode("utf-8")
        request = urllib.request.Request(
            url, data=payload,
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
            raise ProviderError("request failed: {}".format(exc))
        choice = data["choices"][0]
        content = choice["message"].get("content") or ""
        finish_reason = choice.get("finish_reason", "unknown")
        if finish_reason == "length":
            raise ProviderError("output token limit reached")
        if not content.strip():
            raise ProviderError("empty answer (finish_reason={})".format(finish_reason))
        return content


class OfflineGroundedProvider:
    name = "offline-extractive"
    available = True

    def complete_from_results(self, query, results):
        if not results:
            return "当前知识库没有检索到足够信息。请补充问题所属的配置环节、异常现象或相关字段。"
        lines = ["当前处于离线检索模式。根据知识库，与你的问题最相关的信息是：", ""]
        for index, result in enumerate(results[:4], 1):
            text = result.chunk.text.strip().replace("\n", " ")
            if len(text) > 260:
                text = text[:257] + "..."
            lines.append("{}. {} [{}]".format(index, text, result.chunk.chunk_id))
        lines.extend(["", "离线模式只返回证据摘录；配置 API Key 后会生成步骤化回答并执行更完整的核验。"])
        return "\n".join(lines)
