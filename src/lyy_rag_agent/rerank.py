import json
import os
import urllib.error
import urllib.request

from .config import dashscope_workspace_id
from .providers import ProviderError


class DashScopeReranker:
    def __init__(self, config):
        self.config = config
        self.api_key = os.getenv("DASHSCOPE_API_KEY", "")
        self.base_url = self._resolve_base_url()

    @staticmethod
    def _resolve_base_url():
        explicit = os.getenv("DASHSCOPE_RERANK_BASE_URL", "").strip()
        if explicit:
            return explicit.rstrip("/")
        workspace = dashscope_workspace_id()
        if workspace:
            return "https://{}.cn-beijing.maas.aliyuncs.com/compatible-api/v1".format(workspace)
        return ""

    @property
    def available(self):
        offline = os.getenv("LYY_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")
        return bool(self.config.get("enabled") and self.api_key and self.base_url) and not offline

    def rerank(self, query, results):
        if not self.available or not results:
            return results
        payload = json.dumps({
            "model": self.config["model"],
            "query": query,
            "documents": [item.chunk.text for item in results],
            "top_n": min(self.config["top_n"], len(results)),
            "instruct": self.config.get("instruct"),
        }).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/reranks", data=payload,
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
            raise ProviderError("rerank request failed: {}".format(exc))
        reranked = []
        for row in body["results"]:
            item = results[row["index"]]
            item.score = row["relevance_score"]
            item.sources = sorted(set(item.sources + ["qwen3-rerank"]))
            reranked.append(item)
        return reranked
