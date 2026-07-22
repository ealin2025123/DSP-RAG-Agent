"""Optional Tavily web-search tool used by the Agentic RAG graph."""

import json
import os
import urllib.error
import urllib.request

from langchain_core.tools import tool

from .providers import ProviderError


class TavilySearchClient:
    def __init__(self, config):
        self.config = config
        self.api_key = os.getenv("TAVILY_API_KEY", "").strip()

    @property
    def available(self):
        offline = os.getenv("LYY_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")
        return bool(self.config.get("enabled", True) and self.api_key) and not offline

    def search(self, query):
        if not self.available:
            raise ProviderError("Tavily is not configured or offline mode is enabled")
        payload = {
            "query": query,
            "search_depth": self.config.get("search_depth", "basic"),
            "max_results": self.config.get("max_results", 5),
            "include_answer": False,
            "include_raw_content": False,
        }
        include_domains = self.config.get("include_domains", [])
        if include_domains:
            payload["include_domains"] = include_domains
        request = urllib.request.Request(
            self.config.get("endpoint", "https://api.tavily.com/search"),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
            raise ProviderError("Tavily search failed: {}".format(exc))
        results = []
        for item in body.get("results", [])[: self.config.get("max_results", 5)]:
            results.append({
                "title": str(item.get("title") or "Web result"),
                "url": str(item.get("url") or ""),
                "content": str(item.get("content") or ""),
                "score": float(item.get("score") or 0.0),
            })
        return results


def create_tavily_tool(client):
    @tool
    def tavily_search(query: str) -> list:
        """Search current official web information when the local DSP knowledge base is insufficient."""
        return client.search(query)

    return tavily_search

