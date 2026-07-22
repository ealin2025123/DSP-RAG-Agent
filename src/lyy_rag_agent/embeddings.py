import hashlib
import json
import math
import os
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path

from .config import dashscope_workspace_id
from .providers import ProviderError


def _offline():
    return os.getenv("LYY_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")


class DashScopeEmbeddingClient:
    def __init__(self, config):
        self.config = config
        self.api_key = os.getenv("DASHSCOPE_API_KEY", "")
        self.base_url = self._resolve_base_url()

    @staticmethod
    def _resolve_base_url():
        explicit = os.getenv("DASHSCOPE_EMBEDDING_BASE_URL", "").strip()
        if explicit:
            return explicit.rstrip("/")
        workspace = dashscope_workspace_id()
        if workspace:
            return "https://{}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1".format(workspace)
        generic = os.getenv("DASHSCOPE_BASE_URL", "").strip()
        return generic.rstrip("/") if generic else ""

    @property
    def available(self):
        return bool(self.api_key and self.base_url) and not _offline()

    def embed(self, texts):
        if not self.available:
            raise ProviderError("DashScope embedding endpoint is not configured")
        payload = json.dumps({
            "model": self.config["model"],
            "input": texts,
            "dimensions": self.config["dimensions"],
        }).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/embeddings", data=payload,
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
            raise ProviderError("embedding request failed: {}".format(exc))
        ordered = sorted(body["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in ordered]


class EmbeddingCache:
    def __init__(self, db_path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.path)) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS embeddings (chunk_id TEXT PRIMARY KEY, content_hash TEXT NOT NULL, "
                "model TEXT NOT NULL, dimensions INTEGER NOT NULL, vector_json TEXT NOT NULL, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)"
            )

    @staticmethod
    def content_hash(text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, chunk_id, text, model, dimensions):
        with sqlite3.connect(str(self.path)) as conn:
            row = conn.execute(
                "SELECT vector_json FROM embeddings WHERE chunk_id=? AND content_hash=? AND model=? AND dimensions=?",
                (chunk_id, self.content_hash(text), model, dimensions),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, chunk_id, text, model, dimensions, vector):
        with sqlite3.connect(str(self.path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings(chunk_id,content_hash,model,dimensions,vector_json,updated_at) "
                "VALUES(?,?,?,?,?,CURRENT_TIMESTAMP)",
                (chunk_id, self.content_hash(text), model, dimensions, json.dumps(vector, separators=(",", ":"))),
            )


class DenseEmbeddingIndex:
    def __init__(self, chunks, config, db_path):
        self.chunks = chunks
        self.config = config
        self.client = DashScopeEmbeddingClient(config)
        self.cache = EmbeddingCache(db_path)
        self.vectors = {}
        self._load_cached()

    def _load_cached(self):
        self.vectors = {}
        for chunk in self.chunks:
            vector = self.cache.get(chunk.chunk_id, chunk.text, self.config["model"], self.config["dimensions"])
            if vector is not None:
                self.vectors[chunk.chunk_id] = vector

    @property
    def ready(self):
        return len(self.vectors) == len(self.chunks) and bool(self.chunks)

    def build(self, force=False):
        if not self.client.available:
            raise ProviderError("Embedding requires DASHSCOPE_API_KEY and a compatible embedding base URL")
        pending = self.chunks if force else [chunk for chunk in self.chunks if chunk.chunk_id not in self.vectors]
        batch_size = self.config.get("batch_size", 10)
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            vectors = self.client.embed([chunk.text for chunk in batch])
            for chunk, vector in zip(batch, vectors):
                self.cache.put(chunk.chunk_id, chunk.text, self.config["model"], self.config["dimensions"], vector)
        self._load_cached()
        return len(pending)

    @staticmethod
    def _cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
        norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (norm_a * norm_b)

    def search(self, query, top_k):
        if not self.ready or not self.client.available:
            return []
        query_vector = self.client.embed([query])[0]
        scored = [(self._cosine(query_vector, self.vectors[chunk.chunk_id]), chunk) for chunk in self.chunks]
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return scored[:top_k]
