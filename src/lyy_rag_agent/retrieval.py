import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from .core import KnowledgeChunk, RetrievalResult


ASCII_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_-]*|\d+(?:\.\d+)?")
CHINESE = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text):
    text = text.lower()
    tokens = ASCII_WORD.findall(text)
    chars = [ch for ch in text if CHINESE.match(ch)]
    tokens.extend("".join(chars[i:i + 2]) for i in range(max(0, len(chars) - 1)))
    tokens.extend("".join(chars[i:i + 3]) for i in range(max(0, len(chars) - 2)))
    return [token for token in tokens if token]


def expand_query(query):
    """Add common DSP paraphrases without calling an LLM."""
    additions = []
    rules = {
        "没有曝光": "无曝光 曝光过低 投放状态",
        "不曝光": "无曝光 曝光过低",
        "不投放": "无曝光 投放状态",
        "没流量": "曝光过低 曝光规模",
        "点击低": "点击偏低 CTR 素材人群匹配",
        "转化低": "转化偏低 CVR 落地页",
        "素材被拒": "素材审核失败 素材规范",
    }
    for phrase, expansion in rules.items():
        if phrase in query:
            additions.append(expansion)
    return query + (" " + " ".join(additions) if additions else "")


class KnowledgeRepository:
    def __init__(self, jsonl_path):
        self.path = Path(jsonl_path)
        self.chunks = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                self.chunks.append(KnowledgeChunk(row["id"], row["text"], row["metadata"]))
        if not self.chunks:
            raise ValueError("Knowledge base is empty: {}".format(self.path))


class BM25Retriever:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [tokenize(chunk.text) for chunk in chunks]
        self.lengths = [len(doc) for doc in self.docs]
        self.avg_len = sum(self.lengths) / float(len(self.lengths) or 1)
        self.df = Counter()
        for doc in self.docs:
            self.df.update(set(doc))

    def search(self, query, top_k=12):
        query_terms = tokenize(query)
        total = len(self.docs)
        scored = []
        for index, doc in enumerate(self.docs):
            tf = Counter(doc)
            score = 0.0
            for term in query_terms:
                if not tf[term]:
                    continue
                idf = math.log(1.0 + (total - self.df[term] + 0.5) / (self.df[term] + 0.5))
                denom = tf[term] + self.k1 * (1.0 - self.b + self.b * self.lengths[index] / (self.avg_len or 1.0))
                score += idf * tf[term] * (self.k1 + 1.0) / denom
            if score > 0:
                scored.append(RetrievalResult(self.chunks[index], score, ["bm25"]))
        return sorted(scored, key=lambda item: (-item.score, item.chunk.chunk_id))[:top_k]


class CharVectorRetriever:
    """Dependency-free character n-gram cosine retrieval for offline mode."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.vectors = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self.norms = [math.sqrt(sum(v * v for v in vector.values())) for vector in self.vectors]

    def search(self, query, top_k=12):
        q = Counter(tokenize(query))
        q_norm = math.sqrt(sum(v * v for v in q.values())) or 1.0
        scored = []
        for index, vector in enumerate(self.vectors):
            dot = sum(value * vector.get(term, 0) for term, value in q.items())
            score = dot / (q_norm * (self.norms[index] or 1.0))
            if score > 0:
                scored.append(RetrievalResult(self.chunks[index], score, ["char_vector"]))
        return sorted(scored, key=lambda item: (-item.score, item.chunk.chunk_id))[:top_k]


class HybridRetriever:
    def __init__(self, repository, settings, embedding_index=None, reranker=None):
        self.settings = settings
        self.bm25 = BM25Retriever(repository.chunks)
        self.char_vector = CharVectorRetriever(repository.chunks)
        self.embedding_index = embedding_index
        self.reranker = reranker
        self.last_diagnostics = {}

    def search(self, query, use_dense=True, use_rerank=True):
        cfg = self.settings["retrieval"]
        query = expand_query(query)
        bm25_results = self.bm25.search(query, cfg["bm25_top_k"])
        dense_results = []
        if use_dense and self.embedding_index is not None:
            dense_results = [
                RetrievalResult(chunk, score, ["text-embedding-v4"])
                for score, chunk in self.embedding_index.search(query, cfg["dense_top_k"])
            ]
        semantic_results = dense_results or self.char_vector.search(query, cfg["char_vector_top_k"])
        rankings = [bm25_results, semantic_results]
        scores, chunks, sources = defaultdict(float), {}, defaultdict(set)
        for ranking in rankings:
            for rank, result in enumerate(ranking, 1):
                chunk_id = result.chunk.chunk_id
                scores[chunk_id] += 1.0 / (cfg["rrf_k"] + rank)
                chunks[chunk_id] = result.chunk
                sources[chunk_id].update(result.sources)
        fused = [RetrievalResult(chunks[key], score, sorted(sources[key])) for key, score in scores.items()]
        fused.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        candidates = fused[:cfg.get("fusion_candidate_k", cfg["fusion_top_k"])]
        rerank_error = ""
        if use_rerank and self.reranker is not None and self.reranker.available:
            try:
                candidates = self.reranker.rerank(query, candidates)
            except Exception as exc:
                rerank_error = str(exc)
        self.last_diagnostics = {
            "semantic_retriever": "text-embedding-v4" if dense_results else "char_vector",
            "reranker": "qwen3-rerank" if use_rerank and self.reranker is not None and self.reranker.available and not rerank_error else "none",
            "rerank_error": rerank_error,
        }
        return candidates[:cfg["fusion_top_k"]]
