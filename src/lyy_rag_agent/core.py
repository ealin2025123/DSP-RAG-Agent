from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class KnowledgeChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]


@dataclass
class RetrievalResult:
    chunk: KnowledgeChunk
    score: float
    sources: List[str] = field(default_factory=list)


@dataclass
class RouteDecision:
    intent: str
    complexity: str
    provider: str
    keywords: List[str] = field(default_factory=list)


@dataclass
class AgentResponse:
    answer: str
    route: RouteDecision
    citations: List[str]
    retrieved: List[RetrievalResult]
    provider: str
    review_passed: bool
    trace: List[Dict[str, Any]] = field(default_factory=list)

