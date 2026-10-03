"""Типы данных RAG-пайплайна."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.rag.citations import Citation, ParsedAnswer, SourceRef
from src.retrieval.models import RetrievedChunk


@dataclass
class RAGResult:
    """Результат RAG-запроса (или запроса без RAG)."""

    answer: str
    mode: str  # "rag" | "no_rag"
    used_chunks: List[RetrievedChunk] = field(default_factory=list)
    prompt: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    # Поля Task 23 (reranking + query rewrite).
    original_query: str = ""
    rewritten_query: Optional[str] = None
    chunks_before_rerank: List[RetrievedChunk] = field(default_factory=list)
    chunks_after_rerank: List[RetrievedChunk] = field(default_factory=list)
    rewrite_meta: Dict[str, Any] = field(default_factory=dict)
    rerank_meta: Dict[str, Any] = field(default_factory=dict)

    # Поля Task 24 (цитаты, источники, анти-галлюцинации).
    answer_status: str = "ok"  # "ok" | "unknown" | "error"
    parsed_answer: Optional[ParsedAnswer] = None
    citations: List[Citation] = field(default_factory=list)
    sources: List[SourceRef] = field(default_factory=list)
    citations_verified: bool = True
    sources_verified: bool = True

