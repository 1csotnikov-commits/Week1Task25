"""RAG-пайплайн: (rewrite) -> retrieval -> (rerank) -> проверка контекста -> LLM -> парсинг/верификация."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.config import Config
from src.llm.base import LLMProvider
from src.rag.citations import ParsedAnswer, extract_citations, extract_sources, parse_structured_answer
from src.rag.models import RAGResult
from src.rag.prompt_builder import build_messages, render_messages
from src.rag.query_rewrite import QueryRewriter
from src.reranking.base import Reranker
from src.reranking.pipeline import RerankerPipeline
from src.retrieval.base import Retriever
from src.retrieval.models import RetrievedChunk

UNKNOWN_ANSWER = (
    "Не знаю. В предоставленных документах не найдено достаточно информации для ответа. "
    "Уточните, пожалуйста, вопрос или добавьте документы по теме."
)


@dataclass
class RetrievalContext:
    """Промежуточный результат retrieval/rerank/проверки контекста."""

    original_query: str
    rewritten_query: str
    chunks_before: List[RetrievedChunk] = field(default_factory=list)
    chunks_after: List[RetrievedChunk] = field(default_factory=list)
    rewrite_meta: Dict[str, Any] = field(default_factory=dict)
    rerank_meta: Dict[str, Any] = field(default_factory=dict)
    k_retrieve: int = 10
    k_after: int = 5
    is_weak: bool = False


class RAGPipeline:
    """RAG с query rewrite, reranking и анти-галлюцинациями."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
        config: Optional[Config] = None,
        reranker: Optional[Reranker] = None,
        rewriter: Optional[QueryRewriter] = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.config = config
        self.reranker = reranker
        self.rewriter = rewriter
        self._rerank_pipeline = RerankerPipeline(reranker)

    def _cfg(self, dotted: str, default):
        if self.config is not None:
            return self.config.get(dotted, default)
        return default

    def retrieve(
        self,
        question: str,
        top_k_retrieve: Optional[int] = None,
        top_k_after_rerank: Optional[int] = None,
        use_reranker: bool = True,
        use_rewrite: bool = False,
    ) -> RetrievalContext:
        """Выполнить (rewrite) -> retrieval -> (rerank) и проверить слабый контекст."""
        k_retrieve = int(top_k_retrieve) if top_k_retrieve is not None else int(self._cfg("retrieval.top_k_retrieve", 10))
        k_after = int(top_k_after_rerank) if top_k_after_rerank is not None else int(self._cfg("reranking.top_k_after_rerank", 5))

        original_query = question
        rewritten_query = question
        rewrite_meta = {"method": "none", "tokens": 0, "elapsed_ms": 0}
        if use_rewrite and self.rewriter is not None:
            rr = self.rewriter.rewrite(question)
            rewritten_query = rr.rewritten_query
            rewrite_meta = {
                "method": rr.method,
                "tokens": rr.meta.get("tokens", 0),
                "elapsed_ms": rr.meta.get("elapsed_ms", 0),
            }

        chunks_before = self.retriever.search(rewritten_query, k_retrieve)

        if use_reranker:
            chunks_after, rerank_meta = self._rerank_pipeline.apply(chunks_before, rewritten_query, top_k_after=k_after)
        else:
            chunks_after = list(chunks_before)
            rerank_meta = {
                "provider": "none",
                "before": len(chunks_before),
                "after": len(chunks_after),
                "removed": 0,
                "threshold": None,
            }

        return RetrievalContext(
            original_query=original_query,
            rewritten_query=rewritten_query,
            chunks_before=chunks_before,
            chunks_after=chunks_after,
            rewrite_meta=rewrite_meta,
            rerank_meta=rerank_meta,
            k_retrieve=k_retrieve,
            k_after=k_after,
            is_weak=self._is_weak_context(chunks_after),
        )

    def run(
        self,
        question: str,
        top_k_retrieve: Optional[int] = None,
        top_k_after_rerank: Optional[int] = None,
        use_reranker: bool = True,
        use_rewrite: bool = False,
    ) -> RAGResult:
        ctx = self.retrieve(question, top_k_retrieve, top_k_after_rerank, use_reranker, use_rewrite)
        if ctx.is_weak:
            return self._unknown_result(ctx)

        chunks_after = ctx.chunks_after
        messages = build_messages(ctx.original_query, chunks_after)
        prompt_text = render_messages(messages)
        response = self.llm.complete(messages)

        parsed = parse_structured_answer(response.text)
        citations = extract_citations(response.text, chunks_after)
        sources = extract_sources(response.text, chunks_after)
        citations_verified = all(c.verified for c in citations) if citations else True
        sources_verified = all(s.verified for s in sources) if sources else True

        return RAGResult(
            answer=parsed.answer_text,
            mode="rag",
            used_chunks=chunks_after,
            prompt=prompt_text,
            original_query=ctx.original_query,
            rewritten_query=ctx.rewritten_query if ctx.rewritten_query != ctx.original_query else None,
            chunks_before_rerank=ctx.chunks_before,
            chunks_after_rerank=chunks_after,
            rewrite_meta=ctx.rewrite_meta,
            rerank_meta=ctx.rerank_meta,
            answer_status="ok",
            parsed_answer=parsed,
            citations=citations,
            sources=sources,
            citations_verified=citations_verified,
            sources_verified=sources_verified,
            meta={
                "model": response.model,
                "tokens_in": response.prompt_tokens,
                "tokens_out": response.completion_tokens,
                "elapsed_ms": response.elapsed_ms,
                "top_k_retrieve": ctx.k_retrieve,
                "top_k_after_rerank": ctx.k_after,
                "num_chunks": len(chunks_after),
                "num_citations": len(citations),
                "num_sources": len(sources),
                "parse_failed": parsed.parse_failed,
            },
        )

    def _is_weak_context(self, chunks_after) -> bool:
        min_chunks = int(self._cfg("rag.unknown_threshold.min_chunks", 1))
        min_score = float(self._cfg("rag.unknown_threshold.min_score", 0.35))
        if len(chunks_after) < min_chunks:
            return True
        best_score = max((c.score for c in chunks_after), default=0.0)
        return best_score < min_score

    def _unknown_result(self, ctx: RetrievalContext) -> RAGResult:
        parsed = ParsedAnswer(answer_text=UNKNOWN_ANSWER, raw=UNKNOWN_ANSWER, parse_failed=False)
        return RAGResult(
            answer=UNKNOWN_ANSWER,
            mode="rag",
            used_chunks=[],
            prompt="",
            original_query=ctx.original_query,
            rewritten_query=ctx.rewritten_query if ctx.rewritten_query != ctx.original_query else None,
            chunks_before_rerank=ctx.chunks_before,
            chunks_after_rerank=ctx.chunks_after,
            rewrite_meta=ctx.rewrite_meta,
            rerank_meta=ctx.rerank_meta,
            answer_status="unknown",
            parsed_answer=parsed,
            citations=[],
            sources=[],
            citations_verified=True,
            sources_verified=True,
            meta={
                "model": "",
                "tokens_in": 0,
                "tokens_out": 0,
                "elapsed_ms": 0,
                "top_k_retrieve": ctx.k_retrieve,
                "top_k_after_rerank": ctx.k_after,
                "num_chunks": len(ctx.chunks_after),
                "num_citations": 0,
                "num_sources": 0,
                "parse_failed": False,
            },
        )

