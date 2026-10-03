"""Эвристики сравнения ответов с ожиданиями (без LLM-as-judge).

Это грубая автоматическая оценка: пересечение ключевых слов, подстрока, Жаккар.
Точное сравнение пользователь делает вручную.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

STOP_WORDS = set(
    "и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее кажется сейчас были куда зачем сказать всех".split()
)

_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+")


def tokenize(text: str) -> List[str]:
    return [w for w in _TOKEN_RE.findall(text.lower()) if w not in STOP_WORDS and len(w) > 1]


def keyword_overlap(answer: str, expected: str) -> float:
    a = set(tokenize(answer))
    e = set(tokenize(expected))
    if not e:
        return 0.0
    return len(a & e) / len(e)


def jaccard(answer: str, expected: str) -> float:
    a = set(tokenize(answer))
    e = set(tokenize(expected))
    if not (a | e):
        return 0.0
    return len(a & e) / len(a | e)


def substring_match(answer: str, expected: str) -> bool:
    return bool(expected.strip()) and expected.strip().lower() in answer.lower()


def score_answer(answer: str, expected: str) -> Dict[str, Any]:
    return {
        "keyword_overlap": round(keyword_overlap(answer, expected), 4),
        "jaccard": round(jaccard(answer, expected), 4),
        "substring_match": substring_match(answer, expected),
    }


def is_hit(score: Dict[str, Any]) -> bool:
    """Считается ли ответ «попавшим» в ожидание (грубо)."""
    return bool(score.get("substring_match")) or float(score.get("keyword_overlap", 0.0)) >= 0.5


def source_hit(used_chunks: List, expected_sources: List[str]) -> Dict[str, Any]:
    """Проверить попадание использованных чанков в ожидаемые источники."""
    used: set = set()
    for c in used_chunks:
        used.add(c.source)
        if c.section:
            used.add(f"{c.source}#{c.section}")
    expected = set(expected_sources or [])
    if not expected:
        return {"expected_sources": [], "matched": [], "hit": None}
    matched = sorted(used & expected)
    return {"expected_sources": sorted(expected), "matched": matched, "hit": bool(matched)}


def evaluate_mode(rag_result, question) -> Dict[str, Any]:
    """Оценить ответ одного режима против ожидания вопроса."""
    score = score_answer(rag_result.answer, question.expected_answer)
    src = source_hit(rag_result.used_chunks, question.expected_sources)
    return {
        "answer": rag_result.answer,
        "score": score,
        "hit": is_hit(score),
        "sources": src,
        "time_ms": rag_result.meta.get("elapsed_ms", 0),
        "tokens_in": rag_result.meta.get("tokens_in", 0),
        "tokens_out": rag_result.meta.get("tokens_out", 0),
        "used_chunks": [
            {"chunk_id": c.chunk_id, "source": c.source, "section": c.section, "score": round(c.score, 4)}
            for c in rag_result.used_chunks
        ],
        "num_chunks_before": len(rag_result.chunks_before_rerank),
        "num_chunks_after": len(rag_result.chunks_after_rerank),
        # Task 24: источники и цитаты.
        "answer_status": rag_result.answer_status,
        "has_sources": bool(rag_result.sources),
        "has_citations": bool(rag_result.citations),
        "citations_verified": rag_result.citations_verified,
        "sources_verified": rag_result.sources_verified,
        "num_citations": len(rag_result.citations),
        "num_sources": len(rag_result.sources),
    }


def aggregate_modes(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Агрегировать метрики по всем режимам (строки уже содержат 'modes')."""
    n = len(rows)
    if not n:
        return {"num_questions": 0, "per_mode": {}}

    modes = list(rows[0]["modes"].keys())
    per_mode: Dict[str, Any] = {}
    for mode in modes:
        hits = sum(1 for r in rows if r["modes"][mode]["hit"])
        src_hit = sum(1 for r in rows if r["modes"][mode]["sources"]["hit"] is True)
        src_applicable = sum(1 for r in rows if r["modes"][mode]["sources"]["hit"] is not None)
        has_sources = sum(1 for r in rows if r["modes"][mode]["has_sources"])
        has_citations = sum(1 for r in rows if r["modes"][mode]["has_citations"])
        cit_verified = sum(1 for r in rows if r["modes"][mode]["citations_verified"])
        src_verified = sum(1 for r in rows if r["modes"][mode]["sources_verified"])
        unknown = sum(1 for r in rows if r["modes"][mode]["answer_status"] == "unknown")
        per_mode[mode] = {
            "hit_rate": round(hits / n, 4),
            "source_hit_rate": round(src_hit / src_applicable, 4) if src_applicable else 0.0,
            "avg_time_ms": round(sum(r["modes"][mode]["time_ms"] for r in rows) / n, 1),
            "avg_tokens_in": round(sum(r["modes"][mode]["tokens_in"] for r in rows) / n, 1),
            "avg_chunks_before": round(sum(r["modes"][mode]["num_chunks_before"] for r in rows) / n, 2),
            "avg_chunks_after": round(sum(r["modes"][mode]["num_chunks_after"] for r in rows) / n, 2),
            "sources_present_rate": round(has_sources / n, 4),
            "citations_present_rate": round(has_citations / n, 4),
            "citations_verified_rate": round(cit_verified / n, 4),
            "sources_verified_rate": round(src_verified / n, 4),
            "unknown_rate": round(unknown / n, 4),
            "avg_citations": round(sum(r["modes"][mode]["num_citations"] for r in rows) / n, 2),
            "avg_sources": round(sum(r["modes"][mode]["num_sources"] for r in rows) / n, 2),
        }
    return {"num_questions": n, "per_mode": per_mode}
