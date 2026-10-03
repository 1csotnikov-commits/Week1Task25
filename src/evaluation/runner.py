"""Прогон контрольных вопросов в нескольких режимах (RAG / No-RAG / варианты)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.evaluation.comparator import aggregate_modes, evaluate_mode
from src.evaluation.questions import Question
from src.evaluation.report import render_report
from src.rag.models import RAGResult

DEFAULT_MODES = ("no_rag", "rag_no_rerank", "rag")
ALL_MODES = ("no_rag", "rag_no_rerank", "rag", "rag_no_rewrite", "rag_full")


@dataclass
class QuestionResult:
    """Результат одного вопроса по нескольким режимам."""

    question: Question
    modes: Dict[str, RAGResult]


def mode_flags(mode: str, config):
    """Вернуть (use_reranker, use_rewrite) для RAG-режима, либо None для no_rag."""
    if mode == "no_rag":
        return None
    if mode == "rag_no_rerank":
        return (False, False)
    if mode == "rag":
        return (bool(config.get("reranking.enabled", True)), bool(config.get("rewrite.enabled", False)))
    if mode == "rag_no_rewrite":
        return (True, False)
    if mode == "rag_full":
        return (True, True)
    raise ValueError(f"Неизвестный режим: {mode!r}. Доступно: {', '.join(ALL_MODES)}")


def run_evaluation(
    questions: List[Question],
    rag_pipeline,
    no_rag_pipeline,
    modes,
    config,
    top_k_retrieve: Optional[int] = None,
    top_k_after_rerank: Optional[int] = None,
) -> List[QuestionResult]:
    results = []
    for q in questions:
        mode_results: Dict[str, RAGResult] = {}
        for mode in modes:
            flags = mode_flags(mode, config)
            if mode == "no_rag":
                mode_results[mode] = no_rag_pipeline.run(q.question)
            else:
                use_reranker, use_rewrite = flags
                mode_results[mode] = rag_pipeline.run(
                    q.question,
                    top_k_retrieve=top_k_retrieve,
                    top_k_after_rerank=top_k_after_rerank,
                    use_reranker=use_reranker,
                    use_rewrite=use_rewrite,
                )
        results.append(QuestionResult(question=q, modes=mode_results))
    return results


def build_payload(
    results: List[QuestionResult],
    index_name: Optional[str] = None,
    top_k_retrieve: Optional[int] = None,
    top_k_after_rerank: Optional[int] = None,
    modes=None,
    check_citations: Optional[bool] = None,
    check_sources: Optional[bool] = None,
    check_quote_match: Optional[bool] = None,
) -> dict:
    modes_list = list(modes) if modes else (list(results[0].modes.keys()) if results else [])
    rows = []
    for r in results:
        rows.append({
            "id": r.question.id,
            "question": r.question.question,
            "expected_answer": r.question.expected_answer,
            "expected_sources": r.question.expected_sources,
            "modes": {mode: evaluate_mode(rag_result, r.question) for mode, rag_result in r.modes.items()},
        })
    return {
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "index_name": index_name,
        "top_k_retrieve": top_k_retrieve,
        "top_k_after_rerank": top_k_after_rerank,
        "check_citations": check_citations,
        "check_sources": check_sources,
        "check_quote_match": check_quote_match,
        "modes": modes_list,
        "aggregate": aggregate_modes(rows),
        "questions": rows,
    }


def save_results(
    results: List[QuestionResult],
    results_dir: Union[str, Path],
    index_name: Optional[str] = None,
    top_k_retrieve: Optional[int] = None,
    top_k_after_rerank: Optional[int] = None,
    modes=None,
    check_citations: Optional[bool] = None,
    check_sources: Optional[bool] = None,
    check_quote_match: Optional[bool] = None,
):
    """Сохранить результаты в JSON и Markdown. Возвращает (json_path, md_path, payload)."""
    payload = build_payload(results, index_name, top_k_retrieve, top_k_after_rerank, modes, check_citations, check_sources, check_quote_match)
    d = Path(results_dir)
    d.mkdir(parents=True, exist_ok=True)
    ts = payload["timestamp"]
    json_path = d / f"{ts}.json"
    md_path = d / f"{ts}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_report(payload, "markdown"), encoding="utf-8")
    return json_path, md_path, payload


def load_payload(path: Union[str, Path]) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

