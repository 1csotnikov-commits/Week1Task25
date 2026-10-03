"""Цитаты, источники и парсинг структурированного ответа (День 24).

Формат ответа LLM:
    ОТВЕТ:
    <текст>
    ИСТОЧНИКИ:
    - source=<file> | chunk_id=<id>
    ЦИТАТЫ:
    - "<дословный фрагмент>" | source=<file> | chunk_id=<id>
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from src.retrieval.models import RetrievedChunk


@dataclass
class Citation:
    """Дословная цитата из чанка."""

    text: str
    source: str = ""
    section: Optional[str] = None
    chunk_id: str = ""
    verified: bool = False


@dataclass
class SourceRef:
    """Ссылка на источник из ответа LLM."""

    source: str = ""
    section: Optional[str] = None
    chunk_id: str = ""
    verified: bool = False


@dataclass
class ParsedAnswer:
    """Структурированный ответ: текст + источники + цитаты."""

    answer_text: str
    sources: List[SourceRef] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    raw: str = ""
    parse_failed: bool = False


_BLOCK_NAMES = ("ОТВЕТ", "ИСТОЧНИКИ", "ЦИТАТЫ")
_HEADER_RE = re.compile(r"(?im)^\s*(?:#{1,6}\s*)?(ОТВЕТ|ИСТОЧНИКИ|ЦИТАТЫ)\s*:?\s*$")


def _split_blocks(raw: str) -> dict:
    """Разбить сырой ответ на блоки по заголовкам ОТВЕТ / ИСТОЧНИКИ / ЦИТАТЫ."""
    matches = list(_HEADER_RE.finditer(raw))
    blocks = {}
    for i, m in enumerate(matches):
        name = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        blocks[name] = raw[start:end].strip()
    return blocks


def _extract_field(line: str, key: str) -> str:
    m = re.search(re.escape(key) + r"\s*=\s*([^|]+)", line, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _parse_source_line(line: str) -> Optional[SourceRef]:
    line = line.strip().lstrip("-*••\t ").strip()
    if not line:
        return None
    source = _extract_field(line, "source")
    chunk_id = _extract_field(line, "chunk_id")
    if not source:
        # fallback: первый токен до '|' считаем source
        source = line.split("|")[0].strip()
    return SourceRef(source=source, section=None, chunk_id=chunk_id, verified=False)


def _parse_citation_line(line: str) -> Optional[Citation]:
    line = line.strip().lstrip("-*••\t ").strip()
    if not line:
        return None
    m = re.search(r'"([^"]*)"', line)
    text = m.group(1) if m else line
    source = _extract_field(line, "source")
    chunk_id = _extract_field(line, "chunk_id")
    return Citation(text=text.strip(), source=source, section=None, chunk_id=chunk_id, verified=False)


def parse_structured_answer(raw_answer: str) -> ParsedAnswer:
    """Разобрать ответ LLM на три части. При неудаче — fallback на весь текст."""
    raw = (raw_answer or "").strip()
    blocks = _split_blocks(raw)

    if "ОТВЕТ" not in blocks:
        return ParsedAnswer(answer_text=raw, sources=[], citations=[], raw=raw, parse_failed=True)

    answer_text = blocks["ОТВЕТ"].strip()
    sources = [_parse_source_line(ln) for ln in blocks.get("ИСТОЧНИКИ", "").splitlines()]
    citations = [_parse_citation_line(ln) for ln in blocks.get("ЦИТАТЫ", "").splitlines()]
    sources = [s for s in sources if s is not None]
    citations = [c for c in citations if c is not None]
    return ParsedAnswer(answer_text=answer_text, sources=sources, citations=citations, raw=raw, parse_failed=False)


def _verify_citation(cit: Citation, chunks: List[RetrievedChunk]) -> None:
    chunk_ids = {c.chunk_id for c in chunks}
    if cit.chunk_id and cit.chunk_id in chunk_ids:
        cit.verified = True
        return
    q = cit.text.strip().lower()
    if q and any(q in c.text.lower() for c in chunks):
        cit.verified = True


def extract_citations(answer: str, chunks: List[RetrievedChunk]) -> List[Citation]:
    """Извлечь цитаты и пометить verified/unverified (анти-галлюцинация)."""
    parsed = parse_structured_answer(answer)
    for cit in parsed.citations:
        _verify_citation(cit, chunks)
    return parsed.citations


def extract_sources(answer: str, chunks: List[RetrievedChunk]) -> List[SourceRef]:
    """Извлечь источники и пометить verified/unverified."""
    parsed = parse_structured_answer(answer)
    chunk_ids = {c.chunk_id for c in chunks}
    source_names = {c.source for c in chunks}
    for src in parsed.sources:
        if src.chunk_id and src.chunk_id in chunk_ids:
            src.verified = True
        elif src.source and src.source in source_names:
            src.verified = True
    return parsed.sources


__all__ = [
    "Citation",
    "SourceRef",
    "ParsedAnswer",
    "parse_structured_answer",
    "extract_citations",
    "extract_sources",
]
