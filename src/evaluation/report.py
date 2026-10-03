"""Отрисовка отчётов об оценке (console / markdown / json) с разбивкой по режимам."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union


def render_report(payload: Dict[str, Any], fmt: str = "console") -> str:
    fmt = (fmt or "console").lower()
    if fmt == "json":
        return json.dumps(payload, ensure_ascii=False, indent=2)
    if fmt == "markdown":
        return _render_markdown(payload)
    if fmt == "console":
        return _render_console(payload)
    raise ValueError(f"Неизвестный формат: {fmt!r}. Доступно: console, markdown, json")


def render_report_file(path: Union[str, Path], fmt: str = "console") -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return render_report(payload, fmt)


def _short(text: str, limit: int = 180) -> str:
    t = " ".join(text.split())
    return t if len(t) <= limit else t[:limit] + "…"


def _render_console(payload: Dict[str, Any]) -> str:
    agg = payload.get("aggregate", {})
    modes = payload.get("modes", [])
    lines = ["===== Отчёт об оценке =====", ""]
    lines.append(f"Индекс: {payload.get('index_name') or '-'}")
    lines.append(f"Время: {payload.get('timestamp') or '-'}")
    lines.append(f"top_k_retrieve: {payload.get('top_k_retrieve')}")
    lines.append(f"top_k_after_rerank: {payload.get('top_k_after_rerank')}")
    lines.append("")
    lines.append("--- Сводка по режимам ---")
    lines.append(f"{'Режим':<16} {'HitRate':>8} {'SrcHit':>8} {'AvgTime':>9} {'AvgTokIn':>9} {'Chunks':>14}")
    per = agg.get("per_mode", {})
    for mode in modes:
        m = per.get(mode, {})
        src = m.get("source_hit_rate")
        src_s = f"{src:.2%}" if src is not None else "-"
        chunks = f"{m.get('avg_chunks_before', 0)}/{m.get('avg_chunks_after', 0)}"
        lines.append(
            f"{mode:<16} {m.get('hit_rate', 0):>8.2%} {src_s:>8} "
            f"{m.get('avg_time_ms', 0):>8.0f}ms {m.get('avg_tokens_in', 0):>9.1f} {chunks:>14}"
        )
    lines.append("")
    lines.append("--- Источники и цитаты ---")
    lines.append(f"{'Режим':<16} {'Sources':>8} {'Cit':>8} {'CitVerif':>9} {'SrcVerif':>9} {'Unknown':>9} {'AvgCit':>7} {'AvgSrc':>7}")
    for mode in modes:
        m = per.get(mode, {})
        lines.append(
            f"{mode:<16} {m.get('sources_present_rate', 0):>8.2%} {m.get('citations_present_rate', 0):>8.2%} "
            f"{m.get('citations_verified_rate', 0):>9.2%} {m.get('sources_verified_rate', 0):>9.2%} "
            f"{m.get('unknown_rate', 0):>9.2%} {m.get('avg_citations', 0):>7.1f} {m.get('avg_sources', 0):>7.1f}"
        )
    lines.append("")
    lines.append("--- Вопросы ---")
    for row in payload.get("questions", []):
        lines.append(f"[{row['id']}] {row['question']}")
        lines.append(f"  Ожидание: {row['expected_answer']}")
        if row.get("expected_sources"):
            lines.append(f"  Источники: {', '.join(row['expected_sources'])}")
        for mode in modes:
            m = row.get("modes", {}).get(mode)
            if not m:
                continue
            mark = "попал" if m["hit"] else "мимо"
            src = m["sources"]
            src_s = ""
            if row.get("expected_sources"):
                src_s = f" | источники: {'OK' if src['hit'] else 'мимо'} ({', '.join(src['matched']) or '-'})"
            lines.append(f"  [{mode}] ({mark}): {_short(m['answer'])}{src_s}")
        lines.append("")
    return "\n".join(lines)


def _render_markdown(payload: Dict[str, Any]) -> str:
    agg = payload.get("aggregate", {})
    modes = payload.get("modes", [])
    lines = ["# Отчёт об оценке", ""]
    lines.append(f"- Индекс: `{payload.get('index_name') or '-'}`")
    lines.append(f"- Время: {payload.get('timestamp') or '-'}")
    lines.append("")
    lines.append("## Сводка по режимам")
    lines.append("| Режим | HitRate | SrcHit | AvgTime | AvgTokIn | Chunks(before/after) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    per = agg.get("per_mode", {})
    for mode in modes:
        m = per.get(mode, {})
        src = m.get("source_hit_rate")
        src_s = f"{src:.2%}" if src is not None else "-"
        chunks = f"{m.get('avg_chunks_before', 0)}/{m.get('avg_chunks_after', 0)}"
        lines.append(
            f"| {mode} | {m.get('hit_rate', 0):.2%} | {src_s} | "
            f"{m.get('avg_time_ms', 0):.0f}ms | {m.get('avg_tokens_in', 0):.1f} | {chunks} |"
        )
    lines.append("")
    lines.append("## Источники и цитаты")
    lines.append("| Режим | Sources | Citations | CitVerified | SrcVerified | Unknown | AvgCit | AvgSrc |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for mode in modes:
        m = per.get(mode, {})
        lines.append(
            f"| {mode} | {m.get('sources_present_rate', 0):.2%} | {m.get('citations_present_rate', 0):.2%} | "
            f"{m.get('citations_verified_rate', 0):.2%} | {m.get('sources_verified_rate', 0):.2%} | "
            f"{m.get('unknown_rate', 0):.2%} | {m.get('avg_citations', 0):.1f} | {m.get('avg_sources', 0):.1f} |"
        )
    lines.append("")
    for row in payload.get("questions", []):
        lines.append(f"## {row['id']}. {row['question']}")
        lines.append("")
        lines.append(f"**Ожидание:** {row['expected_answer']}")
        if row.get("expected_sources"):
            lines.append(f"**Источники:** {', '.join(row['expected_sources'])}")
        lines.append("")
        for mode in modes:
            m = row.get("modes", {}).get(mode)
            if not m:
                continue
            mark = "попал" if m["hit"] else "мимо"
            lines.append(f"### {mode} ({mark})")
            lines.append("")
            lines.append(f"> {m['answer']}")
            lines.append("")
    return "\n".join(lines)

