"""Точка входа CLI (интерактивный REPL) пайплайна индексации."""

from __future__ import annotations

import shlex
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from src.cli.commands import (
    EVAL_MODE_CHOICES,
    FORMAT_CHOICES,
    RERANKER_CHOICES,
    REPORT_FORMAT_CHOICES,
    REWRITER_CHOICES,
    STRATEGY_CHOICES,
    build_help,
    list_index_subcommands,
)
from src.chat import ChatPipeline, ChatSession, HistoryStore, make_state_updater
from src.chat.chat_pipeline import render_state
from src.chat.commands import build_chat_help, parse_chat_command
from src.chat.scenarios import list_scenarios, load_scenario, run_scenario, save_scenario_results
from src.chat.task_state import TaskState
from src.config import Config
from src.embeddings.nomic_provider import NomicProvider
from src.evaluation.generator import generate_questions
from src.evaluation.questions import load_questions, save_questions
from src.evaluation.report import render_report, render_report_file
from src.evaluation.runner import ALL_MODES, DEFAULT_MODES, run_evaluation, save_results
from src.indexing.faiss_sqlite_store import FAISSSQLiteStore
from src.indexing.pipeline import build_index
from src.llm.deepseek_provider import DeepSeekProvider
from src.rag.no_rag_pipeline import NoRAGPipeline
from src.rag.pipeline import RAGPipeline
from src.rag.query_rewrite import NoOpRewriter, make_rewriter
from src.reranking import make_reranker
from src.reranking.pipeline import RerankerPipeline
from src.retrieval.vector_retriever import VectorRetriever
from src.stats.chunk_stats import compute_stats, render, render_compare

BANNER = "Пайплайн индексации документов. Введите /help для списка команд, /exit для выхода."


def parse_args(tokens: List[str]) -> Tuple[List[str], dict]:
    """Разобрать аргументы: позиционные + флаги --flag value / --flag=value."""
    positional: List[str] = []
    flags: dict = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            key = token[2:]
            if "=" in key:
                k, v = key.split("=", 1)
                flags[k] = v
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                flags[key] = tokens[i + 1]
                i += 1
            else:
                flags[key] = True
        else:
            positional.append(token)
        i += 1
    return positional, flags


class App:
    """Состояние CLI-приложения."""

    def __init__(self, config: Config):
        self.config = config
        self.debug = False
        self.store = FAISSSQLiteStore(config.index_path)


def _require_positional(positional: List[str], n: int, usage: str) -> None:
    if len(positional) < n:
        raise ValueError(f"Недостаточно аргументов. Использование: {usage}")


def _load_stats(app: App, index_name: str) -> dict:
    info = app.store.info(index_name)
    app.store.load(index_name)
    chunks = app.store.list_chunks()
    documents = app.store.get_documents()
    strategy = info.get("strategy") or app.store._strategy or ""
    elapsed = float(info.get("indexing_time_seconds") or 0.0)
    return compute_stats(documents, chunks, elapsed, strategy, index_name)


# ---------------------------------------------------------------------- #
# Обработчики команд
# ---------------------------------------------------------------------- #
def _cmd_help(args: List[str]) -> str:
    if args:
        return build_help(args[0])
    return build_help()


def _cmd_debug(app: App, args: List[str]) -> str:
    _require_positional(args, 1, "/debug on|off")
    mode = args[0].lower()
    if mode == "on":
        app.debug = True
    elif mode == "off":
        app.debug = False
    else:
        raise ValueError(f"Режим отладки: on или off, получено {mode!r}")
    return f"Отладка: {'включена' if app.debug else 'выключена'}"


def _cmd_index(app: App, args: List[str]) -> str:
    if not args:
        return build_help("/index")
    sub = args[0]
    rest = args[1:]

    if sub == "build":
        return _cmd_build(app, rest)
    if sub == "list":
        return _cmd_list(app)
    if sub == "info":
        _require_positional(rest, 1, "/index info <index_name>")
        return _cmd_info(app, rest[0])
    if sub == "stats":
        _require_positional(rest, 1, "/index stats <index_name> [--format ...]")
        pos, flags = parse_args(rest)
        return _cmd_stats(app, pos[0], flags)
    if sub == "compare":
        _require_positional(rest, 2, "/index compare <index1> <index2> [--format ...]")
        pos, flags = parse_args(rest)
        return _cmd_compare(app, pos[0], pos[1], flags)
    if sub == "show-chunk":
        _require_positional(rest, 2, "/index show-chunk <index_name> <chunk_id>")
        return _cmd_show_chunk(app, rest[0], rest[1])
    if sub == "list-chunks":
        _require_positional(rest, 1, "/index list-chunks <index_name> [--source ...] [--section ...] [--limit N]")
        pos, flags = parse_args(rest)
        return _cmd_list_chunks(app, pos[0], flags)
    if sub == "delete":
        _require_positional(rest, 1, "/index delete <index_name>")
        return _cmd_delete(app, rest[0])

    raise ValueError(f"Неизвестная подкоманда /index: {sub!r}. Доступно: {', '.join(list_index_subcommands())}")


def _cmd_build(app: App, args: List[str]) -> str:
    pos, flags = parse_args(args)
    if len(pos) > 1:
        raise ValueError(
            "Лишние позиционные аргументы. Стратегию и имя задавайте через флаги. "
            "Правильный синтаксис: /index build <path> [--strategy fixed|structural] [--name <index_name>] [--config <path>]. "
            f"Лишние аргументы: {' '.join(repr(a) for a in pos[1:])}"
        )
    config = app.config
    if flags.get("config"):
        config = Config.load(flags["config"])

    documents_path = Path(pos[0]) if pos else config.documents_path
    strategy = flags.get("strategy") or config.default_strategy
    if strategy not in STRATEGY_CHOICES:
        raise ValueError(f"Неизвестная стратегия: {strategy!r}. Доступно: {', '.join(STRATEGY_CHOICES)}")

    index_name = flags.get("name") or f"{strategy}-{datetime.now():%Y%m%d-%H%M%S}"

    result = build_index(documents_path, strategy, index_name, config)
    stats = compute_stats(result.documents, result.chunks, result.elapsed, strategy, index_name)
    return render(stats, "console")


def _cmd_list(app: App) -> str:
    indices = app.store.list_indices()
    if not indices:
        return "Индексов нет. Постройте индекс: /index build <path>"
    lines = ["Доступные индексы:"]
    for info in indices:
        lines.append(
            f"  {info['name']:<25} strategy={info['strategy']:<10} "
            f"docs={info['num_documents']} chunks={info['num_chunks']} store={info['store_type']}"
        )
    return "\n".join(lines)


def _cmd_info(app: App, index_name: str) -> str:
    info = app.store.info(index_name)
    lines = [f"Индекс: {info['name']}"]
    for key in ("strategy", "model", "provider", "dim", "store_type", "num_documents", "num_chunks", "indexing_time_seconds", "created_at", "path"):
        lines.append(f"  {key}: {info.get(key, '')}")
    return "\n".join(lines)


def _cmd_stats(app: App, index_name: str, flags: dict) -> str:
    fmt = flags.get("format") or "console"
    if fmt not in FORMAT_CHOICES:
        raise ValueError(f"Неизвестный формат: {fmt!r}. Доступно: {', '.join(FORMAT_CHOICES)}")

    stats = _load_stats(app, index_name)
    output = render(stats, fmt)
    if fmt in ("json", "csv", "markdown"):
        ext = {"json": "json", "csv": "csv", "markdown": "md"}[fmt]
        out_path = app.store._index_dir(index_name) / f"stats.{ext}"
        out_path.write_text(output, encoding="utf-8")
        return output + f"\n\nОтчёт сохранён: {out_path}"
    return output


def _cmd_compare(app: App, a: str, b: str, flags: dict) -> str:
    fmt = flags.get("format") or "console"
    if fmt not in FORMAT_CHOICES:
        raise ValueError(f"Неизвестный формат: {fmt!r}. Доступно: {', '.join(FORMAT_CHOICES)}")

    stats_a = _load_stats(app, a)
    stats_b = _load_stats(app, b)
    output = render_compare(stats_a, stats_b, fmt)
    if fmt in ("json", "csv", "markdown"):
        ext = {"json": "json", "csv": "csv", "markdown": "md"}[fmt]
        out_path = app.config.index_path / f"compare_{a}_vs_{b}.{ext}"
        out_path.write_text(output, encoding="utf-8")
        return output + f"\n\nОтчёт сохранён: {out_path}"
    return output


def _cmd_show_chunk(app: App, index_name: str, chunk_id: str) -> str:
    app.store.load(index_name)
    chunk = app.store.get_chunk(chunk_id)
    if chunk is None:
        raise ValueError(f"Чанк '{chunk_id}' не найден в индексе '{index_name}'")

    lines = ["--- Текст чанка ---", chunk.text, "", "--- Метаданные ---"]
    lines.append(f"chunk_id: {chunk.chunk_id}")
    lines.append(f"source: {chunk.source}")
    lines.append(f"title: {chunk.title}")
    lines.append(f"section: {chunk.section}")
    lines.append(f"strategy: {chunk.strategy}")
    lines.append(f"char_count: {chunk.char_count}")
    lines.append(f"token_count (эвристика): {chunk.token_count}")
    lines.append(f"start_char: {chunk.start_char}")
    lines.append(f"end_char: {chunk.end_char}")
    if chunk.metadata:
        lines.append(f"metadata: {chunk.metadata}")

    if app.debug:
        emb = app.store.get_embedding(chunk_id)
        if emb is not None:
            n = 10
            lines.append("")
            lines.append(f"Эмбеддинг (первые {n} значений): {[round(float(x), 4) for x in emb[:n]]}")
    return "\n".join(lines)


def _cmd_list_chunks(app: App, index_name: str, flags: dict) -> str:
    app.store.load(index_name)
    source = flags.get("source")
    section = flags.get("section")
    limit = None
    if flags.get("limit"):
        limit = int(flags["limit"])

    chunks = app.store.list_chunks(source=source, section=section, limit=limit)
    if not chunks:
        return "Чанков не найдено."
    lines = [f"Чанков: {len(chunks)}"]
    for c in chunks:
        lines.append(
            f"  {c.chunk_id:<40} source={c.source:<20} section={c.section or '-':<20} chars={c.char_count}"
        )
    return "\n".join(lines)


def _cmd_delete(app: App, index_name: str) -> str:
    app.store.delete(index_name)
    return f"Индекс '{index_name}' удалён"


# ---------------------------------------------------------------------- #
# Вспомогательные фабрики (Dependency Injection)
# ---------------------------------------------------------------------- #
def _make_llm(config: Config) -> DeepSeekProvider:
    provider = config.get("llm.provider", "deepseek")
    if provider != "deepseek":
        raise ValueError(f"Неизвестный LLM-провайдер: {provider!r}. Доступен только 'deepseek'.")
    return DeepSeekProvider(
        model=config.get("llm.model", "deepseek-chat"),
        base_url=config.get("llm.base_url", "https://api.deepseek.com"),
        temperature=float(config.get("llm.temperature", 0.3)),
        max_tokens=int(config.get("llm.max_tokens", 1024)),
        timeout=float(config.get("llm.timeout", 60.0)),
    )


def _make_embedder(config: Config) -> NomicProvider:
    return NomicProvider(
        model_name=config.embedding_model,
        device=config.get("embedding.device", "cpu"),
        batch_size=config.embedding_batch_size,
        trust_remote_code=bool(config.get("embedding.trust_remote_code", True)),
    )


def _make_retriever(app: App, index_name: str) -> VectorRetriever:
    return VectorRetriever(store=app.store, embedder=_make_embedder(app.config), index_name=index_name)


def _resolve_index(config: Config, flags: dict) -> str:
    index = flags.get("index") or config.get("rag.default_index", "")
    if not index:
        raise ValueError("Не указан индекс. Укажите --index <index_name> или задайте rag.default_index в config.yaml.")
    return index


def _resolve_reranker(app: App, flags: dict):
    """Создать реранкер по флагу --reranker или конфигу."""
    flag = flags.get("reranker")
    if flag is not None:
        return make_reranker(flag, app.config)
    if bool(app.config.get("reranking.enabled", True)):
        return make_reranker(app.config.get("reranking.provider", "threshold"), app.config)
    return None


def _resolve_rewriter(app: App, flags: dict, llm):
    """Создать переписыватель запроса по флагу --rewrite или конфигу."""
    flag = flags.get("rewrite")
    if flag is not None:
        return make_rewriter(flag, app.config, llm)
    if bool(app.config.get("rewrite.enabled", False)):
        return make_rewriter(app.config.get("rewrite.provider", "none"), app.config, llm)
    return None


def _parse_on_off(flags: dict, key: str, default: bool = True) -> bool:
    if key not in flags:
        return default
    v = str(flags[key]).lower()
    if v in ("on", "true", "1", "yes"):
        return True
    if v in ("off", "false", "0", "no"):
        return False
    raise ValueError(f"Флаг --{key}: on|off, получено {flags[key]!r}")


def _render_questions_short(questions) -> str:
    lines = []
    for q in questions:
        lines.append(f"  {q.id}: {q.question}")
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# Команды /ask, /ask-raw, /ask-compare
# ---------------------------------------------------------------------- #
def _cmd_ask(app: App, args: List[str]) -> str:
    pos, flags = parse_args(args)
    _require_positional(pos, 1, "/ask \"<вопрос>\" [--index <index_name>] [--top-k-retrieve N] [--reranker ...] [--rewrite ...]")
    question = pos[0]
    index_name = _resolve_index(app.config, flags)
    top_k_retrieve = int(flags["top-k-retrieve"]) if flags.get("top-k-retrieve") else None
    top_k_after = int(flags["top-k-after-rerank"]) if flags.get("top-k-after-rerank") else None
    debug = bool(flags.get("debug")) or app.debug

    llm = _make_llm(app.config)
    reranker = _resolve_reranker(app, flags)
    rewriter = _resolve_rewriter(app, flags, llm)
    pipeline = RAGPipeline(_make_retriever(app, index_name), llm, app.config, reranker=reranker, rewriter=rewriter)
    result = pipeline.run(
        question,
        top_k_retrieve=top_k_retrieve,
        top_k_after_rerank=top_k_after,
        use_reranker=reranker is not None,
        use_rewrite=rewriter is not None,
    )

    lines = [f"Статус: {result.answer_status}", ""]
    lines.append("--- Ответ ---")
    lines.append(result.answer)

    lines.append("")
    lines.append("--- Источники ---")
    if result.sources:
        for s in result.sources:
            mark = "verified" if s.verified else "unverified"
            lines.append(f"  [{mark}] {s.source} | chunk_id={s.chunk_id}")
    else:
        lines.append("  (нет)")

    lines.append("")
    lines.append("--- Цитаты ---")
    if result.citations:
        for c in result.citations:
            mark = "verified" if c.verified else "unverified"
            lines.append(f"  [{mark}] \"{c.text}\" | {c.source} | chunk_id={c.chunk_id}")
    else:
        lines.append("  (нет)")

    if debug:
        lines.append("")
        lines.append("--- Полный структурированный ответ LLM ---")
        lines.append(result.parsed_answer.raw if result.parsed_answer else "")
        lines.append("")
        lines.append(f"--- Rewrite: {result.rewrite_meta.get('method', 'none')} ---")
        if result.rewritten_query:
            lines.append(f"  Переписанный запрос: {result.rewritten_query}")
        lines.append("")
        lines.append(f"--- Rerank: {result.rerank_meta.get('provider', 'none')} ---")
        lines.append(f"  Чанков до/после: {result.rerank_meta.get('before', 0)} -> {result.rerank_meta.get('after', 0)} (отсечено {result.rerank_meta.get('removed', 0)})")
        lines.append("")
        lines.append("--- Чанки до rerank ---")
        for c in result.chunks_before_rerank:
            lines.append(f"  [{c.rank}] {c.source} | {c.section or '-'} | {c.chunk_id} | score={c.score:.4f}")
        lines.append("")
        lines.append("--- Чанки после rerank ---")
        for c in result.chunks_after_rerank:
            lines.append(f"  [{c.rank}] {c.source} | {c.section or '-'} | {c.chunk_id} | score={c.score:.4f}")
        lines.append("")
        lines.append("--- Полный промт ---")
        lines.append(result.prompt)
    return "\n".join(lines)


def _cmd_ask_raw(app: App, args: List[str]) -> str:
    pos, flags = parse_args(args)
    _require_positional(pos, 1, "/ask-raw \"<вопрос>\" [--debug]")
    question = pos[0]
    debug = bool(flags.get("debug")) or app.debug

    pipeline = NoRAGPipeline(_make_llm(app.config), app.config)
    result = pipeline.run(question)

    lines = ["--- Ответ (No-RAG) ---", result.answer]
    if debug:
        lines.append("")
        lines.append("--- Полный промт ---")
        lines.append(result.prompt)
    return "\n".join(lines)


def _cmd_ask_compare(app: App, args: List[str]) -> str:
    pos, flags = parse_args(args)
    _require_positional(pos, 1, "/ask-compare \"<вопрос>\" [--index <index_name>] [--top-k-retrieve N]")
    question = pos[0]
    index_name = _resolve_index(app.config, flags)
    top_k_retrieve = int(flags["top-k-retrieve"]) if flags.get("top-k-retrieve") else None

    llm = _make_llm(app.config)
    reranker = _resolve_reranker(app, flags)
    rewriter = _resolve_rewriter(app, flags, llm)
    pipeline = RAGPipeline(_make_retriever(app, index_name), llm, app.config, reranker=reranker, rewriter=rewriter)
    no_rag = NoRAGPipeline(llm, app.config)

    no_rag_res = no_rag.run(question)
    rag_no_rerank = pipeline.run(question, top_k_retrieve=top_k_retrieve, use_reranker=False, use_rewrite=False)
    rag = pipeline.run(question, top_k_retrieve=top_k_retrieve, use_reranker=reranker is not None, use_rewrite=rewriter is not None)

    lines = [f"===== Сравнение режимов =====", "", f"Вопрос: {question}", ""]
    lines.append("--- no_rag ---")
    lines.append(no_rag_res.answer)
    lines.append("")
    lines.append("--- rag_no_rerank ---")
    lines.append(rag_no_rerank.answer)
    lines.append("")
    lines.append("--- rag (с reranker) ---")
    lines.append(rag.answer)
    lines.append("")
    lines.append("--- Источники (rag) ---")
    if not rag.used_chunks:
        lines.append("  (чанки не найдены)")
    for c in rag.used_chunks:
        lines.append(f"  [{c.rank}] {c.source} | {c.section or '-'} | {c.chunk_id} | score={c.score:.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# Команды /rerank, /rewrite
# ---------------------------------------------------------------------- #
def _cmd_rerank(app: App, args: List[str]) -> str:
    if not args:
        return build_help("/rerank")
    sub = args[0]
    rest = args[1:]
    if sub == "test":
        return _cmd_rerank_test(app, rest)
    raise ValueError(f"Неизвестная подкоманда /rerank: {sub!r}. Доступно: test")


def _cmd_rerank_test(app: App, args: List[str]) -> str:
    pos, flags = parse_args(args)
    _require_positional(pos, 1, "/rerank test \"<запрос>\" [--index ...] [--top-k-retrieve N] [--reranker ...]")
    query = pos[0]
    index_name = _resolve_index(app.config, flags)
    k_retrieve = int(flags["top-k-retrieve"]) if flags.get("top-k-retrieve") else int(app.config.get("retrieval.top_k_retrieve", 10))

    reranker = _resolve_reranker(app, flags)
    chunks_before = _make_retriever(app, index_name).search(query, k_retrieve)

    lines = [f"=== Ранкер: {reranker.name if reranker else 'none'} ===", f"Запрос: {query}", ""]
    lines.append(f"--- До фильтра ({len(chunks_before)} чанков) ---")
    for c in chunks_before:
        lines.append(f"  [{c.rank}] {c.source} | {c.section or '-'} | {c.chunk_id} | score={c.score:.4f}")

    if reranker is None:
        lines.append("")
        lines.append("Ранкер не задан (none) — фильтрация не выполнялась.")
        return "\n".join(lines)

    chunks_after, meta = RerankerPipeline(reranker).apply(chunks_before, query)
    lines.append("")
    lines.append(f"--- После фильтра ({meta['after']} чанков, отсечено {meta['removed']}) ---")
    for c in chunks_after:
        lines.append(f"  [{c.rank}] {c.source} | {c.section or '-'} | {c.chunk_id} | score={c.score:.4f}")
    return "\n".join(lines)


def _cmd_rewrite(app: App, args: List[str]) -> str:
    if not args:
        return build_help("/rewrite")
    sub = args[0]
    rest = args[1:]
    if sub == "test":
        return _cmd_rewrite_test(app, rest)
    raise ValueError(f"Неизвестная подкоманда /rewrite: {sub!r}. Доступно: test")


def _cmd_rewrite_test(app: App, args: List[str]) -> str:
    pos, flags = parse_args(args)
    _require_positional(pos, 1, "/rewrite test \"<запрос>\" [--provider none|heuristic|llm]")
    query = pos[0]
    provider = flags.get("provider") or app.config.get("rewrite.provider", "none")

    rewriter = make_rewriter(provider, app.config, _make_llm(app.config))
    if rewriter is None:
        rewriter = NoOpRewriter()

    result = rewriter.rewrite(query)
    lines = [f"=== Rewrite: {result.method} ===", ""]
    lines.append(f"Оригинал:  {result.original_query}")
    lines.append(f"Переписан: {result.rewritten_query}")
    if result.meta:
        lines.append(f"Мета: {result.meta}")
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# Команды /eval
# ---------------------------------------------------------------------- #
def _cmd_eval(app: App, args: List[str]) -> str:
    if not args:
        return build_help("/eval")
    sub = args[0]
    rest = args[1:]
    if sub == "generate":
        return _cmd_eval_generate(app, rest)
    if sub == "run":
        _require_positional(rest, 1, "/eval run <index_name> [--questions <path>] [--mode <mode>] [--compare-all] [--top-k-retrieve N] [--top-k-after-rerank N]")
        pos, flags = parse_args(rest)
        return _cmd_eval_run(app, pos[0], flags)
    if sub == "report":
        _require_positional(rest, 1, "/eval report <result_file> [--format ...]")
        pos, flags = parse_args(rest)
        return _cmd_eval_report(app, pos[0], flags)
    raise ValueError(f"Неизвестная подкоманда /eval: {sub!r}. Доступно: generate, run, report")


def _cmd_eval_generate(app: App, args: List[str]) -> str:
    _pos, flags = parse_args(args)
    index_name = _resolve_index(app.config, flags)
    count = int(flags.get("count", 10))
    out = flags.get("out") or app.config.get("evaluation.questions", "data/eval/questions.yaml")
    force = bool(flags.get("force"))

    llm = _make_llm(app.config)
    questions = generate_questions(app.store, llm, index_name, count=count)

    out_path = Path(out)
    if out_path.exists() and not force:
        out_path = out_path.with_name(f"{out_path.stem}.generated.yaml")
        print(
            f"ПРЕДУПРЕЖДЕНИЕ: файл '{out}' уже существует. "
            f"Сохраняю в '{out_path}' (используйте --force для перезаписи)."
        )
    save_questions(questions, out_path)

    return f"Сгенерировано {len(questions)} вопросов и сохранено в: {out_path}\n\n{_render_questions_short(questions)}"


def _cmd_eval_run(app: App, index_name: str, flags: dict) -> str:
    questions_path = flags.get("questions") or app.config.get("evaluation.questions", "data/eval/questions.yaml")
    top_k_retrieve = int(flags["top-k-retrieve"]) if flags.get("top-k-retrieve") else None
    top_k_after = int(flags["top-k-after-rerank"]) if flags.get("top-k-after-rerank") else None

    questions = load_questions(questions_path)
    if not questions:
        raise ValueError(f"В '{questions_path}' нет вопросов.")

    llm = _make_llm(app.config)
    reranker = _resolve_reranker(app, flags)
    rewriter = _resolve_rewriter(app, flags, llm)
    rag = RAGPipeline(_make_retriever(app, index_name), llm, app.config, reranker=reranker, rewriter=rewriter)
    no_rag = NoRAGPipeline(llm, app.config)

    if flags.get("compare-all"):
        modes = list(ALL_MODES)
    elif flags.get("mode"):
        mode = flags["mode"]
        if mode not in EVAL_MODE_CHOICES:
            raise ValueError(f"Неизвестный режим: {mode!r}. Доступно: {', '.join(EVAL_MODE_CHOICES)}")
        modes = [mode]
    else:
        modes = list(DEFAULT_MODES)

    check_citations = _parse_on_off(flags, "check-citations", True)
    check_sources = _parse_on_off(flags, "check-sources", True)
    check_quote_match = _parse_on_off(flags, "check-quote-match", True)

    results = run_evaluation(questions, rag, no_rag, modes, app.config, top_k_retrieve=top_k_retrieve, top_k_after_rerank=top_k_after)
    results_dir = app.config.resolve_path(app.config.get("evaluation.results_dir", "data/eval/results"))
    json_path, md_path, payload = save_results(
        results, results_dir, index_name=index_name, top_k_retrieve=top_k_retrieve, top_k_after_rerank=top_k_after, modes=modes,
        check_citations=check_citations, check_sources=check_sources, check_quote_match=check_quote_match,
    )

    return render_report(payload, "console") + f"\n\nСохранено:\n  {json_path}\n  {md_path}"


def _cmd_eval_report(app: App, result_file: str, flags: dict) -> str:
    fmt = flags.get("format") or "console"
    if fmt not in REPORT_FORMAT_CHOICES:
        raise ValueError(f"Неизвестный формат: {fmt!r}. Доступно: {', '.join(REPORT_FORMAT_CHOICES)}")
    return render_report_file(result_file, fmt)


# ---------------------------------------------------------------------- #
# Команды /chat
# ---------------------------------------------------------------------- #
def _build_chat_pipeline(app: App, index_name: str, flags: dict) -> ChatPipeline:
    llm = _make_llm(app.config)
    reranker = _resolve_reranker(app, flags)
    rewriter = _resolve_rewriter(app, flags, llm)
    rag = RAGPipeline(_make_retriever(app, index_name), llm, app.config, reranker=reranker, rewriter=rewriter)
    updater = make_state_updater(
        flags.get("state-updater") or app.config.get("chat.state_updater.provider", "heuristic"),
        app.config,
        llm,
    )
    return ChatPipeline(rag, updater, app.config)


def _chat_sessions_store(app: App) -> HistoryStore:
    return HistoryStore(app.config.resolve_path("data/chat/sessions"))


def _print_chat_result(result, debug: bool) -> None:
    rag = result.rag_result
    print(f"[{rag.answer_status}] {rag.answer}")
    if rag.sources:
        print("  Источники:")
        for s in rag.sources:
            mark = "verified" if s.verified else "unverified"
            print(f"    [{mark}] {s.source} | chunk_id={s.chunk_id}")
    if rag.citations:
        print("  Цитаты:")
        for c in rag.citations:
            mark = "verified" if c.verified else "unverified"
            print(f"    [{mark}] \"{c.text}\" | {c.source} | chunk_id={c.chunk_id}")
    if result.state_updates:
        print(f"  (state: {', '.join(result.state_updates)})")


def _run_chat_repl(app: App, session: ChatSession, pipeline: ChatPipeline) -> None:
    print(f"Чат '{session.name}' (индекс: {session.index_name}). /help — команды, /exit — выход.")
    while True:
        try:
            raw = input("chat> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        line = raw.strip()
        if not line:
            continue
        parsed = parse_chat_command(line)
        if parsed is None:
            result = pipeline.send(session, line)
            _print_chat_result(result, app.debug)
            continue
        name, args = parsed
        if name == "exit":
            break
        if name == "help":
            print(build_chat_help())
        elif name == "state_show":
            print(render_state(session.task_state))
        elif name == "state_edit":
            print(f"task_state.json: {session.store.session_dir(session.name) / 'task_state.json'}")
        elif name == "state_clear":
            session.task_state = TaskState()
            session.save()
            print("Task state очищен.")
        elif name == "history_show":
            n = int(args[0]) if args else 10
            for m in session.recent_messages(n):
                print(f"[{m.role}] {m.content}")
        elif name == "history_clear":
            session.clear_history()
            print("История очищена.")
        elif name == "unknown":
            print(f"Неизвестная команда чата: {args[0]}. /help для списка.")
    session.save()


def _cmd_chat(app: App, args: List[str]) -> str:
    if not args:
        return build_help("/chat")
    sub = args[0]
    rest = args[1:]
    if sub == "new":
        _require_positional(rest, 1, "/chat new <session_name> --index <index_name>")
        pos, flags = parse_args(rest)
        return _cmd_chat_new(app, pos[0], flags)
    if sub == "load":
        _require_positional(rest, 1, "/chat load <session_name>")
        return _cmd_chat_load(app, rest[0])
    if sub == "list":
        return _cmd_chat_list(app)
    if sub == "info":
        _require_positional(rest, 1, "/chat info <session_name>")
        return _cmd_chat_info(app, rest[0])
    if sub == "delete":
        _require_positional(rest, 1, "/chat delete <session_name>")
        return _cmd_chat_delete(app, rest[0])
    if sub == "scenario":
        if not rest:
            return build_help("/chat scenario run")
        if rest[0] == "run":
            _require_positional(rest, 2, "/chat scenario run <scenario_file> [--check-goal] [--check-sources] [--check-citations]")
            pos, flags = parse_args(rest[1:])
            return _cmd_chat_scenario_run(app, pos[0], flags)
        if rest[0] == "list":
            return _cmd_chat_scenario_list(app)
        raise ValueError(f"Неизвестная подкоманда /chat scenario: {rest[0]!r}. Доступно: run, list")
    raise ValueError(f"Неизвестная подкоманда /chat: {sub!r}. Доступно: new, load, list, info, delete, scenario run, scenario list")


def _cmd_chat_new(app: App, session_name: str, flags: dict) -> str:
    index_name = flags.get("index") or app.config.get("chat.default_index_name", "")
    if not index_name:
        raise ValueError("Не указан индекс. Укажите --index <index_name> или chat.default_index_name в config.yaml.")
    store = _chat_sessions_store(app)
    if store.exists(session_name):
        raise ValueError(f"Сессия '{session_name}' уже существует. Используйте /chat load {session_name}.")
    session = ChatSession.create(store, session_name, index_name)
    pipeline = _build_chat_pipeline(app, index_name, flags)
    _run_chat_repl(app, session, pipeline)
    return f"Сессия '{session_name}' сохранена."


def _cmd_chat_load(app: App, session_name: str) -> str:
    session = ChatSession.load(_chat_sessions_store(app), session_name)
    pipeline = _build_chat_pipeline(app, session.index_name, {})
    _run_chat_repl(app, session, pipeline)
    return f"Сессия '{session_name}' сохранена."


def _cmd_chat_list(app: App) -> str:
    metas = _chat_sessions_store(app).list_sessions()
    if not metas:
        return "Сессий нет."
    lines = ["Сессии:"]
    for m in metas:
        lines.append(f"  {m.name:<20} index={m.index_name:<22} msgs={m.message_count} updated={m.updated_at}")
    return "\n".join(lines)


def _cmd_chat_info(app: App, session_name: str) -> str:
    session = ChatSession.load(_chat_sessions_store(app), session_name)
    lines = [f"Сессия: {session.name}", f"  index: {session.index_name}", f"  сообщений: {len(session.history)}", "", "Task state:"]
    lines.append(render_state(session.task_state))
    return "\n".join(lines)


def _cmd_chat_delete(app: App, session_name: str) -> str:
    _chat_sessions_store(app).delete_session(session_name)
    return f"Сессия '{session_name}' удалена."


def _cmd_chat_scenario_run(app: App, scenario_file: str, flags: dict) -> str:
    scenarios_dir = app.config.resolve_path(app.config.get("chat.scenarios.path", "data/chat/scenarios"))
    path = Path(scenario_file)
    if not path.is_absolute() and (scenarios_dir / scenario_file).exists():
        path = scenarios_dir / scenario_file
    scenario = load_scenario(path)
    index_name = scenario.index_name or flags.get("index") or app.config.get("chat.default_index_name", "")
    if not index_name:
        raise ValueError("Сценарий не содержит index_name и не указан --index.")

    store = _chat_sessions_store(app)
    session_name = f"{scenario.name}_{datetime.now():%Y%m%d-%H%M%S}"
    session = ChatSession.create(store, session_name, index_name)
    pipeline = _build_chat_pipeline(app, index_name, flags)

    checks = {
        "check_goal": bool(flags.get("check-goal")),
        "check_sources": bool(flags.get("check-sources")),
        "check_citations": bool(flags.get("check-citations")),
    }
    report = run_scenario(pipeline, scenario, session, checks)
    results_dir = app.config.resolve_path(app.config.get("chat.scenarios.results_path", "data/chat/scenarios/results"))
    json_path, md_path = save_scenario_results(report, results_dir)
    return f"Сценарий '{scenario.name}' пройден ({len(report['turns'])} ходов).\n  {json_path}\n  {md_path}"


def _cmd_chat_scenario_list(app: App) -> str:
    scenarios_dir = app.config.resolve_path(app.config.get("chat.scenarios.path", "data/chat/scenarios"))
    files = list_scenarios(scenarios_dir)
    if not files:
        return "Сценариев нет."
    return "Сценарии:\n  " + "\n  ".join(files)


# ---------------------------------------------------------------------- #
# Диспетчер и точка входа
# ---------------------------------------------------------------------- #
def handle_command(line: str, app: App) -> Tuple[bool, str]:
    """Разобрать и выполнить одну команду. Возвращает (выход, вывод)."""
    tokens = shlex.split(line)
    if not tokens:
        return False, ""

    cmd = tokens[0]
    args = tokens[1:]

    if cmd == "/help":
        return False, _cmd_help(args)
    if cmd == "/exit":
        return True, "До свидания!"
    if cmd == "/debug":
        return False, _cmd_debug(app, args)
    if cmd == "/index":
        return False, _cmd_index(app, args)
    if cmd == "/ask":
        return False, _cmd_ask(app, args)
    if cmd == "/ask-raw":
        return False, _cmd_ask_raw(app, args)
    if cmd == "/ask-compare":
        return False, _cmd_ask_compare(app, args)
    if cmd == "/rerank":
        return False, _cmd_rerank(app, args)
    if cmd == "/rewrite":
        return False, _cmd_rewrite(app, args)
    if cmd == "/eval":
        return False, _cmd_eval(app, args)
    if cmd == "/chat":
        return False, _cmd_chat(app, args)

    return False, f"Неизвестная команда: {cmd!r}. Введите /help для списка команд."


def run(app: App) -> None:
    """Интерактивный цикл REPL."""
    print(BANNER)
    while True:
        try:
            raw = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\n/exit")
            break

        line = raw.strip()
        if not line:
            continue
        if not line.startswith("/"):
            print("Команды начинаются с '/'. Введите /help для списка команд.")
            continue

        try:
            should_exit, output = handle_command(line, app)
            if output:
                print(output)
            if should_exit:
                break
        except Exception as exc:  # noqa: BLE001
            print(f"Ошибка: {exc}")
            if app.debug:
                traceback.print_exc()


def main() -> None:
    config = Config.load()
    app = App(config)
    run(app)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
