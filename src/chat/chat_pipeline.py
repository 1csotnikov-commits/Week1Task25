"""ChatPipeline: retrieval + RAG + учёт истории и task state."""

from __future__ import annotations

from typing import List, Optional

from src.chat.models import ChatResult, Message
from src.chat.session import ChatSession
from src.chat.state_updater import StateUpdater
from src.chat.task_state import TaskState
from src.config import Config
from src.rag.citations import extract_citations, extract_sources, parse_structured_answer
from src.rag.models import RAGResult
from src.rag.pipeline import RAGPipeline
from src.rag.prompt_builder import format_chunk, render_messages
from src.retrieval.models import RetrievedChunk

CHAT_SYSTEM_PROMPT = (
    "Ты — ассистент, отвечающий на вопросы по документам. Отвечай на русском языке, "
    "используя ТОЛЬКО предоставленный ниже контекст. Не выдумывай факты, которых нет в контексте.\n\n"
    "Учитывай «Память задачи»: не переспрашивай то, что уже уточнено, продолжай вести диалог к цели.\n\n"
    "Формат ответа — строго три блока:\n\n"
    "ОТВЕТ:\n<текст ответа>\n\n"
    "ИСТОЧНИКИ:\n- source=<имя файла> | chunk_id=<id>\n\n"
    "ЦИТАТЫ:\n- \"<дословный фрагмент>\" | source=<имя файла> | chunk_id=<id>\n\n"
    "Правила:\n"
    "1. В блоке ЦИТАТЫ — 1–2 коротких дословных фрагмента (5–15 слов), не весь текст чанка.\n"
    "2. Каждая цитата/источник ОБЯЗАТЕЛЬНО с source=<файл> и chunk_id=<id> из контекста.\n"
    "3. Если контекста недостаточно — в блоке ОТВЕТ напиши ровно: "
    "\"Не знаю. В предоставленных документах не найдено достаточно информации. Уточните, пожалуйста, вопрос.\" "
    "и оставь блоки ИСТОЧНИКИ и ЦИТАТЫ пустыми.\n"
    "4. Если ответа нет в контексте — нельзя додумывать."
)


def render_state(state: TaskState) -> str:
    lines = [f"- Цель диалога: {state.goal or '(не задана)'}"]
    if state.facts:
        lines.append("- Уже уточнено:")
        lines.extend(f"  - {f}" for f in state.facts)
    if state.constraints:
        lines.append("- Ограничения:")
        lines.extend(f"  - {c}" for c in state.constraints)
    if state.terms:
        lines.append("- Термины:")
        lines.extend(f"  - {t}: {d}" for t, d in state.terms.items())
    if state.open_questions:
        lines.append("- Открытые вопросы:")
        lines.extend(f"  - {q}" for q in state.open_questions)
    return "\n".join(lines)


def render_history(messages: List[Message]) -> str:
    if not messages:
        return "(история пуста)"
    out = []
    for m in messages:
        role = "Пользователь" if m.role == "user" else "Ассистент"
        out.append(f"{role}: {m.content}")
    return "\n".join(out)


def render_chunks(chunks: List[RetrievedChunk]) -> str:
    if not chunks:
        return "(контекст отсутствует)"
    return "\n\n".join(format_chunk(i + 1, c) for i, c in enumerate(chunks))


def build_chat_messages(state: TaskState, history: List[Message], chunks: List[RetrievedChunk], question: str) -> List[dict]:
    system = CHAT_SYSTEM_PROMPT + "\n\nПамять задачи:\n" + render_state(state)
    user = (
        "История диалога:\n" + render_history(history) + "\n\n"
        "Контекст:\n" + render_chunks(chunks) + "\n\n"
        "Вопрос: " + question
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _rag_result_meta(rag_result: RAGResult) -> dict:
    return {
        "answer_status": rag_result.answer_status,
        "mode": rag_result.mode,
        "num_chunks": len(rag_result.used_chunks),
        "num_citations": len(rag_result.citations),
        "num_sources": len(rag_result.sources),
        "citations_verified": rag_result.citations_verified,
        "sources_verified": rag_result.sources_verified,
        "citations": [{"text": c.text, "source": c.source, "chunk_id": c.chunk_id, "verified": c.verified} for c in rag_result.citations],
        "sources": [{"source": s.source, "chunk_id": s.chunk_id, "verified": s.verified} for s in rag_result.sources],
        "rerank": rag_result.rerank_meta,
    }


class ChatPipeline:
    """Мини-чат: история + task state + retrieval + RAG-ответ."""

    def __init__(self, rag_pipeline: RAGPipeline, state_updater: StateUpdater, config: Optional[Config] = None):
        self.rag = rag_pipeline
        self.state_updater = state_updater
        self.config = config
        self.llm = rag_pipeline.llm

    def _cfg(self, dotted: str, default):
        if self.config is not None:
            return self.config.get(dotted, default)
        return default

    def send(self, session: ChatSession, user_message: str) -> ChatResult:
        session.add_message(Message(role="user", content=user_message))
        state_before = TaskState.from_dict(session.task_state.to_dict())

        update = self.state_updater.update(user_message, session.task_state, None)
        state_updates = session.task_state.apply_update(update)

        use_rewrite = bool(self._cfg("rewrite.enabled", False))
        ctx = self.rag.retrieve(user_message, use_reranker=True, use_rewrite=use_rewrite)

        if ctx.is_weak:
            rag_result = self.rag._unknown_result(ctx)
        else:
            history_window = int(self._cfg("chat.history_window", 6))
            recent = session.history[-(history_window + 1):-1]
            messages = build_chat_messages(session.task_state, recent, ctx.chunks_after, user_message)
            prompt_text = render_messages(messages)
            response = self.llm.complete(messages)

            parsed = parse_structured_answer(response.text)
            citations = extract_citations(response.text, ctx.chunks_after)
            sources = extract_sources(response.text, ctx.chunks_after)
            rag_result = RAGResult(
                answer=parsed.answer_text,
                mode="rag",
                used_chunks=ctx.chunks_after,
                prompt=prompt_text,
                original_query=ctx.original_query,
                rewritten_query=ctx.rewritten_query if ctx.rewritten_query != ctx.original_query else None,
                chunks_before_rerank=ctx.chunks_before,
                chunks_after_rerank=ctx.chunks_after,
                rewrite_meta=ctx.rewrite_meta,
                rerank_meta=ctx.rerank_meta,
                answer_status="ok",
                parsed_answer=parsed,
                citations=citations,
                sources=sources,
                citations_verified=all(c.verified for c in citations) if citations else True,
                sources_verified=all(s.verified for s in sources) if sources else True,
                meta={
                    "model": response.model,
                    "tokens_in": response.prompt_tokens,
                    "tokens_out": response.completion_tokens,
                    "elapsed_ms": response.elapsed_ms,
                    "top_k_retrieve": ctx.k_retrieve,
                    "top_k_after_rerank": ctx.k_after,
                    "num_chunks": len(ctx.chunks_after),
                    "num_citations": len(citations),
                    "num_sources": len(sources),
                    "parse_failed": parsed.parse_failed,
                },
            )

        assistant_msg = Message(role="assistant", content=rag_result.answer, meta={"rag": _rag_result_meta(rag_result)})
        session.add_message(assistant_msg)

        if self.state_updater.name == "llm" and self._cfg("chat.state_updater.update_after", "pair") == "pair":
            update2 = self.state_updater.update(user_message, session.task_state, rag_result.answer)
            state_updates += session.task_state.apply_update(update2)

        session.save()
        return ChatResult(
            message=assistant_msg,
            rag_result=rag_result,
            state_before=state_before,
            state_after=session.task_state,
            state_updates=state_updates,
        )

