"""StateUpdater: обновление task state по ходу диалога (heuristic | llm)."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Optional

from src.chat.task_state import TaskState, TaskStateUpdate
from src.llm.base import LLMProvider


class StateUpdater(ABC):
    """Абстракция обновления памяти задачи."""

    name: str = "base"

    @abstractmethod
    def update(self, user_message: str, current_state: TaskState, assistant_answer: Optional[str] = None) -> TaskStateUpdate:
        raise NotImplementedError


class HeuristicUpdater(StateUpdater):
    """Быстрое обновление по маркерам в сообщении пользователя."""

    name = "heuristic"

    _GOAL_RE = re.compile(r"\b(?:цель|задача)\s*[:\-—]\s*([^.!?\n]+)", re.IGNORECASE)
    _FACT_RE = re.compile(r"\b(?:запомни|важно|не забудь|уточняю|учти)\b\s*[:\-—]\s*([^.!?\n]+)", re.IGNORECASE)
    _CONSTRAINT_RE = re.compile(r"\b(?:ограничение|требование)\b\s*[:\-—]\s*([^.!?\n]+)", re.IGNORECASE)
    _TERM_RE = re.compile(r"\bтермин\s*[:\-—]?\s*([^=—:\n]{1,60}?)\s*[=—]\s*([^.!?\n]+)", re.IGNORECASE)
    _QUESTION_RE = re.compile(r"\b(?:непонятно|не ясно|не знаю)\b", re.IGNORECASE)

    def update(self, user_message: str, current_state: TaskState, assistant_answer: Optional[str] = None) -> TaskStateUpdate:
        upd = TaskStateUpdate()
        msg = (user_message or "").strip()

        m = self._GOAL_RE.search(msg)
        if m:
            upd.goal = self._clean(m.group(1))

        for m in self._FACT_RE.finditer(msg):
            upd.new_facts.append(self._clean(m.group(1)))

        for m in self._CONSTRAINT_RE.finditer(msg):
            upd.new_constraints.append(self._clean(m.group(1)))

        for m in self._TERM_RE.finditer(msg):
            term = m.group(1).strip().strip('"').strip()
            definition = self._clean(m.group(2))
            if term and definition:
                upd.new_terms[term] = definition

        if self._QUESTION_RE.search(msg) and "?" in msg:
            for sent in re.split(r"[.!?]\s+", msg):
                sent = sent.strip()
                if sent.endswith("?") and len(sent) > 8:
                    upd.new_open_questions.append(sent)

        return upd

    @staticmethod
    def _clean(text: str) -> str:
        return text.strip().strip('"').rstrip(".,;:").strip()


class LLMUpdater(StateUpdater):
    """Точное обновление через LLM (DeepSeek)."""

    name = "llm"

    SYSTEM = (
        "Ты обновляешь память задачи диалога. Тебе дана текущая память и последнее сообщение "
        "пользователя (и, возможно, ответ ассистента). Верни JSON с полями: "
        "goal (строка или null, если не менялось), new_facts (список), new_constraints (список), "
        "new_terms (словарь термин->определение), resolved_open_questions (список), "
        "new_open_questions (список). Не дублируй уже существующее. Если ничего не изменилось — "
        "верни пустые структуры. Отвечай ТОЛЬКО JSON-объектом."
    )

    def __init__(self, llm: LLMProvider, temperature: float = 0.0, max_tokens: int = 500):
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens

    def update(self, user_message: str, current_state: TaskState, assistant_answer: Optional[str] = None) -> TaskStateUpdate:
        user_content = (
            f"Текущая память задачи (JSON):\n{json.dumps(current_state.to_dict(), ensure_ascii=False)}\n\n"
            f"Сообщение пользователя:\n{user_message}\n"
        )
        if assistant_answer:
            user_content += f"\nОтвет ассистента:\n{assistant_answer}\n"

        try:
            response = self.llm.complete(
                [{"role": "system", "content": self.SYSTEM}, {"role": "user", "content": user_content}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            data = self._parse_json(response.text)
        except Exception as exc:  # noqa: BLE001
            print(f"ПРЕДУПРЕЖДЕНИЕ: не удалось обновить task state через LLM: {exc}")
            return TaskStateUpdate()

        return TaskStateUpdate(
            goal=data.get("goal"),
            new_facts=[str(f) for f in (data.get("new_facts") or [])],
            new_constraints=[str(c) for c in (data.get("new_constraints") or [])],
            new_terms={str(k): str(v) for k, v in (data.get("new_terms") or {}).items()},
            resolved_open_questions=[str(q) for q in (data.get("resolved_open_questions") or [])],
            new_open_questions=[str(q) for q in (data.get("new_open_questions") or [])],
        )

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = (text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("ожидался JSON-объект")
        return data


def make_state_updater(provider: str, config=None, llm: Optional[LLMProvider] = None) -> StateUpdater:
    """Создать обновлятель task state по имени провайдера."""
    provider = (provider or "heuristic").lower()
    if provider == "heuristic":
        return HeuristicUpdater()
    if provider == "llm":
        if llm is None:
            raise ValueError("LLMUpdater требует LLMProvider (передайте llm).")
        temperature = 0.0
        max_tokens = 500
        if config is not None:
            temperature = float(config.get("chat.state_updater.llm.temperature", 0.0))
            max_tokens = int(config.get("chat.state_updater.llm.max_tokens", 500))
        return LLMUpdater(llm, temperature=temperature, max_tokens=max_tokens)
    raise ValueError(f"Неизвестный state_updater: {provider!r}. Доступно: heuristic, llm")


__all__ = ["StateUpdater", "HeuristicUpdater", "LLMUpdater", "make_state_updater"]

