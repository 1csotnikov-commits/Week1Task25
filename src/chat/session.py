"""ChatSession: история + task state + сохранение на диск."""

from __future__ import annotations

import uuid
from typing import List

from src.chat.history import HistoryStore
from src.chat.models import ChatSessionMeta, Message
from src.chat.task_state import TaskState, now_iso


class ChatSession:
    """Сессия диалога: хранит историю и task state, умеет сохраняться/загружаться."""

    def __init__(self, store: HistoryStore, name: str, index_name: str):
        self.store = store
        self.name = name
        self.index_name = index_name
        self.meta = ChatSessionMeta(
            session_id=str(uuid.uuid4()),
            name=name,
            index_name=index_name,
            created_at=now_iso(),
            updated_at=now_iso(),
            message_count=0,
        )
        self.history: List[Message] = []
        self.task_state = TaskState()

    @classmethod
    def create(cls, store: HistoryStore, name: str, index_name: str) -> "ChatSession":
        if store.exists(name):
            raise ValueError(f"Сессия '{name}' уже существует. Используйте /chat load {name}.")
        session = cls(store, name, index_name)
        session.save()
        return session

    @classmethod
    def load(cls, store: HistoryStore, name: str) -> "ChatSession":
        meta = store.load_meta(name)
        session = cls(store, meta.name, meta.index_name)
        session.meta = meta
        session.history = store.load_history(name)
        session.task_state = store.load_task_state(name)
        return session

    def add_message(self, message: Message) -> None:
        message.timestamp = message.timestamp or now_iso()
        self.history.append(message)
        self.meta.message_count = len(self.history)
        self.meta.updated_at = now_iso()
        # append-only в JSONL
        self.store.append_message(self.name, message)

    def recent_messages(self, n: int) -> List[Message]:
        if n <= 0:
            return []
        return self.history[-n:]

    def clear_history(self) -> None:
        self.history = []
        self.store.clear_history(self.name)
        self.meta.message_count = 0
        self.meta.updated_at = now_iso()
        self.save()

    def save(self) -> None:
        self.store.save_meta(self.meta)
        self.store.save_task_state(self.name, self.task_state)
