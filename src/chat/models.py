"""Модели данных мини-чата."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.chat.task_state import TaskState, now_iso
from src.rag.models import RAGResult


@dataclass
class Message:
    """Сообщение диалога."""

    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp or now_iso(),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", ""),
            meta=dict(data.get("meta") or {}),
        )


@dataclass
class ChatSessionMeta:
    """Метаданные сессии."""

    session_id: str
    name: str
    index_name: str
    created_at: str = ""
    updated_at: str = ""
    message_count: int = 0

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "name": self.name,
            "index_name": self.index_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": self.message_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChatSessionMeta":
        return cls(
            session_id=data.get("session_id", ""),
            name=data.get("name", ""),
            index_name=data.get("index_name", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            message_count=int(data.get("message_count", 0)),
        )


@dataclass
class ChatResult:
    """Результат одного хода чата."""

    message: Message
    rag_result: RAGResult
    state_before: TaskState
    state_after: TaskState
    state_updates: List[str] = field(default_factory=list)
