"""HistoryStore: JSON-файлы сессий (meta.json, history.jsonl, task_state.json)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List

from src.chat.models import ChatSessionMeta, Message
from src.chat.task_state import TaskState


class HistoryStore:
    """Хранит сессии на диске: метаданные, история (JSONL) и task state."""

    def __init__(self, root):
        self.root = Path(root)

    def session_dir(self, name: str) -> Path:
        return self.root / name

    def exists(self, name: str) -> bool:
        return (self.session_dir(name) / "meta.json").exists()

    def save_meta(self, meta: ChatSessionMeta) -> None:
        d = self.session_dir(meta.name)
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def load_meta(self, name: str) -> ChatSessionMeta:
        p = self.session_dir(name) / "meta.json"
        if not p.exists():
            raise FileNotFoundError(f"Сессия '{name}' не найдена.")
        return ChatSessionMeta.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def append_message(self, name: str, message: Message) -> None:
        p = self.session_dir(name) / "history.jsonl"
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(message.to_dict(), ensure_ascii=False) + "\n")

    def load_history(self, name: str) -> List[Message]:
        p = self.session_dir(name) / "history.jsonl"
        if not p.exists():
            return []
        messages = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                messages.append(Message.from_dict(json.loads(line)))
        return messages

    def clear_history(self, name: str) -> None:
        p = self.session_dir(name) / "history.jsonl"
        p.write_text("", encoding="utf-8")

    def save_task_state(self, name: str, state: TaskState) -> None:
        p = self.session_dir(name) / "task_state.json"
        p.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def load_task_state(self, name: str) -> TaskState:
        p = self.session_dir(name) / "task_state.json"
        if not p.exists():
            return TaskState()
        return TaskState.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def list_sessions(self) -> List[ChatSessionMeta]:
        if not self.root.exists():
            return []
        metas = []
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and (d / "meta.json").exists():
                metas.append(ChatSessionMeta.from_dict(json.loads((d / "meta.json").read_text(encoding="utf-8"))))
        return metas

    def delete_session(self, name: str) -> None:
        d = self.session_dir(name)
        if d.exists():
            shutil.rmtree(d)
        else:
            raise FileNotFoundError(f"Сессия '{name}' не найдена.")
