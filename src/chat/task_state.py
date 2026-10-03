"""TaskState — «память задачи» диалога + обновления."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class TaskState:
    """Компактная выжимка контекста диалога."""

    goal: Optional[str] = None
    facts: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    terms: Dict[str, str] = field(default_factory=dict)
    open_questions: List[str] = field(default_factory=list)
    updated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "facts": list(self.facts),
            "constraints": list(self.constraints),
            "terms": dict(self.terms),
            "open_questions": list(self.open_questions),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskState":
        data = data or {}
        return cls(
            goal=data.get("goal"),
            facts=list(data.get("facts") or []),
            constraints=list(data.get("constraints") or []),
            terms=dict(data.get("terms") or {}),
            open_questions=list(data.get("open_questions") or []),
            updated_at=data.get("updated_at", ""),
        )

    def apply_update(self, update: "TaskStateUpdate") -> List[str]:
        """Применить обновление; вернуть список строк-описаний изменений."""
        changes: List[str] = []
        if update.goal and update.goal != self.goal:
            self.goal = update.goal
            changes.append(f"goal: {update.goal}")
        for f in update.new_facts:
            if f not in self.facts:
                self.facts.append(f)
                changes.append(f"fact: {f}")
        for c in update.new_constraints:
            if c not in self.constraints:
                self.constraints.append(c)
                changes.append(f"constraint: {c}")
        for term, definition in update.new_terms.items():
            if term not in self.terms:
                self.terms[term] = definition
                changes.append(f"term: {term} = {definition}")
        for q in update.resolved_open_questions:
            if q in self.open_questions:
                self.open_questions.remove(q)
                changes.append(f"resolved: {q}")
        for q in update.new_open_questions:
            if q not in self.open_questions:
                self.open_questions.append(q)
                changes.append(f"open_question: {q}")
        if changes:
            self.updated_at = now_iso()
        return changes


@dataclass
class TaskStateUpdate:
    """Набор изменений task state."""

    goal: Optional[str] = None
    new_facts: List[str] = field(default_factory=list)
    new_constraints: List[str] = field(default_factory=list)
    new_terms: Dict[str, str] = field(default_factory=dict)
    resolved_open_questions: List[str] = field(default_factory=list)
    new_open_questions: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (
            self.goal
            or self.new_facts
            or self.new_constraints
            or self.new_terms
            or self.resolved_open_questions
            or self.new_open_questions
        )

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "new_facts": list(self.new_facts),
            "new_constraints": list(self.new_constraints),
            "new_terms": dict(self.new_terms),
            "resolved_open_questions": list(self.resolved_open_questions),
            "new_open_questions": list(self.new_open_questions),
        }
