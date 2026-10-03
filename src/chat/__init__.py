"""Мини-чат с RAG + памятью (реализация Дня 25)."""

from __future__ import annotations

from src.chat.chat_pipeline import ChatPipeline
from src.chat.history import HistoryStore
from src.chat.models import ChatResult, ChatSessionMeta, Message
from src.chat.scenarios import load_scenario, list_scenarios, run_scenario, save_scenario_results
from src.chat.session import ChatSession
from src.chat.state_updater import HeuristicUpdater, LLMUpdater, StateUpdater, make_state_updater
from src.chat.task_state import TaskState, TaskStateUpdate

__all__ = [
    "ChatPipeline",
    "HistoryStore",
    "ChatResult",
    "ChatSessionMeta",
    "Message",
    "ChatSession",
    "TaskState",
    "TaskStateUpdate",
    "StateUpdater",
    "HeuristicUpdater",
    "LLMUpdater",
    "make_state_updater",
    "load_scenario",
    "list_scenarios",
    "run_scenario",
    "save_scenario_results",
]
