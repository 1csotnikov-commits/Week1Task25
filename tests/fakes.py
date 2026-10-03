"""Общие фейки и хелперы для тестов (без реальной модели и API)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embeddings.base import EMBEDDING_FORMAT_VERSION, EmbeddingProvider  # noqa: E402
from src.indexing.faiss_sqlite_store import FAISSSQLiteStore  # noqa: E402
from src.llm.base import LLMProvider, LLMResponse  # noqa: E402
from src.models import Chunk, Document  # noqa: E402


class FakeEmbedder(EmbeddingProvider):
    """Детерминированные эмбеддинги; запоминает все переданные тексты."""

    name = "fake"

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.calls = []

    @staticmethod
    def _stable_hash(text: str) -> int:
        h = 0
        for ch in text:
            h = (h * 31 + ord(ch)) % 1000000007
        return h

    def encode(self, texts, batch_size=None):
        self.calls.extend(list(texts))
        arr = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            arr[i, self._stable_hash(t) % self.dim] = 1.0
        norm = np.linalg.norm(arr, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return arr / norm


class FakeLLM(LLMProvider):
    """Мок LLM: возвращает фиксированный ответ, запоминает сообщения."""

    name = "fake"

    def __init__(self, answer: str = "Ответ мок-LLM", prompt_tokens: int = 10, completion_tokens: int = 5):
        self.answer = answer
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.messages_history = []

    def complete(self, messages, temperature=None, max_tokens=None, timeout=None):
        self.messages_history.append(messages)
        return LLMResponse(
            text=self.answer,
            model="fake-model",
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.prompt_tokens + self.completion_tokens,
            elapsed_ms=1,
        )


def build_temp_index(root: Path, name: str = "idx", n_chunks: int = 3, dim: int = 8) -> FAISSSQLiteStore:
    """Построить маленький индекс с известными векторами (без модели)."""
    store = FAISSSQLiteStore(root)
    store.begin(name, "fixed", "model", "nomic", dim, embedding_format=EMBEDDING_FORMAT_VERSION)
    store.add_document(Document(source="a.txt", title="a", text="hello", format="txt"))
    for i in range(n_chunks):
        chunk = Chunk(
            chunk_id=f"c{i}",
            source="a.txt",
            title="a",
            text=f"текст чанка {i}",
            start_char=0,
            end_char=5,
            strategy="fixed",
            section="s" if i == 0 else "",
        )
        v = np.ones(dim, dtype=np.float32)
        v[i % dim] = 3.0
        v = v / np.linalg.norm(v)
        store.add(chunk, v)
    store.finalize(num_documents=1)
    return store
