"""Тесты RAG-пайплайна и режима без RAG."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.no_rag_pipeline import NoRAGPipeline  # noqa: E402
from src.rag.pipeline import RAGPipeline  # noqa: E402
from src.retrieval.vector_retriever import VectorRetriever  # noqa: E402
from src.reranking import ThresholdFilter  # noqa: E402
from tests.fakes import FakeEmbedder, FakeLLM, build_temp_index  # noqa: E402


class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _retriever(self) -> VectorRetriever:
        store = build_temp_index(self.root, "idx", n_chunks=4, dim=8)
        return VectorRetriever(store=store, embedder=FakeEmbedder(dim=8), index_name="idx")

    def test_rag_run(self):
        llm = FakeLLM(answer="Ответ с контекстом")
        pipeline = RAGPipeline(self._retriever(), llm)
        result = pipeline.run("Какой вопрос?", top_k_retrieve=2, use_reranker=False)

        self.assertEqual(result.answer, "Ответ с контекстом")
        self.assertEqual(result.mode, "rag")
        self.assertEqual(len(result.used_chunks), 2)
        self.assertIn("Контекст:", result.prompt)
        self.assertIn("Какой вопрос?", result.prompt)
        self.assertIn("текст чанка", result.prompt)
        self.assertEqual(result.meta["top_k_retrieve"], 2)
        self.assertEqual(result.meta["num_chunks"], 2)
        self.assertEqual(result.meta["model"], "fake-model")
        # новые поля
        self.assertEqual(result.original_query, "Какой вопрос?")
        self.assertEqual(len(result.chunks_before_rerank), 2)
        self.assertEqual(len(result.chunks_after_rerank), 2)

    def test_rag_run_with_reranker(self):
        llm = FakeLLM(answer="Ответ")
        pipeline = RAGPipeline(self._retriever(), llm, reranker=ThresholdFilter(min_score=0.0, min_keep=1))
        result = pipeline.run("Вопрос", top_k_retrieve=4, top_k_after_rerank=2, use_reranker=True)
        self.assertLessEqual(len(result.used_chunks), 2)
        self.assertEqual(result.rerank_meta["provider"], "threshold")
        self.assertEqual(result.rerank_meta["before"], 4)
        self.assertEqual(len(result.chunks_before_rerank), 4)

    def test_no_rag_run(self):
        llm = FakeLLM(answer="Ответ без контекста")
        pipeline = NoRAGPipeline(llm)
        result = pipeline.run("Какой вопрос?")

        self.assertEqual(result.mode, "no_rag")
        self.assertEqual(result.used_chunks, [])
        self.assertIn("Какой вопрос?", result.prompt)
        self.assertNotIn("Контекст:", result.prompt)

    def test_unknown_when_no_chunks(self):
        from src.rag.pipeline import RAGPipeline as RP

        class EmptyRetriever:
            def search(self, query, top_k):
                return []

        llm = FakeLLM(answer="не должен вызываться")
        pipeline = RP(EmptyRetriever(), llm)
        result = pipeline.run("вопрос", use_reranker=False)
        self.assertEqual(result.answer_status, "unknown")
        self.assertIn("Не знаю", result.answer)
        self.assertEqual(result.used_chunks, [])
        self.assertEqual(len(llm.messages_history), 0)

    def test_unknown_when_low_score(self):
        from src.config import Config
        from src.models import Chunk
        from src.retrieval.models import RetrievedChunk

        class LowRetriever:
            def search(self, query, top_k):
                return [RetrievedChunk(Chunk("c0", "a.txt", "t", "текст", 0, 4, "fixed", ""), 0.1, 1)]

        llm = FakeLLM(answer="не должен вызываться")
        pipeline = RAGPipeline(
            LowRetriever(),
            llm,
            config=Config({"rag": {"unknown_threshold": {"min_score": 0.5, "min_chunks": 1}}}),
        )
        result = pipeline.run("вопрос", use_reranker=False)
        self.assertEqual(result.answer_status, "unknown")
        self.assertEqual(len(llm.messages_history), 0)


if __name__ == "__main__":
    unittest.main()
