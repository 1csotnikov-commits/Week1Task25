"""Тесты мини-чата: task state, история, state updater, ChatPipeline, сценарии."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chat.history import HistoryStore  # noqa: E402
from src.chat.models import Message  # noqa: E402
from src.chat.session import ChatSession  # noqa: E402
from src.chat.state_updater import HeuristicUpdater, LLMUpdater  # noqa: E402
from src.chat.task_state import TaskState, TaskStateUpdate  # noqa: E402
from tests.fakes import FakeLLM  # noqa: E402

STRUCTURED_ANSWER = (
    "ОТВЕТ:\nЭто ответ на вопрос.\n\n"
    "ИСТОЧНИКИ:\n- source=a.txt | chunk_id=c0\n\n"
    'ЦИТАТЫ:\n- "текст чанка" | source=a.txt | chunk_id=c0\n'
)


class TestTaskState(unittest.TestCase):
    def test_serialize(self):
        s = TaskState(goal="цель", facts=["ф1"], constraints=["огр"], terms={"Т": "опр"}, open_questions=["в?"])
        s2 = TaskState.from_dict(s.to_dict())
        self.assertEqual(s2.goal, "цель")
        self.assertEqual(s2.facts, ["ф1"])
        self.assertEqual(s2.terms, {"Т": "опр"})

    def test_apply_update(self):
        s = TaskState()
        u = TaskStateUpdate(goal="новая цель", new_facts=["ф1"], new_constraints=["огр"], new_terms={"Т": "опр"}, new_open_questions=["в?"])
        changes = s.apply_update(u)
        self.assertEqual(s.goal, "новая цель")
        self.assertIn("ф1", s.facts)
        self.assertIn("огр", s.constraints)
        self.assertGreaterEqual(len(changes), 4)


class TestHistoryStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_append_and_restore(self):
        store = HistoryStore(self.root / "sessions")
        session = ChatSession.create(store, "s1", "idx")
        session.add_message(Message(role="user", content="привет"))
        session.add_message(Message(role="assistant", content="здравствуйте"))
        session.task_state.goal = "цель"
        session.save()

        s2 = ChatSession.load(store, "s1")
        self.assertEqual(len(s2.history), 2)
        self.assertEqual(s2.history[0].content, "привет")
        self.assertEqual(s2.index_name, "idx")
        self.assertEqual(s2.task_state.goal, "цель")


class TestHeuristicUpdater(unittest.TestCase):
    def test_markers(self):
        upd = HeuristicUpdater().update(
            "Цель — внедрить маркировку. Запомни: мы торгуем обувью. Ограничение: только Android. Термин: ГИС МТ = система мониторинга.",
            TaskState(),
        )
        self.assertEqual(upd.goal, "внедрить маркировку")
        self.assertIn("мы торгуем обувью", upd.new_facts)
        self.assertIn("только Android", upd.new_constraints)
        self.assertIn("ГИС МТ", upd.new_terms)


class TestLLMUpdater(unittest.TestCase):
    def test_parse_good(self):
        llm = FakeLLM(answer='{"goal": "новая цель", "new_facts": ["ф1"], "new_constraints": [], "new_terms": {}, "resolved_open_questions": [], "new_open_questions": []}')
        upd = LLMUpdater(llm).update("вопрос", TaskState())
        self.assertEqual(upd.goal, "новая цель")
        self.assertIn("ф1", upd.new_facts)

    def test_parse_bad(self):
        llm = FakeLLM(answer="не json")
        upd = LLMUpdater(llm).update("вопрос", TaskState())
        self.assertTrue(upd.is_empty())


class TestChatPipeline(unittest.TestCase):
    def setUp(self):
        from src.chat.chat_pipeline import ChatPipeline
        from src.config import Config
        from src.indexing.faiss_sqlite_store import FAISSSQLiteStore
        from src.rag.pipeline import RAGPipeline
        from src.retrieval.vector_retriever import VectorRetriever
        from src.reranking import ThresholdFilter
        from tests.fakes import FakeEmbedder, build_temp_index

        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        store = HistoryStore(self.root / "sessions")
        self.session = ChatSession.create(store, "s1", "idx")
        build_temp_index(self.root / "index", "idx", n_chunks=4, dim=8)
        retriever = VectorRetriever(FAISSSQLiteStore(self.root / "index"), FakeEmbedder(8), "idx")
        config = Config({
            "retrieval": {"top_k_retrieve": 10},
            "reranking": {"enabled": True, "provider": "threshold", "top_k_after_rerank": 5, "threshold": {"min_score": 0.0, "min_keep": 1}},
            "rag": {"unknown_threshold": {"min_score": 0.0, "min_chunks": 0}},
            "rewrite": {"enabled": False},
            "chat": {"history_window": 6, "state_updater": {"provider": "heuristic", "update_after": "pair"}},
        })
        self.llm = FakeLLM(answer=STRUCTURED_ANSWER)
        rag = RAGPipeline(retriever, self.llm, config, reranker=ThresholdFilter(0.0, 1))
        self.pipeline = ChatPipeline(rag, HeuristicUpdater(), config)

    def tearDown(self):
        self._tmp.cleanup()

    def test_normal_answer(self):
        result = self.pipeline.send(self.session, "вопрос")
        self.assertEqual(result.rag_result.answer_status, "ok")
        self.assertEqual(result.rag_result.answer, "Это ответ на вопрос.")
        self.assertTrue(result.rag_result.sources)
        self.assertTrue(result.rag_result.citations)

    def test_state_in_prompt(self):
        self.session.task_state.goal = "моя цель"
        result = self.pipeline.send(self.session, "вопрос")
        self.assertIn("моя цель", result.rag_result.prompt)

    def test_history_in_prompt(self):
        self.pipeline.send(self.session, "первый вопрос")
        result = self.pipeline.send(self.session, "второй вопрос")
        self.assertIn("первый вопрос", result.rag_result.prompt)

    def test_state_updates_on_markers(self):
        result = self.pipeline.send(self.session, "Цель — разобраться в маркировке. Что такое ТС ПИоТ?")
        self.assertEqual(result.state_after.goal, "разобраться в маркировке")
        self.assertGreaterEqual(len(result.state_updates), 1)


class TestChatPrompt(unittest.TestCase):
    def test_structure(self):
        from src.chat.chat_pipeline import build_chat_messages
        from src.models import Chunk
        from src.retrieval.models import RetrievedChunk

        state = TaskState(goal="цель", facts=["факт"])
        msgs = [Message(role="user", content="вопрос 1"), Message(role="assistant", content="ответ 1")]
        chunks = [RetrievedChunk(Chunk("c0", "a.txt", "t", "текст чанка", 0, 10, "fixed", ""), 0.9, 1)]
        messages = build_chat_messages(state, msgs, chunks, "текущий вопрос")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("цель", messages[0]["content"])
        self.assertIn("вопрос 1", messages[1]["content"])
        self.assertIn("текущий вопрос", messages[1]["content"])


class TestScenario(unittest.TestCase):
    def test_run_scenario(self):
        from src.chat.chat_pipeline import ChatPipeline
        from src.chat.scenarios import Scenario, ScenarioMessage, run_scenario
        from src.config import Config
        from src.indexing.faiss_sqlite_store import FAISSSQLiteStore
        from src.rag.pipeline import RAGPipeline
        from src.retrieval.vector_retriever import VectorRetriever
        from src.reranking import ThresholdFilter
        from tests.fakes import FakeEmbedder, build_temp_index

        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        try:
            store = HistoryStore(root / "sessions")
            session = ChatSession.create(store, "sc", "idx")
            build_temp_index(root / "index", "idx", n_chunks=4, dim=8)
            retriever = VectorRetriever(FAISSSQLiteStore(root / "index"), FakeEmbedder(8), "idx")
            config = Config({
                "retrieval": {"top_k_retrieve": 10},
                "reranking": {"threshold": {"min_score": 0.0, "min_keep": 1}},
                "rag": {"unknown_threshold": {"min_score": 0.0, "min_chunks": 0}},
                "rewrite": {"enabled": False},
                "chat": {"history_window": 6},
            })
            llm = FakeLLM(answer=STRUCTURED_ANSWER)
            rag = RAGPipeline(retriever, llm, config, reranker=ThresholdFilter(0.0, 1))
            pipeline = ChatPipeline(rag, HeuristicUpdater(), config)
            scenario = Scenario(
                name="sc", description="d", index_name="idx",
                messages=[
                    ScenarioMessage("m1", "user", "Цель — разобраться."),
                    ScenarioMessage("m2", "user", "Что такое ТС ПИоТ?"),
                    ScenarioMessage("m3", "user", "Итог?"),
                ],
            )
            report = run_scenario(pipeline, scenario, session, {"check_goal": True, "check_sources": True, "check_citations": True})
            self.assertEqual(len(report["turns"]), 3)
            self.assertTrue(report["checks"]["goal"])
            self.assertEqual(report["checks"]["sources"]["passed"], 3)
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()

