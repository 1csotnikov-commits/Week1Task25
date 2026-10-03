"""Тесты парсинга структурированного ответа и верификации цитат/источников."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import Chunk  # noqa: E402
from src.rag.citations import (  # noqa: E402
    extract_citations,
    extract_sources,
    parse_structured_answer,
)
from src.retrieval.models import RetrievedChunk  # noqa: E402


def make_chunk(chunk_id: str, source: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(Chunk(chunk_id, source, "t", text, 0, len(text), "fixed", ""), 0.9, 1)


class TestCitations(unittest.TestCase):
    def test_parse_correct(self):
        raw = (
            "ОТВЕТ:\nОтвет на вопрос.\n\n"
            "ИСТОЧНИКИ:\n- source=a.txt | chunk_id=c0\n\n"
            'ЦИТАТЫ:\n- "фрагмент текста" | source=a.txt | chunk_id=c0\n'
        )
        p = parse_structured_answer(raw)
        self.assertFalse(p.parse_failed)
        self.assertEqual(p.answer_text, "Ответ на вопрос.")
        self.assertEqual(len(p.sources), 1)
        self.assertEqual(p.sources[0].source, "a.txt")
        self.assertEqual(len(p.citations), 1)
        self.assertEqual(p.citations[0].text, "фрагмент текста")

    def test_parse_broken(self):
        raw = "просто текст без структуры"
        p = parse_structured_answer(raw)
        self.assertTrue(p.parse_failed)
        self.assertEqual(p.answer_text, "просто текст без структуры")
        self.assertEqual(p.sources, [])
        self.assertEqual(p.citations, [])

    def test_extract_citations_verified(self):
        chunks = [make_chunk("c0", "a.txt", "это фрагмент текста для цитаты")]
        raw = 'ОТВЕТ:\nОтвет.\nЦИТАТЫ:\n- "фрагмент текста" | source=a.txt | chunk_id=c0\n'
        citations = extract_citations(raw, chunks)
        self.assertEqual(len(citations), 1)
        self.assertTrue(citations[0].verified)

    def test_extract_citations_unverified(self):
        chunks = [make_chunk("c0", "a.txt", "совсем другой текст")]
        raw = 'ОТВЕТ:\nОтвет.\nЦИТАТЫ:\n- "несуществующий фрагмент" | source=a.txt | chunk_id=c9\n'
        citations = extract_citations(raw, chunks)
        self.assertEqual(len(citations), 1)
        self.assertFalse(citations[0].verified)

    def test_extract_sources_verified(self):
        chunks = [make_chunk("c0", "a.txt", "текст")]
        raw = "ОТВЕТ:\nОтвет.\nИСТОЧНИКИ:\n- source=a.txt | chunk_id=c0\n"
        sources = extract_sources(raw, chunks)
        self.assertTrue(sources[0].verified)

    def test_extract_sources_unverified(self):
        chunks = [make_chunk("c0", "a.txt", "текст")]
        raw = "ОТВЕТ:\nОтвет.\nИСТОЧНИКИ:\n- source=zzz.txt | chunk_id=c9\n"
        sources = extract_sources(raw, chunks)
        self.assertFalse(sources[0].verified)

    def test_citation_backfills_chunk_id(self):
        chunks = [make_chunk("c0", "a.txt", "это фрагмент текста для цитаты")]
        # Цитата без source/chunk_id — должна быть verified и получить backfill.
        raw = 'ОТВЕТ:\nОтвет.\nЦИТАТЫ:\n- "фрагмент текста"\n'
        citations = extract_citations(raw, chunks)
        self.assertTrue(citations[0].verified)
        self.assertEqual(citations[0].chunk_id, "c0")
        self.assertEqual(citations[0].source, "a.txt")


if __name__ == "__main__":
    unittest.main()
