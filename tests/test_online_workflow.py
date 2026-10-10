"""Small offline checks; no model downloads or external API calls."""

from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from multimodal_rag.keyword_index import KeywordIndex, build_keyword_index, file_sha256
from multimodal_rag.online_graph import OnlineRAG
from multimodal_rag.online_retrieval import ChunkStore, fuse_hits
from multimodal_rag.settings import Settings


class FakeDense:
    def search(self, query, *, include_visual):
        return [{"chunk_id": "doc-1", "score": 0.9}]


class FakeReranker:
    def rerank(self, query, hits, chunks):
        return hits[:5]


class FakeLLM:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self.answer)


class OnlineWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.chunks_path = self.root / "chunks.jsonl"
        chunks = [{
            "id": "doc-1",
            "text": "Revenue increased to 100 units.",
            "metadata": {"source": "report.pdf", "page": 7, "page_label": "5", "section": "Financials", "modality": "text"},
        }, {
            "id": "doc-2",
            "text": "The preceding year had 90 units.",
            "metadata": {"source": "report.pdf", "page": 7, "page_label": "5", "section": "Financials", "modality": "text"},
        }]
        self.chunks_path.write_text("\n".join(json.dumps(c) for c in chunks) + "\n", encoding="utf-8")
        self.db_path = self.root / "search.sqlite3"
        build_keyword_index(self.chunks_path, self.db_path)
        self.keyword = KeywordIndex(self.db_path, expected_sha256=file_sha256(self.chunks_path))
        self.settings = Settings(
            chunks_path=self.chunks_path, vector_db_dir=self.root, keyword_db_path=self.db_path,
            access_token="test-secret", llm_provider="ollama", llm_model="fake",
            ollama_base_url="http://localhost", rerank_model="fake", device="cpu", tracing_enabled=False,
        )

    def tearDown(self):
        self.keyword.close()
        self.temp.cleanup()

    def test_keyword_index_and_checksum(self):
        hits = self.keyword.search("revenue")
        self.assertEqual(hits[0]["chunk_id"], "doc-1")
        with self.assertRaises(ValueError):
            KeywordIndex(self.db_path, expected_sha256="wrong")

    def test_fusion_deduplicates(self):
        hits = fuse_hits([{"chunk_id": "doc-1"}], [{"chunk_id": "doc-1"}, {"chunk_id": "doc-2"}])
        self.assertEqual([hit["chunk_id"] for hit in hits], ["doc-1", "doc-2"])

    def test_graph_auth_retrieval_and_citations(self):
        llm = FakeLLM("Revenue increased to 100 units [1].")
        rag = OnlineRAG(
            self.settings, chunks=ChunkStore(self.chunks_path), dense=FakeDense(),
            keyword=self.keyword, reranker=FakeReranker(), llm=llm,
        )
        with self.assertRaises(PermissionError):
            asyncio.run(rag.ask("revenue", "wrong"))
        self.assertEqual(llm.calls, [])
        result = asyncio.run(rag.ask("What was revenue?", "test-secret"))
        self.assertIn("[1]", result["answer"])
        self.assertEqual(result["citations"][0]["page"], 7)
        prompt = str(llm.calls[0])
        self.assertNotIn("test-secret", prompt)
        rag.close()

    def test_uncited_output_is_not_presented_as_grounded(self):
        rag = OnlineRAG(
            self.settings, chunks=ChunkStore(self.chunks_path), dense=FakeDense(),
            keyword=self.keyword, reranker=FakeReranker(), llm=FakeLLM("Revenue was 100."),
        )
        result = asyncio.run(rag.ask("revenue", "test-secret"))
        self.assertIn("couldn't produce a reliably cited answer", result["answer"])
        rag.close()

    def test_model_supplied_markdown_image_is_removed(self):
        rag = OnlineRAG(
            self.settings, chunks=ChunkStore(self.chunks_path), dense=FakeDense(),
            keyword=self.keyword, reranker=FakeReranker(),
            llm=FakeLLM("Revenue increased to 100 units [1].\n![Unverified image](missing.png)"),
        )
        result = asyncio.run(rag.ask("revenue", "test-secret"))
        self.assertIn("[1]", result["answer"])
        self.assertNotIn("![", result["answer"])
        rag.close()

    def test_dense_and_keyword_branches_run_concurrently(self):
        barrier = threading.Barrier(2, timeout=3)

        class ParallelDense:
            def search(self, query, *, include_visual):
                barrier.wait()
                return [{"chunk_id": "doc-1"}]

        class ParallelKeyword:
            def search(self, query):
                barrier.wait()
                return [{"chunk_id": "doc-1"}]

            def close(self):
                pass

        rag = OnlineRAG(
            self.settings, chunks=ChunkStore(self.chunks_path), dense=ParallelDense(),
            keyword=ParallelKeyword(), reranker=FakeReranker(),
            llm=FakeLLM("Revenue increased [1]."),
        )
        try:
            result = asyncio.run(rag.ask("revenue", "test-secret"))
            self.assertIn("[1]", result["answer"])
        finally:
            rag.close()


if __name__ == "__main__":
    unittest.main()
