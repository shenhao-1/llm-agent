"""Unit tests for sparse BM25 retrieval orchestration."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from core import Chunk, RetrievalResult
from core.query_engine import SparseRetriever
from core.trace import TraceContext
from ingestion.embedding import SparseEncoder
from ingestion.storage import BM25Indexer
from libs.vector_store import BaseVectorStore


class FakeVectorStore(BaseVectorStore):
    """In-memory record store used to hydrate sparse results."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        super().__init__({})
        self.records_by_id = {str(record["id"]): dict(record) for record in records}
        self.calls: list[list[str]] = []

    def upsert(
        self,
        records: list[Mapping[str, Any]],
        trace: Any | None = None,
    ) -> None:
        self.validate_records(records)

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        self.validate_query_args(vector, top_k, filters)
        return []

    def get_by_ids(
        self,
        ids: list[str],
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        self.validate_ids(ids)
        self.calls.append(list(ids))
        return [self.records_by_id[item_id] for item_id in ids if item_id in self.records_by_id]


def test_sparse_retriever_returns_expected_chunk_with_text_and_metadata(tmp_path) -> None:
    chunks = [
        _chunk("python python retrieval", "chunk-python", 0),
        _chunk("dense vector search", "chunk-dense", 1),
        _chunk("image captioning workflow", "chunk-image", 2),
    ]
    bm25 = BM25Indexer(tmp_path).build(SparseEncoder().encode(chunks), persist=False)
    store = FakeVectorStore(
        [
            {
                "id": chunk.id,
                "text": chunk.text,
                "metadata": dict(chunk.metadata),
            }
            for chunk in chunks
        ]
    )

    results = SparseRetriever(
        bm25_indexer=bm25,
        vector_store=store,
    ).retrieve(["python"], top_k=2)

    assert results == [
        RetrievalResult(
            chunk_id="chunk-python",
            score=results[0].score,
            text="python python retrieval",
            metadata={"source_path": "docs/sample.md", "chunk_index": 0},
        )
    ]
    assert results[0].score > 0
    assert store.calls == [["chunk-python"]]


def test_sparse_retriever_preserves_bm25_order_and_top_k(tmp_path) -> None:
    chunks = [
        _chunk("alpha unique", "chunk-a", 0),
        _chunk("beta unique", "chunk-b", 1),
        _chunk("gamma unique", "chunk-c", 2),
    ]
    bm25 = BM25Indexer(tmp_path).build(SparseEncoder().encode(chunks), persist=False)
    store = FakeVectorStore(
        [{"id": chunk.id, "text": chunk.text, "metadata": dict(chunk.metadata)} for chunk in chunks]
    )

    results = SparseRetriever(bm25_indexer=bm25, vector_store=store).retrieve(
        ["alpha", "beta"],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk_id in {"chunk-a", "chunk-b"}
    assert store.calls == [[results[0].chunk_id]]


def test_sparse_retriever_merges_hit_metadata_with_store_metadata(tmp_path) -> None:
    chunk = _chunk("metadata needle", "chunk-1", 0)
    bm25 = BM25Indexer(tmp_path).build(SparseEncoder().encode([chunk]), persist=False)
    store = FakeVectorStore(
        [
            {
                "id": "chunk-1",
                "text": "hydrated text",
                "metadata": {"source_path": "docs/hydrated.md", "section": "intro"},
            }
        ]
    )

    result = SparseRetriever(bm25_indexer=bm25, vector_store=store).retrieve(
        ["needle"],
        top_k=1,
    )[0]

    assert result.text == "hydrated text"
    assert result.metadata == {
        "source_path": "docs/hydrated.md",
        "chunk_index": 0,
        "section": "intro",
    }


def test_trace_records_sparse_retrieval_stage(tmp_path) -> None:
    chunk = _chunk("trace needle", "chunk-1", 0)
    bm25 = BM25Indexer(tmp_path).build(SparseEncoder().encode([chunk]), persist=False)
    store = FakeVectorStore(
        [{"id": "chunk-1", "text": chunk.text, "metadata": dict(chunk.metadata)}]
    )
    trace = TraceContext()

    SparseRetriever(bm25_indexer=bm25, vector_store=store).retrieve(
        ["needle"],
        top_k=1,
        trace=trace,
    )

    assert trace.stages[-1]["name"] == "retrieve"
    assert trace.stages[-1]["method"] == "sparse_retriever"
    assert trace.stages[-1]["details"] == {
        "keyword_count": 1,
        "top_k": 1,
        "result_count": 1,
    }


def test_sparse_retriever_rejects_invalid_inputs(tmp_path) -> None:
    retriever = SparseRetriever(
        bm25_indexer=BM25Indexer(tmp_path),
        vector_store=FakeVectorStore([]),
    )

    with pytest.raises(ValueError, match="keywords"):
        retriever.retrieve([], top_k=1)
    with pytest.raises(ValueError, match=r"keywords\[0\]"):
        retriever.retrieve([""], top_k=1)
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve(["needle"], top_k=0)
    with pytest.raises(ValueError, match="settings"):
        SparseRetriever()


def _chunk(text: str, chunk_id: str, chunk_index: int) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": "docs/sample.md", "chunk_index": chunk_index},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-1",
    )
