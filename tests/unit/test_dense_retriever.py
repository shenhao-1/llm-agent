"""Unit tests for dense retrieval orchestration."""

from __future__ import annotations

import json
from typing import Any, Mapping

import pytest

from core import RetrievalResult
from core.query_engine import DenseRetriever
from core.trace import TraceContext
from libs.embedding import BaseEmbedding
from libs.vector_store import BaseVectorStore


class FakeEmbedding(BaseEmbedding):
    """Fake embedding client that records query text."""

    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        super().__init__({})
        self.vectors = [[0.1, 0.2, 0.3]] if vectors is None else vectors
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        self.validate_texts(texts)
        self.calls.append(texts)
        return self.vectors


class FakeVectorStore(BaseVectorStore):
    """Fake vector store that records query calls."""

    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        super().__init__({})
        self.results = results or []
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "vector": list(vector),
                "top_k": top_k,
                "filters": dict(filters) if filters else None,
            }
        )
        return self.results[:top_k]

    def get_by_ids(
        self,
        ids: list[str],
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        self.validate_ids(ids)
        by_id = {str(result.get("id", result.get("chunk_id"))): result for result in self.results}
        return [by_id[item_id] for item_id in ids if item_id in by_id]


def test_retrieval_result_is_json_serializable_and_roundtrips() -> None:
    result = RetrievalResult(
        chunk_id="chunk-1",
        score=0.8,
        text="retrieved text",
        metadata={"source_path": "docs/sample.md"},
    )

    payload = result.to_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert RetrievalResult.from_dict(payload) == result


def test_dense_retriever_embeds_query_and_calls_vector_store_with_filters() -> None:
    embedding = FakeEmbedding(vectors=[[1, 2, 3]])
    store = FakeVectorStore(
        [
            {
                "id": "chunk-1",
                "text": "alpha result",
                "metadata": {"source_path": "docs/a.md"},
                "score": 0.91,
            },
            {
                "id": "chunk-2",
                "text": "beta result",
                "metadata": {"source_path": "docs/b.md"},
                "score": 0.82,
            },
        ]
    )

    results = DenseRetriever(
        embedding_client=embedding,
        vector_store=store,
    ).retrieve("hybrid search", top_k=1, filters={"collection": "docs"})

    assert embedding.calls == [["hybrid search"]]
    assert store.calls == [
        {"vector": [1.0, 2.0, 3.0], "top_k": 1, "filters": {"collection": "docs"}}
    ]
    assert results == [
        RetrievalResult(
            chunk_id="chunk-1",
            score=0.91,
            text="alpha result",
            metadata={"source_path": "docs/a.md"},
        )
    ]


def test_dense_retriever_converts_distance_to_similarity_score() -> None:
    store = FakeVectorStore(
        [
            {
                "id": "chunk-1",
                "text": "distance result",
                "metadata": {"source_path": "docs/a.md"},
                "distance": 0.25,
            }
        ]
    )

    result = DenseRetriever(
        embedding_client=FakeEmbedding(),
        vector_store=store,
    ).retrieve("query", top_k=1)[0]

    assert result.score == pytest.approx(0.8)


def test_dense_retriever_accepts_chunk_id_field_from_store() -> None:
    store = FakeVectorStore(
        [
            {
                "chunk_id": "chunk-x",
                "text": "chunk id result",
                "metadata": {"source_path": "docs/x.md"},
            }
        ]
    )

    result = DenseRetriever(
        embedding_client=FakeEmbedding(),
        vector_store=store,
    ).retrieve("query", top_k=1)[0]

    assert result.chunk_id == "chunk-x"
    assert result.score == 0.0


def test_trace_records_dense_retrieval_stage() -> None:
    trace = TraceContext()
    store = FakeVectorStore(
        [{"id": "chunk-1", "text": "text", "metadata": {"source_path": "docs/a.md"}}]
    )

    DenseRetriever(
        embedding_client=FakeEmbedding(),
        vector_store=store,
    ).retrieve("query", top_k=1, filters={"collection": "docs"}, trace=trace)

    assert trace.stages[-1]["name"] == "dense_retrieval"
    assert trace.stages[-1]["method"] == "dense_retriever"
    assert trace.stages[-1]["details"] == {
        "top_k": 1,
        "result_count": 1,
        "filters": {"collection": "docs"},
    }


def test_invalid_inputs_are_rejected() -> None:
    retriever = DenseRetriever(
        embedding_client=FakeEmbedding(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(ValueError, match="query"):
        retriever.retrieve("", top_k=1)
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("query", top_k=0)
    with pytest.raises(ValueError, match="filters"):
        retriever.retrieve("query", top_k=1, filters=["bad"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exactly one"):
        DenseRetriever(
            embedding_client=FakeEmbedding(vectors=[]),
            vector_store=FakeVectorStore(),
        ).retrieve("query", top_k=1)


def test_constructor_requires_settings_or_dependencies() -> None:
    with pytest.raises(ValueError, match="settings"):
        DenseRetriever()
