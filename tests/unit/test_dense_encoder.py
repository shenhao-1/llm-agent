"""Unit tests for dense chunk encoding."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from core import Chunk, ChunkRecord
from core.trace import TraceContext
from ingestion.embedding import DenseEncoder
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory


class FakeEmbedding(BaseEmbedding):
    """Fake embedding provider with deterministic vectors."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        vectors: list[list[float]] | None = None,
    ) -> None:
        super().__init__(config)
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        self.validate_texts(texts)
        self.calls.append(texts)
        if self.vectors is not None:
            return self.vectors
        return [[float(index), float(len(text))] for index, text in enumerate(texts)]


@pytest.fixture(autouse=True)
def reset_embedding_registry() -> Any:
    original = dict(EmbeddingFactory._providers)
    EmbeddingFactory._providers = {"fake": FakeEmbedding}
    yield
    EmbeddingFactory._providers = original


def test_encode_returns_chunk_records_with_dense_vectors() -> None:
    chunks = [_chunk("alpha", "chunk-1"), _chunk("beta", "chunk-2")]
    embedding = FakeEmbedding()

    records = DenseEncoder(embedding=embedding).encode(chunks)

    assert all(isinstance(record, ChunkRecord) for record in records)
    assert [record.id for record in records] == ["chunk-1", "chunk-2"]
    assert [record.text for record in records] == ["alpha", "beta"]
    assert [record.dense_vector for record in records] == [[0.0, 5.0], [1.0, 4.0]]
    assert records[0].metadata["source_path"] == "docs/sample.md"
    assert embedding.calls == [["alpha", "beta"]]


def test_encoder_can_create_embedding_from_settings() -> None:
    chunks = [_chunk("alpha")]

    records = DenseEncoder({"embedding": {"provider": "fake"}}).encode(chunks)

    assert records[0].dense_vector == [0.0, 5.0]


def test_encode_empty_chunk_list_returns_empty_without_embedding_call() -> None:
    embedding = FakeEmbedding()

    records = DenseEncoder(embedding=embedding).encode([])

    assert records == []
    assert embedding.calls == []


def test_encode_rejects_non_chunk_entries() -> None:
    with pytest.raises(ValueError, match=r"chunks\[0\]"):
        DenseEncoder(embedding=FakeEmbedding()).encode(["bad"])


def test_encode_rejects_vector_count_mismatch() -> None:
    chunks = [_chunk("alpha"), _chunk("beta", "chunk-2")]
    embedding = FakeEmbedding(vectors=[[1.0, 2.0]])

    with pytest.raises(ValueError, match="output count"):
        DenseEncoder(embedding=embedding).encode(chunks)


def test_encode_rejects_inconsistent_vector_dimensions() -> None:
    chunks = [_chunk("alpha"), _chunk("beta", "chunk-2")]
    embedding = FakeEmbedding(vectors=[[1.0, 2.0], [3.0]])

    with pytest.raises(ValueError, match="consistent dimensions"):
        DenseEncoder(embedding=embedding).encode(chunks)


def test_encode_rejects_non_numeric_vector_values() -> None:
    chunks = [_chunk("alpha")]
    embedding = FakeEmbedding(vectors=[[1.0, "bad"]])  # type: ignore[list-item]

    with pytest.raises(ValueError, match=r"vectors\[0\]\[1\]"):
        DenseEncoder(embedding=embedding).encode(chunks)


def test_trace_records_dense_embedding_stage() -> None:
    trace = TraceContext()

    DenseEncoder(embedding=FakeEmbedding()).encode([_chunk("alpha")], trace=trace)

    assert trace.stages[-1]["name"] == "embed"
    assert trace.stages[-1]["method"] == "dense_encoder"
    assert trace.stages[-1]["details"] == {
        "chunk_count": 1,
        "vector_count": 1,
        "dimension": 2,
    }


def _chunk(text: str, chunk_id: str = "chunk-1") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": "docs/sample.md", "chunk_index": 0},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-1",
    )
