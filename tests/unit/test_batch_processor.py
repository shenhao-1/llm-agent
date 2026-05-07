"""Unit tests for batch encoding orchestration."""

from __future__ import annotations

from typing import Any

import pytest

from core import Chunk, ChunkRecord
from core.trace import TraceContext
from ingestion.embedding import (
    BatchProcessingResult,
    BatchProcessor,
    DenseEncoder,
    SparseEncoder,
    SparseEncodingResult,
)


class FakeDenseEncoder:
    """Fake dense encoder that records batch boundaries."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, chunks: list[Chunk], trace: Any | None = None) -> list[ChunkRecord]:
        self.calls.append([chunk.id for chunk in chunks])
        return [
            ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=dict(chunk.metadata),
                dense_vector=[float(len(chunk.text)), float(index)],
            )
            for index, chunk in enumerate(chunks)
        ]


class FakeSparseEncoder:
    """Fake sparse encoder with predictable sparse stats."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(
        self, chunks: list[Chunk], trace: Any | None = None
    ) -> SparseEncodingResult:
        self.calls.append([chunk.id for chunk in chunks])
        records = [
            ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=dict(chunk.metadata),
                sparse_vector={chunk.id: 1.0},
            )
            for chunk in chunks
        ]
        return SparseEncodingResult(
            records=records,
            corpus_stats={
                "doc_count": len(chunks),
                "avg_doc_length": 1.0 if chunks else 0.0,
                "document_frequency": {chunk.id: 1 for chunk in chunks},
                "doc_lengths": {chunk.id: 1 for chunk in chunks},
            },
        )


def test_batch_size_two_splits_five_chunks_into_three_stable_batches() -> None:
    dense = FakeDenseEncoder()
    sparse = FakeSparseEncoder()
    chunks = [_chunk(index) for index in range(5)]

    result = BatchProcessor(
        dense_encoder=dense,  # type: ignore[arg-type]
        sparse_encoder=sparse,  # type: ignore[arg-type]
        batch_size=2,
    ).process(chunks)

    assert isinstance(result, BatchProcessingResult)
    assert result.batch_count == 3
    assert [batch.size for batch in result.batches] == [2, 2, 1]
    assert [[chunk.id for chunk in batch.chunks] for batch in result.batches] == [
        ["chunk-0", "chunk-1"],
        ["chunk-2", "chunk-3"],
        ["chunk-4"],
    ]
    assert dense.calls == sparse.calls
    assert dense.calls == [["chunk-0", "chunk-1"], ["chunk-2", "chunk-3"], ["chunk-4"]]


def test_process_aggregates_dense_records_and_sparse_stats() -> None:
    chunks = [_chunk(index) for index in range(3)]

    result = BatchProcessor(
        dense_encoder=FakeDenseEncoder(),  # type: ignore[arg-type]
        sparse_encoder=FakeSparseEncoder(),  # type: ignore[arg-type]
        batch_size=2,
    ).process(chunks)

    assert [record.id for record in result.dense_records] == [
        "chunk-0",
        "chunk-1",
        "chunk-2",
    ]
    assert [record.id for record in result.sparse_result.records] == [
        "chunk-0",
        "chunk-1",
        "chunk-2",
    ]
    assert result.sparse_result.corpus_stats == {
        "doc_count": 3,
        "avg_doc_length": 1.0,
        "document_frequency": {"chunk-0": 1, "chunk-1": 1, "chunk-2": 1},
        "doc_lengths": {"chunk-0": 1, "chunk-1": 1, "chunk-2": 1},
    }
    assert all(batch.elapsed_ms >= 0 for batch in result.batches)
    assert result.elapsed_ms >= 0


def test_iter_batches_returns_empty_for_empty_input() -> None:
    processor = BatchProcessor(
        dense_encoder=FakeDenseEncoder(),  # type: ignore[arg-type]
        sparse_encoder=FakeSparseEncoder(),  # type: ignore[arg-type]
        batch_size=2,
    )

    assert processor.iter_batches([]) == []
    result = processor.process([])
    assert result.batches == []
    assert result.dense_records == []
    assert result.sparse_result.corpus_stats["doc_count"] == 0


def test_batch_size_can_come_from_settings() -> None:
    processor = BatchProcessor(
        {"ingestion": {"batch_processor": {"batch_size": 3}}},
        dense_encoder=FakeDenseEncoder(),  # type: ignore[arg-type]
        sparse_encoder=FakeSparseEncoder(),  # type: ignore[arg-type]
    )

    assert [len(batch) for batch in processor.iter_batches([_chunk(i) for i in range(7)])] == [
        3,
        3,
        1,
    ]


def test_invalid_batch_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        BatchProcessor(
            dense_encoder=FakeDenseEncoder(),  # type: ignore[arg-type]
            sparse_encoder=FakeSparseEncoder(),  # type: ignore[arg-type]
            batch_size=0,
        )


def test_process_rejects_non_chunk_entries() -> None:
    with pytest.raises(ValueError, match=r"chunks\[0\]"):
        BatchProcessor(
            dense_encoder=FakeDenseEncoder(),  # type: ignore[arg-type]
            sparse_encoder=FakeSparseEncoder(),  # type: ignore[arg-type]
            batch_size=2,
        ).process(["bad"])


def test_trace_records_batch_processor_stage() -> None:
    trace = TraceContext()

    BatchProcessor(
        dense_encoder=FakeDenseEncoder(),  # type: ignore[arg-type]
        sparse_encoder=FakeSparseEncoder(),  # type: ignore[arg-type]
        batch_size=2,
    ).process([_chunk(0), _chunk(1), _chunk(2)], trace=trace)

    assert trace.stages[-1]["name"] == "embed"
    assert trace.stages[-1]["method"] == "batch_processor"
    assert trace.stages[-1]["details"]["chunk_count"] == 3
    assert trace.stages[-1]["details"]["batch_count"] == 2
    assert trace.stages[-1]["details"]["batch_size"] == 2
    assert trace.stages[-1]["details"]["elapsed_ms"] >= 0


def test_default_constructor_uses_real_encoder_types_when_settings_are_valid() -> None:
    processor = BatchProcessor(
        {"embedding": {"provider": "openai", "api_key": "test", "model": "m"}},
        batch_size=1,
    )

    assert isinstance(processor.dense_encoder, DenseEncoder)
    assert isinstance(processor.sparse_encoder, SparseEncoder)


def _chunk(index: int) -> Chunk:
    text = f"chunk text {index}"
    return Chunk(
        id=f"chunk-{index}",
        text=text,
        metadata={"source_path": "docs/sample.md", "chunk_index": index},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-1",
    )
