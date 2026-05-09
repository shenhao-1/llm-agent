"""Unit tests for idempotent dense-vector upserts."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import pytest

from core import ChunkRecord
from core.trace import TraceContext
from ingestion.storage import VectorUpserter, VectorUpsertResult
from libs.vector_store import BaseVectorStore


class FakeVectorStore(BaseVectorStore):
    """Dict-backed vector store that mimics idempotent upsert by id."""

    def __init__(self) -> None:
        super().__init__({})
        self.records_by_id: dict[str, dict[str, Any]] = {}
        self.calls: list[list[str]] = []

    def upsert(
        self,
        records: list[Mapping[str, Any]],
        trace: Any | None = None,
    ) -> None:
        self.validate_records(records)
        self.calls.append([str(record["id"]) for record in records])
        for record in records:
            self.records_by_id[str(record["id"])] = dict(record)

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        self.validate_query_args(vector, top_k, filters)
        return list(self.records_by_id.values())[:top_k]

    def get_by_ids(
        self,
        ids: list[str],
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        self.validate_ids(ids)
        return [self.records_by_id[item_id] for item_id in ids if item_id in self.records_by_id]


def test_same_chunk_upsert_twice_uses_same_id_without_duplicate_records() -> None:
    store = FakeVectorStore()
    upserter = VectorUpserter(vector_store=store)
    record = _record("chunk-a", "same content", [0.1, 0.2], chunk_index=0)

    first = upserter.upsert([record])
    second = upserter.upsert([record])

    assert isinstance(first, VectorUpsertResult)
    assert first.ids == second.ids
    assert first.ids == [_expected_id("docs/sample.md", 0, "same content")]
    assert len(store.records_by_id) == 1
    assert store.calls == [first.ids, second.ids]


def test_content_change_changes_deterministic_id() -> None:
    upserter = VectorUpserter(vector_store=FakeVectorStore())
    original = _record("chunk-a", "old content", [0.1], chunk_index=2)
    changed = _record("chunk-a", "new content", [0.9], chunk_index=2)

    assert upserter.prepare_records([original])[0]["id"] != upserter.prepare_records(
        [changed]
    )[0]["id"]


def test_batch_upsert_preserves_input_order_and_payload_shape() -> None:
    store = FakeVectorStore()
    upserter = VectorUpserter(vector_store=store)
    records = [
        _record("chunk-a", "alpha", [1, 0], chunk_index=0),
        _record("chunk-b", "beta", [0, 1], chunk_index=1),
        _record("chunk-c", "gamma", [0.5, 0.5], chunk_index=2),
    ]

    result = upserter.upsert(records)

    assert result.ids == [
        _expected_id("docs/sample.md", 0, "alpha"),
        _expected_id("docs/sample.md", 1, "beta"),
        _expected_id("docs/sample.md", 2, "gamma"),
    ]
    assert store.calls == [result.ids]
    assert [record["id"] for record in result.records] == result.ids
    assert result.records[0]["vector"] == [1.0, 0.0]
    assert result.records[0]["text"] == "alpha"
    assert result.records[0]["metadata"]["source_chunk_id"] == "chunk-a"
    assert result.records[0]["metadata"]["vector_id"] == result.ids[0]


def test_empty_batch_is_noop() -> None:
    store = FakeVectorStore()

    result = VectorUpserter(vector_store=store).upsert([])

    assert result.count == 0
    assert result.ids == []
    assert store.calls == []


def test_invalid_dense_record_is_rejected() -> None:
    upserter = VectorUpserter(vector_store=FakeVectorStore())

    with pytest.raises(ValueError, match="dense_vector is required"):
        upserter.prepare_records(
            [
                ChunkRecord(
                    id="chunk-a",
                    text="missing vector",
                    metadata={"source_path": "docs/sample.md", "chunk_index": 0},
                )
            ]
        )
    with pytest.raises(ValueError, match="metadata.chunk_index"):
        upserter.prepare_records(
            [
                ChunkRecord(
                    id="chunk-a",
                    text="bad metadata",
                    metadata={"source_path": "docs/sample.md"},
                    dense_vector=[0.1],
                )
            ]
        )


def test_trace_records_vector_upsert_stage() -> None:
    trace = TraceContext()
    record = _record("chunk-a", "trace content", [0.1], chunk_index=0)

    result = VectorUpserter(vector_store=FakeVectorStore()).upsert([record], trace=trace)

    assert trace.stages[-1]["name"] == "store"
    assert trace.stages[-1]["method"] == "vector_upserter"
    assert trace.stages[-1]["details"] == {"record_count": 1, "ids": result.ids}


def _record(
    record_id: str,
    text: str,
    vector: list[float],
    *,
    chunk_index: int,
) -> ChunkRecord:
    return ChunkRecord(
        id=record_id,
        text=text,
        metadata={"source_path": "docs/sample.md", "chunk_index": chunk_index},
        dense_vector=vector,
    )


def _expected_id(source_path: str, chunk_index: int, text: str) -> str:
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    raw = f"{source_path}:{chunk_index}:{content_hash[:8]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
