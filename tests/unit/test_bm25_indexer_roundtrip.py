"""Unit tests for BM25 index build, persistence, and incremental updates."""

from __future__ import annotations

import math

import pytest

from core import Chunk
from ingestion.embedding import SparseEncoder
from ingestion.storage import BM25Indexer, INDEX_FILENAME


def test_build_save_load_and_query_returns_stable_top_ids(tmp_path) -> None:
    sparse_result = SparseEncoder().encode(
        [
            _chunk("python python retrieval", "chunk-a", 0),
            _chunk("dense vector search", "chunk-b", 1),
            _chunk("image caption search", "chunk-c", 2),
        ]
    )
    indexer = BM25Indexer(tmp_path).build(sparse_result)

    assert (tmp_path / INDEX_FILENAME).exists()
    assert [item["chunk_id"] for item in indexer.query("python", top_k=3)] == [
        "chunk-a"
    ]

    loaded = BM25Indexer(tmp_path).load()

    assert loaded.to_dict() == indexer.to_dict()
    assert [item["chunk_id"] for item in loaded.query("python", top_k=3)] == [
        "chunk-a"
    ]


def test_idf_matches_spec_formula_for_known_corpus(tmp_path) -> None:
    sparse_result = SparseEncoder().encode(
        [
            _chunk("alpha unique", "chunk-a", 0),
            _chunk("alpha beta", "chunk-b", 1),
            _chunk("alpha gamma", "chunk-c", 2),
        ]
    )

    indexer = BM25Indexer(tmp_path).build(sparse_result, persist=False)

    assert indexer.idf("unique") == pytest.approx(math.log((3 - 1 + 0.5) / (1 + 0.5)))
    assert indexer.idf("alpha") == pytest.approx(math.log((3 - 3 + 0.5) / (3 + 0.5)))
    assert indexer.index["unique"] == {
        "idf": pytest.approx(math.log(2.5 / 1.5)),
        "postings": [{"chunk_id": "chunk-a", "tf": 1.0, "doc_length": 2}],
    }


def test_rebuild_replaces_existing_index_content(tmp_path) -> None:
    indexer = BM25Indexer(tmp_path)
    indexer.build(
        SparseEncoder().encode([_chunk("legacy token", "chunk-old", 0)])
    )

    indexer.rebuild(
        SparseEncoder().encode([_chunk("fresh token", "chunk-new", 0)])
    )

    assert indexer.query("legacy") == []
    assert [item["chunk_id"] for item in indexer.query("fresh")] == ["chunk-new"]
    assert "chunk-old" not in BM25Indexer(tmp_path).load().documents


def test_incremental_update_adds_and_replaces_records(tmp_path) -> None:
    indexer = BM25Indexer(tmp_path).build(
        SparseEncoder().encode([_chunk("alpha first", "chunk-a", 0)])
    )

    indexer.update(
        SparseEncoder().encode(
            [
                _chunk("beta replacement", "chunk-a", 0),
                _chunk("gamma addition", "chunk-b", 1),
            ]
        )
    )

    loaded = BM25Indexer.from_disk(tmp_path)
    assert loaded.query("alpha") == []
    assert [item["chunk_id"] for item in loaded.query("beta")] == ["chunk-a"]
    assert [item["chunk_id"] for item in loaded.query("gamma")] == ["chunk-b"]
    assert loaded.doc_count == 2


def test_remove_document_deletes_matching_source_and_recomputes_index(tmp_path) -> None:
    indexer = BM25Indexer(tmp_path).build(
        SparseEncoder().encode(
            [
                _chunk("alpha one", "chunk-a", 0),
                _chunk("alpha two", "chunk-b", 1),
                Chunk(
                    id="chunk-c",
                    text="beta notes",
                    metadata={
                        "source_path": "docs/other.md",
                        "collection": "notes",
                        "chunk_index": 0,
                    },
                ),
            ]
        )
    )

    assert indexer.remove_document("docs/sample.md") == 2

    loaded = BM25Indexer(tmp_path).load()
    assert loaded.query("alpha") == []
    assert [item["chunk_id"] for item in loaded.query("beta")] == ["chunk-c"]
    assert loaded.doc_count == 1


def test_invalid_inputs_are_rejected(tmp_path) -> None:
    indexer = BM25Indexer(tmp_path)

    with pytest.raises(ValueError, match="SparseEncodingResult"):
        indexer.build("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="query"):
        indexer.query(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="top_k"):
        indexer.query("alpha", top_k=0)


def _chunk(text: str, chunk_id: str, chunk_index: int) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": "docs/sample.md", "chunk_index": chunk_index},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-1",
    )
