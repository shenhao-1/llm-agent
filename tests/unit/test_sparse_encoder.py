"""Unit tests for sparse term-statistics encoding."""

from __future__ import annotations

import json

import pytest

from core import Chunk, ChunkRecord
from core.trace import TraceContext
from ingestion.embedding import SparseEncoder, SparseEncodingResult


def test_encode_returns_sparse_records_and_corpus_stats() -> None:
    chunks = [
        _chunk("BM25 retrieval retrieval", "chunk-1"),
        _chunk("Dense retrieval", "chunk-2"),
    ]

    result = SparseEncoder().encode(chunks)

    assert isinstance(result, SparseEncodingResult)
    assert all(isinstance(record, ChunkRecord) for record in result.records)
    assert result.records[0].sparse_vector == {"bm25": 1.0, "retrieval": 2.0}
    assert result.records[1].sparse_vector == {"dense": 1.0, "retrieval": 1.0}
    assert result.corpus_stats["doc_count"] == 2
    assert result.corpus_stats["doc_lengths"] == {"chunk-1": 3, "chunk-2": 2}
    assert result.corpus_stats["avg_doc_length"] == 2.5
    assert result.corpus_stats["document_frequency"] == {
        "bm25": 1,
        "dense": 1,
        "retrieval": 2,
    }


def test_tokenization_normalizes_case_and_punctuation() -> None:
    encoder = SparseEncoder()

    tokens = encoder.tokenize("Hybrid-search, HYBRID_search! v2")

    assert tokens == ["hybrid", "search", "hybrid_search", "v2"]


def test_min_token_length_configuration_filters_short_terms() -> None:
    encoder = SparseEncoder({"min_token_length": 3})

    result = encoder.encode([_chunk("a ai rag bm25")])

    assert result.records[0].sparse_vector == {"bm25": 1.0, "rag": 1.0}
    assert result.corpus_stats["doc_lengths"] == {"chunk-1": 2}


def test_empty_text_has_explicit_none_sparse_vector() -> None:
    result = SparseEncoder().encode([_chunk("", "empty")])

    assert result.records[0].sparse_vector is None
    assert result.corpus_stats["doc_lengths"] == {"empty": 0}
    assert result.corpus_stats["avg_doc_length"] == 0.0
    assert result.corpus_stats["document_frequency"] == {}


def test_empty_chunk_list_returns_empty_result() -> None:
    result = SparseEncoder().encode([])

    assert result.records == []
    assert result.corpus_stats == {
        "doc_count": 0,
        "avg_doc_length": 0.0,
        "document_frequency": {},
        "doc_lengths": {},
    }


def test_result_is_json_serializable() -> None:
    result = SparseEncoder().encode([_chunk("json ready")])

    payload = result.to_dict()

    assert json.loads(json.dumps(payload)) == payload


def test_encode_rejects_non_chunk_entries() -> None:
    with pytest.raises(ValueError, match=r"chunks\[0\]"):
        SparseEncoder().encode(["bad"])


def test_invalid_min_token_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="min_token_length"):
        SparseEncoder({"min_token_length": 0})


def test_trace_records_sparse_embedding_stage() -> None:
    trace = TraceContext()

    SparseEncoder().encode([_chunk("trace retrieval")], trace=trace)

    assert trace.stages[-1]["name"] == "embed"
    assert trace.stages[-1]["method"] == "sparse_encoder"
    assert trace.stages[-1]["details"] == {
        "chunk_count": 1,
        "vocabulary_size": 2,
        "avg_doc_length": 2.0,
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
