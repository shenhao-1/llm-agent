"""Unit tests for rule-based query preprocessing."""

from __future__ import annotations

import json

import pytest

from core.query_engine import ProcessedQuery, QueryProcessor
from core.trace import TraceContext


def test_process_extracts_non_empty_keywords_and_empty_filters_dict() -> None:
    result = QueryProcessor().process("How does BM25 hybrid retrieval work?")

    assert isinstance(result, ProcessedQuery)
    assert result.keywords == ["does", "bm25", "hybrid", "retrieval", "work"]
    assert result.filters == {}
    assert result.normalized_query == "How does BM25 hybrid retrieval work?"


def test_process_removes_stopwords_and_deduplicates_in_order() -> None:
    result = QueryProcessor().process("the RAG and rag pipeline for search")

    assert result.keywords == ["rag", "pipeline", "search"]


def test_process_parses_supported_field_filters_from_query() -> None:
    result = QueryProcessor().process(
        'collection:docs language:en doc_type:"pdf file" explain chunking'
    )

    assert result.keywords == ["explain", "chunking"]
    assert result.filters == {
        "collection": "docs",
        "language": "en",
        "doc_type": "pdf file",
    }
    assert "collection:" not in result.normalized_query


def test_explicit_filters_are_merged_and_override_query_filters() -> None:
    result = QueryProcessor().process(
        "collection:docs retrieval",
        filters={"collection": "notes", "access_level": "public"},
    )

    assert result.keywords == ["retrieval"]
    assert result.filters == {
        "collection": "notes",
        "access_level": "public",
    }


def test_unknown_filter_like_token_stays_as_keyword_text() -> None:
    result = QueryProcessor().process("topic:rag collection:docs vector search")

    assert result.filters == {"collection": "docs"}
    assert result.keywords == ["topic", "rag", "vector", "search"]


def test_all_stopword_query_falls_back_to_non_empty_keywords() -> None:
    result = QueryProcessor().process("the and of")

    assert result.keywords == ["the", "and", "of"]
    assert result.filters == {}


def test_processed_query_is_json_serializable() -> None:
    result = QueryProcessor().process("collection:docs BM25 search")

    payload = result.to_dict()

    assert json.loads(json.dumps(payload)) == payload


def test_trace_records_query_process_stage() -> None:
    trace = TraceContext()

    result = QueryProcessor().process("collection:docs BM25 search", trace=trace)

    assert trace.stages[-1]["name"] == "query_process"
    assert trace.stages[-1]["method"] == "query_processor"
    assert trace.stages[-1]["details"] == {
        "keyword_count": len(result.keywords),
        "filters": {"collection": "docs"},
    }


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="query"):
        QueryProcessor().process("")
    with pytest.raises(ValueError, match="filters"):
        QueryProcessor().process("rag", filters=["bad"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_keyword_length"):
        QueryProcessor({"min_keyword_length": 0})
    with pytest.raises(ValueError, match="stopwords"):
        QueryProcessor({"stopwords": "bad"})
