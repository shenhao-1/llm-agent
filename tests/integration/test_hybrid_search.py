"""Integration tests for hybrid retrieval orchestration."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from core import RetrievalResult
from core.query_engine import HybridSearch, QueryProcessor, RRFFusion
from core.trace import TraceContext


class FakeDenseRetriever:
    def __init__(self, results: list[RetrievalResult] | None = None, fail: bool = False) -> None:
        self.results = results or []
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "filters": dict(filters) if filters else {},
            }
        )
        if self.fail:
            raise RuntimeError("dense route failed")
        return self.results[:top_k]


class FakeSparseRetriever:
    def __init__(self, results: list[RetrievalResult] | None = None, fail: bool = False) -> None:
        self.results = results or []
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self,
        keywords: list[str],
        top_k: int,
        trace: Any | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append({"keywords": list(keywords), "top_k": top_k})
        if self.fail:
            raise RuntimeError("sparse route failed")
        return self.results[:top_k]


def test_hybrid_search_returns_fused_top_k_with_text_and_metadata() -> None:
    dense = FakeDenseRetriever(
        [
            _result("a", "dense alpha", {"collection": "docs", "route": "dense"}),
            _result("b", "dense beta", {"collection": "docs"}),
        ]
    )
    sparse = FakeSparseRetriever(
        [
            _result("b", "sparse beta", {"collection": "docs", "topic": "bm25"}),
            _result("c", "sparse gamma", {"collection": "docs"}),
        ]
    )

    results = HybridSearch(
        query_processor=QueryProcessor(),
        dense_retriever=dense,  # type: ignore[arg-type]
        sparse_retriever=sparse,  # type: ignore[arg-type]
        fusion=RRFFusion(k=60),
    ).search("How does BM25 retrieval work?", top_k=2)

    assert [result.chunk_id for result in results] == ["b", "a"]
    assert results[0].text == "dense beta"
    assert results[0].metadata["topic"] == "bm25"
    assert dense.calls[0]["query"] == "How does BM25 retrieval work?"
    assert sparse.calls[0]["keywords"] == ["does", "bm25", "retrieval", "work"]
    assert dense.calls[0]["top_k"] == 4
    assert sparse.calls[0]["top_k"] == 4


def test_hybrid_search_applies_filters_from_query_and_argument() -> None:
    dense = FakeDenseRetriever(
        [
            _result("a", "docs result", {"collection": "docs", "doc_type": "pdf"}),
            _result("b", "notes result", {"collection": "notes", "doc_type": "pdf"}),
            _result("c", "missing collection result", {"doc_type": "pdf"}),
        ]
    )
    sparse = FakeSparseRetriever([])

    results = HybridSearch(
        query_processor=QueryProcessor(),
        dense_retriever=dense,  # type: ignore[arg-type]
        sparse_retriever=sparse,  # type: ignore[arg-type]
        fusion=RRFFusion(k=10),
    ).search(
        "collection:docs retrieval",
        top_k=5,
        filters={"doc_type": "pdf"},
    )

    assert [result.chunk_id for result in results] == ["a", "c"]
    assert dense.calls[0]["query"] == "retrieval"
    assert dense.calls[0]["filters"] == {"collection": "docs", "doc_type": "pdf"}
    assert sparse.calls[0]["keywords"] == ["retrieval"]


def test_hybrid_search_degrades_to_sparse_when_dense_route_fails() -> None:
    dense = FakeDenseRetriever(fail=True)
    sparse = FakeSparseRetriever([_result("s", "sparse only", {"collection": "docs"})])
    trace = TraceContext()

    results = HybridSearch(
        query_processor=QueryProcessor(),
        dense_retriever=dense,  # type: ignore[arg-type]
        sparse_retriever=sparse,  # type: ignore[arg-type]
        fusion=RRFFusion(k=10),
    ).search("fallback retrieval", top_k=3, trace=trace)

    assert [result.chunk_id for result in results] == ["s"]
    assert trace.stages[-1]["name"] == "hybrid_search"
    assert trace.stages[-1]["details"]["fallback"] is True
    assert "dense" in trace.stages[-1]["details"]["errors"]


def test_hybrid_search_degrades_to_dense_when_sparse_route_fails() -> None:
    dense = FakeDenseRetriever([_result("d", "dense only", {"collection": "docs"})])
    sparse = FakeSparseRetriever(fail=True)

    results = HybridSearch(
        query_processor=QueryProcessor(),
        dense_retriever=dense,  # type: ignore[arg-type]
        sparse_retriever=sparse,  # type: ignore[arg-type]
        fusion=RRFFusion(k=10),
    ).search("fallback retrieval", top_k=3)

    assert [result.chunk_id for result in results] == ["d"]


def test_hybrid_search_raises_when_both_routes_fail() -> None:
    search = HybridSearch(
        query_processor=QueryProcessor(),
        dense_retriever=FakeDenseRetriever(fail=True),  # type: ignore[arg-type]
        sparse_retriever=FakeSparseRetriever(fail=True),  # type: ignore[arg-type]
        fusion=RRFFusion(k=10),
    )

    with pytest.raises(RuntimeError, match="both retrieval routes failed"):
        search.search("broken retrieval", top_k=3)


def test_hybrid_search_rejects_invalid_inputs() -> None:
    search = HybridSearch(
        query_processor=QueryProcessor(),
        dense_retriever=FakeDenseRetriever(),  # type: ignore[arg-type]
        sparse_retriever=FakeSparseRetriever(),  # type: ignore[arg-type]
        fusion=RRFFusion(),
    )

    with pytest.raises(ValueError, match="top_k"):
        search.search("query", top_k=0)
    with pytest.raises(ValueError, match="filters"):
        search.search("query", top_k=1, filters=["bad"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="settings"):
        HybridSearch()


def _result(
    chunk_id: str,
    text: str,
    metadata: dict[str, Any],
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=0.0,
        text=text,
        metadata=metadata,
    )
