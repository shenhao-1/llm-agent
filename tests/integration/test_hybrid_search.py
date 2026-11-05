"""Integration tests for hybrid retrieval orchestration."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from core import Chunk, RetrievalResult
from core.query_engine import (
    DenseRetriever,
    HybridSearch,
    QueryProcessor,
    Reranker,
    RRFFusion,
    SparseRetriever,
)
from core.trace import TraceContext
from ingestion.embedding import SparseEncoder
from ingestion.storage import BM25Indexer
from libs.embedding import BaseEmbedding
from libs.reranker import BaseReranker
from libs.vector_store import BaseVectorStore


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


class StaticEmbedding(BaseEmbedding):
    def __init__(self, vector: list[float]) -> None:
        super().__init__({})
        self.vector = [float(value) for value in vector]

    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        self.validate_texts(texts)
        return [list(self.vector) for _ in texts]


class InMemoryVectorStore(BaseVectorStore):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        super().__init__({})
        self.records = [dict(record) for record in records]

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
        filtered = [
            record
            for record in self.records
            if _metadata_matches(record.get("metadata", {}), filters)
        ]
        ranked = sorted(
            filtered,
            key=lambda record: (
                -_similarity(vector, record.get("vector", [])),
                str(record.get("id", "")),
            ),
        )
        return [
            {
                "id": str(record["id"]),
                "text": str(record.get("text", "")),
                "metadata": dict(record.get("metadata", {})),
                "score": _similarity(vector, record.get("vector", [])),
            }
            for record in ranked[:top_k]
        ]

    def get_by_ids(
        self,
        ids: list[str],
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        self.validate_ids(ids)
        records_by_id = {str(record["id"]): record for record in self.records}
        return [
            {
                "id": item_id,
                "text": str(records_by_id[item_id].get("text", "")),
                "metadata": dict(records_by_id[item_id].get("metadata", {})),
                "vector": list(records_by_id[item_id].get("vector", [])),
            }
            for item_id in ids
            if item_id in records_by_id
        ]


class PassthroughRerankerBackend(BaseReranker):
    def __init__(self) -> None:
        super().__init__({})

    def rerank(
        self,
        query: str,
        candidates: list[Mapping[str, Any]],
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        self.validate_query(query)
        self.validate_candidates(candidates)
        return [dict(candidate) for candidate in candidates]


@pytest.mark.parametrize("strategy", ["dense", "sparse"])
def test_single_route_strategy_uses_only_selected_retriever(strategy: str) -> None:
    dense = FakeDenseRetriever([_result("dense-hit", "dense text", {"collection": "docs"})])
    sparse = FakeSparseRetriever([_result("sparse-hit", "sparse text", {"collection": "docs"})])
    search = HybridSearch(
        settings={"retrieval": {"strategy": strategy}},
        dense_retriever=dense,
        sparse_retriever=sparse,
    )

    results = search.search("alpha", top_k=1)

    assert [item.chunk_id for item in results] == [f"{strategy}-hit"]
    assert len(dense.calls) == (1 if strategy == "dense" else 0)
    assert len(sparse.calls) == (1 if strategy == "sparse" else 0)


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


def test_query_trace_contains_processing_retrieval_fusion_and_rerank_stages(
    tmp_path,
) -> None:
    chunks = [
        _chunk("hybrid retrieval trace pipeline", "chunk-1", 0),
        _chunk("hybrid retrieval reference notes", "chunk-2", 1),
    ]
    bm25 = BM25Indexer(tmp_path).build(SparseEncoder().encode(chunks), persist=False)
    store = InMemoryVectorStore(
        [
            {
                "id": "chunk-1",
                "text": chunks[0].text,
                "vector": [1.0, 0.0, 0.0],
                "metadata": dict(chunks[0].metadata),
            },
            {
                "id": "chunk-2",
                "text": chunks[1].text,
                "vector": [0.0, 1.0, 0.0],
                "metadata": dict(chunks[1].metadata),
            },
        ]
    )
    trace = TraceContext(trace_type="query")

    search = HybridSearch(
        query_processor=QueryProcessor(),
        dense_retriever=DenseRetriever(
            embedding_client=StaticEmbedding([1.0, 0.0, 0.0]),
            vector_store=store,
        ),
        sparse_retriever=SparseRetriever(
            bm25_indexer=bm25,
            vector_store=store,
        ),
        fusion=RRFFusion(k=20),
    )
    results = search.search("hybrid retrieval trace", top_k=2, trace=trace)
    reranked = Reranker(backend=PassthroughRerankerBackend()).rerank(
        "hybrid retrieval trace",
        results,
        trace=trace,
    )

    assert [result.chunk_id for result in reranked] == ["chunk-1", "chunk-2"]
    stage_names = {stage["name"] for stage in trace.stages}
    expected = {
        "query_processing",
        "dense_retrieval",
        "sparse_retrieval",
        "fusion",
        "rerank",
    }
    assert expected.issubset(stage_names)

    trace_by_name = {stage["name"]: stage for stage in trace.stages}
    for name in expected:
        stage = trace_by_name[name]
        assert isinstance(stage["elapsed_ms"], float)
        assert stage["elapsed_ms"] >= 0.0
        assert isinstance(stage["method"], str)
        assert stage["method"]

    payload = trace.to_dict()
    assert payload["trace_type"] == "query"
    assert payload["trace_id"] == trace.trace_id


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


def _chunk(text: str, chunk_id: str, chunk_index: int) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": "docs/sample.md", "chunk_index": chunk_index},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-1",
    )


def _similarity(query_vector: list[float], candidate_vector: list[float]) -> float:
    if not candidate_vector or len(query_vector) != len(candidate_vector):
        return 0.0
    distance = sum(
        (float(left) - float(right)) ** 2 for left, right in zip(query_vector, candidate_vector)
    ) ** 0.5
    return 1.0 / (1.0 + distance)


def _metadata_matches(
    metadata: Any,
    filters: Mapping[str, Any] | None,
) -> bool:
    if not filters:
        return True
    if not isinstance(metadata, Mapping):
        return False
    for key, expected in filters.items():
        if key not in metadata or metadata[key] != expected:
            return False
    return True
