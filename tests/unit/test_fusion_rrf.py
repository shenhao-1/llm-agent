"""Unit tests for Reciprocal Rank Fusion."""

from __future__ import annotations

import pytest

from core import RetrievalResult
from core.query_engine import RRFFusion
from core.trace import TraceContext


def test_rrf_fuses_dense_and_sparse_rankings_deterministically() -> None:
    dense = [
        _result("a", "dense a"),
        _result("b", "dense b", {"dense": True}),
        _result("c", "dense c"),
    ]
    sparse = [
        _result("b", "sparse b", {"sparse": True}),
        _result("d", "sparse d"),
        _result("a", "sparse a"),
    ]

    results = RRFFusion(k=60).fuse(dense, sparse)

    assert [result.chunk_id for result in results] == ["b", "a", "d", "c"]
    assert results[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert results[1].score == pytest.approx(1 / 61 + 1 / 63)
    assert results[0].text == "dense b"
    assert results[0].metadata == {"dense": True, "sparse": True}


def test_rrf_uses_configurable_k_and_top_k() -> None:
    dense = [_result("a"), _result("b"), _result("c")]
    sparse = [_result("c"), _result("b"), _result("a")]

    results = RRFFusion(k=10).fuse(dense, sparse, top_k=2)

    assert [result.chunk_id for result in results] == ["a", "c"]
    assert len(results) == 2
    assert results[0].score == pytest.approx(1 / 11 + 1 / 13)


def test_rrf_supports_weighted_routes() -> None:
    dense = [_result("dense-only")]
    sparse = [_result("sparse-only")]

    results = RRFFusion(k=10, dense_weight=2.0, sparse_weight=1.0).fuse(dense, sparse)

    assert [result.chunk_id for result in results] == ["dense-only", "sparse-only"]
    assert results[0].score == pytest.approx(2 / 11)
    assert results[1].score == pytest.approx(1 / 11)


def test_duplicate_ids_in_same_ranking_only_count_first_occurrence() -> None:
    dense = [_result("a"), _result("a"), _result("b")]

    results = RRFFusion(k=10).fuse(dense, [])

    assert [result.chunk_id for result in results] == ["a", "b"]
    assert results[0].score == pytest.approx(1 / 11)
    assert results[1].score == pytest.approx(1 / 13)


def test_empty_inputs_return_empty_list() -> None:
    assert RRFFusion().fuse([], []) == []


def test_trace_records_fusion_stage() -> None:
    trace = TraceContext()

    results = RRFFusion(k=20).fuse([_result("a")], [_result("b")], trace=trace)

    assert [result.chunk_id for result in results] == ["a", "b"]
    assert trace.stages[-1]["name"] == "fusion"
    assert trace.stages[-1]["method"] == "rrf"
    assert trace.stages[-1]["details"] == {
        "dense_count": 1,
        "sparse_count": 1,
        "result_count": 2,
        "k": 20,
    }


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="k"):
        RRFFusion(k=0)
    with pytest.raises(ValueError, match="dense_weight"):
        RRFFusion(dense_weight=0)
    with pytest.raises(ValueError, match="top_k"):
        RRFFusion().fuse([], [], top_k=0)
    with pytest.raises(ValueError, match="dense_results"):
        RRFFusion().fuse(["bad"], [])  # type: ignore[list-item]


def _result(
    chunk_id: str,
    text: str | None = None,
    metadata: dict | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=0.0,
        text=text if text is not None else f"text {chunk_id}",
        metadata=metadata or {},
    )
