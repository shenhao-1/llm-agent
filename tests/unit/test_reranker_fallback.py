"""Unit tests for core reranker fallback behavior."""

from __future__ import annotations

import time
from typing import Any, Mapping

import pytest

from core import RetrievalResult
from core.query_engine import Reranker
from core.trace import TraceContext
from libs.reranker import BaseReranker


class FakeBackend(BaseReranker):
    def __init__(
        self,
        results: list[dict[str, Any]] | None = None,
        *,
        fail: bool = False,
        sleep_seconds: float = 0.0,
    ) -> None:
        super().__init__({})
        self.results = results
        self.fail = fail
        self.sleep_seconds = sleep_seconds
        self.calls: list[dict[str, Any]] = []

    def rerank(
        self,
        query: str,
        candidates: list[Mapping[str, Any]],
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        self.validate_query(query)
        self.validate_candidates(candidates)
        self.calls.append({"query": query, "ids": [candidate["id"] for candidate in candidates]})
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        if self.fail:
            raise RuntimeError("backend failed")
        if self.results is not None:
            return self.results
        return [dict(candidate) for candidate in reversed(candidates)]


def test_reranker_uses_backend_order_and_normalizes_results() -> None:
    backend = FakeBackend(
        [
            {
                "id": "chunk-2",
                "text": "reranked text",
                "score": 0.98,
                "metadata": {"rerank_reason": "better match"},
            },
            {"id": "chunk-1", "score": 0.72},
        ]
    )
    candidates = [
        _result("chunk-1", 0.4, "first", {"source_path": "docs/one.md"}),
        _result("chunk-2", 0.3, "second", {"source_path": "docs/two.md"}),
    ]

    results = Reranker(backend=backend).rerank("find second", candidates)

    assert [result.chunk_id for result in results] == ["chunk-2", "chunk-1"]
    assert results[0].score == 0.98
    assert results[0].text == "reranked text"
    assert results[0].metadata == {
        "source_path": "docs/two.md",
        "rerank_reason": "better match",
    }
    assert backend.calls == [{"query": "find second", "ids": ["chunk-1", "chunk-2"]}]


def test_backend_exception_falls_back_to_original_order_and_marks_metadata() -> None:
    candidates = [
        _result("chunk-1", 0.4, "first", {"source_path": "docs/one.md"}),
        _result("chunk-2", 0.3, "second", {"source_path": "docs/two.md"}),
    ]
    trace = TraceContext()

    results = Reranker(backend=FakeBackend(fail=True)).rerank(
        "query",
        candidates,
        trace=trace,
    )

    assert [result.chunk_id for result in results] == ["chunk-1", "chunk-2"]
    assert all(result.metadata["rerank_fallback"] is True for result in results)
    assert all("backend failed" in result.metadata["rerank_error"] for result in results)
    assert trace.stages[-1]["name"] == "rerank"
    assert trace.stages[-1]["details"]["fallback"] is True
    assert "backend failed" in trace.stages[-1]["details"]["error"]


def test_backend_timeout_falls_back_without_raising() -> None:
    candidates = [_result("chunk-1", 0.4, "first", {"source_path": "docs/one.md"})]

    results = Reranker(
        backend=FakeBackend(sleep_seconds=0.1),
        timeout_seconds=0.01,
    ).rerank("query", candidates)

    assert [result.chunk_id for result in results] == ["chunk-1"]
    assert results[0].metadata["rerank_fallback"] is True
    assert "timed out" in results[0].metadata["rerank_error"]


def test_empty_candidates_return_empty_and_trace_success() -> None:
    trace = TraceContext()

    results = Reranker(backend=FakeBackend()).rerank("query", [], trace=trace)

    assert results == []
    assert trace.stages[-1]["name"] == "rerank"
    assert trace.stages[-1]["details"] == {"fallback": False, "result_count": 0}


def test_invalid_backend_output_falls_back() -> None:
    candidates = [_result("chunk-1", 0.4, "first", {"source_path": "docs/one.md"})]

    results = Reranker(backend=FakeBackend(results=[])).rerank("query", candidates)

    assert results[0].metadata["rerank_fallback"] is True
    assert "no results" in results[0].metadata["rerank_error"]


def test_invalid_inputs_are_rejected() -> None:
    reranker = Reranker(backend=FakeBackend())

    with pytest.raises(ValueError, match="query"):
        reranker.rerank("", [])
    with pytest.raises(ValueError, match="candidates"):
        reranker.rerank("query", "bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"candidates\[0\]"):
        reranker.rerank("query", ["bad"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="settings"):
        Reranker()
    with pytest.raises(ValueError, match="timeout_seconds"):
        Reranker(backend=FakeBackend(), timeout_seconds=0)


def _result(
    chunk_id: str,
    score: float,
    text: str,
    metadata: dict[str, Any],
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text,
        metadata=metadata,
    )
