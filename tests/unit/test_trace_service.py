"""Unit tests for dashboard trace service."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from observability.dashboard.services import TraceService


def test_trace_service_returns_empty_list_when_log_missing(tmp_path: Path) -> None:
    service = TraceService(tmp_path / "missing.jsonl")

    assert service.list_traces("ingestion") == []
    assert service.list_ingestion_summaries() == []


def test_trace_service_filters_sorts_and_summarizes_ingestion_traces(
    tmp_path: Path,
) -> None:
    path = tmp_path / "traces.jsonl"
    _write_jsonl(
        path,
        [
            _trace(
                "query-1",
                "query",
                "2026-06-03T00:00:00+00:00",
                [],
            ),
            _trace(
                "ingest-old",
                "ingestion",
                "2026-06-03T00:01:00+00:00",
                [
                    {
                        "name": "split",
                        "elapsed_ms": 12.5,
                        "method": "DocumentChunker",
                        "details": {"chunk_count": 2},
                    }
                ],
                source_path="docs/old.pdf",
                collection="docs",
                image_count=0,
            ),
            _trace(
                "ingest-new",
                "ingestion",
                "2026-06-03T00:02:00+00:00",
                [
                    {
                        "name": "load",
                        "elapsed_ms": 7.0,
                        "method": "PdfLoader",
                        "details": {"image_count": 1},
                    },
                    {
                        "name": "store_images",
                        "elapsed_ms": 3.0,
                        "method": "ImageStorage",
                        "details": {"image_count": 1},
                    },
                ],
                source_path="docs/new.pdf",
                collection="docs",
                chunk_count=4,
                image_count=1,
            ),
        ],
    )
    path.write_text(
        path.read_text(encoding="utf-8") + "{bad json\n",
        encoding="utf-8",
    )
    service = TraceService(path)

    traces = service.list_traces("ingestion")
    summaries = service.list_ingestion_summaries()

    assert [trace["trace_id"] for trace in traces] == ["ingest-new", "ingest-old"]
    assert summaries[0]["filename"] == "new.pdf"
    assert summaries[0]["collection"] == "docs"
    assert summaries[0]["chunk_count"] == 4
    assert summaries[0]["image_count"] == 1
    assert summaries[0]["status"] == "success"
    assert summaries[0]["stage_count"] == 2


def test_trace_service_returns_stage_durations_and_processing_stats(
    tmp_path: Path,
) -> None:
    trace = _trace(
        "ingest-1",
        "ingestion",
        "2026-06-03T00:00:00+00:00",
        [
            {"name": "load", "elapsed_ms": 1.2, "method": "PdfLoader"},
            {
                "name": "split",
                "elapsed_ms": 2.3,
                "method": "DocumentChunker",
                "details": {"chunk_count": 3},
            },
        ],
        source_path="docs/a.pdf",
        collection="docs",
        image_count=2,
    )
    service = TraceService(tmp_path / "missing.jsonl")

    assert service.stage_durations(trace) == [
        {"stage": "load", "elapsed_ms": 1.2, "method": "PdfLoader"},
        {"stage": "split", "elapsed_ms": 2.3, "method": "DocumentChunker"},
    ]
    assert service.processing_stats(trace) == {
        "chunks": 3,
        "images": 2,
        "skipped": 0,
        "failed": 0,
    }


def test_trace_service_get_trace_and_validation(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    _write_jsonl(path, [_trace("ingest-1", "ingestion", "2026-06-03T00:00:00+00:00", [])])
    service = TraceService(path)

    assert service.get_trace("ingest-1")["trace_id"] == "ingest-1"
    with pytest.raises(ValueError, match="trace not found"):
        service.get_trace("missing")
    with pytest.raises(ValueError, match="trace_type"):
        service.list_traces("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="limit"):
        service.list_traces(limit=0)


def test_trace_service_summarizes_and_filters_query_traces(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    _write_jsonl(
        path,
        [
            _trace(
                "query-old",
                "query",
                "2026-06-03T00:01:00+00:00",
                [
                    {
                        "name": "dense_retrieval",
                        "elapsed_ms": 1.0,
                        "details": {"top_k": 4, "result_count": 2},
                    },
                    {
                        "name": "sparse_retrieval",
                        "elapsed_ms": 2.0,
                        "details": {"top_k": 4, "result_count": 1},
                    },
                    {
                        "name": "rerank",
                        "elapsed_ms": 3.0,
                        "details": {"result_count": 2, "fallback": True},
                    },
                ],
                query="find alpha",
                collection="docs",
                top_k=2,
                result_count=2,
            ),
            _trace(
                "query-new",
                "query",
                "2026-06-03T00:02:00+00:00",
                [],
                query="find beta",
                result_count=0,
                status="empty",
            ),
        ],
    )
    service = TraceService(path)

    summaries = service.list_query_summaries()
    filtered = service.list_query_summaries(keyword="alpha")

    assert [summary["trace_id"] for summary in summaries] == ["query-new", "query-old"]
    assert filtered == [summaries[1]]
    assert summaries[1]["query"] == "find alpha"
    assert summaries[1]["collection"] == "docs"
    assert summaries[1]["top_k"] == 2
    assert summaries[1]["result_count"] == 2
    assert summaries[1]["dense_count"] == 2
    assert summaries[1]["sparse_count"] == 1
    assert summaries[1]["rerank_fallback"] is True


def test_trace_service_returns_query_comparison_and_rerank_changes(tmp_path: Path) -> None:
    trace = _trace(
        "query-1",
        "query",
        "2026-06-03T00:00:00+00:00",
        [
            {
                "name": "dense_retrieval",
                "elapsed_ms": 1.0,
                "method": "dense_retriever",
                "details": {"top_k": 5, "result_count": 3},
            },
            {
                "name": "sparse_retrieval",
                "elapsed_ms": 2.0,
                "method": "sparse_retriever",
                "details": {"top_k": 5, "result_count": 2},
            },
            {
                "name": "rerank",
                "elapsed_ms": 3.0,
                "method": "core_reranker",
                "details": {
                    "result_count": 3,
                    "before": ["a", "b", "c"],
                    "after": ["b", "a", "c"],
                },
            },
        ],
        query="find gamma",
    )
    service = TraceService(tmp_path / "missing.jsonl")

    comparison = service.retrieval_comparison(trace)
    changes = service.rerank_changes(trace)

    assert comparison[0] == {
        "route": "Dense",
        "stage": "dense_retrieval",
        "method": "dense_retriever",
        "top_k": 5,
        "result_count": 3,
        "elapsed_ms": 1.0,
        "fallback": False,
        "error": None,
    }
    assert comparison[1]["route"] == "Sparse"
    assert changes == [
        {"chunk_id": "a", "before_rank": 1, "after_rank": 2, "delta": -1},
        {"chunk_id": "b", "before_rank": 2, "after_rank": 1, "delta": 1},
        {"chunk_id": "c", "before_rank": 3, "after_rank": 3, "delta": 0},
    ]


def _write_jsonl(path: Path, payloads: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(payload, sort_keys=True) for payload in payloads) + "\n",
        encoding="utf-8",
    )


def _trace(
    trace_id: str,
    trace_type: str,
    started_at: str,
    stages: list[dict],
    **extra,
) -> dict:
    payload = {
        "trace_id": trace_id,
        "trace_type": trace_type,
        "started_at": started_at,
        "finished_at": started_at,
        "total_elapsed_ms": 10.0,
        "stages": stages,
    }
    payload.update(extra)
    return payload
