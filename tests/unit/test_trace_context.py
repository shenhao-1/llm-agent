"""Tests for enhanced trace context and collection."""

from __future__ import annotations

import json
import time

import pytest

from core.trace import TraceCollector, TraceContext


def test_trace_context_records_stage_and_serializes_after_finish() -> None:
    trace = TraceContext(trace_type="query")
    trace.record_stage(
        "dense_retrieval",
        method="dense_retriever",
        details={"top_k": 3},
    )
    time.sleep(0.001)
    trace.finish()

    payload = trace.to_dict()

    assert payload["trace_id"] == trace.trace_id
    assert payload["trace_type"] == "query"
    assert payload["started_at"]
    assert payload["finished_at"]
    assert payload["total_elapsed_ms"] >= 0
    assert payload["stages"][0]["name"] == "dense_retrieval"
    assert payload["stages"][0]["method"] == "dense_retriever"
    assert payload["stages"][0]["details"] == {"top_k": 3}
    assert payload["stages"][0]["elapsed_ms"] >= 0
    json.dumps(payload)


def test_elapsed_ms_returns_total_or_stage_elapsed() -> None:
    trace = TraceContext(trace_type="ingestion")
    trace.record_stage("load")
    trace.finish()

    assert trace.elapsed_ms() == pytest.approx(trace.total_elapsed_ms)
    assert trace.elapsed_ms("load") >= 0
    with pytest.raises(KeyError, match="stage not found"):
        trace.elapsed_ms("missing")


def test_finish_is_idempotent() -> None:
    trace = TraceContext()
    trace.finish()
    first_finished_at = trace.finished_at
    first_elapsed = trace.total_elapsed_ms
    time.sleep(0.001)

    trace.finish()

    assert trace.finished_at == first_finished_at
    assert trace.total_elapsed_ms == first_elapsed


def test_trace_type_is_validated() -> None:
    with pytest.raises(ValueError, match="trace_type"):
        TraceContext(trace_type="other")


def test_trace_collector_finishes_collects_and_persists_jsonl(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    trace = TraceContext(trace_type="query")
    trace.record_stage("query_processing")

    collector = TraceCollector(path)
    collector.collect(trace)

    assert trace.finished_at is not None
    assert len(collector.traces) == 1
    line = path.read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    assert payload["trace_id"] == trace.trace_id
    assert payload["trace_type"] == "query"
    assert payload["stages"][0]["name"] == "query_processing"


def test_trace_collector_validates_input() -> None:
    with pytest.raises(ValueError, match="TraceContext"):
        TraceCollector().collect({"bad": "trace"})  # type: ignore[arg-type]
