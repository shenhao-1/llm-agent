"""Tests for JSON Lines trace logging."""

from __future__ import annotations

import json
import logging

import pytest

from observability.logger import JSONFormatter, get_trace_logger, write_trace


def test_json_formatter_outputs_valid_json_with_trace_fields() -> None:
    record = logging.LogRecord(
        name="test.trace",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="trace",
        args=(),
        exc_info=None,
    )
    record.trace = {"trace_id": "trace-1", "trace_type": "query", "stages": []}

    payload = json.loads(JSONFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.trace"
    assert payload["message"] == "trace"
    assert payload["trace_id"] == "trace-1"
    assert payload["trace_type"] == "query"
    assert payload["timestamp"]


def test_get_trace_logger_writes_json_lines_without_duplicate_handlers(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    logger = get_trace_logger(path, name="test.trace.logger")
    same_logger = get_trace_logger(path, name="test.trace.logger")

    assert same_logger is logger
    assert len([handler for handler in logger.handlers if hasattr(handler, "_trace_log_path")]) == 1

    logger.info("trace", extra={"trace": {"trace_id": "trace-1", "trace_type": "query"}})
    for handler in logger.handlers:
        handler.flush()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["trace_type"] == "query"


def test_write_trace_appends_legal_jsonl_with_trace_type(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"

    write_trace(
        {
            "trace_id": "trace-1",
            "trace_type": "ingestion",
            "started_at": "2026-05-11T00:00:00+00:00",
            "finished_at": "2026-05-11T00:00:01+00:00",
            "total_elapsed_ms": 1000.0,
            "stages": [],
        },
        path,
    )
    write_trace({"trace_id": "trace-2", "trace_type": "query", "stages": []}, path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["trace_type"] == "ingestion"
    assert second["trace_type"] == "query"
    assert first["trace_id"] == "trace-1"


def test_write_trace_rejects_non_mapping(tmp_path) -> None:
    with pytest.raises(ValueError, match="trace_dict"):
        write_trace(["bad"], tmp_path / "traces.jsonl")  # type: ignore[arg-type]
