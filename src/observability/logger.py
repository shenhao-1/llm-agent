"""Logging helpers for human-readable logs and trace JSON Lines."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
from pathlib import Path
from typing import Any, Mapping


DEFAULT_TRACE_LOG_PATH = Path("logs/traces.jsonl")


def get_logger(name: str = "modular_rag", level: int = logging.INFO) -> logging.Logger:
    """Return a stderr logger with a stable format and without duplicate handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)

    logger.propagate = False
    return logger


class JSONFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        trace = getattr(record, "trace", None)
        if isinstance(trace, Mapping):
            payload.update(dict(trace))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def get_trace_logger(
    log_path: str | Path = DEFAULT_TRACE_LOG_PATH,
    *,
    name: str = "modular_rag.trace",
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a JSONL file logger for trace records."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    resolved = str(path.resolve())
    for handler in logger.handlers:
        if getattr(handler, "_trace_log_path", None) == resolved:
            handler.setLevel(level)
            return logger

    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(JSONFormatter())
    setattr(handler, "_trace_log_path", resolved)
    logger.addHandler(handler)
    return logger


def write_trace(
    trace_dict: Mapping[str, Any],
    log_path: str | Path = DEFAULT_TRACE_LOG_PATH,
) -> None:
    """Append one trace dictionary to the trace JSONL log."""
    if not isinstance(trace_dict, Mapping):
        raise ValueError("trace_dict must be a mapping")
    logger = get_trace_logger(log_path)
    logger.info("trace", extra={"trace": dict(trace_dict)})


__all__ = [
    "DEFAULT_TRACE_LOG_PATH",
    "JSONFormatter",
    "get_logger",
    "get_trace_logger",
    "write_trace",
]
