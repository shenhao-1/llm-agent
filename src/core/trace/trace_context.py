"""Trace context used by ingestion and query pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import time
from typing import Any
from uuid import uuid4


VALID_TRACE_TYPES = {"query", "ingestion"}


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class TraceContext:
    """Collect stage records for a single pipeline execution."""

    trace_type: str = "query"
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    stages: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = field(default_factory=_utc_now_iso)
    finished_at: str | None = None
    total_elapsed_ms: float | None = None
    _start_perf: float = field(default_factory=time.perf_counter, init=False, repr=False)
    _last_stage_perf: float = field(default_factory=time.perf_counter, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.trace_type not in VALID_TRACE_TYPES:
            raise ValueError("trace_type must be 'query' or 'ingestion'")
        if not isinstance(self.trace_id, str) or not self.trace_id.strip():
            raise ValueError("trace_id must be a non-empty string")

    def record_stage(
        self,
        name: str,
        method: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Append a stage record using the stable trace shape."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("stage name must be a non-empty string")

        now = time.perf_counter()
        record: dict[str, Any] = {
            "name": name.strip(),
            "started_at": _utc_now_iso(),
            "finished_at": _utc_now_iso(),
            "elapsed_ms": max((now - self._last_stage_perf) * 1000, 0.0),
        }
        self._last_stage_perf = now
        if method is not None:
            record["method"] = method
        if details is not None:
            record["details"] = dict(details)
        self.stages.append(record)

    def finish(self) -> None:
        """Mark the trace as finished and freeze total elapsed time."""
        if self.finished_at is not None:
            return
        self.finished_at = _utc_now_iso()
        self.total_elapsed_ms = max((time.perf_counter() - self._start_perf) * 1000, 0.0)

    def elapsed_ms(self, stage_name: str | None = None) -> float:
        """Return total elapsed time or the first matching stage elapsed time."""
        if stage_name is None:
            if self.total_elapsed_ms is not None:
                return float(self.total_elapsed_ms)
            return max((time.perf_counter() - self._start_perf) * 1000, 0.0)
        if not isinstance(stage_name, str) or not stage_name.strip():
            raise ValueError("stage_name must be a non-empty string when provided")
        for stage in self.stages:
            if stage.get("name") == stage_name:
                return float(stage.get("elapsed_ms", 0.0))
        raise KeyError(f"stage not found: {stage_name}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable trace snapshot."""
        total_elapsed = (
            float(self.total_elapsed_ms)
            if self.total_elapsed_ms is not None
            else self.elapsed_ms()
        )
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_elapsed_ms": total_elapsed,
            "stages": [dict(stage) for stage in self.stages],
        }

__all__ = ["TraceContext", "VALID_TRACE_TYPES"]
