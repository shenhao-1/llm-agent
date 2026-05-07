"""Minimal trace context used before Phase F observability expansion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class TraceContext:
    """Collect stage records for a single pipeline execution."""

    trace_type: str = "ingestion"
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    stages: list[dict[str, Any]] = field(default_factory=list)

    def record_stage(
        self,
        name: str,
        method: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Append a stage record using the stable trace shape."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("stage name must be a non-empty string")

        record: dict[str, Any] = {"name": name.strip()}
        if method is not None:
            record["method"] = method
        if details is not None:
            record["details"] = dict(details)
        self.stages.append(record)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable trace snapshot."""
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "stages": [dict(stage) for stage in self.stages],
        }
