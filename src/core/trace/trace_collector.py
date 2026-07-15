"""Trace collection and optional JSONL persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .trace_context import TraceContext


class TraceCollector:
    """Collect finished trace snapshots and optionally append them to JSONL."""

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self.persist_path = Path(persist_path) if persist_path is not None else None
        self.traces: list[dict[str, Any]] = []

    def collect(self, trace: TraceContext) -> None:
        if not isinstance(trace, TraceContext):
            raise ValueError("trace must be a TraceContext")
        trace.finish()
        payload = trace.to_dict()
        self.traces.append(payload)
        if self.persist_path is not None:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with self.persist_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
                handle.write("\n")


__all__ = ["TraceCollector"]
