"""Trace log reader used by dashboard trace pages."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Mapping

from observability.logger import DEFAULT_TRACE_LOG_PATH


class TraceService:
    """Parse traces.jsonl and expose dashboard-friendly trace payloads."""

    def __init__(self, log_path: str | Path = DEFAULT_TRACE_LOG_PATH) -> None:
        self.log_path = Path(log_path)

    def list_traces(
        self,
        trace_type: str | None = None,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return traces sorted newest first."""
        if trace_type is not None and trace_type not in {"query", "ingestion"}:
            raise ValueError("trace_type must be 'query' or 'ingestion'")
        if limit is not None and (not isinstance(limit, int) or limit <= 0):
            raise ValueError("limit must be a positive integer when provided")

        traces = [
            trace
            for trace in self._read_traces()
            if trace_type is None or trace.get("trace_type") == trace_type
        ]
        traces.sort(key=_sort_key, reverse=True)
        return traces[:limit] if limit is not None else traces

    def get_trace(self, trace_id: str) -> dict[str, Any]:
        """Return one trace by id."""
        if not isinstance(trace_id, str) or not trace_id.strip():
            raise ValueError("trace_id must be a non-empty string")
        target = trace_id.strip()
        for trace in self._read_traces():
            if trace.get("trace_id") == target:
                return trace
        raise ValueError(f"trace not found: {target}")

    def list_ingestion_summaries(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Return ingestion trace summaries for history tables."""
        return [
            self.summarize_trace(trace)
            for trace in self.list_traces("ingestion", limit=limit)
        ]

    def list_query_summaries(
        self,
        *,
        keyword: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return query trace summaries, optionally filtered by query text."""
        summaries = [
            self.summarize_query_trace(trace)
            for trace in self.list_traces("query", limit=limit)
        ]
        if keyword is None or not keyword.strip():
            return summaries
        needle = keyword.strip().lower()
        return [
            summary
            for summary in summaries
            if needle in str(summary.get("query", "")).lower()
        ]

    def summarize_trace(self, trace: Mapping[str, Any]) -> dict[str, Any]:
        """Extract stable table fields from a trace payload."""
        _validate_trace(trace)
        source_path = _first_string(
            trace.get("source_path"),
            _stage_detail(trace, "load", "source_path"),
            _stage_detail(trace, "integrity", "source_path"),
        )
        collection = _first_string(
            trace.get("collection"),
            _stage_detail(trace, "load", "collection"),
            _stage_detail(trace, "integrity", "collection"),
        )
        return {
            "trace_id": str(trace["trace_id"]),
            "trace_type": str(trace["trace_type"]),
            "started_at": trace.get("started_at"),
            "finished_at": trace.get("finished_at"),
            "total_elapsed_ms": _float_or_zero(trace.get("total_elapsed_ms")),
            "status": _trace_status(trace),
            "source_path": source_path,
            "filename": Path(source_path).name if source_path else None,
            "collection": collection,
            "chunk_count": _first_int(
                trace.get("chunk_count"),
                _stage_detail(trace, "split", "chunk_count"),
                _stage_detail(trace, "transform", "chunk_count"),
            ),
            "image_count": _first_int(
                trace.get("image_count"),
                _stage_detail(trace, "store_images", "image_count"),
                _stage_detail(trace, "load", "image_count"),
            ),
            "skipped": bool(trace.get("skipped", False)),
            "stage_count": len(trace.get("stages", [])),
        }

    def stage_durations(self, trace: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Return stage timing rows for horizontal bar charts."""
        _validate_trace(trace)
        rows: list[dict[str, Any]] = []
        for stage in trace.get("stages", []):
            if not isinstance(stage, Mapping):
                continue
            name = stage.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            rows.append(
                {
                    "stage": name,
                    "elapsed_ms": _float_or_zero(stage.get("elapsed_ms")),
                    "method": stage.get("method"),
                }
            )
        return rows

    def processing_stats(self, trace: Mapping[str, Any]) -> dict[str, Any]:
        """Return aggregate counters shown on the trace detail page."""
        summary = self.summarize_trace(trace)
        return {
            "chunks": summary["chunk_count"],
            "images": summary["image_count"],
            "skipped": 1 if summary["skipped"] else 0,
            "failed": 1 if summary["status"] == "failed" else 0,
        }

    def summarize_query_trace(self, trace: Mapping[str, Any]) -> dict[str, Any]:
        """Extract stable Query trace fields for history tables."""
        _validate_trace(trace)
        query = _first_string(
            trace.get("query"),
            _stage_detail(trace, "query_processing", "query"),
            _stage_detail(trace, "query_processing", "original_query"),
        )
        collection = _first_string(
            trace.get("collection"),
            _stage_detail(trace, "query_processing", "collection"),
        )
        return {
            "trace_id": str(trace["trace_id"]),
            "trace_type": str(trace["trace_type"]),
            "started_at": trace.get("started_at"),
            "finished_at": trace.get("finished_at"),
            "total_elapsed_ms": _float_or_zero(trace.get("total_elapsed_ms")),
            "status": _trace_status(trace),
            "query": query,
            "collection": collection,
            "top_k": _first_int(
                trace.get("top_k"),
                _stage_detail(trace, "dense_retrieval", "top_k"),
                _stage_detail(trace, "sparse_retrieval", "top_k"),
                _stage_detail(trace, "hybrid_search", "top_k"),
            ),
            "result_count": _first_int(
                trace.get("result_count"),
                _stage_detail(trace, "rerank", "result_count"),
                _stage_detail(trace, "hybrid_search", "result_count"),
                _stage_detail(trace, "fusion", "result_count"),
            ),
            "dense_count": _first_int(
                _stage_detail(trace, "dense_retrieval", "result_count"),
                _stage_detail(trace, "fusion", "dense_count"),
            ),
            "sparse_count": _first_int(
                _stage_detail(trace, "sparse_retrieval", "result_count"),
                _stage_detail(trace, "fusion", "sparse_count"),
            ),
            "rerank_fallback": bool(_stage_detail(trace, "rerank", "fallback")),
            "stage_count": len(trace.get("stages", [])),
        }

    def retrieval_comparison(self, trace: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Return Dense/Sparse/Fusion/Rerank rows for comparison tables."""
        _validate_trace(trace)
        rows: list[dict[str, Any]] = []
        stage_map = {stage.get("name"): stage for stage in trace.get("stages", []) if isinstance(stage, Mapping)}
        for stage_name, label in (
            ("dense_retrieval", "Dense"),
            ("sparse_retrieval", "Sparse"),
            ("fusion", "Fusion"),
            ("rerank", "Rerank"),
        ):
            stage = stage_map.get(stage_name)
            if not isinstance(stage, Mapping):
                continue
            details = stage.get("details")
            if not isinstance(details, Mapping):
                details = {}
            rows.append(
                {
                    "route": label,
                    "stage": stage_name,
                    "method": stage.get("method"),
                    "top_k": _first_int(details.get("top_k")),
                    "result_count": _first_int(
                        details.get("result_count"),
                        details.get("dense_count") if stage_name == "dense_retrieval" else None,
                        details.get("sparse_count") if stage_name == "sparse_retrieval" else None,
                    ),
                    "elapsed_ms": _float_or_zero(stage.get("elapsed_ms")),
                    "fallback": bool(details.get("fallback", False)),
                    "error": details.get("error"),
                }
            )
        return rows

    def rerank_changes(self, trace: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Return available rerank change information.

        Current core traces expose rerank count/fallback metadata. If future traces
        add before/after ranking samples, this method converts them into rows too.
        """
        _validate_trace(trace)
        rerank_stage = _stage(trace, "rerank")
        if rerank_stage is None:
            return []
        details = rerank_stage.get("details")
        if not isinstance(details, Mapping):
            details = {}

        before = details.get("before")
        after = details.get("after")
        if isinstance(before, list) and isinstance(after, list):
            rows: list[dict[str, Any]] = []
            after_positions = {str(chunk_id): index for index, chunk_id in enumerate(after, start=1)}
            for before_rank, chunk_id in enumerate(before, start=1):
                key = str(chunk_id)
                after_rank = after_positions.get(key)
                if after_rank is None:
                    continue
                rows.append(
                    {
                        "chunk_id": key,
                        "before_rank": before_rank,
                        "after_rank": after_rank,
                        "delta": before_rank - after_rank,
                    }
                )
            return rows

        return [
            {
                "chunk_id": None,
                "before_rank": None,
                "after_rank": None,
                "delta": None,
                "result_count": _first_int(details.get("result_count")),
                "fallback": bool(details.get("fallback", False)),
                "error": details.get("error"),
            }
        ]

    def _read_traces(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        traces: list[dict[str, Any]] = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, Mapping) and _is_trace_payload(payload):
                traces.append(dict(payload))
        return traces


def _is_trace_payload(payload: Mapping[str, Any]) -> bool:
    return (
        isinstance(payload.get("trace_id"), str)
        and payload.get("trace_type") in {"query", "ingestion"}
        and isinstance(payload.get("stages", []), list)
    )


def _validate_trace(trace: Mapping[str, Any]) -> None:
    if not isinstance(trace, Mapping) or not _is_trace_payload(trace):
        raise ValueError("trace must be a trace payload mapping")


def _trace_status(trace: Mapping[str, Any]) -> str:
    status = trace.get("status")
    if isinstance(status, str) and status.strip():
        return status.strip()
    if trace.get("error") or _stage_error(trace):
        return "failed"
    if trace.get("finished_at") is None:
        return "running"
    return "success"


def _stage_error(trace: Mapping[str, Any]) -> bool:
    for stage in trace.get("stages", []):
        if not isinstance(stage, Mapping):
            continue
        details = stage.get("details")
        if isinstance(details, Mapping) and (details.get("error") or details.get("error_msg")):
            return True
    return False


def _stage_detail(trace: Mapping[str, Any], stage_name: str, key: str) -> Any:
    stage = _stage(trace, stage_name)
    if stage is not None:
        details = stage.get("details")
        if isinstance(details, Mapping) and key in details:
            return details[key]
    return None


def _stage(trace: Mapping[str, Any], stage_name: str) -> Mapping[str, Any] | None:
    for stage in trace.get("stages", []):
        if isinstance(stage, Mapping) and stage.get("name") == stage_name:
            return stage
    return None


def _first_string(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _first_int(*values: Any) -> int:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
    return 0


def _float_or_zero(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _sort_key(trace: Mapping[str, Any]) -> str:
    for key in ("finished_at", "started_at", "timestamp"):
        value = trace.get(key)
        if isinstance(value, str):
            return value
    return ""


__all__ = ["TraceService"]
