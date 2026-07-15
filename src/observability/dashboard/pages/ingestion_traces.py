"""Dashboard page for inspecting ingestion trace history."""

from __future__ import annotations

from typing import Any

from observability.dashboard.services import TraceService


def render() -> None:
    import streamlit as st

    st.title("Ingestion Traces")
    try:
        service = TraceService()
        summaries = service.list_ingestion_summaries()
    except Exception as exc:
        st.error(f"Unable to read ingestion traces: {exc}")
        return

    if not summaries:
        st.info("No ingestion traces found in logs/traces.jsonl.")
        return

    st.markdown("### History")
    st.dataframe(
        summaries,
        use_container_width=True,
        hide_index=True,
        column_order=[
            "started_at",
            "filename",
            "collection",
            "status",
            "total_elapsed_ms",
            "chunk_count",
            "image_count",
            "skipped",
        ],
    )

    options = {
        _summary_label(summary): str(summary["trace_id"])
        for summary in summaries
    }
    selected_label = st.selectbox("Trace", list(options))
    trace = service.get_trace(options[selected_label])
    _render_trace_detail(st, service, trace)


def _render_trace_detail(st: Any, service: TraceService, trace: dict[str, Any]) -> None:
    summary = service.summarize_trace(trace)
    stats = service.processing_stats(trace)
    durations = service.stage_durations(trace)

    st.markdown("### Trace Detail")
    left, middle, right, status_col = st.columns(4)
    left.metric("Chunks", stats["chunks"])
    middle.metric("Images", stats["images"])
    right.metric("Skipped", stats["skipped"])
    status_col.metric("Failed", stats["failed"])

    st.caption(
        f"{summary.get('filename') or summary['trace_id']} | "
        f"{summary.get('collection') or 'unknown collection'} | "
        f"{summary['status']} | {summary['total_elapsed_ms']:.2f} ms"
    )

    st.markdown("### Stage Duration")
    if durations:
        st.vega_lite_chart(
            durations,
            {
                "mark": "bar",
                "encoding": {
                    "y": {"field": "stage", "type": "nominal", "sort": None},
                    "x": {"field": "elapsed_ms", "type": "quantitative"},
                    "tooltip": [
                        {"field": "stage", "type": "nominal"},
                        {"field": "elapsed_ms", "type": "quantitative"},
                        {"field": "method", "type": "nominal"},
                    ],
                },
            },
            use_container_width=True,
        )
    else:
        st.caption("No stage timing records found.")

    st.markdown("### Stage Details")
    for index, stage in enumerate(trace.get("stages", []), start=1):
        if not isinstance(stage, dict):
            continue
        name = str(stage.get("name", f"stage-{index}"))
        elapsed = float(stage.get("elapsed_ms", 0.0) or 0.0)
        label = f"{index}. {name} ({elapsed:.2f} ms)"
        with st.expander(label):
            st.json(
                {
                    "method": stage.get("method"),
                    "started_at": stage.get("started_at"),
                    "finished_at": stage.get("finished_at"),
                    "elapsed_ms": elapsed,
                    "details": stage.get("details", {}),
                }
            )


def _summary_label(summary: dict[str, Any]) -> str:
    filename = summary.get("filename") or "unknown file"
    collection = summary.get("collection") or "unknown collection"
    status = summary.get("status") or "unknown"
    started_at = summary.get("started_at") or "unknown time"
    return f"{started_at} | {filename} | {collection} | {status}"


if __name__ == "__main__":
    render()
