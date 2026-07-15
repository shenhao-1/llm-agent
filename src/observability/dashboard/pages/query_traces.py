"""Dashboard page for inspecting query trace history."""

from __future__ import annotations

from typing import Any

from observability.dashboard.services import TraceService


def render() -> None:
    import streamlit as st

    st.title("Query Traces")
    keyword = st.text_input("Search query text", value="")

    try:
        service = TraceService()
        summaries = service.list_query_summaries(keyword=keyword)
    except Exception as exc:
        st.error(f"Unable to read query traces: {exc}")
        return

    if not summaries:
        st.info("No query traces found in logs/traces.jsonl.")
        return

    st.markdown("### History")
    st.dataframe(
        summaries,
        use_container_width=True,
        hide_index=True,
        column_order=[
            "started_at",
            "query",
            "collection",
            "status",
            "total_elapsed_ms",
            "top_k",
            "result_count",
            "dense_count",
            "sparse_count",
            "rerank_fallback",
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
    summary = service.summarize_query_trace(trace)
    durations = service.stage_durations(trace)
    comparison = service.retrieval_comparison(trace)
    rerank_changes = service.rerank_changes(trace)

    st.markdown("### Trace Detail")
    left, middle, right, rerank_col = st.columns(4)
    left.metric("Top K", summary["top_k"])
    middle.metric("Results", summary["result_count"])
    right.metric("Dense/Sparse", f"{summary['dense_count']} / {summary['sparse_count']}")
    rerank_col.metric("Rerank Fallback", "yes" if summary["rerank_fallback"] else "no")

    st.caption(
        f"{summary.get('query') or summary['trace_id']} | "
        f"{summary.get('collection') or 'all collections'} | "
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

    st.markdown("### Dense vs Sparse")
    if comparison:
        st.dataframe(comparison, use_container_width=True, hide_index=True)
    else:
        st.caption("No retrieval comparison details found.")

    st.markdown("### Rerank Changes")
    if rerank_changes:
        st.dataframe(rerank_changes, use_container_width=True, hide_index=True)
    else:
        st.caption("No rerank stage found.")

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
    query = summary.get("query") or "unknown query"
    status = summary.get("status") or "unknown"
    started_at = summary.get("started_at") or "unknown time"
    return f"{started_at} | {query} | {status}"


if __name__ == "__main__":
    render()
