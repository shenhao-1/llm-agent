"""Dashboard page for running and reviewing evaluation reports."""

from __future__ import annotations

from typing import Any

from observability.dashboard.services.evaluation_service import (
    DEFAULT_GOLDEN_TEST_SET,
    EvaluationService,
)


def render() -> None:
    import streamlit as st

    st.title("Evaluation Panel")
    try:
        service = EvaluationService()
    except Exception as exc:
        st.error(f"Unable to initialize evaluation service: {exc}")
        return

    backend = st.selectbox(
        "Evaluation Backend",
        service.available_backends(),
        format_func=lambda value: {
            "custom": "Custom",
            "ragas": "Ragas",
            "all": "All",
        }.get(value, value),
    )
    test_set_path = st.text_input("Golden test set", value=str(DEFAULT_GOLDEN_TEST_SET))
    top_k = st.number_input("Top K override", min_value=1, value=5, step=1)

    if st.button("Run evaluation", type="primary"):
        try:
            report = service.run_evaluation(
                backend=backend,
                test_set_path=test_set_path,
                top_k=int(top_k),
            )
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
        else:
            st.success("Evaluation completed.")
            _render_report(st, report)

    st.divider()
    _render_history(st, service)


def _render_report(st: Any, report: dict[str, Any]) -> None:
    metrics = report.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    st.markdown("### Metrics")
    metric_items = sorted(metrics.items())
    if metric_items:
        columns = st.columns(min(len(metric_items), 4))
        for index, (name, value) in enumerate(metric_items):
            columns[index % len(columns)].metric(name, f"{float(value):.4f}")
        st.bar_chart(
            [{"metric": key, "value": float(value)} for key, value in metric_items],
            x="metric",
            y="value",
        )
    else:
        st.info("No metrics returned.")

    st.markdown("### Query Details")
    results = report.get("results", [])
    if isinstance(results, list) and results:
        st.dataframe(_detail_rows(results), use_container_width=True, hide_index=True)
        for result in results:
            if isinstance(result, dict):
                with st.expander(str(result.get("query", "query"))):
                    st.json(result)
    else:
        st.info("No per-query details returned.")


def _render_history(st: Any, service: EvaluationService) -> None:
    st.markdown("### Evaluation History")
    try:
        rows = service.history_rows(limit=20)
    except Exception as exc:
        st.error(f"Unable to read evaluation history: {exc}")
        return

    if not rows:
        st.info("No evaluation history found yet.")
        return

    st.dataframe(rows, use_container_width=True, hide_index=True)
    metric_names = [
        key
        for key in rows[0]
        if key not in {"created_at", "backend", "test_set", "case_count"}
    ]
    if metric_names:
        selected_metric = st.selectbox("Trend metric", metric_names)
        st.line_chart(
            [
                {
                    "created_at": row.get("created_at"),
                    "value": float(row.get(selected_metric, 0.0) or 0.0),
                }
                for row in reversed(rows)
            ],
            x="created_at",
            y="value",
        )


def _detail_rows(results: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        metrics = result.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        row = {
            "query": result.get("query"),
            "retrieved_ids": ", ".join(map(str, result.get("retrieved_ids", []))),
            "expected_chunk_ids": ", ".join(
                map(str, result.get("expected_chunk_ids", []))
            ),
        }
        row.update({str(key): value for key, value in metrics.items()})
        rows.append(row)
    return rows


if __name__ == "__main__":
    render()
