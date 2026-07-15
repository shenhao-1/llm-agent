"""Unit tests for Dashboard evaluation panel helpers."""

from __future__ import annotations

from observability.dashboard.pages.evaluation_panel import _detail_rows, render


def test_detail_rows_flattens_eval_results() -> None:
    rows = _detail_rows(
        [
            {
                "query": "find docs",
                "retrieved_ids": ["chunk-1", "chunk-2"],
                "expected_chunk_ids": ["chunk-2"],
                "metrics": {"hit_rate": 1.0, "mrr": 0.5},
            },
            "invalid",
        ]
    )

    assert rows == [
        {
            "query": "find docs",
            "retrieved_ids": "chunk-1, chunk-2",
            "expected_chunk_ids": "chunk-2",
            "hit_rate": 1.0,
            "mrr": 0.5,
        }
    ]


def test_render_entrypoint_is_importable() -> None:
    assert callable(render)
