"""Streamlit Dashboard smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.types import Chunk
from ingestion.embedding import SparseEncoder
from ingestion.storage import BM25Indexer


ROOT = Path(__file__).resolve().parents[2]
PAGES = [
    ("overview.py", "System Overview"),
    ("data_browser.py", "Data Browser"),
    ("ingestion_manager.py", "Ingestion Manager"),
    ("ingestion_traces.py", "Ingestion Traces"),
    ("query_traces.py", "Query Traces"),
    ("evaluation_panel.py", "Evaluation Panel"),
]


@pytest.mark.parametrize(("page_name", "expected_title"), PAGES)
def test_dashboard_page_loads_without_streamlit_errors(
    page_name: str,
    expected_title: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    _write_dashboard_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)

    page_path = ROOT / "src" / "observability" / "dashboard" / "pages" / page_name
    app = streamlit_testing.AppTest.from_file(str(page_path)).run(timeout=5)

    assert not app.exception
    assert not app.error
    assert app.title
    assert app.title[0].value == expected_title


def _write_dashboard_fixture(root: Path) -> None:
    _write_settings(root / "config" / "settings.yaml")
    _write_data_index(root / "data")
    _write_traces(root / "logs" / "traces.jsonl")
    _write_evaluation_history(root / "logs" / "evaluation_reports.jsonl")


def _write_settings(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "llm:",
                "  provider: openai",
                "embedding:",
                "  provider: openai",
                "  model: text-embedding-3-small",
                "  api_key: test-key",
                "vector_store:",
                "  provider: chroma",
                "  persist_directory: data/db/chroma",
                "retrieval:",
                "  strategy: hybrid",
                "  top_k: 5",
                "rerank:",
                "  provider: none",
                "evaluation:",
                "  provider: custom",
                "observability:",
                "  level: INFO",
                "  traces_path: logs/traces.jsonl",
            ]
        ),
        encoding="utf-8",
    )


def _write_data_index(data_dir: Path) -> None:
    chunk = Chunk(
        id="chunk-dashboard-smoke",
        text="Dashboard smoke test document with trace and evaluation data.",
        metadata={
            "source_path": "docs/dashboard_smoke.pdf",
            "collection": "docs",
            "chunk_index": 0,
            "doc_hash": "doc-dashboard-smoke",
        },
    )
    db_dir = data_dir / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    BM25Indexer(db_dir / "bm25").build(SparseEncoder().encode([chunk]))
    (db_dir / "vectors.json").write_text(
        json.dumps(
            [
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "vector": [1.0, 2.0, 3.0],
                    "metadata": dict(chunk.metadata),
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_traces(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    traces = [
        {
            "trace_id": "ingestion-smoke",
            "trace_type": "ingestion",
            "started_at": "2026-07-15T10:00:00+08:00",
            "finished_at": "2026-07-15T10:00:01+08:00",
            "total_elapsed_ms": 1000.0,
            "source_path": "docs/dashboard_smoke.pdf",
            "collection": "docs",
            "status": "success",
            "chunk_count": 1,
            "image_count": 0,
            "stages": [
                {
                    "name": "load",
                    "method": "smoke",
                    "elapsed_ms": 100.0,
                    "details": {
                        "source_path": "docs/dashboard_smoke.pdf",
                        "collection": "docs",
                    },
                },
                {
                    "name": "split",
                    "method": "smoke",
                    "elapsed_ms": 200.0,
                    "details": {"chunk_count": 1},
                },
            ],
        },
        {
            "trace_id": "query-smoke",
            "trace_type": "query",
            "started_at": "2026-07-15T10:01:00+08:00",
            "finished_at": "2026-07-15T10:01:01+08:00",
            "total_elapsed_ms": 500.0,
            "query": "dashboard smoke",
            "collection": "docs",
            "top_k": 3,
            "result_count": 1,
            "status": "success",
            "stages": [
                {
                    "name": "dense_retrieval",
                    "method": "smoke",
                    "elapsed_ms": 120.0,
                    "details": {"top_k": 3, "result_count": 1},
                },
                {
                    "name": "sparse_retrieval",
                    "method": "smoke",
                    "elapsed_ms": 80.0,
                    "details": {"top_k": 3, "result_count": 1},
                },
                {
                    "name": "rerank",
                    "method": "smoke",
                    "elapsed_ms": 50.0,
                    "details": {"fallback": False, "result_count": 1},
                },
            ],
        },
    ]
    path.write_text(
        "\n".join(json.dumps(trace, ensure_ascii=False) for trace in traces),
        encoding="utf-8",
    )


def _write_evaluation_history(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "created_at": "2026-07-15T02:00:00+00:00",
                "backend": "custom",
                "test_set": "tests/fixtures/golden_test_set.json",
                "metrics": {"hit_rate": 1.0, "mrr": 1.0},
                "results": [
                    {
                        "query": "dashboard smoke",
                        "retrieved_ids": ["chunk-dashboard-smoke"],
                        "expected_chunk_ids": ["chunk-dashboard-smoke"],
                        "metrics": {"hit_rate": 1.0, "mrr": 1.0},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
