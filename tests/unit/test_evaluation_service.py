"""Unit tests for Dashboard evaluation service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.settings import Settings
from core.types import RetrievalResult
from observability.dashboard.services.evaluation_service import EvaluationService


class FakeSearch:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, top_k: int, filters=None, trace=None):
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        return [
            RetrievalResult(
                chunk_id="chunk-1",
                score=0.9,
                text="matched text",
                metadata={"source_path": "docs/a.pdf"},
            )
        ][:top_k]


class FakeEvaluator:
    def evaluate(self, query: str, retrieved_ids: list[str], golden_ids: list[str], trace=None):
        return {
            "hit_rate": 1.0 if set(retrieved_ids) & set(golden_ids) else 0.0,
            "mrr": 1.0,
        }


def test_run_evaluation_persists_report_history(tmp_path: Path) -> None:
    golden = _write_golden_set(tmp_path)
    history = tmp_path / "history.jsonl"
    fake_search = FakeSearch()
    captured_settings: list[dict[str, Any]] = []
    service = EvaluationService(
        settings_path=tmp_path / "settings.yaml",
        data_dir=tmp_path / "data",
        history_path=history,
        settings_loader=lambda path: _settings(),
        search_builder=lambda settings, data_dir: fake_search,
        evaluator_factory=lambda settings: _capture_evaluator(settings, captured_settings),
    )

    payload = service.run_evaluation(
        backend="custom",
        test_set_path=golden,
        top_k=3,
    )

    assert payload["backend"] == "custom"
    assert payload["test_set"] == str(golden)
    assert payload["metrics"] == {"hit_rate": 1.0, "mrr": 1.0}
    assert payload["results"][0]["retrieved_ids"] == ["chunk-1"]
    assert fake_search.calls == [
        {"query": "query one", "top_k": 3, "filters": {"collection": "docs"}}
    ]
    assert captured_settings[0]["evaluation"] == {"provider": "custom"}
    assert json.loads(history.read_text(encoding="utf-8").strip())["backend"] == "custom"


def test_run_evaluation_builds_composite_backend_config(tmp_path: Path) -> None:
    golden = _write_golden_set(tmp_path)
    captured_settings: list[dict[str, Any]] = []
    service = EvaluationService(
        history_path=tmp_path / "history.jsonl",
        settings_loader=lambda path: _settings(),
        search_builder=lambda settings, data_dir: FakeSearch(),
        evaluator_factory=lambda settings: _capture_evaluator(settings, captured_settings),
    )

    payload = service.run_evaluation(backend="all", test_set_path=golden)

    assert payload["backend"] == "all"
    assert captured_settings[0]["evaluation"] == {
        "backends": ["custom", "ragas"],
        "prefix_metrics": True,
    }


def test_history_rows_returns_latest_reports_first(tmp_path: Path) -> None:
    history = tmp_path / "history.jsonl"
    history.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "created_at": "2026-06-03T01:00:00+00:00",
                        "backend": "custom",
                        "test_set": "a.json",
                        "metrics": {"hit_rate": 0.5},
                        "results": [{}, {}],
                    }
                ),
                "not json",
                json.dumps(
                    {
                        "created_at": "2026-06-03T02:00:00+00:00",
                        "backend": "ragas",
                        "test_set": "b.json",
                        "metrics": {"hit_rate": 1.0, "mrr": 0.75},
                        "results": [{}],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    service = EvaluationService(history_path=history)

    rows = service.history_rows(limit=2)

    assert rows[0]["backend"] == "ragas"
    assert rows[0]["case_count"] == 1
    assert rows[0]["mrr"] == 0.75
    assert rows[1]["backend"] == "custom"


def test_run_evaluation_rejects_invalid_backend(tmp_path: Path) -> None:
    service = EvaluationService(history_path=tmp_path / "history.jsonl")

    try:
        service.run_evaluation(backend="unknown")
    except ValueError as exc:
        assert "backend must be one of" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def _capture_evaluator(settings: dict[str, Any], sink: list[dict[str, Any]]) -> FakeEvaluator:
    sink.append(settings)
    return FakeEvaluator()


def _write_golden_set(tmp_path: Path) -> Path:
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps(
            {
                "test_cases": [
                    {
                        "query": "query one",
                        "expected_chunk_ids": ["chunk-1"],
                        "collection": "docs",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def _settings() -> Settings:
    return Settings(
        llm={"provider": "openai"},
        embedding={"provider": "openai"},
        vector_store={"provider": "chroma"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"level": "info"},
    )
