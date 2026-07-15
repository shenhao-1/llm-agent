"""Unit tests for golden-set evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path

from core.types import RetrievalResult
from observability.evaluation import EvalReport, EvalRunner, load_golden_test_set


class FakeSearch:
    def __init__(self, results_by_query: dict[str, list[RetrievalResult]]) -> None:
        self.results_by_query = results_by_query
        self.calls: list[dict[str, object]] = []

    def search(self, query: str, top_k: int, filters=None, trace=None):
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        return self.results_by_query.get(query, [])[:top_k]


class FakeEvaluator:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def evaluate(self, query: str, retrieved_ids: list[str], golden_ids: list[str], trace=None):
        self.calls.append(
            {
                "query": query,
                "retrieved_ids": list(retrieved_ids),
                "golden_ids": list(golden_ids),
            }
        )
        hit = 1.0 if set(retrieved_ids) & set(golden_ids) else 0.0
        mrr = 0.0
        for index, chunk_id in enumerate(retrieved_ids, start=1):
            if chunk_id in set(golden_ids):
                mrr = 1.0 / index
                break
        return {"hit_rate": hit, "mrr": mrr}


def test_load_golden_test_set_accepts_mapping_payload(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps(
            {
                "test_cases": [
                    {
                        "query": "find docs",
                        "expected_chunk_ids": ["chunk-1"],
                        "expected_sources": ["docs/a.pdf"],
                        "collection": "docs",
                        "top_k": 3,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    cases = load_golden_test_set(path)

    assert len(cases) == 1
    assert cases[0].query == "find docs"
    assert cases[0].expected_chunk_ids == ["chunk-1"]
    assert cases[0].expected_sources == ["docs/a.pdf"]
    assert cases[0].collection == "docs"
    assert cases[0].top_k == 3


def test_eval_runner_runs_search_evaluator_and_aggregates_metrics(tmp_path: Path) -> None:
    test_set = tmp_path / "golden.json"
    test_set.write_text(
        json.dumps(
            {
                "test_cases": [
                    {
                        "query": "match first",
                        "expected_chunk_ids": ["chunk-1"],
                        "expected_sources": ["docs/one.pdf"],
                        "collection": "docs",
                    },
                    {
                        "query": "match second",
                        "expected_chunk_ids": ["chunk-3"],
                        "expected_sources": ["docs/three.pdf"],
                        "top_k": 2,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    search = FakeSearch(
        {
            "match first": [
                _result("chunk-1", "docs/one.pdf"),
                _result("chunk-2", "docs/two.pdf"),
            ],
            "match second": [
                _result("chunk-2", "docs/two.pdf"),
                _result("chunk-3", "docs/three.pdf"),
            ],
        }
    )
    evaluator = FakeEvaluator()
    runner = EvalRunner(
        {"retrieval": {"top_k": 5}},
        hybrid_search=search,
        evaluator=evaluator,  # type: ignore[arg-type]
    )

    report = runner.run(test_set)

    assert isinstance(report, EvalReport)
    assert report.metrics == {"hit_rate": 1.0, "mrr": 0.75}
    assert report.hit_rate == 1.0
    assert report.mrr == 0.75
    assert search.calls == [
        {"query": "match first", "top_k": 5, "filters": {"collection": "docs"}},
        {"query": "match second", "top_k": 2, "filters": None},
    ]
    assert evaluator.calls[0]["retrieved_ids"] == ["chunk-1", "chunk-2"]
    assert report.results[1].sources == ["docs/two.pdf", "docs/three.pdf"]
    assert report.to_dict()["results"][0]["query"] == "match first"


def test_load_golden_test_set_validates_shape(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"test_cases": []}), encoding="utf-8")

    try:
        load_golden_test_set(path)
    except ValueError as exc:
        assert "non-empty test_cases" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_repository_golden_test_set_is_valid() -> None:
    cases = load_golden_test_set("tests/fixtures/golden_test_set.json")

    assert len(cases) >= 2
    assert all(case.query for case in cases)


def _result(chunk_id: str, source_path: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=1.0,
        text=f"text for {chunk_id}",
        metadata={"source_path": source_path},
    )
