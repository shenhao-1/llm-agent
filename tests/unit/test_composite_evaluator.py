"""Unit tests for composite evaluator orchestration."""

from __future__ import annotations

from typing import Any

import pytest

from core.trace import TraceContext
from libs.evaluator import BaseEvaluator, CompositeEvaluator, EvaluatorFactory


class FakeEvaluator(BaseEvaluator):
    def __init__(
        self,
        name: str,
        metrics: dict[str, float],
        calls: list[str] | None = None,
    ) -> None:
        super().__init__({"provider": name})
        self.metrics = dict(metrics)
        self.calls = calls if calls is not None else []

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: Any | None = None,
    ) -> dict[str, float]:
        self.validate_query(query)
        self.validate_id_list(retrieved_ids, "retrieved_ids")
        self.validate_id_list(golden_ids, "golden_ids")
        self.calls.append(f"{self.config['provider']}:{query}")
        return dict(self.metrics)


class BadMetricsEvaluator(BaseEvaluator):
    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: Any | None = None,
    ) -> dict[str, float]:
        return {"bad": "value"}  # type: ignore[return-value]


@pytest.fixture(autouse=True)
def reset_factory_registry() -> Any:
    original = dict(EvaluatorFactory._providers)
    EvaluatorFactory._providers = dict(original)
    yield
    EvaluatorFactory._providers = original


def test_composite_evaluator_runs_all_and_merges_prefixed_metrics() -> None:
    calls: list[str] = []
    evaluator = CompositeEvaluator(
        [
            FakeEvaluator("alpha", {"score": 0.7}, calls),
            FakeEvaluator("beta", {"score": 0.9, "mrr": 0.5}, calls),
        ]
    )
    trace = TraceContext(trace_type="query")

    metrics = evaluator.evaluate(
        query="find docs",
        retrieved_ids=["doc-1", "doc-2"],
        golden_ids=["doc-2"],
        trace=trace,
    )

    assert metrics == {
        "alpha.score": 0.7,
        "beta.mrr": 0.5,
        "beta.score": 0.9,
    }
    assert sorted(calls) == ["alpha:find docs", "beta:find docs"]
    assert trace.stages[-1]["name"] == "evaluation"
    assert trace.stages[-1]["method"] == "composite"
    assert trace.stages[-1]["details"]["evaluator_count"] == 2


def test_composite_evaluator_can_merge_without_prefix_when_keys_are_unique() -> None:
    evaluator = CompositeEvaluator(
        [
            FakeEvaluator("alpha", {"hit_rate": 1.0}),
            FakeEvaluator("beta", {"faithfulness": 0.8}),
        ],
        {"prefix_metrics": False},
    )

    assert evaluator.evaluate("query", ["doc-1"], ["doc-1"]) == {
        "faithfulness": 0.8,
        "hit_rate": 1.0,
    }


def test_composite_evaluator_rejects_invalid_inputs_and_metrics() -> None:
    with pytest.raises(ValueError, match="non-empty list"):
        CompositeEvaluator([])
    with pytest.raises(ValueError, match="BaseEvaluator"):
        CompositeEvaluator(["bad"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="max_workers"):
        CompositeEvaluator([FakeEvaluator("alpha", {"score": 1.0})], {"max_workers": 0})
    with pytest.raises(ValueError, match="must be numeric"):
        CompositeEvaluator([BadMetricsEvaluator()]).evaluate(
            "query",
            ["doc-1"],
            ["doc-1"],
        )


def test_composite_evaluator_rejects_duplicate_unprefixed_metrics() -> None:
    evaluator = CompositeEvaluator(
        [
            FakeEvaluator("alpha", {"score": 0.7}),
            FakeEvaluator("beta", {"score": 0.9}),
        ],
        {"prefix_metrics": False},
    )

    with pytest.raises(ValueError, match="duplicate metric key"):
        evaluator.evaluate("query", ["doc-1"], ["doc-1"])


def test_factory_builds_composite_from_backend_names() -> None:
    EvaluatorFactory.register_provider("alpha", _provider("alpha", {"score": 0.7}))
    EvaluatorFactory.register_provider("beta", _provider("beta", {"quality": 0.9}))

    evaluator = EvaluatorFactory.create(
        {"evaluation": {"backends": ["alpha", "beta"], "prefix_metrics": True}}
    )

    assert isinstance(evaluator, CompositeEvaluator)
    assert evaluator.evaluate("query", ["doc-1"], ["doc-1"]) == {
        "alpha.score": 0.7,
        "beta.quality": 0.9,
    }


def test_factory_builds_composite_from_backend_mappings() -> None:
    EvaluatorFactory.register_provider("alpha", _provider("alpha", {"score": 0.7}))
    evaluator = EvaluatorFactory.create(
        {
            "evaluation": {
                "backends": [{"provider": "alpha", "weight": 2}],
                "prefix_metrics": True,
            }
        }
    )

    assert isinstance(evaluator, CompositeEvaluator)
    assert evaluator.evaluators[0].config["weight"] == 2


def test_factory_rejects_invalid_backends_config() -> None:
    with pytest.raises(ValueError, match="non-empty list"):
        EvaluatorFactory.create({"evaluation": {"backends": []}})
    with pytest.raises(ValueError, match="provider"):
        EvaluatorFactory.create({"evaluation": {"backends": [{}]}})
    with pytest.raises(ValueError, match="provider name or mapping"):
        EvaluatorFactory.create({"evaluation": {"backends": [123]}})


def _provider(name: str, metrics: dict[str, float]) -> type[FakeEvaluator]:
    class Provider(FakeEvaluator):
        def __init__(self, config=None) -> None:
            super().__init__(name, metrics)
            if isinstance(config, dict):
                self.config.update(config)
            self.config["provider"] = name

    return Provider
