"""Unit tests for the Ragas evaluator adapter."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from core.trace import TraceContext
from libs.evaluator import EvaluatorFactory, RagasEvaluator
from observability.evaluation.ragas_evaluator import DEFAULT_RAGAS_METRICS


class FakeRagasRunner:
    def __init__(self, metrics: Mapping[str, float] | None = None) -> None:
        self.metrics = dict(
            metrics
            or {
                "faithfulness": 0.8,
                "answer_relevancy": 0.7,
                "context_precision": 0.6,
            }
        )
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        payload: Mapping[str, Any],
        *,
        metrics: list[str],
        config: Mapping[str, Any],
        trace: Any | None = None,
    ) -> Mapping[str, Any]:
        self.calls.append(
            {
                "payload": dict(payload),
                "metrics": list(metrics),
                "config": dict(config),
            }
        )
        return self.metrics


def test_ragas_evaluator_uses_injected_runner_and_returns_metrics() -> None:
    runner = FakeRagasRunner()
    evaluator = RagasEvaluator(
        {
            "metrics": list(DEFAULT_RAGAS_METRICS),
            "contexts": {"doc-1": "Context text for doc 1"},
            "generated_answer": "Answer from the RAG pipeline.",
            "ground_truth": "Reference answer.",
        },
        runner=runner,
    )
    trace = TraceContext(trace_type="query")

    metrics = evaluator.evaluate(
        query="Where is doc 1?",
        retrieved_ids=["doc-1"],
        golden_ids=["doc-1"],
        trace=trace,
    )

    assert metrics == {
        "faithfulness": 0.8,
        "answer_relevancy": 0.7,
        "context_precision": 0.6,
    }
    assert runner.calls[0]["metrics"] == list(DEFAULT_RAGAS_METRICS)
    assert runner.calls[0]["payload"]["contexts"] == ["Context text for doc 1"]
    assert runner.calls[0]["payload"]["answer"] == "Answer from the RAG pipeline."
    assert runner.calls[0]["payload"]["ground_truth"] == "Reference answer."
    assert trace.stages[-1]["name"] == "evaluation"
    assert trace.stages[-1]["method"] == "ragas"


def test_ragas_evaluator_factory_routes_ragas_provider() -> None:
    evaluator = EvaluatorFactory.create(
        {"evaluation": {"provider": "ragas", "metrics": ["faithfulness"]}}
    )

    assert isinstance(evaluator, RagasEvaluator)
    assert evaluator.metrics == ("faithfulness",)


def test_ragas_evaluator_reports_missing_optional_dependency(monkeypatch) -> None:
    from observability.evaluation import ragas_evaluator as module

    def fail_import(metric_names: list[str]):
        raise ImportError("missing ragas")

    monkeypatch.setattr(module, "_import_ragas", fail_import)
    evaluator = RagasEvaluator()

    with pytest.raises(ImportError, match="requires optional dependencies"):
        evaluator.evaluate(
            query="find docs",
            retrieved_ids=["doc-1"],
            golden_ids=["doc-1"],
        )


def test_ragas_evaluator_validates_inputs_and_metric_config() -> None:
    with pytest.raises(ValueError, match="metrics"):
        RagasEvaluator({"metrics": "faithfulness"})
    with pytest.raises(ValueError, match="golden_ids"):
        RagasEvaluator(runner=FakeRagasRunner()).evaluate(
            query="find docs",
            retrieved_ids=["doc-1"],
            golden_ids=[],
        )
    with pytest.raises(ValueError, match="retrieved_ids"):
        RagasEvaluator(runner=FakeRagasRunner()).evaluate(
            query="find docs",
            retrieved_ids=[""],
            golden_ids=["doc-1"],
        )


def test_ragas_evaluator_rejects_non_numeric_metric() -> None:
    evaluator = RagasEvaluator(runner=FakeRagasRunner({"faithfulness": "bad"}))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="must be numeric"):
        evaluator.evaluate(
            query="find docs",
            retrieved_ids=["doc-1"],
            golden_ids=["doc-1"],
        )
