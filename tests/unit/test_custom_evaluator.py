"""Unit tests for custom evaluator and factory routing."""

from __future__ import annotations

from typing import Any

import pytest

from core.settings import Settings
from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator
from libs.evaluator.evaluator_factory import EvaluatorFactory


class FakeEvaluator(BaseEvaluator):
    """Simple fake evaluator used for routing tests."""

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
        return {"score": 1.0}


@pytest.fixture(autouse=True)
def reset_factory_registry() -> Any:
    """Keep factory registry isolated between tests while preserving custom fallback."""
    original = dict(EvaluatorFactory._providers)
    EvaluatorFactory._providers = {"custom": CustomEvaluator}
    yield
    EvaluatorFactory._providers = original


@pytest.mark.unit
def test_create_returns_custom_evaluator_for_custom_provider() -> None:
    settings = {"evaluation": {"provider": "custom"}}

    instance = EvaluatorFactory.create(settings)

    assert isinstance(instance, CustomEvaluator)


@pytest.mark.unit
def test_create_routes_to_registered_provider_with_settings_dataclass() -> None:
    EvaluatorFactory.register_provider("fake", FakeEvaluator)
    settings = Settings(
        llm={"provider": "dummy"},
        embedding={"provider": "dummy"},
        vector_store={"provider": "dummy"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "fake"},
        observability={"level": "INFO"},
    )

    instance = EvaluatorFactory.create(settings)

    assert isinstance(instance, FakeEvaluator)


@pytest.mark.unit
def test_create_raises_on_unknown_provider() -> None:
    settings = {"evaluation": {"provider": "not-registered"}}

    with pytest.raises(ValueError, match="Unknown evaluator provider"):
        EvaluatorFactory.create(settings)


@pytest.mark.unit
def test_custom_evaluator_returns_hit_rate_and_mrr_metrics() -> None:
    evaluator = CustomEvaluator()

    metrics = evaluator.evaluate(
        query="find vector store docs",
        retrieved_ids=["doc-3", "doc-1", "doc-2"],
        golden_ids=["doc-1", "doc-9"],
    )

    assert metrics["hit_rate"] == 1.0
    assert metrics["mrr"] == 0.5
    assert metrics["precision_at_k"] == pytest.approx(1 / 3)
    assert metrics["recall_at_k"] == 0.5


@pytest.mark.unit
def test_custom_evaluator_returns_zero_when_no_match_found() -> None:
    evaluator = CustomEvaluator()

    metrics = evaluator.evaluate(
        query="find reranker docs",
        retrieved_ids=["doc-3", "doc-4"],
        golden_ids=["doc-1", "doc-2"],
    )

    assert metrics["hit_rate"] == 0.0
    assert metrics["mrr"] == 0.0
    assert metrics["precision_at_k"] == 0.0
    assert metrics["recall_at_k"] == 0.0


@pytest.mark.unit
def test_custom_evaluator_handles_empty_retrieved_ids() -> None:
    metrics = CustomEvaluator().evaluate(
        query="find evaluator docs",
        retrieved_ids=[],
        golden_ids=["doc-1"],
    )

    assert metrics == {
        "hit_rate": 0.0,
        "mrr": 0.0,
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
    }


@pytest.mark.unit
def test_custom_evaluator_deduplicates_matches_for_precision_and_recall() -> None:
    metrics = CustomEvaluator().evaluate(
        query="find evaluator docs",
        retrieved_ids=["doc-1", "doc-1", "doc-2"],
        golden_ids=["doc-1", "doc-2"],
    )

    assert metrics["hit_rate"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["precision_at_k"] == pytest.approx(2 / 3)
    assert metrics["recall_at_k"] == 1.0


@pytest.mark.unit
def test_custom_evaluator_rejects_empty_golden_ids() -> None:
    with pytest.raises(ValueError, match="golden_ids must be a non-empty list"):
        CustomEvaluator().evaluate(
            query="find evaluator docs",
            retrieved_ids=["doc-1"],
            golden_ids=[],
        )


@pytest.mark.unit
def test_custom_evaluator_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        CustomEvaluator().evaluate(
            query=" ",
            retrieved_ids=["doc-1"],
            golden_ids=["doc-1"],
        )


@pytest.mark.unit
def test_custom_evaluator_rejects_invalid_retrieved_id() -> None:
    with pytest.raises(ValueError, match=r"retrieved_ids\[1\]"):
        CustomEvaluator().evaluate(
            query="find evaluator docs",
            retrieved_ids=["doc-1", ""],
            golden_ids=["doc-1"],
        )


@pytest.mark.unit
def test_custom_evaluator_rejects_non_list_golden_ids() -> None:
    with pytest.raises(ValueError, match="golden_ids must be a list"):
        CustomEvaluator().evaluate(
            query="find evaluator docs",
            retrieved_ids=["doc-1"],
            golden_ids="doc-1",  # type: ignore[arg-type]
        )
