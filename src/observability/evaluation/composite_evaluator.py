"""Composite evaluator that runs multiple evaluator backends."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Mapping, TYPE_CHECKING

from libs.evaluator.base_evaluator import BaseEvaluator

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class CompositeEvaluator(BaseEvaluator):
    """Run multiple evaluators and merge their metrics into one payload."""

    def __init__(
        self,
        evaluators: list[BaseEvaluator],
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        self.evaluators = _validate_evaluators(evaluators)
        self.prefix_metrics = bool(self.config.get("prefix_metrics", True))
        max_workers = self.config.get("max_workers", len(self.evaluators))
        if not isinstance(max_workers, int) or max_workers <= 0:
            raise ValueError("max_workers must be a positive integer")
        self.max_workers = min(max_workers, len(self.evaluators))

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: "TraceContext | None" = None,
    ) -> dict[str, float]:
        self.validate_query(query)
        self.validate_id_list(retrieved_ids, "retrieved_ids")
        self.validate_id_list(golden_ids, "golden_ids")

        results: list[tuple[str, dict[str, float]]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    evaluator.evaluate,
                    query,
                    list(retrieved_ids),
                    list(golden_ids),
                    trace,
                ): evaluator
                for evaluator in self.evaluators
            }
            for future in as_completed(futures):
                evaluator = futures[future]
                metrics = future.result()
                _validate_metrics(metrics, evaluator)
                results.append((_evaluator_key(evaluator), dict(metrics)))

        merged: dict[str, float] = {}
        for key, metrics in sorted(results, key=lambda item: item[0]):
            for metric_name, value in metrics.items():
                output_key = f"{key}.{metric_name}" if self.prefix_metrics else metric_name
                if output_key in merged:
                    raise ValueError(f"duplicate metric key: {output_key}")
                merged[output_key] = float(value)

        if trace is not None:
            trace.record_stage(
                "evaluation",
                method="composite",
                details={
                    "evaluator_count": len(self.evaluators),
                    "metric_count": len(merged),
                    "evaluators": [_evaluator_key(evaluator) for evaluator in self.evaluators],
                },
            )
        return merged


def _validate_evaluators(evaluators: Any) -> list[BaseEvaluator]:
    if not isinstance(evaluators, list) or not evaluators:
        raise ValueError("evaluators must be a non-empty list")
    for idx, evaluator in enumerate(evaluators):
        if not isinstance(evaluator, BaseEvaluator):
            raise ValueError(f"evaluators[{idx}] must be a BaseEvaluator")
    return list(evaluators)


def _validate_metrics(metrics: Any, evaluator: BaseEvaluator) -> None:
    if not isinstance(metrics, Mapping) or not metrics:
        raise ValueError(f"{_evaluator_key(evaluator)} returned no metrics")
    for name, value in metrics.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{_evaluator_key(evaluator)} returned invalid metric name")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{_evaluator_key(evaluator)} metric {name} must be numeric")


def _evaluator_key(evaluator: BaseEvaluator) -> str:
    provider = evaluator.config.get("provider")
    if isinstance(provider, str) and provider.strip():
        return provider.strip().lower()
    name = evaluator.__class__.__name__
    if name.endswith("Evaluator"):
        name = name[: -len("Evaluator")]
    return name.lower() or "evaluator"


__all__ = ["CompositeEvaluator"]
