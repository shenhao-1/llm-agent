"""Ragas-backed evaluator adapter."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, TYPE_CHECKING

from libs.evaluator.base_evaluator import BaseEvaluator

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


DEFAULT_RAGAS_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
)


class RagasRunner(Protocol):
    """Callable runner used to make Ragas evaluation injectable in tests."""

    def __call__(
        self,
        payload: Mapping[str, Any],
        *,
        metrics: list[str],
        config: Mapping[str, Any],
        trace: "TraceContext | None" = None,
    ) -> Mapping[str, Any]:
        ...


class RagasEvaluator(BaseEvaluator):
    """Wrap Ragas while preserving the project BaseEvaluator contract."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        runner: RagasRunner | None = None,
    ) -> None:
        super().__init__(config)
        self.metrics = _configured_metrics(self.config)
        self.runner = runner

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
        if not golden_ids:
            raise ValueError("golden_ids must be a non-empty list")

        payload = self._payload(query, retrieved_ids, golden_ids)
        if self.runner is not None:
            raw_metrics = self.runner(
                payload,
                metrics=list(self.metrics),
                config=dict(self.config),
                trace=trace,
            )
        else:
            raw_metrics = _run_ragas(payload, list(self.metrics), self.config)

        metrics = _normalize_metrics(raw_metrics, self.metrics)
        if trace is not None:
            trace.record_stage(
                "evaluation",
                method="ragas",
                details={
                    "metrics": list(metrics),
                    "retrieved_count": len(retrieved_ids),
                    "golden_count": len(golden_ids),
                },
            )
        return metrics

    def _payload(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
    ) -> dict[str, Any]:
        contexts = _resolve_contexts(retrieved_ids, self.config.get("contexts"))
        answer = _first_string(
            self.config.get("generated_answer"),
            self.config.get("answer"),
            " ".join(contexts),
        )
        ground_truth = _first_string(
            self.config.get("ground_truth"),
            self.config.get("reference"),
            " ".join(golden_ids),
        )
        return {
            "question": query.strip(),
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
            "retrieved_ids": list(retrieved_ids),
            "golden_ids": list(golden_ids),
        }


def _run_ragas(
    payload: Mapping[str, Any],
    metric_names: list[str],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    try:
        ragas_evaluate, metric_objects, dataset_factory = _import_ragas(metric_names)
    except ImportError as exc:
        raise ImportError(
            "RagasEvaluator requires optional dependencies 'ragas' and 'datasets'. "
            "Install them or inject a runner for tests/offline evaluation."
        ) from exc

    dataset = dataset_factory(
        {
            "question": [payload["question"]],
            "answer": [payload["answer"]],
            "contexts": [payload["contexts"]],
            "ground_truth": [payload["ground_truth"]],
        }
    )
    kwargs = _ragas_kwargs(config)
    return ragas_evaluate(dataset, metrics=metric_objects, **kwargs)


def _import_ragas(metric_names: list[str]) -> tuple[Callable[..., Any], list[Any], Callable[[dict[str, Any]], Any]]:
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas import metrics as ragas_metrics

    metric_objects: list[Any] = []
    for name in metric_names:
        metric = getattr(ragas_metrics, name, None)
        if metric is None:
            raise ValueError(f"Unsupported Ragas metric: {name}")
        metric_objects.append(metric)
    return ragas_evaluate, metric_objects, Dataset.from_dict


def _ragas_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if config.get("llm") is not None:
        kwargs["llm"] = config["llm"]
    if config.get("embeddings") is not None:
        kwargs["embeddings"] = config["embeddings"]
    if config.get("raise_exceptions") is not None:
        kwargs["raise_exceptions"] = bool(config["raise_exceptions"])
    return kwargs


def _configured_metrics(config: Mapping[str, Any]) -> tuple[str, ...]:
    configured = config.get("metrics", DEFAULT_RAGAS_METRICS)
    if not isinstance(configured, (list, tuple, set)):
        raise ValueError("ragas metrics must be a sequence of strings")
    metrics: list[str] = []
    seen: set[str] = set()
    for item in configured:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("ragas metric names must be non-empty strings")
        normalized = item.strip().lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        metrics.append(normalized)
    if not metrics:
        raise ValueError("ragas metrics must not be empty")
    return tuple(metrics)


def _resolve_contexts(
    retrieved_ids: list[str],
    configured_contexts: Any,
) -> list[str]:
    if isinstance(configured_contexts, Mapping):
        contexts = [
            str(configured_contexts.get(item_id, item_id))
            for item_id in retrieved_ids
        ]
    elif isinstance(configured_contexts, list):
        contexts = [str(item) for item in configured_contexts]
    else:
        contexts = list(retrieved_ids)
    return contexts or [""]


def _normalize_metrics(
    raw_metrics: Any,
    expected_metrics: tuple[str, ...],
) -> dict[str, float]:
    payload = _metrics_mapping(raw_metrics)
    normalized: dict[str, float] = {}
    for metric_name in expected_metrics:
        value = payload.get(metric_name)
        if value is None:
            value = payload.get(metric_name.replace("_", " "))
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"ragas metric {metric_name} must be numeric")
        normalized[metric_name] = float(value)
    if not normalized:
        raise ValueError("ragas evaluation returned no configured metrics")
    return normalized


def _metrics_mapping(raw_metrics: Any) -> Mapping[str, Any]:
    if isinstance(raw_metrics, Mapping):
        return raw_metrics
    if hasattr(raw_metrics, "to_pandas"):
        frame = raw_metrics.to_pandas()
        if len(frame.index) == 0:
            return {}
        return dict(frame.iloc[0].to_dict())
    if hasattr(raw_metrics, "scores"):
        scores = raw_metrics.scores
        if isinstance(scores, list) and scores:
            first = scores[0]
            if isinstance(first, Mapping):
                return first
        if isinstance(scores, Mapping):
            return scores
    raise ValueError("ragas evaluation result must be a mapping-like object")


def _first_string(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


__all__ = ["DEFAULT_RAGAS_METRICS", "RagasEvaluator", "RagasRunner"]
