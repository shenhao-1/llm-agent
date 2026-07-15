"""Golden-test evaluation runner for retrieval quality."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, TYPE_CHECKING

from core.types import RetrievalResult
from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.evaluator_factory import EvaluatorFactory

if TYPE_CHECKING:
    from core.settings import Settings
    from core.trace.trace_context import TraceContext


class SearchProtocol(Protocol):
    def search(
        self,
        query: str,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: "TraceContext | None" = None,
    ) -> list[RetrievalResult]:
        ...


@dataclass(frozen=True)
class EvalCase:
    """One golden test case."""

    query: str
    expected_chunk_ids: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)
    collection: str | None = None
    top_k: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "expected_chunk_ids": list(self.expected_chunk_ids),
            "expected_sources": list(self.expected_sources),
            "collection": self.collection,
            "top_k": self.top_k,
        }


@dataclass(frozen=True)
class EvalCaseResult:
    """Evaluation output for one golden query."""

    query: str
    retrieved_ids: list[str]
    expected_chunk_ids: list[str]
    expected_sources: list[str]
    metrics: dict[str, float]
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "retrieved_ids": list(self.retrieved_ids),
            "expected_chunk_ids": list(self.expected_chunk_ids),
            "expected_sources": list(self.expected_sources),
            "sources": list(self.sources),
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class EvalReport:
    """Aggregate evaluation report."""

    metrics: dict[str, float]
    results: list[EvalCaseResult]

    @property
    def hit_rate(self) -> float:
        return float(self.metrics.get("hit_rate", 0.0))

    @property
    def mrr(self) -> float:
        return float(self.metrics.get("mrr", 0.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "results": [result.to_dict() for result in self.results],
        }


class EvalRunner:
    """Run golden retrieval cases and aggregate evaluator metrics."""

    def __init__(
        self,
        settings: "Settings | Mapping[str, Any] | None" = None,
        hybrid_search: SearchProtocol | None = None,
        evaluator: BaseEvaluator | None = None,
        *,
        default_top_k: int | None = None,
    ) -> None:
        if hybrid_search is None:
            raise ValueError("hybrid_search must be provided")
        self.settings = settings or {}
        self.hybrid_search = hybrid_search
        self.evaluator = evaluator or EvaluatorFactory.create(self.settings)
        self.default_top_k = (
            default_top_k
            if default_top_k is not None
            else _extract_default_top_k(self.settings)
        )
        if not isinstance(self.default_top_k, int) or self.default_top_k <= 0:
            raise ValueError("default_top_k must be a positive integer")

    def run(
        self,
        test_set_path: str | Path,
        trace: "TraceContext | None" = None,
    ) -> EvalReport:
        cases = load_golden_test_set(test_set_path)
        results: list[EvalCaseResult] = []
        for case in cases:
            result = self._run_case(case, trace)
            results.append(result)
        return EvalReport(metrics=_aggregate_metrics(results), results=results)

    def _run_case(
        self,
        case: EvalCase,
        trace: "TraceContext | None",
    ) -> EvalCaseResult:
        top_k = case.top_k or self.default_top_k
        filters = {"collection": case.collection} if case.collection else None
        retrieved = self.hybrid_search.search(
            case.query,
            top_k=top_k,
            filters=filters,
            trace=trace,
        )
        retrieved_ids = [result.chunk_id for result in retrieved]
        expected_ids = _expected_ids(case)
        metrics = self.evaluator.evaluate(
            query=case.query,
            retrieved_ids=retrieved_ids,
            golden_ids=expected_ids,
            trace=trace,
        )
        return EvalCaseResult(
            query=case.query,
            retrieved_ids=retrieved_ids,
            expected_chunk_ids=list(case.expected_chunk_ids),
            expected_sources=list(case.expected_sources),
            metrics=dict(metrics),
            sources=[_source_from_result(result) for result in retrieved],
        )


def load_golden_test_set(path: str | Path) -> list[EvalCase]:
    """Read and validate a golden test set file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raw_cases = payload
    elif isinstance(payload, Mapping):
        raw_cases = payload.get("test_cases")
    else:
        raise ValueError("golden test set must be a list or mapping")

    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("golden test set must contain non-empty test_cases")

    cases: list[EvalCase] = []
    for idx, item in enumerate(raw_cases):
        if not isinstance(item, Mapping):
            raise ValueError(f"test_cases[{idx}] must be a mapping")
        query = item.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"test_cases[{idx}].query must be a non-empty string")
        expected_chunk_ids = _string_list(
            item.get("expected_chunk_ids", []),
            f"test_cases[{idx}].expected_chunk_ids",
        )
        expected_sources = _string_list(
            item.get("expected_sources", []),
            f"test_cases[{idx}].expected_sources",
        )
        if not expected_chunk_ids and not expected_sources:
            raise ValueError(
                f"test_cases[{idx}] must define expected_chunk_ids or expected_sources"
            )
        collection = item.get("collection")
        if collection is not None and (
            not isinstance(collection, str) or not collection.strip()
        ):
            raise ValueError(f"test_cases[{idx}].collection must be a non-empty string")
        top_k = item.get("top_k")
        if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
            raise ValueError(f"test_cases[{idx}].top_k must be a positive integer")
        cases.append(
            EvalCase(
                query=query.strip(),
                expected_chunk_ids=expected_chunk_ids,
                expected_sources=expected_sources,
                collection=collection.strip() if isinstance(collection, str) else None,
                top_k=top_k,
            )
        )
    return cases


def _expected_ids(case: EvalCase) -> list[str]:
    if case.expected_chunk_ids:
        return list(case.expected_chunk_ids)
    return [f"source:{source}" for source in case.expected_sources]


def _source_from_result(result: RetrievalResult) -> str:
    source = result.metadata.get("source_path", "")
    return str(source) if source is not None else ""


def _aggregate_metrics(results: list[EvalCaseResult]) -> dict[str, float]:
    if not results:
        raise ValueError("results must be non-empty")
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for result in results:
        for key, value in result.metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {
        key: totals[key] / counts[key]
        for key in sorted(totals)
    }


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    result: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name}[{idx}] must be a non-empty string")
        result.append(item.strip())
    return result


def _extract_default_top_k(settings: "Settings | Mapping[str, Any]") -> int:
    retrieval: Any = {}
    if hasattr(settings, "retrieval"):
        retrieval = getattr(settings, "retrieval")
    elif isinstance(settings, Mapping):
        retrieval = settings.get("retrieval", {})
    if isinstance(retrieval, Mapping):
        top_k = retrieval.get("top_k", 5)
    else:
        top_k = 5
    return int(top_k) if isinstance(top_k, int) else 5


__all__ = [
    "EvalCase",
    "EvalCaseResult",
    "EvalReport",
    "EvalRunner",
    "load_golden_test_set",
]
