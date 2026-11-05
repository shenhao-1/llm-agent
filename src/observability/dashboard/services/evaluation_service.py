"""Dashboard service for running and browsing evaluation reports."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from core.settings import load_settings
from libs.evaluator import EvaluatorFactory
from observability.evaluation.eval_runner import EvalReport, EvalRunner


DEFAULT_EVAL_HISTORY_PATH = Path("logs/evaluation_reports.jsonl")
DEFAULT_GOLDEN_TEST_SET = Path("tests/fixtures/golden_test_set.json")


class EvaluationService:
    """Build and run evaluation workflows for Dashboard pages."""

    def __init__(
        self,
        settings_path: str | Path = "config/settings.yaml",
        *,
        data_dir: str | Path = "data",
        history_path: str | Path = DEFAULT_EVAL_HISTORY_PATH,
        settings_loader: Callable[[str | Path], Any] = load_settings,
        search_builder: Callable[[Any, Path], Any] | None = None,
        evaluator_factory: Callable[[Any], Any] = EvaluatorFactory.create,
    ) -> None:
        self.settings_path = Path(settings_path)
        self.data_dir = Path(data_dir)
        self.history_path = Path(history_path)
        self.settings_loader = settings_loader
        self.search_builder = search_builder or _default_search_builder
        self.evaluator_factory = evaluator_factory

    def available_backends(self) -> list[str]:
        return ["custom", "ragas", "all"]

    def run_evaluation(
        self,
        *,
        backend: str,
        test_set_path: str | Path = DEFAULT_GOLDEN_TEST_SET,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Run an evaluation and append the report to history."""
        clean_backend = _validate_backend(backend)
        if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
            raise ValueError("top_k must be a positive integer when provided")

        settings = self.settings_loader(self.settings_path)
        search = self.search_builder(settings, self.data_dir)
        evaluator = self.evaluator_factory(_settings_for_backend(settings, clean_backend))
        runner = EvalRunner(
            settings=settings,
            hybrid_search=search,
            evaluator=evaluator,
            default_top_k=top_k,
        )
        report = runner.run(test_set_path)
        payload = _history_payload(
            report,
            backend=clean_backend,
            test_set_path=test_set_path,
        )
        self._append_history(payload)
        return payload

    def list_history(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        if limit is not None and (not isinstance(limit, int) or limit <= 0):
            raise ValueError("limit must be a positive integer when provided")
        if not self.history_path.exists():
            return []
        reports: list[dict[str, Any]] = []
        for line in self.history_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, Mapping) and isinstance(payload.get("metrics"), Mapping):
                reports.append(dict(payload))
        reports.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        return reports[:limit] if limit is not None else reports

    def history_rows(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for report in self.list_history(limit=limit):
            metrics = report.get("metrics", {})
            if not isinstance(metrics, Mapping):
                metrics = {}
            row = {
                "created_at": report.get("created_at"),
                "backend": report.get("backend"),
                "test_set": report.get("test_set"),
                "case_count": len(report.get("results", []))
                if isinstance(report.get("results"), list)
                else 0,
            }
            row.update({str(key): value for key, value in metrics.items()})
            rows.append(row)
        return rows

    def _append_history(self, payload: Mapping[str, Any]) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _settings_for_backend(settings: Any, backend: str) -> dict[str, Any]:
    payload = _settings_to_mapping(settings)
    evaluation = dict(payload.get("evaluation", {}))
    if backend == "all":
        evaluation["backends"] = ["custom", "ragas"]
        evaluation["prefix_metrics"] = True
        evaluation.pop("provider", None)
    else:
        evaluation["provider"] = backend
        evaluation.pop("backends", None)
    payload["evaluation"] = evaluation
    return payload


def _settings_to_mapping(settings: Any) -> dict[str, Any]:
    if hasattr(settings, "to_dict"):
        payload = settings.to_dict()
        if not isinstance(payload, Mapping):
            raise ValueError("settings.to_dict() must return a mapping")
        return dict(payload)
    if isinstance(settings, Mapping):
        return dict(settings)
    sections = (
        "llm",
        "embedding",
        "vector_store",
        "retrieval",
        "rerank",
        "evaluation",
        "observability",
    )
    if all(hasattr(settings, section) for section in sections):
        return {section: dict(getattr(settings, section)) for section in sections}
    raise ValueError("settings must be a mapping or Settings-like object")


def _history_payload(
    report: EvalReport,
    *,
    backend: str,
    test_set_path: str | Path,
) -> dict[str, Any]:
    payload = report.to_dict()
    payload.update(
        {
            "created_at": datetime.now(UTC).isoformat(),
            "backend": backend,
            "test_set": str(test_set_path),
        }
    )
    return payload


def _validate_backend(backend: str) -> str:
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError("backend must be a non-empty string")
    clean = backend.strip().lower()
    if clean not in {"custom", "ragas", "all"}:
        raise ValueError("backend must be one of: custom, ragas, all")
    return clean


def _default_search_builder(settings: Any, data_dir: Path) -> Any:
    from core.runtime import build_search

    return build_search(settings, data_dir)


__all__ = [
    "DEFAULT_EVAL_HISTORY_PATH",
    "DEFAULT_GOLDEN_TEST_SET",
    "EvaluationService",
]
