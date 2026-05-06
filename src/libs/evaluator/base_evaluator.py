"""Base contract for all evaluator providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class BaseEvaluator(ABC):
    """Abstract evaluator contract used by evaluation workflows."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: TraceContext | None = None,
    ) -> dict[str, float]:
        """Return evaluation metrics for one query result set."""

    @staticmethod
    def validate_query(query: str) -> None:
        """Validate evaluator query input."""
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if not query.strip():
            raise ValueError("query must be a non-empty string")

    @staticmethod
    def validate_id_list(values: list[str], field_name: str) -> None:
        """Validate retrieved or golden id lists."""
        if not isinstance(values, list):
            raise ValueError(f"{field_name} must be a list")

        for idx, value in enumerate(values):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name}[{idx}] must be a non-empty string")
