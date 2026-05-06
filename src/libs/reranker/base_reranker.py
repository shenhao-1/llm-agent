"""Base contract for all reranker providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class BaseReranker(ABC):
    """Abstract reranker contract used by retrieval and response layers."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[Mapping[str, Any]],
        trace: TraceContext | None = None,
    ) -> list[dict[str, Any]]:
        """Return candidates in ranked order for a query."""

    @staticmethod
    def validate_query(query: str) -> None:
        """Validate reranker query input."""
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if not query.strip():
            raise ValueError("query must be a non-empty string")

    @staticmethod
    def validate_candidates(candidates: list[Mapping[str, Any]]) -> None:
        """Validate the minimal candidate shape shared by rerankers."""
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("candidates must be a non-empty list")

        for idx, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise ValueError(f"candidates[{idx}] must be a mapping")

            candidate_id = candidate.get("id")
            if not isinstance(candidate_id, str) or not candidate_id.strip():
                raise ValueError(f"candidates[{idx}].id must be a non-empty string")

            text = candidate.get("text")
            if not isinstance(text, str):
                raise ValueError(f"candidates[{idx}].text must be a string")

            metadata = candidate.get("metadata")
            if metadata is not None and not isinstance(metadata, Mapping):
                raise ValueError(f"candidates[{idx}].metadata must be a mapping")


class NoneReranker(BaseReranker):
    """Fallback reranker that preserves the original order."""

    def rerank(
        self,
        query: str,
        candidates: list[Mapping[str, Any]],
        trace: TraceContext | None = None,
    ) -> list[dict[str, Any]]:
        self.validate_query(query)
        self.validate_candidates(candidates)
        return [dict(candidate) for candidate in candidates]
