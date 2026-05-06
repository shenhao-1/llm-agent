"""Base contract for all embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class BaseEmbedding(ABC):
    """Abstract embedding contract used by ingestion and retrieval layers."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})

    @abstractmethod
    def embed(
        self, texts: list[str], trace: TraceContext | None = None
    ) -> list[list[float]]:
        """Return vectors for a batch of texts."""

    @staticmethod
    def validate_texts(texts: list[str]) -> None:
        """Validate batched input shape for embedding operations."""
        if not isinstance(texts, list) or not texts:
            raise ValueError("texts must be a non-empty list")

        for idx, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"texts[{idx}] must be a string")
