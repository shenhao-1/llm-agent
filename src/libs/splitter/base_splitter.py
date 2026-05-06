"""Base contract for all splitter providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class BaseSplitter(ABC):
    """Abstract splitter contract used in ingestion pipeline."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})

    @abstractmethod
    def split_text(self, text: str, trace: TraceContext | None = None) -> list[str]:
        """Split a single text input into chunks."""

    @staticmethod
    def validate_text(text: str) -> None:
        """Validate splitter input text shape."""
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        if not text.strip():
            raise ValueError("text must be a non-empty string")
