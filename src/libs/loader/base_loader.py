"""Base contract for source document loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

from core.types import Document

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class BaseLoader(ABC):
    """Abstract loader contract for converting files into Documents."""

    supported_extensions: tuple[str, ...] = ()

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})

    @abstractmethod
    def load(self, path: str | Path, trace: TraceContext | None = None) -> Document:
        """Parse a source file into a Document."""

    def validate_source_path(self, path: str | Path) -> Path:
        """Validate a source path and return a normalized Path."""
        source_path = Path(path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        if self.supported_extensions:
            extension = source_path.suffix.lower()
            if extension not in self.supported_extensions:
                supported = ", ".join(self.supported_extensions)
                raise ValueError(
                    f"Unsupported file extension '{extension}'. Supported: {supported}"
                )

        return source_path
