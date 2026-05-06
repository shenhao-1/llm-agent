"""Base contract for all vector store providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Real
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class BaseVectorStore(ABC):
    """Abstract vector store contract used by ingestion and retrieval layers."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})

    @abstractmethod
    def upsert(
        self,
        records: list[Mapping[str, Any]],
        trace: TraceContext | None = None,
    ) -> None:
        """Store or update vector records."""

    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: TraceContext | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top-k matching records for a query vector."""

    @staticmethod
    def validate_records(records: list[Mapping[str, Any]]) -> None:
        """Validate the minimal record shape shared by all vector stores."""
        if not isinstance(records, list) or not records:
            raise ValueError("records must be a non-empty list")

        for idx, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise ValueError(f"records[{idx}] must be a mapping")

            record_id = record.get("id")
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError(f"records[{idx}].id must be a non-empty string")

            vector = record.get("vector")
            BaseVectorStore.validate_vector(vector, field_name=f"records[{idx}].vector")

            metadata = record.get("metadata")
            if metadata is not None and not isinstance(metadata, Mapping):
                raise ValueError(f"records[{idx}].metadata must be a mapping")

    @staticmethod
    def validate_vector(vector: Any, field_name: str = "vector") -> None:
        """Validate vector input shared by upsert and query operations."""
        if not isinstance(vector, list) or not vector:
            raise ValueError(f"{field_name} must be a non-empty list")

        for idx, value in enumerate(vector):
            if not isinstance(value, Real):
                raise ValueError(f"{field_name}[{idx}] must be a number")

    @staticmethod
    def validate_query_args(
        vector: list[float], top_k: int, filters: Mapping[str, Any] | None = None
    ) -> None:
        """Validate common query arguments used by all vector stores."""
        BaseVectorStore.validate_vector(vector)

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        if filters is not None and not isinstance(filters, Mapping):
            raise ValueError("filters must be a mapping when provided")
