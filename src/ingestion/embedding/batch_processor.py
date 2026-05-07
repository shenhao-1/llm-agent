"""Batch orchestration for dense and sparse encoding."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping, TYPE_CHECKING

from core.types import Chunk, ChunkRecord

from .dense_encoder import DenseEncoder
from .sparse_encoder import SparseEncoder, SparseEncodingResult

if TYPE_CHECKING:
    from core.settings import Settings
    from core.trace.trace_context import TraceContext


@dataclass(frozen=True)
class BatchResult:
    """Encoding output for one batch."""

    batch_index: int
    chunks: list[Chunk]
    dense_records: list[ChunkRecord]
    sparse_result: SparseEncodingResult
    elapsed_ms: float

    @property
    def size(self) -> int:
        return len(self.chunks)


@dataclass(frozen=True)
class BatchProcessingResult:
    """Complete encoding output across all batches."""

    batches: list[BatchResult]
    dense_records: list[ChunkRecord]
    sparse_result: SparseEncodingResult
    elapsed_ms: float

    @property
    def batch_count(self) -> int:
        return len(self.batches)


class BatchProcessor:
    """Split chunks into stable batches and run dense/sparse encoders."""

    def __init__(
        self,
        settings: "Settings | Mapping[str, Any] | None" = None,
        dense_encoder: DenseEncoder | None = None,
        sparse_encoder: SparseEncoder | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.settings = settings or {}
        config = self._extract_config(self.settings)
        configured_batch_size = config.get("batch_size", 32) if batch_size is None else batch_size
        if not isinstance(configured_batch_size, int) or configured_batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        self.batch_size = configured_batch_size
        self.dense_encoder = dense_encoder or DenseEncoder(self.settings)
        self.sparse_encoder = sparse_encoder or SparseEncoder(config.get("sparse", {}))

    def process(
        self, chunks: list[Chunk], trace: "TraceContext | None" = None
    ) -> BatchProcessingResult:
        """Encode chunks in batches and return aggregate dense/sparse outputs."""
        self._validate_chunks(chunks)
        total_start = time.perf_counter()
        batches: list[BatchResult] = []

        for batch_index, batch_chunks in enumerate(self.iter_batches(chunks)):
            batch_start = time.perf_counter()
            dense_records = self.dense_encoder.encode(batch_chunks, trace=trace)
            sparse_result = self.sparse_encoder.encode(batch_chunks, trace=trace)
            elapsed_ms = (time.perf_counter() - batch_start) * 1000
            batches.append(
                BatchResult(
                    batch_index=batch_index,
                    chunks=batch_chunks,
                    dense_records=dense_records,
                    sparse_result=sparse_result,
                    elapsed_ms=elapsed_ms,
                )
            )

        dense_records = [
            record for batch in batches for record in batch.dense_records
        ]
        sparse_result = self._combine_sparse_results(
            [batch.sparse_result for batch in batches]
        )
        elapsed_ms = (time.perf_counter() - total_start) * 1000

        if trace is not None:
            trace.record_stage(
                "embed",
                method="batch_processor",
                details={
                    "chunk_count": len(chunks),
                    "batch_count": len(batches),
                    "batch_size": self.batch_size,
                    "elapsed_ms": elapsed_ms,
                },
            )

        return BatchProcessingResult(
            batches=batches,
            dense_records=dense_records,
            sparse_result=sparse_result,
            elapsed_ms=elapsed_ms,
        )

    def iter_batches(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        """Return stable, order-preserving batches."""
        self._validate_chunks(chunks)
        return [
            chunks[index : index + self.batch_size]
            for index in range(0, len(chunks), self.batch_size)
        ]

    @staticmethod
    def _combine_sparse_results(
        results: list[SparseEncodingResult],
    ) -> SparseEncodingResult:
        records: list[ChunkRecord] = []
        document_frequency: dict[str, int] = {}
        doc_lengths: dict[str, int] = {}

        for result in results:
            records.extend(result.records)
            for term, count in result.corpus_stats.get("document_frequency", {}).items():
                document_frequency[term] = document_frequency.get(term, 0) + int(count)
            doc_lengths.update(result.corpus_stats.get("doc_lengths", {}))

        doc_count = len(records)
        total_terms = sum(doc_lengths.values())
        return SparseEncodingResult(
            records=records,
            corpus_stats={
                "doc_count": doc_count,
                "avg_doc_length": (total_terms / doc_count) if doc_count else 0.0,
                "document_frequency": dict(sorted(document_frequency.items())),
                "doc_lengths": doc_lengths,
            },
        )

    @staticmethod
    def _extract_config(settings: "Settings | Mapping[str, Any]") -> dict[str, Any]:
        if hasattr(settings, "ingestion"):
            ingestion = getattr(settings, "ingestion")
            if isinstance(ingestion, Mapping):
                batch_processor = ingestion.get("batch_processor")
                if isinstance(batch_processor, Mapping):
                    return dict(batch_processor)

        if isinstance(settings, Mapping):
            batch_processor = settings.get("batch_processor")
            if isinstance(batch_processor, Mapping):
                return dict(batch_processor)

            ingestion = settings.get("ingestion")
            if isinstance(ingestion, Mapping):
                nested = ingestion.get("batch_processor")
                if isinstance(nested, Mapping):
                    return dict(nested)

        return {}

    @staticmethod
    def _validate_chunks(chunks: Any) -> None:
        if not isinstance(chunks, list):
            raise ValueError("chunks must be a list")
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise ValueError(f"chunks[{idx}] must be a Chunk")


__all__ = ["BatchProcessor", "BatchProcessingResult", "BatchResult"]
