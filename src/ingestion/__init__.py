"""Ingestion pipeline package."""

from .document_manager import (
    CollectionStats,
    DeleteResult,
    DocumentDetail,
    DocumentInfo,
    DocumentManager,
)
from .pipeline import IngestionPipeline, IngestionPipelineError, IngestionResult

__all__ = [
    "CollectionStats",
    "DeleteResult",
    "DocumentDetail",
    "DocumentInfo",
    "DocumentManager",
    "IngestionPipeline",
    "IngestionPipelineError",
    "IngestionResult",
]
