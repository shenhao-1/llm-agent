"""Namespace bridge to allow imports before editable install."""

from __future__ import annotations

from pathlib import Path
import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)

_src_pkg = Path(__file__).resolve().parent.parent / "src" / __name__
if _src_pkg.exists():
    __path__.append(str(_src_pkg))

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
