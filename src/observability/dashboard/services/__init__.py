"""Dashboard service helpers with lazy imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config_service import ComponentConfig, ConfigService
    from .data_service import DataService
    from .evaluation_service import EvaluationService
    from .ingestion_service import IngestionService
    from .trace_service import TraceService

__all__ = [
    "ComponentConfig",
    "ConfigService",
    "DataService",
    "EvaluationService",
    "IngestionService",
    "TraceService",
]


def __getattr__(name: str) -> Any:
    if name in {"ComponentConfig", "ConfigService"}:
        from .config_service import ComponentConfig, ConfigService

        return {"ComponentConfig": ComponentConfig, "ConfigService": ConfigService}[name]
    if name == "DataService":
        from .data_service import DataService

        return DataService
    if name == "EvaluationService":
        from .evaluation_service import EvaluationService

        return EvaluationService
    if name == "IngestionService":
        from .ingestion_service import IngestionService

        return IngestionService
    if name == "TraceService":
        from .trace_service import TraceService

        return TraceService
    raise AttributeError(name)
