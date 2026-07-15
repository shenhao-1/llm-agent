"""Configuration helpers for the Streamlit dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from core.settings import Settings, load_settings


SENSITIVE_KEY_PARTS = ("api_key", "apikey", "token", "secret", "password")


@dataclass(frozen=True)
class ComponentConfig:
    """Dashboard-friendly description of one configured component."""

    name: str
    provider: str
    settings: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "settings": dict(self.settings),
        }


class ConfigService:
    """Load settings and format component configuration for display."""

    def __init__(self, settings_path: str | Path = "config/settings.yaml") -> None:
        self.settings_path = Path(settings_path)

    def load(self) -> Settings:
        return load_settings(self.settings_path)

    def get_component_configs(self) -> list[ComponentConfig]:
        settings = self.load()
        sections: list[tuple[str, Mapping[str, Any], str]] = [
            ("LLM", settings.llm, "provider"),
            ("Embedding", settings.embedding, "provider"),
            ("Vector Store", settings.vector_store, "provider"),
            ("Retrieval", settings.retrieval, "strategy"),
            ("Rerank", settings.rerank, "provider"),
            ("Evaluation", settings.evaluation, "provider"),
            ("Observability", settings.observability, "level"),
        ]
        return [
            ComponentConfig(
                name=name,
                provider=str(section.get(provider_key, "unknown")),
                settings=_redact_mapping(section),
            )
            for name, section, provider_key in sections
        ]

    def get_summary(self) -> dict[str, Any]:
        components = self.get_component_configs()
        return {
            "settings_path": str(self.settings_path),
            "component_count": len(components),
            "components": [component.to_dict() for component in components],
        }


def _redact_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in values.items():
        normalized_key = str(key)
        if _is_sensitive_key(normalized_key):
            redacted[normalized_key] = "***"
            continue
        if isinstance(value, Mapping):
            redacted[normalized_key] = _redact_mapping(value)
            continue
        redacted[normalized_key] = value
    return redacted


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in SENSITIVE_KEY_PARTS)


__all__ = ["ComponentConfig", "ConfigService"]
