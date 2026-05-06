"""Factory for constructing splitter providers from settings."""

from __future__ import annotations

from typing import Any, Mapping

from core.settings import Settings

from .base_splitter import BaseSplitter


class SplitterFactory:
    """Create splitter implementations based on provider name."""

    _providers: dict[str, type[BaseSplitter]] = {}

    @classmethod
    def register_provider(cls, name: str, provider_cls: type[BaseSplitter]) -> None:
        """Register a splitter provider class for a provider key."""
        if not name or not isinstance(name, str):
            raise ValueError("Provider name must be a non-empty string")
        if not issubclass(provider_cls, BaseSplitter):
            raise TypeError("provider_cls must inherit from BaseSplitter")

        cls._providers[name.strip().lower()] = provider_cls

    @classmethod
    def create(cls, settings: Settings | Mapping[str, Any]) -> BaseSplitter:
        """Instantiate a splitter provider from settings."""
        cls._ensure_default_providers_registered()
        splitter_config = cls._extract_splitter_config(settings)
        provider_name = str(splitter_config.get("provider", "")).strip().lower()

        if not provider_name:
            raise ValueError("Missing splitter provider in settings: splitter.provider")

        provider_cls = cls._providers.get(provider_name)
        if provider_cls is None:
            known = ", ".join(sorted(cls._providers)) or "<none>"
            raise ValueError(
                "Unknown splitter provider "
                f"'{provider_name}'. Registered providers: {known}"
            )

        return provider_cls(splitter_config)

    @staticmethod
    def _extract_splitter_config(settings: Settings | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(settings, Settings):
            splitter = settings.retrieval.get("splitter")
            if isinstance(splitter, Mapping):
                return dict(splitter)
            raise ValueError("Missing or invalid settings section: retrieval.splitter")

        if not isinstance(settings, Mapping):
            raise TypeError("settings must be Settings or Mapping")

        splitter = settings.get("splitter")
        if isinstance(splitter, Mapping):
            return dict(splitter)

        retrieval = settings.get("retrieval")
        if isinstance(retrieval, Mapping):
            nested = retrieval.get("splitter")
            if isinstance(nested, Mapping):
                return dict(nested)

        raise ValueError("Missing or invalid settings section: splitter")

    @classmethod
    def _ensure_default_providers_registered(cls) -> None:
        from .recursive_splitter import RecursiveSplitter

        cls._providers.setdefault("recursive", RecursiveSplitter)
