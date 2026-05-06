"""Factory for constructing vector store providers from settings."""

from __future__ import annotations

from typing import Any, Mapping

from core.settings import Settings

from .base_vector_store import BaseVectorStore


class VectorStoreFactory:
    """Create vector store implementations based on provider name."""

    _providers: dict[str, type[BaseVectorStore]] = {}

    @classmethod
    def register_provider(cls, name: str, provider_cls: type[BaseVectorStore]) -> None:
        """Register a vector store provider class for a provider key."""
        if not name or not isinstance(name, str):
            raise ValueError("Provider name must be a non-empty string")
        if not issubclass(provider_cls, BaseVectorStore):
            raise TypeError("provider_cls must inherit from BaseVectorStore")

        cls._providers[name.strip().lower()] = provider_cls

    @classmethod
    def create(cls, settings: Settings | Mapping[str, Any]) -> BaseVectorStore:
        """Instantiate a vector store provider from settings."""
        cls._ensure_default_providers_registered()
        vector_store_config = cls._extract_vector_store_config(settings)
        provider_name = str(vector_store_config.get("provider", "")).strip().lower()

        if not provider_name:
            raise ValueError(
                "Missing vector store provider in settings: vector_store.provider"
            )

        provider_cls = cls._providers.get(provider_name)
        if provider_cls is None:
            known = ", ".join(sorted(cls._providers)) or "<none>"
            raise ValueError(
                "Unknown vector store provider "
                f"'{provider_name}'. Registered providers: {known}"
            )

        return provider_cls(vector_store_config)

    @staticmethod
    def _extract_vector_store_config(
        settings: Settings | Mapping[str, Any],
    ) -> dict[str, Any]:
        if isinstance(settings, Settings):
            return dict(settings.vector_store)

        if not isinstance(settings, Mapping):
            raise TypeError("settings must be Settings or Mapping")

        vector_store = settings.get("vector_store")
        if not isinstance(vector_store, Mapping):
            raise ValueError("Missing or invalid settings section: vector_store")

        return dict(vector_store)

    @classmethod
    def _ensure_default_providers_registered(cls) -> None:
        from .chroma_store import ChromaStore

        cls._providers.setdefault("chroma", ChromaStore)
