"""Factory for constructing embedding providers from settings."""

from __future__ import annotations

from typing import Any, Mapping

from core.settings import Settings

from .base_embedding import BaseEmbedding


class EmbeddingFactory:
    """Create embedding implementations based on provider name."""

    _providers: dict[str, type[BaseEmbedding]] = {}

    @classmethod
    def register_provider(cls, name: str, provider_cls: type[BaseEmbedding]) -> None:
        """Register an embedding provider class for a provider key."""
        if not name or not isinstance(name, str):
            raise ValueError("Provider name must be a non-empty string")
        if not issubclass(provider_cls, BaseEmbedding):
            raise TypeError("provider_cls must inherit from BaseEmbedding")

        cls._providers[name.strip().lower()] = provider_cls

    @classmethod
    def create(cls, settings: Settings | Mapping[str, Any]) -> BaseEmbedding:
        """Instantiate an embedding provider from settings."""
        cls._ensure_default_providers_registered()
        embedding_config = cls._extract_embedding_config(settings)
        provider_name = str(embedding_config.get("provider", "")).strip().lower()

        if not provider_name:
            raise ValueError("Missing embedding provider in settings: embedding.provider")

        provider_cls = cls._providers.get(provider_name)
        if provider_cls is None:
            known = ", ".join(sorted(cls._providers)) or "<none>"
            raise ValueError(
                "Unknown embedding provider "
                f"'{provider_name}'. Registered providers: {known}"
            )

        return provider_cls(embedding_config)

    @staticmethod
    def _extract_embedding_config(settings: Settings | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(settings, Settings):
            return dict(settings.embedding)

        if not isinstance(settings, Mapping):
            raise TypeError("settings must be Settings or Mapping")

        embedding = settings.get("embedding")
        if not isinstance(embedding, Mapping):
            raise ValueError("Missing or invalid settings section: embedding")

        return dict(embedding)

    @classmethod
    def _ensure_default_providers_registered(cls) -> None:
        from .azure_embedding import AzureEmbedding
        from .ollama_embedding import OllamaEmbedding
        from .openai_embedding import OpenAIEmbedding

        defaults: dict[str, type[BaseEmbedding]] = {
            "openai": OpenAIEmbedding,
            "azure": AzureEmbedding,
            "ollama": OllamaEmbedding,
        }
        for name, provider_cls in defaults.items():
            cls._providers.setdefault(name, provider_cls)
