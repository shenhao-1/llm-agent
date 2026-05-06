"""Factory for constructing reranker providers from settings."""

from __future__ import annotations

from typing import Any, Mapping

from core.settings import Settings

from .base_reranker import BaseReranker, NoneReranker


class RerankerFactory:
    """Create reranker implementations based on provider name."""

    _providers: dict[str, type[BaseReranker]] = {"none": NoneReranker}

    @classmethod
    def register_provider(cls, name: str, provider_cls: type[BaseReranker]) -> None:
        """Register a reranker provider class for a provider key."""
        if not name or not isinstance(name, str):
            raise ValueError("Provider name must be a non-empty string")
        if not issubclass(provider_cls, BaseReranker):
            raise TypeError("provider_cls must inherit from BaseReranker")

        cls._providers[name.strip().lower()] = provider_cls

    @classmethod
    def create(cls, settings: Settings | Mapping[str, Any]) -> BaseReranker:
        """Instantiate a reranker provider from settings."""
        cls._ensure_default_providers_registered()
        rerank_config = cls._extract_rerank_config(settings)
        provider_name = str(rerank_config.get("provider", "none")).strip().lower() or "none"

        provider_cls = cls._providers.get(provider_name)
        if provider_cls is None:
            known = ", ".join(sorted(cls._providers)) or "<none>"
            raise ValueError(
                f"Unknown reranker provider '{provider_name}'. Registered providers: {known}"
            )

        return provider_cls(rerank_config)

    @staticmethod
    def _extract_rerank_config(settings: Settings | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(settings, Settings):
            return dict(settings.rerank)

        if not isinstance(settings, Mapping):
            raise TypeError("settings must be Settings or Mapping")

        rerank = settings.get("rerank")
        if not isinstance(rerank, Mapping):
            raise ValueError("Missing or invalid settings section: rerank")

        return dict(rerank)

    @classmethod
    def _ensure_default_providers_registered(cls) -> None:
        from .cross_encoder_reranker import CrossEncoderReranker
        from .llm_reranker import LLMReranker

        cls._providers.setdefault("llm", LLMReranker)
        cls._providers.setdefault("cross_encoder", CrossEncoderReranker)
