"""Factory for constructing LLM providers from settings."""

from __future__ import annotations

from typing import Any, Mapping

from core.settings import Settings

from .base_llm import BaseLLM
from .base_vision_llm import BaseVisionLLM


class LLMFactory:
    """Create LLM implementations based on provider name."""

    _providers: dict[str, type[BaseLLM]] = {}
    _vision_providers: dict[str, type[BaseVisionLLM]] = {}

    @classmethod
    def register_provider(cls, name: str, provider_cls: type[BaseLLM]) -> None:
        """Register an LLM provider class for a provider key."""
        if not name or not isinstance(name, str):
            raise ValueError("Provider name must be a non-empty string")
        if not issubclass(provider_cls, BaseLLM):
            raise TypeError("provider_cls must inherit from BaseLLM")

        cls._providers[name.strip().lower()] = provider_cls

    @classmethod
    def register_vision_provider(
        cls, name: str, provider_cls: type[BaseVisionLLM]
    ) -> None:
        """Register a vision-capable LLM provider class for a provider key."""
        if not name or not isinstance(name, str):
            raise ValueError("Provider name must be a non-empty string")
        if not issubclass(provider_cls, BaseVisionLLM):
            raise TypeError("provider_cls must inherit from BaseVisionLLM")

        cls._vision_providers[name.strip().lower()] = provider_cls

    @classmethod
    def create(cls, settings: Settings | Mapping[str, Any]) -> BaseLLM:
        """Instantiate a provider from settings."""
        cls._ensure_default_providers_registered()
        llm_config = cls._extract_llm_config(settings)
        provider_name = str(llm_config.get("provider", "")).strip().lower()

        if not provider_name:
            raise ValueError("Missing LLM provider in settings: llm.provider")

        provider_cls = cls._providers.get(provider_name)
        if provider_cls is None:
            known = ", ".join(sorted(cls._providers)) or "<none>"
            raise ValueError(
                f"Unknown LLM provider '{provider_name}'. Registered providers: {known}"
            )

        return provider_cls(llm_config)

    @classmethod
    def create_vision_llm(
        cls, settings: Settings | Mapping[str, Any]
    ) -> BaseVisionLLM:
        """Instantiate a vision-capable provider from settings."""
        cls._ensure_default_providers_registered()
        llm_config = cls._extract_llm_config(settings)
        provider_name = str(llm_config.get("provider", "")).strip().lower()

        if not provider_name:
            raise ValueError("Missing LLM provider in settings: llm.provider")

        provider_cls = cls._vision_providers.get(provider_name)
        if provider_cls is None:
            known = ", ".join(sorted(cls._vision_providers)) or "<none>"
            raise ValueError(
                "Unknown vision LLM provider "
                f"'{provider_name}'. Registered providers: {known}"
            )

        return provider_cls(llm_config)

    @staticmethod
    def _extract_llm_config(settings: Settings | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(settings, Settings):
            return dict(settings.llm)

        if not isinstance(settings, Mapping):
            raise TypeError("settings must be Settings or Mapping")

        llm = settings.get("llm")
        if not isinstance(llm, Mapping):
            raise ValueError("Missing or invalid settings section: llm")

        return dict(llm)

    @classmethod
    def _ensure_default_providers_registered(cls) -> None:
        from .azure_llm import AzureLLM
        from .azure_vision_llm import AzureVisionLLM
        from .deepseek_llm import DeepSeekLLM
        from .ollama_llm import OllamaLLM
        from .openai_llm import OpenAILLM

        defaults: dict[str, type[BaseLLM]] = {
            "openai": OpenAILLM,
            "azure": AzureLLM,
            "deepseek": DeepSeekLLM,
            "ollama": OllamaLLM,
        }
        for name, provider_cls in defaults.items():
            cls._providers.setdefault(name, provider_cls)

        cls._vision_providers.setdefault("azure", AzureVisionLLM)
