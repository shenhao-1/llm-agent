"""Factory for constructing evaluator providers from settings."""

from __future__ import annotations

from typing import Any, Mapping

from core.settings import Settings

from .base_evaluator import BaseEvaluator
from .custom_evaluator import CustomEvaluator


class EvaluatorFactory:
    """Create evaluator implementations based on provider name."""

    _providers: dict[str, type[BaseEvaluator]] = {"custom": CustomEvaluator}

    @classmethod
    def register_provider(cls, name: str, provider_cls: type[BaseEvaluator]) -> None:
        """Register an evaluator provider class for a provider key."""
        if not name or not isinstance(name, str):
            raise ValueError("Provider name must be a non-empty string")
        if not issubclass(provider_cls, BaseEvaluator):
            raise TypeError("provider_cls must inherit from BaseEvaluator")

        cls._providers[name.strip().lower()] = provider_cls

    @classmethod
    def create(cls, settings: Settings | Mapping[str, Any]) -> BaseEvaluator:
        """Instantiate an evaluator provider from settings."""
        evaluation_config = cls._extract_evaluation_config(settings)
        provider_name = str(evaluation_config.get("provider", "custom")).strip().lower() or "custom"

        provider_cls = cls._providers.get(provider_name)
        if provider_cls is None:
            known = ", ".join(sorted(cls._providers)) or "<none>"
            raise ValueError(
                f"Unknown evaluator provider '{provider_name}'. Registered providers: {known}"
            )

        return provider_cls(evaluation_config)

    @staticmethod
    def _extract_evaluation_config(settings: Settings | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(settings, Settings):
            return dict(settings.evaluation)

        if not isinstance(settings, Mapping):
            raise TypeError("settings must be Settings or Mapping")

        evaluation = settings.get("evaluation")
        if not isinstance(evaluation, Mapping):
            raise ValueError("Missing or invalid settings section: evaluation")

        return dict(evaluation)
