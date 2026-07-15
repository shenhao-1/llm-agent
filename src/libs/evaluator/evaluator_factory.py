"""Factory for constructing evaluator providers from settings."""

from __future__ import annotations

from typing import Any, Mapping

from core.settings import Settings

from .base_evaluator import BaseEvaluator
from .custom_evaluator import CustomEvaluator
from .ragas_evaluator import RagasEvaluator


class EvaluatorFactory:
    """Create evaluator implementations based on provider name."""

    _providers: dict[str, type[BaseEvaluator]] = {
        "custom": CustomEvaluator,
        "ragas": RagasEvaluator,
    }

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
        backends = evaluation_config.get("backends")
        if backends is not None:
            return cls._create_composite(evaluation_config, backends)

        provider_name = str(evaluation_config.get("provider", "custom")).strip().lower() or "custom"

        return cls._create_provider(provider_name, evaluation_config)

    @classmethod
    def _create_composite(
        cls,
        evaluation_config: dict[str, Any],
        backends: Any,
    ):
        from observability.evaluation.composite_evaluator import CompositeEvaluator

        if not isinstance(backends, list) or not backends:
            raise ValueError("evaluation.backends must be a non-empty list")
        evaluators: list[BaseEvaluator] = []
        for idx, backend in enumerate(backends):
            backend_config = cls._backend_config(evaluation_config, backend, idx)
            provider_name = str(backend_config.get("provider", "")).strip().lower()
            evaluators.append(cls._create_provider(provider_name, backend_config))
        return CompositeEvaluator(evaluators, evaluation_config)

    @classmethod
    def _create_provider(
        cls,
        provider_name: str,
        evaluation_config: Mapping[str, Any],
    ) -> BaseEvaluator:
        clean_name = provider_name.strip().lower() or "custom"
        provider_cls = cls._providers.get(clean_name)
        if provider_cls is None:
            known = ", ".join(sorted(cls._providers)) or "<none>"
            raise ValueError(
                f"Unknown evaluator provider '{clean_name}'. Registered providers: {known}"
            )
        return provider_cls(evaluation_config)

    @staticmethod
    def _backend_config(
        evaluation_config: Mapping[str, Any],
        backend: Any,
        idx: int,
    ) -> dict[str, Any]:
        if isinstance(backend, str) and backend.strip():
            config = dict(evaluation_config)
            config.pop("backends", None)
            config["provider"] = backend.strip().lower()
            return config
        if isinstance(backend, Mapping):
            provider = backend.get("provider")
            if not isinstance(provider, str) or not provider.strip():
                raise ValueError(f"evaluation.backends[{idx}].provider must be a non-empty string")
            config = dict(evaluation_config)
            config.pop("backends", None)
            config.update(dict(backend))
            config["provider"] = provider.strip().lower()
            return config
        raise ValueError(f"evaluation.backends[{idx}] must be a provider name or mapping")

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
