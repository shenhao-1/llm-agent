"""Azure OpenAI embedding provider."""

from __future__ import annotations

from typing import Any, Mapping
from urllib.parse import quote, urlencode

from .openai_embedding import OpenAICompatibleEmbedding


class AzureEmbedding(OpenAICompatibleEmbedding):
    """Azure OpenAI provider using deployment-scoped embedding API."""

    provider_name = "azure"
    api_key_env_var = "AZURE_OPENAI_API_KEY"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        normalized = dict(config or {})
        if "model" not in normalized and normalized.get("deployment_name"):
            normalized["model"] = normalized["deployment_name"]
        if "base_url" not in normalized and normalized.get("azure_endpoint"):
            normalized["base_url"] = normalized["azure_endpoint"]

        super().__init__(normalized)
        self.deployment_name = self.model
        self.api_version = str(self.config.get("api_version", "2024-10-21")).strip()
        if not self.api_version:
            raise ValueError("azure config missing api_version")

    def _build_url(self) -> str:
        deployment = quote(self.deployment_name, safe="")
        query = urlencode({"api-version": self.api_version})
        return (
            f"{self.base_url.rstrip('/')}/openai/deployments/"
            f"{deployment}/embeddings?{query}"
        )

    def _build_headers(self) -> dict[str, str]:
        return {"api-key": self.api_key, "Content-Type": "application/json"}

    def _optional_payload_fields(self) -> dict[str, Any]:
        return {}

    def embed(self, texts: list[str], trace=None) -> list[list[float]]:
        return super().embed(texts, trace=trace)


__all__ = ["AzureEmbedding"]
