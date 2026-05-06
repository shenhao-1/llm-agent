"""Azure OpenAI chat completion provider."""

from __future__ import annotations

from typing import Any, Mapping
from urllib.parse import quote, urlencode

from .openai_llm import OpenAICompatibleLLM


class AzureLLM(OpenAICompatibleLLM):
    """Azure OpenAI provider using deployment-scoped chat completions."""

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
            f"{deployment}/chat/completions?{query}"
        )

    def _build_headers(self) -> dict[str, str]:
        return {"api-key": self.api_key, "Content-Type": "application/json"}

    def _build_payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload: dict[str, Any] = {"messages": messages}
        payload.update(self._optional_payload_fields())
        return payload


__all__ = ["AzureLLM"]
