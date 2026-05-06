"""DeepSeek chat completion provider."""

from __future__ import annotations

from .openai_llm import OpenAICompatibleLLM


class DeepSeekLLM(OpenAICompatibleLLM):
    """DeepSeek provider using the OpenAI-compatible chat API."""

    provider_name = "deepseek"
    default_base_url = "https://api.deepseek.com/v1"
    api_key_env_var = "DEEPSEEK_API_KEY"


__all__ = ["DeepSeekLLM"]
