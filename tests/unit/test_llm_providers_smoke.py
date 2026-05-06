"""Smoke tests for OpenAI-compatible LLM providers."""

from __future__ import annotations

import io
import json
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

from libs.llm.azure_llm import AzureLLM
from libs.llm.deepseek_llm import DeepSeekLLM
from libs.llm.llm_factory import LLMFactory
from libs.llm.openai_llm import OpenAILLM


class DummyResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "DummyResponse":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


@pytest.fixture(autouse=True)
def reset_factory_registry() -> Any:
    """Keep LLM factory registry isolated between tests."""
    original = dict(LLMFactory._providers)
    LLMFactory._providers.clear()
    yield
    LLMFactory._providers = original


@pytest.mark.unit
def test_factory_routes_openai_provider() -> None:
    llm = LLMFactory.create(
        {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "test-key",
            }
        }
    )

    assert isinstance(llm, OpenAILLM)


@pytest.mark.unit
def test_factory_routes_azure_provider() -> None:
    llm = LLMFactory.create(
        {
            "llm": {
                "provider": "azure",
                "deployment_name": "gpt-4o-mini",
                "azure_endpoint": "https://example.openai.azure.com",
                "api_key": "azure-key",
                "api_version": "2024-10-21",
            }
        }
    )

    assert isinstance(llm, AzureLLM)


@pytest.mark.unit
def test_factory_routes_deepseek_provider() -> None:
    llm = LLMFactory.create(
        {
            "llm": {
                "provider": "deepseek",
                "model": "deepseek-chat",
                "api_key": "deepseek-key",
            }
        }
    )

    assert isinstance(llm, DeepSeekLLM)


@pytest.mark.unit
def test_openai_llm_posts_expected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> DummyResponse:
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return DummyResponse({"choices": [{"message": {"content": "hello world"}}]})

    monkeypatch.setattr("libs.llm.openai_llm.urlopen", fake_urlopen)

    llm = OpenAILLM(
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "secret-key",
            "temperature": 0.2,
        }
    )

    result = llm.chat([{"role": "user", "content": "hi"}])

    assert result == "hello world"
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["body"]["model"] == "gpt-4o-mini"
    assert captured["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["body"]["temperature"] == 0.2
    assert captured["timeout"] == 30.0
    assert captured["headers"]["Authorization"] == "Bearer secret-key"


@pytest.mark.unit
def test_azure_llm_uses_deployment_scoped_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> DummyResponse:
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return DummyResponse({"choices": [{"message": {"content": "azure reply"}}]})

    monkeypatch.setattr("libs.llm.openai_llm.urlopen", fake_urlopen)

    llm = AzureLLM(
        {
            "provider": "azure",
            "deployment_name": "gpt-4o-mini",
            "azure_endpoint": "https://example.openai.azure.com",
            "api_key": "azure-secret",
            "api_version": "2024-10-21",
        }
    )

    result = llm.chat([{"role": "user", "content": "hello"}])

    assert result == "azure reply"
    assert (
        captured["url"]
        == "https://example.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-10-21"
    )
    assert captured["body"] == {"messages": [{"role": "user", "content": "hello"}]}
    headers = {key.lower(): value for key, value in captured["headers"].items()}
    assert headers["api-key"] == "azure-secret"


@pytest.mark.unit
def test_provider_error_includes_provider_name_and_http_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: Any, timeout: float) -> DummyResponse:
        raise HTTPError(
            url=request.full_url,
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b'{"error": {"message": "bad api key"}}'),
        )

    monkeypatch.setattr("libs.llm.openai_llm.urlopen", fake_urlopen)

    llm = OpenAILLM(
        {"provider": "openai", "model": "gpt-4o-mini", "api_key": "bad-key"}
    )

    with pytest.raises(RuntimeError, match=r"openai request failed \(HTTP 401\): bad api key"):
        llm.chat([{"role": "user", "content": "hi"}])


@pytest.mark.unit
def test_provider_error_includes_provider_name_and_error_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: Any, timeout: float) -> DummyResponse:
        raise URLError("connection refused")

    monkeypatch.setattr("libs.llm.openai_llm.urlopen", fake_urlopen)

    llm = DeepSeekLLM(
        {"provider": "deepseek", "model": "deepseek-chat", "api_key": "key"}
    )

    with pytest.raises(
        RuntimeError,
        match=r"deepseek request failed \(URLError\): connection refused",
    ):
        llm.chat([{"role": "user", "content": "hi"}])
