"""Unit tests for Ollama LLM provider."""

from __future__ import annotations

import io
import json
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

from libs.llm.llm_factory import LLMFactory
from libs.llm.ollama_llm import OllamaLLM


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
def test_factory_routes_ollama_provider() -> None:
    llm = LLMFactory.create(
        {
            "llm": {
                "provider": "ollama",
                "model": "llama3.1:8b",
            }
        }
    )

    assert isinstance(llm, OllamaLLM)
    assert llm.base_url == "http://localhost:11434"


@pytest.mark.unit
def test_ollama_llm_posts_expected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> DummyResponse:
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return DummyResponse({"message": {"content": "ollama reply"}})

    monkeypatch.setattr("libs.llm.ollama_llm.urlopen", fake_urlopen)

    llm = OllamaLLM(
        {
            "provider": "ollama",
            "model": "llama3.1:8b",
            "options": {"temperature": 0.1},
        }
    )

    result = llm.chat([{"role": "user", "content": "hi"}])

    assert result == "ollama reply"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["body"] == {
        "model": "llama3.1:8b",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "options": {"temperature": 0.1},
    }
    assert captured["timeout"] == 30.0


@pytest.mark.unit
def test_ollama_llm_surfaces_connection_failure_without_sensitive_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: Any, timeout: float) -> DummyResponse:
        raise URLError("connection refused")

    monkeypatch.setattr("libs.llm.ollama_llm.urlopen", fake_urlopen)

    llm = OllamaLLM(
        {
            "provider": "ollama",
            "model": "llama3.1:8b",
            "base_url": "http://127.0.0.1:11434",
        }
    )

    with pytest.raises(
        RuntimeError,
        match=r"ollama request failed \(URLError\): connection refused",
    ) as exc_info:
        llm.chat([{"role": "user", "content": "hi"}])

    assert "127.0.0.1:11434" not in str(exc_info.value)


@pytest.mark.unit
def test_ollama_llm_surfaces_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, timeout: float) -> DummyResponse:
        raise HTTPError(
            url=request.full_url,
            code=500,
            msg="Server Error",
            hdrs=None,
            fp=io.BytesIO(b'{"error": "model not found"}'),
        )

    monkeypatch.setattr("libs.llm.ollama_llm.urlopen", fake_urlopen)

    llm = OllamaLLM({"provider": "ollama", "model": "missing-model"})

    with pytest.raises(RuntimeError, match=r"ollama request failed \(HTTP 500\): model not found"):
        llm.chat([{"role": "user", "content": "hi"}])
