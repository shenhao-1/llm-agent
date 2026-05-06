"""Unit tests for Ollama embedding provider."""

from __future__ import annotations

import io
import json
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.ollama_embedding import OllamaEmbedding


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
    """Keep embedding factory registry isolated between tests."""
    original = dict(EmbeddingFactory._providers)
    EmbeddingFactory._providers.clear()
    yield
    EmbeddingFactory._providers = original


@pytest.mark.unit
def test_factory_routes_ollama_embedding_provider() -> None:
    embedding = EmbeddingFactory.create(
        {
            "embedding": {
                "provider": "ollama",
                "model": "nomic-embed-text",
            }
        }
    )

    assert isinstance(embedding, OllamaEmbedding)
    assert embedding.base_url == "http://localhost:11434"


@pytest.mark.unit
def test_ollama_embedding_posts_one_request_per_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_bodies: list[dict[str, Any]] = []
    captured_urls: list[str] = []

    responses = iter(
        [
            DummyResponse({"embedding": [0.1, 0.2]}),
            DummyResponse({"embedding": [0.3, 0.4]}),
        ]
    )

    def fake_urlopen(request: Any, timeout: float) -> DummyResponse:
        captured_urls.append(request.full_url)
        captured_bodies.append(json.loads(request.data.decode("utf-8")))
        return next(responses)

    monkeypatch.setattr("libs.embedding.ollama_embedding.urlopen", fake_urlopen)

    embedding = OllamaEmbedding(
        {
            "provider": "ollama",
            "model": "nomic-embed-text",
            "options": {"num_ctx": 2048},
        }
    )

    vectors = embedding.embed(["alpha", "beta"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert captured_urls == [
        "http://localhost:11434/api/embeddings",
        "http://localhost:11434/api/embeddings",
    ]
    assert captured_bodies == [
        {
            "model": "nomic-embed-text",
            "prompt": "alpha",
            "options": {"num_ctx": 2048},
        },
        {
            "model": "nomic-embed-text",
            "prompt": "beta",
            "options": {"num_ctx": 2048},
        },
    ]


@pytest.mark.unit
def test_ollama_embedding_rejects_overlong_input() -> None:
    embedding = OllamaEmbedding(
        {
            "provider": "ollama",
            "model": "nomic-embed-text",
            "max_input_length": 5,
        }
    )

    with pytest.raises(ValueError, match=r"ollama texts\[0\] exceeds max_input_length=5"):
        embedding.embed(["too-long"])


@pytest.mark.unit
def test_ollama_embedding_surfaces_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: Any, timeout: float) -> DummyResponse:
        raise URLError("connection refused")

    monkeypatch.setattr("libs.embedding.ollama_embedding.urlopen", fake_urlopen)

    embedding = OllamaEmbedding({"provider": "ollama", "model": "nomic-embed-text"})

    with pytest.raises(
        RuntimeError,
        match=r"ollama request failed \(URLError\): connection refused",
    ):
        embedding.embed(["alpha"])


@pytest.mark.unit
def test_ollama_embedding_surfaces_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, timeout: float) -> DummyResponse:
        raise HTTPError(
            url=request.full_url,
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=io.BytesIO(b'{"error": "model not found"}'),
        )

    monkeypatch.setattr("libs.embedding.ollama_embedding.urlopen", fake_urlopen)

    embedding = OllamaEmbedding({"provider": "ollama", "model": "missing-model"})

    with pytest.raises(RuntimeError, match=r"ollama request failed \(HTTP 404\): model not found"):
        embedding.embed(["alpha"])
