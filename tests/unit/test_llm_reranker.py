"""Unit tests for LLM reranker provider."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from libs.llm.base_llm import BaseLLM
from libs.reranker.llm_reranker import LLMReranker
from libs.reranker.reranker_factory import RerankerFactory


class FakeLLM(BaseLLM):
    def __init__(self, response: str) -> None:
        super().__init__({})
        self.response = response
        self.last_messages: list[dict[str, str]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.last_messages = messages
        return self.response


@pytest.fixture(autouse=True)
def reset_factory_registry():
    original = dict(RerankerFactory._providers)
    RerankerFactory._providers = {"none": RerankerFactory._providers.get("none")}
    yield
    RerankerFactory._providers = original


@pytest.fixture
def prompt_file(tmp_path: Path) -> Path:
    path = tmp_path / "rerank.txt"
    path.write_text(
        "Rank the candidates for query: {query}\nCandidates:\n{candidates}",
        encoding="utf-8",
    )
    return path


@pytest.mark.unit
def test_factory_routes_llm_provider(prompt_file: Path) -> None:
    reranker = RerankerFactory.create(
        {
            "rerank": {
                "provider": "llm",
                "model": "gpt-4o-mini",
                "api_key": "test-key",
                "prompt_path": str(prompt_file),
                "llm": {"provider": "openai", "model": "gpt-4o-mini", "api_key": "test-key"},
            }
        }
    )

    assert isinstance(reranker, LLMReranker)


@pytest.mark.unit
def test_llm_reranker_returns_ranked_candidates(prompt_file: Path) -> None:
    llm = FakeLLM(json.dumps({"ranked_ids": ["chunk-2", "chunk-1"]}))
    reranker = LLMReranker(
        {"prompt_path": str(prompt_file), "provider": "llm", "model": "ignored"},
        llm=llm,
    )

    results = reranker.rerank(
        "find beta",
        [
            {"id": "chunk-1", "text": "alpha"},
            {"id": "chunk-2", "text": "beta"},
        ],
    )

    assert [item["id"] for item in results] == ["chunk-2", "chunk-1"]
    assert llm.last_messages
    assert "find beta" in llm.last_messages[0]["content"]
    assert '"id": "chunk-1"' in llm.last_messages[0]["content"]


@pytest.mark.unit
def test_llm_reranker_rejects_non_json_response(prompt_file: Path) -> None:
    reranker = LLMReranker(
        {"prompt_path": str(prompt_file), "provider": "llm", "model": "ignored"},
        llm=FakeLLM("not-json"),
    )

    with pytest.raises(ValueError, match="llm reranker response must be valid JSON"):
        reranker.rerank("find beta", [{"id": "chunk-1", "text": "alpha"}])


@pytest.mark.unit
def test_llm_reranker_rejects_mismatched_ids(prompt_file: Path) -> None:
    reranker = LLMReranker(
        {"prompt_path": str(prompt_file), "provider": "llm", "model": "ignored"},
        llm=FakeLLM(json.dumps({"ranked_ids": ["chunk-1", "missing"]})),
    )

    with pytest.raises(ValueError, match="llm reranker ranked_ids must exactly match candidate ids"):
        reranker.rerank(
            "find beta",
            [
                {"id": "chunk-1", "text": "alpha"},
                {"id": "chunk-2", "text": "beta"},
            ],
        )
