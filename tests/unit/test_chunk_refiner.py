"""Unit tests for chunk refinement transforms."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from core import Chunk
from core.trace import TraceContext
from ingestion.transform import BaseTransform, ChunkRefiner
from libs.llm.base_llm import BaseLLM


class FakeLLM(BaseLLM):
    """Fake LLM used to verify prompt calls."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.validate_messages(messages)
        self.calls.append(messages)
        return "LLM refined text"


class FailingLLM(BaseLLM):
    """Fake LLM that simulates provider failure."""

    def chat(self, messages: list[dict[str, str]]) -> str:
        raise RuntimeError("provider unavailable")


class ExplodingRefiner(ChunkRefiner):
    """Refiner that simulates an unexpected per-chunk processing error."""

    def _rule_based_refine(self, text: str) -> str:
        if "explode" in text:
            raise RuntimeError("bad chunk")
        return super()._rule_based_refine(text)


def test_base_transform_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseTransform()


@pytest.mark.parametrize("case", json.loads(Path("tests/fixtures/noisy_chunks.json").read_text()))
def test_rule_based_refine_handles_noisy_fixtures(case: dict[str, Any]) -> None:
    refined = ChunkRefiner()._rule_based_refine(case["input"])

    if "expected" in case:
        assert refined == case["expected"]
    if "expected_contains" in case:
        assert case["expected_contains"] in refined
    for forbidden in case.get("expected_not_contains", []):
        assert forbidden not in refined


def test_transform_marks_rule_refinement_in_metadata() -> None:
    chunks = [_chunk("Confidential\n\nUseful text\n\nPage 1")]

    refined = ChunkRefiner().transform(chunks)

    assert refined[0].text == "Useful text"
    assert refined[0].metadata["refined_by"] == "rule"
    assert refined[0].id == chunks[0].id
    assert refined[0].source_ref == chunks[0].source_ref


def test_transform_with_mock_llm_returns_rewritten_text_and_marks_metadata() -> None:
    llm = FakeLLM()
    chunk = _chunk("Noisy text")

    refined = ChunkRefiner(
        {"ingestion": {"chunk_refiner": {"use_llm": True}}},
        llm=llm,
    ).transform([chunk])

    assert refined[0].text == "LLM refined text"
    assert refined[0].metadata["refined_by"] == "llm"
    assert "Noisy text" in llm.calls[0][1]["content"]


def test_llm_failure_falls_back_to_rule_result() -> None:
    chunk = _chunk("Confidential\n\nUseful text")

    refined = ChunkRefiner(
        {"ingestion": {"chunk_refiner": {"use_llm": True}}},
        llm=FailingLLM(),
    ).transform([chunk])

    assert refined[0].text == "Useful text"
    assert refined[0].metadata["refined_by"] == "rule"
    assert "provider unavailable" in refined[0].metadata["fallback_reason"]


def test_use_llm_false_does_not_call_passed_llm() -> None:
    llm = FakeLLM()

    refined = ChunkRefiner(
        {"ingestion": {"chunk_refiner": {"use_llm": False}}},
        llm=llm,
    ).transform([_chunk("Useful text")])

    assert refined[0].metadata["refined_by"] == "rule"
    assert llm.calls == []


def test_single_chunk_exception_preserves_original_and_continues() -> None:
    chunks = [_chunk("explode"), _chunk("Confidential\n\nkeep", chunk_id="chunk-2")]

    refined = ExplodingRefiner().transform(chunks)

    assert refined[0].text == "explode"
    assert refined[0].metadata["refined_by"] == "original"
    assert refined[0].metadata["refine_error"] == "bad chunk"
    assert refined[1].text == "keep"
    assert refined[1].metadata["refined_by"] == "rule"


def test_trace_records_transform_stage() -> None:
    trace = TraceContext()

    ChunkRefiner().transform([_chunk("Useful text")], trace=trace)

    assert trace.stages[-1]["name"] == "transform"
    assert trace.stages[-1]["method"] == "chunk_refiner"
    assert trace.stages[-1]["details"]["chunk_count"] == 1


def test_prompt_loader_appends_text_placeholder_when_missing(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Refine this chunk.", encoding="utf-8")

    template = ChunkRefiner._load_prompt(prompt_path)

    assert "{text}" in template


def test_transform_rejects_non_chunk_entries() -> None:
    with pytest.raises(ValueError, match=r"chunks\[0\]"):
        ChunkRefiner().transform(["bad"])


def _chunk(text: str, chunk_id: str = "chunk-1") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": "docs/sample.md"},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-1",
    )
