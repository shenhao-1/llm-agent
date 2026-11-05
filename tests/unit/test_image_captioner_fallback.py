"""Unit tests for image captioner fallback behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from core import Chunk
from core.trace import TraceContext
from ingestion.transform import ImageCaptioner
from libs.llm.base_vision_llm import BaseVisionLLM


class FakeVisionLLM(BaseVisionLLM):
    """Fake vision model that records calls and returns a caption."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self.calls: list[tuple[str, str | bytes]] = []

    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: Any | None = None,
    ) -> str:
        self.validate_text(text)
        self.validate_image_input(image_path)
        self.calls.append((text, image_path))
        return "A small architecture diagram."


class FailingVisionLLM(BaseVisionLLM):
    """Fake vision model that simulates provider failure."""

    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: Any | None = None,
    ) -> str:
        raise RuntimeError("vision provider failed")


def test_chunk_without_images_is_returned_unchanged() -> None:
    chunk = _chunk(metadata={"source_path": "docs/sample.md"})

    result = ImageCaptioner().transform([chunk])[0]

    assert result == chunk


def test_disabled_captioner_preserves_refs_and_marks_unprocessed() -> None:
    chunk = _chunk_with_image()
    vision = FakeVisionLLM()

    result = ImageCaptioner(
        {"ingestion": {"image_captioner": {"enabled": False}}},
        vision_llm=vision,
    ).transform([chunk])[0]

    assert result.metadata["image_refs"] == ["img-1"]
    assert result.metadata["has_unprocessed_images"] is True
    assert result.metadata["image_caption_status"] == "disabled"
    assert "image_captions" not in result.metadata
    assert vision.calls == []


def test_enabled_captioner_writes_caption_to_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake-image-bytes")
    chunk = _chunk_with_image(image_path=str(image_path))
    vision = FakeVisionLLM()

    result = ImageCaptioner(
        {"ingestion": {"image_captioner": {"enabled": True}}},
        vision_llm=vision,
    ).transform([chunk])[0]

    assert result.metadata["has_unprocessed_images"] is False
    assert result.metadata["image_caption_status"] == "success"
    assert result.metadata["image_captions"] == {
        "img-1": "A small architecture diagram."
    }
    assert result.metadata["images"][0]["caption"] == "A small architecture diagram."
    assert "A small architecture diagram." in result.text
    assert vision.calls[0][1] == str(image_path)
    assert "Chunk context" in vision.calls[0][0]
    assert "Chunk with [IMAGE: img-1]" in vision.calls[0][0]


def test_enabled_captioner_handles_provider_failure_without_raising(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake-image-bytes")
    chunk = _chunk_with_image(image_path=str(image_path))

    result = ImageCaptioner(
        {"ingestion": {"image_captioner": {"enabled": True}}},
        vision_llm=FailingVisionLLM(),
    ).transform([chunk])[0]

    assert result.metadata["has_unprocessed_images"] is True
    assert result.metadata["image_caption_status"] == "failed"
    assert "vision provider failed" in result.metadata["image_caption_errors"]["img-1"]
    assert "image_captions" not in result.metadata


def test_missing_image_metadata_is_reported_for_ref() -> None:
    chunk = _chunk(
        metadata={
            "source_path": "docs/sample.pdf",
            "image_refs": ["img-missing"],
            "images": [],
        }
    )

    result = ImageCaptioner(
        {"ingestion": {"image_captioner": {"enabled": True}}},
        vision_llm=FakeVisionLLM(),
    ).transform([chunk])[0]

    assert result.metadata["has_unprocessed_images"] is True
    assert result.metadata["image_caption_errors"]["img-missing"] == "image metadata not found"


def test_trace_records_image_captioner_stage(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake-image-bytes")
    trace = TraceContext()

    ImageCaptioner(
        {"ingestion": {"image_captioner": {"enabled": True}}},
        vision_llm=FakeVisionLLM(),
    ).transform([_chunk_with_image(image_path=str(image_path))], trace=trace)

    assert trace.stages[-1]["name"] == "transform"
    assert trace.stages[-1]["method"] == "image_captioner"
    assert trace.stages[-1]["details"]["captioned_images"] == 1


def test_transform_rejects_non_chunk_entries() -> None:
    with pytest.raises(ValueError, match=r"chunks\[0\]"):
        ImageCaptioner().transform(["bad"])


def _chunk(
    text: str = "Chunk text",
    metadata: dict[str, Any] | None = None,
) -> Chunk:
    return Chunk(
        id="chunk-1",
        text=text,
        metadata=metadata or {"source_path": "docs/sample.md"},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-1",
    )


def _chunk_with_image(image_path: str = "data/images/default/img-1.png") -> Chunk:
    text = "Chunk with [IMAGE: img-1]"
    return _chunk(
        text=text,
        metadata={
            "source_path": "docs/sample.pdf",
            "image_refs": ["img-1"],
            "images": [
                {
                    "id": "img-1",
                    "path": image_path,
                    "page": 0,
                    "text_offset": 11,
                    "text_length": len("[IMAGE: img-1]"),
                    "position": {},
                }
            ],
        },
    )
