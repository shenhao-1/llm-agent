"""Unit tests for Document to Chunk adaptation."""

from __future__ import annotations

from typing import Any

import pytest

from core import Chunk, Document
from ingestion.chunking import DocumentChunker
from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import SplitterFactory


class FakeSplitter(BaseSplitter):
    """Configurable splitter for DocumentChunker contract tests."""

    def split_text(self, text: str, trace: Any | None = None) -> list[str]:
        self.validate_text(text)
        if "|" in text:
            return [part.strip() for part in text.split("|") if part.strip()]

        chunk_size = int(self.config.get("chunk_size", len(text)))
        return [
            text[index : index + chunk_size]
            for index in range(0, len(text), chunk_size)
            if text[index : index + chunk_size].strip()
        ]


@pytest.fixture(autouse=True)
def reset_splitter_registry() -> Any:
    original = dict(SplitterFactory._providers)
    SplitterFactory._providers = {"fake": FakeSplitter}
    yield
    SplitterFactory._providers = original


def test_split_document_uses_splitter_factory_configuration() -> None:
    document = _document(text="abcdef")

    chunks = DocumentChunker({"splitter": {"provider": "fake", "chunk_size": 2}}).split_document(
        document
    )

    assert [chunk.text for chunk in chunks] == ["ab", "cd", "ef"]


def test_split_document_outputs_contract_chunks_with_offsets_and_source_ref() -> None:
    document = _document(text="alpha | beta | gamma")

    chunks = DocumentChunker({"splitter": {"provider": "fake"}}).split_document(document)

    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert [chunk.text for chunk in chunks] == ["alpha", "beta", "gamma"]
    assert [chunk.source_ref for chunk in chunks] == ["doc-1", "doc-1", "doc-1"]
    assert [(chunk.start_offset, chunk.end_offset) for chunk in chunks] == [
        (0, 5),
        (8, 12),
        (15, 20),
    ]
    assert all(chunk.to_dict()["metadata"]["source_path"] == "docs/sample.md" for chunk in chunks)


def test_chunk_ids_are_unique_and_deterministic() -> None:
    document = _document(text="alpha | beta | gamma")
    chunker = DocumentChunker({"splitter": {"provider": "fake"}})

    first_ids = [chunk.id for chunk in chunker.split_document(document)]
    second_ids = [chunk.id for chunk in chunker.split_document(document)]

    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids))
    assert first_ids[0].startswith("doc-1_0000_")


def test_chunk_metadata_inherits_document_metadata_and_adds_index() -> None:
    document = _document(
        text="alpha | beta",
        metadata={
            "source_path": "docs/sample.md",
            "title": "Sample",
            "language": "en",
        },
    )

    chunks = DocumentChunker({"splitter": {"provider": "fake"}}).split_document(document)

    assert chunks[0].metadata == {
        "source_path": "docs/sample.md",
        "title": "Sample",
        "language": "en",
        "chunk_index": 0,
    }
    assert chunks[1].metadata["chunk_index"] == 1


def test_image_metadata_is_distributed_only_to_referencing_chunk() -> None:
    text = "before [IMAGE: img-1] | middle | after [IMAGE: img-2]"
    document = _document(
        text=text,
        metadata={
            "source_path": "docs/sample.pdf",
            "images": [
                _image_ref("img-1", text_offset=7),
                _image_ref("img-2", text_offset=38),
            ],
        },
    )

    chunks = DocumentChunker({"splitter": {"provider": "fake"}}).split_document(document)

    assert chunks[0].metadata["image_refs"] == ["img-1"]
    assert [image["id"] for image in chunks[0].metadata["images"]] == ["img-1"]
    assert "images" not in chunks[1].metadata
    assert "image_refs" not in chunks[1].metadata
    assert chunks[2].metadata["image_refs"] == ["img-2"]
    assert [image["id"] for image in chunks[2].metadata["images"]] == ["img-2"]


def test_chunk_output_is_serializable_with_complete_fields() -> None:
    document = _document(text="alpha | beta")

    payload = DocumentChunker({"splitter": {"provider": "fake"}}).split_document(document)[0].to_dict()

    assert list(payload) == [
        "id",
        "text",
        "metadata",
        "start_offset",
        "end_offset",
        "source_ref",
    ]
    assert payload["metadata"]["chunk_index"] == 0


def test_split_document_rejects_non_document_input() -> None:
    with pytest.raises(TypeError, match="document must be a Document"):
        DocumentChunker({"splitter": {"provider": "fake"}}).split_document("bad")


def _document(
    text: str,
    metadata: dict[str, Any] | None = None,
) -> Document:
    return Document(
        id="doc-1",
        text=text,
        metadata=metadata or {"source_path": "docs/sample.md"},
    )


def _image_ref(image_id: str, text_offset: int) -> dict[str, Any]:
    return {
        "id": image_id,
        "path": f"data/images/default/{image_id}.png",
        "page": 0,
        "text_offset": text_offset,
        "text_length": len(f"[IMAGE: {image_id}]"),
        "position": {},
    }
