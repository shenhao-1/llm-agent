"""Unit tests for shared core data contracts.  单元测试共享核心数据契约。"""

from __future__ import annotations

import json

import pytest

from core import Chunk, ChunkRecord, Document, image_placeholder


def test_document_serializes_to_stable_dict_and_json() -> None:
    document = Document(
        id="doc-1",
        text="Intro\n[IMAGE: img-1]\nBody",
        metadata={
            "source_path": "docs/sample.pdf",
            "title": "Sample",
            "images": [
                {
                    "id": "img-1",
                    "path": "data/images/default/img-1.png",
                    "page": 0,
                    "text_offset": 6,
                    "text_length": len("[IMAGE: img-1]"),
                    "position": {"x0": 1, "y0": 2, "x1": 3, "y1": 4},
                }
            ],
        },
    )

    payload = document.to_dict()

    assert list(payload) == ["id", "text", "metadata"]
    assert payload["metadata"]["source_path"] == "docs/sample.pdf"
    assert payload["metadata"]["images"][0]["id"] == "img-1"
    assert json.loads(json.dumps(payload)) == payload


def test_document_round_trips_from_dict() -> None:
    payload = {
        "id": "doc-1",
        "text": "content",
        "metadata": {"source_path": "docs/sample.md", "language": "en"},
    }

    document = Document.from_dict(payload)

    assert document.to_dict() == payload


def test_document_requires_source_path_metadata() -> None:
    with pytest.raises(ValueError, match=r"Document\.metadata\.source_path"):
        Document(id="doc-1", text="content", metadata={})


def test_document_validates_image_metadata_shape() -> None:
    with pytest.raises(ValueError, match=r"Document\.metadata\.images\[0\]\.text_offset"):
        Document(
            id="doc-1",
            text=image_placeholder("img-1"),
            metadata={
                "source_path": "docs/sample.pdf",
                "images": [
                    {
                        "id": "img-1",
                        "path": "data/images/default/img-1.png",
                        "text_offset": -1,
                        "text_length": 14,
                    }
                ],
            },
        )


def test_image_placeholder_uses_canonical_format() -> None:
    assert image_placeholder("docabc_0_001") == "[IMAGE: docabc_0_001]"


def test_chunk_serializes_offsets_and_source_ref() -> None:
    chunk = Chunk(
        id="doc-1_0000_abcd1234",
        text="chunk text",
        metadata={"source_path": "docs/sample.md", "doc_id": "doc-1"},
        start_offset=5,
        end_offset=15,
        source_ref="doc-1",
    )

    payload = chunk.to_dict()

    assert list(payload) == [
        "id",
        "text",
        "metadata",
        "start_offset",
        "end_offset",
        "source_ref",
    ]
    assert Chunk.from_dict(payload) == chunk


def test_chunk_rejects_invalid_offset_range() -> None:
    with pytest.raises(ValueError, match="end_offset"):
        Chunk(
            id="chunk-1",
            text="chunk text",
            metadata={"source_path": "docs/sample.md"},
            start_offset=10,
            end_offset=5,
        )


def test_chunk_record_serializes_dense_and_sparse_vectors() -> None:
    record = ChunkRecord(
        id="chunk-1",
        text="chunk text",
        metadata={"source_path": "docs/sample.md"},
        dense_vector=[0.1, 0.2],
        sparse_vector={"chunk": 1.0, "text": 0.5},
    )

    payload = record.to_dict()

    assert list(payload) == [
        "id",
        "text",
        "metadata",
        "dense_vector",
        "sparse_vector",
    ]
    assert json.loads(json.dumps(payload)) == payload
    assert ChunkRecord.from_dict(payload) == record


def test_chunk_record_accepts_missing_vectors_for_later_pipeline_stages() -> None:
    record = ChunkRecord(
        id="chunk-1",
        text="chunk text",
        metadata={"source_path": "docs/sample.md"},
    )

    assert record.to_dict()["dense_vector"] is None
    assert record.to_dict()["sparse_vector"] is None


def test_chunk_record_rejects_non_numeric_vector_values() -> None:
    with pytest.raises(ValueError, match=r"dense_vector\[1\]"):
        ChunkRecord(
            id="chunk-1",
            text="chunk text",
            metadata={"source_path": "docs/sample.md"},
            dense_vector=[0.1, "bad"],
        )
