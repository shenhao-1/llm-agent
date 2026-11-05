"""Integration tests for Chroma store roundtrip behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from libs.vector_store.chroma_store import ChromaStore
from libs.vector_store.vector_store_factory import VectorStoreFactory


@pytest.fixture(autouse=True)
def reset_factory_registry():
    original = dict(VectorStoreFactory._providers)
    VectorStoreFactory._providers.clear()
    yield
    VectorStoreFactory._providers = original


@pytest.mark.integration
def test_factory_routes_chroma_provider(tmp_path: Path) -> None:
    store = VectorStoreFactory.create(
        {
            "vector_store": {
                "provider": "chroma",
                "persist_directory": str(tmp_path / "chroma"),
                "collection_name": "factory-test",
            }
        }
    )

    assert isinstance(store, ChromaStore)


@pytest.mark.integration
def test_chroma_store_upsert_and_query_roundtrip(tmp_path: Path) -> None:
    store = ChromaStore(
        {
            "provider": "chroma",
            "persist_directory": str(tmp_path / "chroma"),
            "collection_name": "roundtrip",
        }
    )

    store.upsert(
        [
            {
                "id": "chunk-1",
                "text": "alpha vector",
                "vector": [1.0, 0.0],
                "metadata": {"collection": "docs", "topic": "alpha"},
            },
            {
                "id": "chunk-2",
                "text": "beta vector",
                "vector": [0.0, 1.0],
                "metadata": {"collection": "docs", "topic": "beta"},
            },
            {
                "id": "chunk-3",
                "text": "gamma vector",
                "vector": [0.8, 0.2],
                "metadata": {"collection": "notes", "topic": "alpha"},
            },
        ]
    )

    results = store.query([1.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0]["id"] == "chunk-1"
    assert results[0]["text"] == "alpha vector"
    assert results[0]["metadata"]["collection"] == "docs"


@pytest.mark.integration
def test_chroma_store_preserves_nested_image_metadata(tmp_path: Path) -> None:
    store = ChromaStore({"persist_directory": str(tmp_path / "chroma")})
    images = [{"id": "figure-1", "path": "images/figure.png"}]
    store.upsert([{"id": "chunk", "text": "figure", "vector": [1.0, 0.0],
                   "metadata": {"collection": "docs", "images": images, "tags": ["diagram"]}}])

    assert store.get_by_ids(["chunk"])[0]["metadata"]["images"] == images
    assert store.query([1.0, 0.0], 1)[0]["metadata"]["tags"] == ["diagram"]


@pytest.mark.integration
def test_chroma_store_respects_top_k(tmp_path: Path) -> None:
    store = ChromaStore(
        {
            "provider": "chroma",
            "persist_directory": str(tmp_path / "chroma"),
            "collection_name": "top-k",
        }
    )

    store.upsert(
        [
            {"id": "a", "text": "A", "vector": [1.0, 0.0], "metadata": {"kind": "x"}},
            {"id": "b", "text": "B", "vector": [0.9, 0.1], "metadata": {"kind": "x"}},
            {"id": "c", "text": "C", "vector": [0.8, 0.2], "metadata": {"kind": "x"}},
        ]
    )

    results = store.query([1.0, 0.0], top_k=1)

    assert [item["id"] for item in results] == ["a"]


@pytest.mark.integration
def test_chroma_store_supports_metadata_filters(tmp_path: Path) -> None:
    store = ChromaStore(
        {
            "provider": "chroma",
            "persist_directory": str(tmp_path / "chroma"),
            "collection_name": "filters",
        }
    )

    store.upsert(
        [
            {
                "id": "doc-1",
                "text": "docs alpha",
                "vector": [1.0, 0.0],
                "metadata": {"collection": "docs", "topic": "alpha"},
            },
            {
                "id": "doc-2",
                "text": "notes alpha",
                "vector": [1.0, 0.0],
                "metadata": {"collection": "notes", "topic": "alpha"},
            },
        ]
    )

    results = store.query([1.0, 0.0], top_k=5, filters={"collection": "notes"})

    assert [item["id"] for item in results] == ["doc-2"]


@pytest.mark.integration
def test_chroma_store_get_by_ids_returns_text_and_metadata(tmp_path: Path) -> None:
    store = ChromaStore(
        {
            "provider": "chroma",
            "persist_directory": str(tmp_path / "chroma"),
            "collection_name": "get-by-ids",
        }
    )
    store.upsert(
        [
            {
                "id": "chunk-1",
                "text": "first text",
                "vector": [1.0, 0.0],
                "metadata": {"source_path": "docs/one.md"},
            },
            {
                "id": "chunk-2",
                "text": "second text",
                "vector": [0.0, 1.0],
                "metadata": {"source_path": "docs/two.md"},
            },
        ]
    )

    records = store.get_by_ids(["chunk-2", "missing", "chunk-1"])

    assert [record["id"] for record in records] == ["chunk-2", "chunk-1"]
    assert records[0]["text"] == "second text"
    assert records[0]["metadata"] == {"source_path": "docs/two.md"}


@pytest.mark.integration
def test_chroma_store_returns_collection_stats(tmp_path: Path) -> None:
    store = ChromaStore(
        {
            "provider": "chroma",
            "persist_directory": str(tmp_path / "chroma"),
            "collection_name": "stats",
            "distance": "cosine",
        }
    )
    store.upsert(
        [
            {
                "id": "chunk-1",
                "text": "stats text",
                "vector": [1.0, 0.0],
                "metadata": {"source_path": "docs/stats.md"},
            }
        ]
    )

    stats = store.get_collection_stats()

    assert stats == {
        "provider": "chroma",
        "collection_name": "stats",
        "persist_directory": str(tmp_path / "chroma"),
        "distance": "cosine",
        "record_count": 1,
    }


@pytest.mark.integration
def test_chroma_store_delete_by_metadata_removes_matching_records(tmp_path: Path) -> None:
    store = ChromaStore(
        {
            "provider": "chroma",
            "persist_directory": str(tmp_path / "chroma"),
            "collection_name": "delete-metadata",
        }
    )
    store.upsert(
        [
            {
                "id": "chunk-1",
                "text": "delete me",
                "vector": [1.0, 0.0],
                "metadata": {"source_path": "docs/a.pdf", "collection": "docs"},
            },
            {
                "id": "chunk-2",
                "text": "keep me",
                "vector": [0.0, 1.0],
                "metadata": {"source_path": "docs/b.pdf", "collection": "docs"},
            },
        ]
    )

    deleted = store.delete_by_metadata(
        {"source_path": "docs/a.pdf", "collection": "docs"}
    )

    assert deleted == 1
    assert store.get_by_ids(["chunk-1", "chunk-2"]) == [
        {
            "id": "chunk-2",
            "text": "keep me",
            "metadata": {"source_path": "docs/b.pdf", "collection": "docs"},
        }
    ]


@pytest.mark.integration
def test_chroma_store_get_by_metadata_returns_matching_records(tmp_path: Path) -> None:
    store = ChromaStore(
        {
            "provider": "chroma",
            "persist_directory": str(tmp_path / "chroma"),
            "collection_name": "get-metadata",
        }
    )
    store.upsert(
        [
            {
                "id": "chunk-1",
                "text": "docs text",
                "vector": [1.0, 0.0],
                "metadata": {"collection": "docs", "source_path": "docs/a.pdf"},
            },
            {
                "id": "chunk-2",
                "text": "notes text",
                "vector": [0.0, 1.0],
                "metadata": {"collection": "notes", "source_path": "notes/b.pdf"},
            },
        ]
    )

    records = store.get_by_metadata({"collection": "docs"})

    assert records == [
        {
            "id": "chunk-1",
            "text": "docs text",
            "metadata": {"collection": "docs", "source_path": "docs/a.pdf"},
        }
    ]
