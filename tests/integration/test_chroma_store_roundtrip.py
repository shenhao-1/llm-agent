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
