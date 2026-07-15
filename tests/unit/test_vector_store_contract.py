"""Unit tests for vector store contract and factory routing."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from core.settings import Settings
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.chroma_store import ChromaStore
from libs.vector_store.vector_store_factory import VectorStoreFactory


class FakeVectorStore(BaseVectorStore):
    """Simple fake vector store used for contract and routing tests."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self._records: list[dict[str, Any]] = []

    def upsert(self, records: list[Mapping[str, Any]], trace: Any | None = None) -> None:
        self.validate_records(records)
        self._records = [dict(record) for record in records]

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        self.validate_query_args(vector, top_k, filters)
        return [dict(record) for record in self._records[:top_k]]

    def get_by_ids(
        self,
        ids: list[str],
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        self.validate_ids(ids)
        records_by_id = {str(record["id"]): record for record in self._records}
        return [dict(records_by_id[item_id]) for item_id in ids if item_id in records_by_id]

    def delete_by_metadata(self, filters: Mapping[str, Any]) -> int:
        if not isinstance(filters, Mapping) or not filters:
            raise ValueError("filters must be a non-empty mapping")
        before = len(self._records)
        self._records = [
            record
            for record in self._records
            if not _metadata_matches(record.get("metadata", {}), filters)
        ]
        return before - len(self._records)


class FakeChromaCollection:
    def __init__(self, ids: list[str]) -> None:
        self.ids = list(ids)
        self.deleted_ids: list[str] = []
        self.where: Any = None

    def get(self, where: Any = None):
        self.where = where
        return {"ids": list(self.ids)}

    def delete(self, ids: list[str]) -> None:
        self.deleted_ids.extend(ids)


@pytest.fixture(autouse=True)
def reset_factory_registry() -> Any:
    """Keep factory registry isolated between tests."""
    original = dict(VectorStoreFactory._providers)
    VectorStoreFactory._providers.clear()
    yield
    VectorStoreFactory._providers = original


@pytest.mark.unit
def test_create_routes_to_registered_provider_with_mapping_settings() -> None:
    VectorStoreFactory.register_provider("fake", FakeVectorStore)
    settings = {"vector_store": {"provider": "fake", "persist_directory": "tmp/db"}}

    instance = VectorStoreFactory.create(settings)

    assert isinstance(instance, FakeVectorStore)
    assert instance.config["persist_directory"] == "tmp/db"


@pytest.mark.unit
def test_create_routes_to_registered_provider_with_settings_dataclass() -> None:
    VectorStoreFactory.register_provider("fake", FakeVectorStore)
    settings = Settings(
        llm={"provider": "dummy"},
        embedding={"provider": "dummy"},
        vector_store={"provider": "fake"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"level": "INFO"},
    )

    instance = VectorStoreFactory.create(settings)

    assert isinstance(instance, FakeVectorStore)


@pytest.mark.unit
def test_create_raises_on_unknown_provider() -> None:
    settings = {"vector_store": {"provider": "not-registered"}}

    with pytest.raises(ValueError, match="Unknown vector store provider"):
        VectorStoreFactory.create(settings)


@pytest.mark.unit
def test_validate_records_accepts_minimal_valid_shape() -> None:
    FakeVectorStore().validate_records(
        [
            {
                "id": "chunk-1",
                "vector": [0.1, 0.2, 0.3],
                "metadata": {"source_path": "doc.md"},
            }
        ]
    )


@pytest.mark.unit
def test_validate_records_rejects_missing_id() -> None:
    with pytest.raises(ValueError, match=r"records\[0\]\.id"):
        FakeVectorStore().validate_records([{"vector": [0.1, 0.2]}])


@pytest.mark.unit
def test_validate_records_rejects_non_numeric_vector_entries() -> None:
    with pytest.raises(ValueError, match=r"records\[0\]\.vector\[1\]"):
        FakeVectorStore().validate_records(
            [{"id": "chunk-1", "vector": [0.1, "bad"]}]
        )


@pytest.mark.unit
def test_validate_query_args_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        FakeVectorStore().validate_query_args([0.1, 0.2], 0)


@pytest.mark.unit
def test_query_returns_up_to_top_k_records() -> None:
    store = FakeVectorStore()
    store.upsert(
        [
            {"id": "chunk-1", "vector": [1.0, 0.0]},
            {"id": "chunk-2", "vector": [0.0, 1.0]},
        ]
    )

    results = store.query([1.0, 0.0], top_k=1)

    assert [result["id"] for result in results] == ["chunk-1"]


@pytest.mark.unit
def test_get_by_ids_returns_matching_records_in_requested_order() -> None:
    store = FakeVectorStore()
    store.upsert(
        [
            {"id": "chunk-1", "vector": [1.0, 0.0], "text": "one"},
            {"id": "chunk-2", "vector": [0.0, 1.0], "text": "two"},
        ]
    )

    results = store.get_by_ids(["chunk-2", "missing", "chunk-1"])

    assert [result["id"] for result in results] == ["chunk-2", "chunk-1"]


@pytest.mark.unit
def test_validate_ids_rejects_invalid_ids() -> None:
    with pytest.raises(ValueError, match=r"ids\[0\]"):
        FakeVectorStore().validate_ids([""])


@pytest.mark.unit
def test_delete_by_metadata_removes_matching_records_and_returns_count() -> None:
    store = FakeVectorStore()
    store.upsert(
        [
            {
                "id": "chunk-1",
                "vector": [1.0, 0.0],
                "metadata": {"source_path": "docs/a.pdf", "collection": "docs"},
            },
            {
                "id": "chunk-2",
                "vector": [0.0, 1.0],
                "metadata": {"source_path": "docs/b.pdf", "collection": "docs"},
            },
        ]
    )

    deleted = store.delete_by_metadata({"source_path": "docs/a.pdf"})

    assert deleted == 1
    assert [record["id"] for record in store.query([1.0], top_k=10)] == ["chunk-2"]


@pytest.mark.unit
def test_delete_by_metadata_rejects_empty_filters() -> None:
    with pytest.raises(ValueError, match="filters must be a non-empty mapping"):
        FakeVectorStore().delete_by_metadata({})


@pytest.mark.unit
def test_chroma_delete_by_metadata_returns_zero_without_matches() -> None:
    store = object.__new__(ChromaStore)
    store._collection = FakeChromaCollection([])

    deleted = store.delete_by_metadata({"source_path": "docs/missing.pdf"})

    assert deleted == 0
    assert store._collection.deleted_ids == []


@pytest.mark.unit
def test_chroma_delete_by_metadata_deletes_all_matching_ids_with_and_filter() -> None:
    store = object.__new__(ChromaStore)
    store._collection = FakeChromaCollection(["chunk-1", "chunk-2"])

    deleted = store.delete_by_metadata(
        {"source_path": "docs/a.pdf", "collection": "docs"}
    )

    assert deleted == 2
    assert store._collection.where == {
        "$and": [{"source_path": "docs/a.pdf"}, {"collection": "docs"}]
    }
    assert store._collection.deleted_ids == ["chunk-1", "chunk-2"]


def _metadata_matches(metadata: Any, filters: Mapping[str, Any]) -> bool:
    if not isinstance(metadata, Mapping):
        return False
    return all(metadata.get(key) == value for key, value in filters.items())
