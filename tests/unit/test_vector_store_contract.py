"""Unit tests for vector store contract and factory routing."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from core.settings import Settings
from libs.vector_store.base_vector_store import BaseVectorStore
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
