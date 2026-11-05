"""Chroma vector store provider."""

from __future__ import annotations

from importlib import import_module
import json
from pathlib import Path
from typing import Any, Mapping

from .base_vector_store import BaseVectorStore


class ChromaStore(BaseVectorStore):
    """Minimal Chroma-backed vector store for upsert and vector query."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        persist_directory = str(self.config.get("persist_directory", "")).strip()
        if not persist_directory:
            raise ValueError("chroma config missing persist_directory")

        self.persist_directory = str(Path(persist_directory))
        self.collection_name = (
            str(self.config.get("collection_name", "default")).strip() or "default"
        )
        self.distance = str(self.config.get("distance", "cosine")).strip() or "cosine"

        chromadb = self._load_chromadb()
        client = chromadb.PersistentClient(path=self.persist_directory)
        self._collection = client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance},
        )

    def upsert(self, records: list[Mapping[str, Any]], trace=None) -> None:
        self.validate_records(records)

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for record in records:
            ids.append(str(record["id"]))
            embeddings.append([float(value) for value in record["vector"]])
            documents.append(str(record.get("text", "")))
            metadata = record.get("metadata")
            metadatas.append(_encode_metadata(metadata) if isinstance(metadata, Mapping) else {})

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace=None,
    ) -> list[dict[str, Any]]:
        self.validate_query_args(vector, top_k, filters)

        result = self._collection.query(
            query_embeddings=[[float(value) for value in vector]],
            n_results=top_k,
            where=_metadata_where(filters),
            include=["documents", "metadatas", "distances"],
        )

        ids = self._first_row(result.get("ids"))
        documents = self._first_row(result.get("documents"))
        metadatas = self._first_row(result.get("metadatas"))
        distances = self._first_row(result.get("distances"))

        items: list[dict[str, Any]] = []
        for index, item_id in enumerate(ids):
            item: dict[str, Any] = {"id": item_id}
            if index < len(documents):
                item["text"] = documents[index]
            if index < len(metadatas) and isinstance(metadatas[index], Mapping):
                item["metadata"] = _decode_metadata(metadatas[index])
            if index < len(distances):
                item["distance"] = float(distances[index])
            items.append(item)
        return items

    def get_by_ids(self, ids: list[str], trace=None) -> list[dict[str, Any]]:
        self.validate_ids(ids)
        if not ids:
            return []

        result = self._collection.get(
            ids=[str(item_id) for item_id in ids],
            include=["documents", "metadatas"],
        )
        result_ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        records_by_id: dict[str, dict[str, Any]] = {}
        for index, item_id in enumerate(result_ids):
            record: dict[str, Any] = {"id": str(item_id)}
            if index < len(documents):
                record["text"] = documents[index]
            if index < len(metadatas) and isinstance(metadatas[index], Mapping):
                record["metadata"] = _decode_metadata(metadatas[index])
            records_by_id[str(item_id)] = record

        return [records_by_id[item_id] for item_id in ids if item_id in records_by_id]

    def get_by_metadata(
        self,
        filters: Mapping[str, Any] | None = None,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return records matching exact metadata filters."""
        if limit is not None and (not isinstance(limit, int) or limit <= 0):
            raise ValueError("limit must be a positive integer when provided")
        kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
        where = _metadata_where(filters)
        if where is not None:
            kwargs["where"] = where
        if limit is not None:
            kwargs["limit"] = limit

        result = self._collection.get(**kwargs)
        result_ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        records: list[dict[str, Any]] = []
        for index, item_id in enumerate(result_ids):
            record: dict[str, Any] = {"id": str(item_id)}
            if index < len(documents):
                record["text"] = documents[index]
            if index < len(metadatas) and isinstance(metadatas[index], Mapping):
                record["metadata"] = _decode_metadata(metadatas[index])
            records.append(record)
        return records

    def get_collection_stats(self) -> dict[str, Any]:
        """Return lightweight collection stats for dashboard overview pages."""
        return {
            "provider": "chroma",
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "distance": self.distance,
            "record_count": int(self._collection.count()),
        }

    def delete_by_metadata(self, filters: Mapping[str, Any]) -> int:
        """Delete records matching exact metadata filters and return count."""
        if not isinstance(filters, Mapping) or not filters:
            raise ValueError("filters must be a non-empty mapping")
        where = _metadata_where(filters)
        matches = self._collection.get(where=where)
        ids = [str(item_id) for item_id in matches.get("ids", [])]
        if not ids:
            return 0
        self._collection.delete(ids=ids)
        return len(ids)

    @staticmethod
    def _first_row(values: Any) -> list[Any]:
        if isinstance(values, list) and values:
            first = values[0]
            if isinstance(first, list):
                return first
        return []

    @staticmethod
    def _load_chromadb():
        try:
            return import_module("chromadb")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "chromadb is required for ChromaStore; install project dependencies first"
            ) from exc


def _metadata_where(filters: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not filters:
        return None
    items = [{str(key): value} for key, value in filters.items()]
    if len(items) == 1:
        return items[0]
    return {"$and": items}


_JSON_PREFIX = "__rag_json__:"


def _encode_metadata(metadata: Mapping[str, Any]) -> dict[str, str | int | float | bool]:
    encoded: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)) and not (
            isinstance(value, str) and value.startswith(_JSON_PREFIX)
        ):
            encoded[str(key)] = value
        else:
            encoded[str(key)] = _JSON_PREFIX + json.dumps(value, ensure_ascii=False)
    return encoded


def _decode_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    decoded = dict(metadata)
    for key, value in decoded.items():
        if isinstance(value, str) and value.startswith(_JSON_PREFIX):
            decoded[key] = json.loads(value[len(_JSON_PREFIX):])
    return decoded


__all__ = ["ChromaStore"]
