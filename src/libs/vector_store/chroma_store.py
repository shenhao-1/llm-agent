"""Chroma vector store provider."""

from __future__ import annotations

from importlib import import_module
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
            metadatas.append(dict(metadata) if isinstance(metadata, Mapping) else {})

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
            where=dict(filters) if filters else None,
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
                item["metadata"] = dict(metadatas[index])
            if index < len(distances):
                item["distance"] = float(distances[index])
            items.append(item)
        return items

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


__all__ = ["ChromaStore"]
