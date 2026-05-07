"""Idempotent vector upsert adapter for dense ChunkRecords."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Real
from typing import Any, Mapping, TYPE_CHECKING

from core.types import ChunkRecord
from libs.vector_store import BaseVectorStore, VectorStoreFactory

if TYPE_CHECKING:
    from core.settings import Settings
    from core.trace.trace_context import TraceContext


@dataclass(frozen=True)
class VectorUpsertResult:
    """Summary of one vector upsert operation."""

    ids: list[str]
    records: list[dict[str, Any]]

    @property
    def count(self) -> int:
        return len(self.ids)


class VectorUpserter:
    """Convert dense ChunkRecords into deterministic vector-store upserts."""

    def __init__(
        self,
        settings: "Settings | Mapping[str, Any] | None" = None,
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        if vector_store is None and settings is None:
            raise ValueError("settings or vector_store must be provided")

        self.settings = settings or {}
        self.vector_store = vector_store or VectorStoreFactory.create(self.settings)
    #主入口 
    def upsert(
        self,
        records: list[ChunkRecord],
        trace: "TraceContext | None" = None,
    ) -> VectorUpsertResult:
        """Upsert dense records and return the deterministic ids used.
        输入 ChunkRecord 列表
            ↓
        prepare_records()
            ↓
        转换成 vector store record
            ↓
        vector_store.upsert()
            ↓
        返回 VectorUpsertResult
        """
        vector_records = self.prepare_records(records)
        if not vector_records:
            result = VectorUpsertResult(ids=[], records=[])
            self._record_trace(trace, result)
            return result

        self.vector_store.upsert(vector_records, trace=trace)
        result = VectorUpsertResult(
            ids=[record["id"] for record in vector_records],
            records=vector_records,
        )
        self._record_trace(trace, result)
        return result
    # 转换函数
    def prepare_records(self, records: list[ChunkRecord]) -> list[dict[str, Any]]:
        """Return vector-store records without writing them.
        
        """
        self._validate_records(records)
        vector_records: list[dict[str, Any]] = []
        for record in records:
            vector_id = self.stable_id(record)
            content_hash = self._content_hash(record)
            metadata = dict(record.metadata)
            metadata.update(
                {
                    "source_chunk_id": record.id,
                    "vector_id": vector_id,
                    "content_hash": content_hash,
                }
            )
            vector_records.append(
                {
                    "id": vector_id,
                    "text": record.text,
                    "vector": [float(value) for value in record.dense_vector or []],
                    "metadata": metadata,
                }
            )
        return vector_records

    @classmethod
    def stable_id(cls, record: ChunkRecord) -> str:
        """Generate hash(source_path + chunk_index + content_hash[:8])."""
        cls._validate_record(record)
        metadata = record.metadata
        source_path = str(metadata["source_path"])
        chunk_index = int(metadata["chunk_index"])
        content_prefix = cls._content_hash(record)[:8]
        raw = f"{source_path}:{chunk_index}:{content_prefix}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _content_hash(record: ChunkRecord) -> str:
        metadata_hash = record.metadata.get("content_hash")
        if isinstance(metadata_hash, str) and metadata_hash.strip():
            return metadata_hash.strip().lower()
        return hashlib.sha256(record.text.encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_records(records: Any) -> None:
        """
        records 必须是 list
        每个元素必须是 ChunkRecord
        每个 ChunkRecord 必须有 dense_vector
        metadata 必须有 source_path 和 chunk_index
        """
        if not isinstance(records, list):
            raise ValueError("records must be a list")
        for idx, record in enumerate(records):
            try:
                VectorUpserter._validate_record(record)
            except ValueError as exc:
                raise ValueError(f"records[{idx}]: {exc}") from exc

    @staticmethod
    def _validate_record(record: Any) -> None:
        #这也是为了保证输入的 record 是合法的 ChunkRecord，并且包含必要的字段和格式要求。
        if not isinstance(record, ChunkRecord):
            raise ValueError("record must be a ChunkRecord")
        if record.dense_vector is None:
            raise ValueError("dense_vector is required")
        if not record.dense_vector:
            raise ValueError("dense_vector must be a non-empty list")
        for idx, value in enumerate(record.dense_vector):
            if not isinstance(value, Real):
                raise ValueError(f"dense_vector[{idx}] must be a number")

        source_path = record.metadata.get("source_path")
        if not isinstance(source_path, str) or not source_path.strip():
            raise ValueError("metadata.source_path must be a non-empty string")

        chunk_index = record.metadata.get("chunk_index")
        if not isinstance(chunk_index, int) or chunk_index < 0:
            raise ValueError("metadata.chunk_index must be a non-negative integer")

    @staticmethod # 是给后面可观测用的
    def _record_trace(
        trace: "TraceContext | None",
        result: VectorUpsertResult,
    ) -> None:
        if trace is None:
            return
        trace.record_stage(
            "store",
            method="vector_upserter",
            details={"record_count": result.count, "ids": list(result.ids)},
        )


__all__ = ["VectorUpsertResult", "VectorUpserter"]
