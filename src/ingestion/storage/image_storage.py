"""Filesystem image storage with a SQLite lookup index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


DEFAULT_IMAGE_ROOT = Path("data/images")
DEFAULT_IMAGE_INDEX_DB = Path("data/db/image_index.db")


@dataclass(frozen=True)
class ImageRecord:
    """Indexed image file metadata."""

    image_id: str
    file_path: str
    collection: str
    doc_hash: str | None = None
    page_num: int | None = None
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "file_path": self.file_path,
            "collection": self.collection,
            "doc_hash": self.doc_hash,
            "page_num": self.page_num,
            "created_at": self.created_at,
        }


class ImageStorage:
    """Store image files under a collection directory and index them in SQLite."""

    def __init__(
        self,
        image_root: str | Path = DEFAULT_IMAGE_ROOT,
        db_path: str | Path = DEFAULT_IMAGE_INDEX_DB,
    ) -> None:
        self.image_root = Path(image_root)
        self.db_path = Path(db_path)
        self.image_root.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()

    def save_image(
        self,
        *,
        image_id: str,
        source_path: str | Path,
        collection: str = "default",
        doc_hash: str | None = None,
        page_num: int | None = None,
        extension: str | None = None,
        trace: "TraceContext | None" = None,
    ) -> ImageRecord:
        """Copy a source image file into managed storage and index it."""
        normalized_id = _validate_name(image_id, "image_id")
        normalized_collection = _validate_name(collection, "collection")
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f"Image source file not found: {source}")

        suffix = _normalize_extension(extension or source.suffix or ".bin")
        target = self._target_path(normalized_collection, normalized_id, suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)

        record = self._upsert_record(
            image_id=normalized_id,
            file_path=target,
            collection=normalized_collection,
            doc_hash=doc_hash,
            page_num=page_num,
        )
        self._record_trace(trace, "save_image", record)
        return record

    def save_bytes(
        self,
        *,
        image_id: str,
        data: bytes,
        collection: str = "default",
        doc_hash: str | None = None,
        page_num: int | None = None,
        extension: str = ".png",
        trace: "TraceContext | None" = None,
    ) -> ImageRecord:
        """Write image bytes into managed storage and index them."""
        normalized_id = _validate_name(image_id, "image_id")
        normalized_collection = _validate_name(collection, "collection")
        if not isinstance(data, bytes) or not data:
            raise ValueError("data must be non-empty bytes")

        suffix = _normalize_extension(extension)
        target = self._target_path(normalized_collection, normalized_id, suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

        record = self._upsert_record(
            image_id=normalized_id,
            file_path=target,
            collection=normalized_collection,
            doc_hash=doc_hash,
            page_num=page_num,
        )
        self._record_trace(trace, "save_bytes", record)
        return record

    def get_record(self, image_id: str) -> ImageRecord | None:
        """Return one indexed image record by id."""
        normalized_id = _validate_name(image_id, "image_id")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM image_index WHERE image_id = ?",
                (normalized_id,),
            ).fetchone()
        return self._row_to_record(row) if row is not None else None

    def get_path(self, image_id: str) -> Path | None:
        """Return the indexed image path, if present."""
        record = self.get_record(image_id)
        return Path(record.file_path) if record is not None else None

    def list_by_collection(self, collection: str) -> list[ImageRecord]:
        """Return indexed images for a collection in deterministic order."""
        normalized_collection = _validate_name(collection, "collection")
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM image_index
                WHERE collection = ?
                ORDER BY image_id
                """,
                (normalized_collection,),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def journal_mode(self) -> str:
        """Return active SQLite journal mode for diagnostics/tests."""
        with self._connect() as conn:
            row = conn.execute("PRAGMA journal_mode").fetchone()
        return str(row[0]).lower()

    def _target_path(self, collection: str, image_id: str, suffix: str) -> Path:
        return self.image_root / collection / f"{image_id}{suffix}"

    def _upsert_record(
        self,
        *,
        image_id: str,
        file_path: Path,
        collection: str,
        doc_hash: str | None,
        page_num: int | None,
    ) -> ImageRecord:
        if page_num is not None and (not isinstance(page_num, int) or page_num < 0):
            raise ValueError("page_num must be a non-negative integer when provided")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO image_index (
                    image_id, file_path, collection, doc_hash, page_num
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(image_id) DO UPDATE SET
                    file_path = excluded.file_path,
                    collection = excluded.collection,
                    doc_hash = excluded.doc_hash,
                    page_num = excluded.page_num
                """,
                (
                    image_id,
                    str(file_path),
                    collection,
                    _normalize_optional_text(doc_hash),
                    page_num,
                ),
            )
        record = self.get_record(image_id)
        if record is None:
            raise RuntimeError(f"failed to index image: {image_id}")
        return record

    def _initialize_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS image_index (
                    image_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    collection TEXT,
                    doc_hash TEXT,
                    page_num INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_collection
                ON image_index(collection)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_doc_hash
                ON image_index(doc_hash)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _row_to_record(row: Mapping[str, Any]) -> ImageRecord:
        return ImageRecord(
            image_id=str(row["image_id"]),
            file_path=str(row["file_path"]),
            collection=str(row["collection"] or ""),
            doc_hash=row["doc_hash"],
            page_num=row["page_num"],
            created_at=str(row["created_at"]) if row["created_at"] is not None else None,
        )

    @staticmethod
    def _record_trace(
        trace: "TraceContext | None",
        method: str,
        record: ImageRecord,
    ) -> None:
        if trace is None:
            return
        trace.record_stage(
            "store",
            method=f"image_storage.{method}",
            details={
                "image_id": record.image_id,
                "collection": record.collection,
                "file_path": record.file_path,
            },
        )


def _validate_name(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    normalized = value.strip()
    path = Path(normalized)
    if path.name != normalized or normalized in {".", ".."}:
        raise ValueError(f"{field_name} must not contain path separators")
    return normalized


def _normalize_extension(extension: str) -> str:
    if not isinstance(extension, str) or not extension.strip():
        raise ValueError("extension must be a non-empty string")
    suffix = extension.strip().lower()
    return suffix if suffix.startswith(".") else f".{suffix}"


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("doc_hash must be a string when provided")
    stripped = value.strip()
    return stripped or None


__all__ = [
    "DEFAULT_IMAGE_INDEX_DB",
    "DEFAULT_IMAGE_ROOT",
    "ImageRecord",
    "ImageStorage",
]
