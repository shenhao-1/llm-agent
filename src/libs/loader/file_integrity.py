"""File integrity checks and ingestion history persistence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping


DEFAULT_INGESTION_HISTORY_DB = Path("data/db/ingestion_history.db")


class FileIntegrityChecker(ABC):
    """Contract for deciding whether a source file can be skipped."""
    # 生成文件的 SHA256 哈希值，并使用持久化存储记录成功和失败的摄取历史，以避免重复处理相同的文件。
    @abstractmethod
    def compute_sha256(self, path: str | Path) -> str:
        """Return the SHA256 hex digest for a file."""

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        """Return True when a file hash has already been ingested successfully."""

    @abstractmethod
    def mark_success(
        self,
        file_hash: str,
        file_path: str | Path,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Record a successful ingestion for a file hash."""

    @abstractmethod
    def mark_failed(
        self,
        file_hash: str,
        error_msg: str,
        file_path: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Record a failed ingestion for a file hash."""


class SQLiteIntegrityChecker(FileIntegrityChecker):
    """SQLite-backed ingestion history with WAL enabled for concurrent writes."""

    def __init__(self, db_path: str | Path = DEFAULT_INGESTION_HISTORY_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()

    def compute_sha256(self, path: str | Path) -> str:
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def should_skip(self, file_hash: str) -> bool:
        normalized_hash = _validate_file_hash(file_hash)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM ingestion_history WHERE file_hash = ?",
                (normalized_hash,),
            ).fetchone()
        return row is not None and row["status"] == "success"

    def mark_success(
        self,
        file_hash: str,
        file_path: str | Path,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        normalized_hash = _validate_file_hash(file_hash)
        normalized_path = _validate_file_path(file_path)
        metadata_json = _metadata_to_json(metadata)
        now = _utc_now()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_history (
                    file_hash, file_path, status, error_msg, metadata_json,
                    created_at, updated_at, success_at, failed_at
                )
                VALUES (?, ?, 'success', NULL, ?, ?, ?, ?, NULL)
                ON CONFLICT(file_hash) DO UPDATE SET
                    file_path = excluded.file_path,
                    status = 'success',
                    error_msg = NULL,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at,
                    success_at = excluded.success_at,
                    failed_at = NULL
                """,
                (normalized_hash, normalized_path, metadata_json, now, now, now),
            )

    def mark_failed(
        self,
        file_hash: str,
        error_msg: str,
        file_path: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        normalized_hash = _validate_file_hash(file_hash)
        if not isinstance(error_msg, str) or not error_msg.strip():
            raise ValueError("error_msg must be a non-empty string")
        normalized_path = str(file_path) if file_path is not None else ""
        metadata_json = _metadata_to_json(metadata)
        now = _utc_now()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_history (
                    file_hash, file_path, status, error_msg, metadata_json,
                    created_at, updated_at, success_at, failed_at
                )
                VALUES (?, ?, 'failed', ?, ?, ?, ?, NULL, ?)
                ON CONFLICT(file_hash) DO UPDATE SET
                    file_path = excluded.file_path,
                    status = 'failed',
                    error_msg = excluded.error_msg,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at,
                    success_at = NULL,
                    failed_at = excluded.failed_at
                """,
                (
                    normalized_hash,
                    normalized_path,
                    error_msg.strip(),
                    metadata_json,
                    now,
                    now,
                    now,
                ),
            )

    def get_record(self, file_hash: str) -> dict[str, Any] | None:
        """Return an ingestion history row for tests and diagnostics."""
        normalized_hash = _validate_file_hash(file_hash)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM ingestion_history WHERE file_hash = ?",
                (normalized_hash,),
            ).fetchone()

        if row is None:
            return None
        record = dict(row)
        metadata_json = record.pop("metadata_json")
        record["metadata"] = json.loads(metadata_json) if metadata_json else {}
        return record

    def journal_mode(self) -> str:
        """Return the active SQLite journal mode."""
        with self._connect() as conn:
            row = conn.execute("PRAGMA journal_mode").fetchone()
        return str(row[0]).lower()

    def _initialize_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_history (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('success', 'failed')),
                    error_msg TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    success_at TEXT,
                    failed_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ingestion_history_status
                ON ingestion_history(status)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn


def _validate_file_hash(file_hash: Any) -> str:
    if not isinstance(file_hash, str) or not file_hash.strip():
        raise ValueError("file_hash must be a non-empty string")
    return file_hash.strip()


def _validate_file_path(file_path: str | Path) -> str:
    path_text = str(file_path)
    if not path_text.strip():
        raise ValueError("file_path must be a non-empty path")
    return path_text


def _metadata_to_json(metadata: Mapping[str, Any] | None) -> str:
    if metadata is None:
        return "{}"
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping when provided")
    return json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
