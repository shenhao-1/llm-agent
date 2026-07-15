"""Unit tests for file integrity checks and ingestion history."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from libs.loader import SQLiteIntegrityChecker


def test_compute_sha256_is_stable_for_same_file(tmp_path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("stable content", encoding="utf-8")
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")

    first = checker.compute_sha256(source)
    second = checker.compute_sha256(source)

    assert first == second
    assert len(first) == 64


def test_compute_sha256_changes_when_file_changes(tmp_path) -> None:
    source = tmp_path / "sample.txt"
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")

    source.write_text("one", encoding="utf-8")
    first = checker.compute_sha256(source)
    source.write_text("two", encoding="utf-8")
    second = checker.compute_sha256(source)

    assert first != second


def test_compute_sha256_rejects_missing_file(tmp_path) -> None:
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")

    with pytest.raises(FileNotFoundError):
        checker.compute_sha256(tmp_path / "missing.pdf")


def test_should_skip_is_true_after_success(tmp_path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("content", encoding="utf-8")
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")
    file_hash = checker.compute_sha256(source)

    assert checker.should_skip(file_hash) is False

    checker.mark_success(file_hash, source, metadata={"collection": "default"})

    assert checker.should_skip(file_hash) is True
    record = checker.get_record(file_hash)
    assert record is not None
    assert record["status"] == "success"
    assert record["file_path"] == str(source)
    assert record["metadata"] == {"collection": "default"}


def test_failed_record_does_not_skip_and_can_be_replaced_by_success(tmp_path) -> None:
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")
    file_hash = "abc123"

    checker.mark_failed(file_hash, "parse failed")

    assert checker.should_skip(file_hash) is False
    failed_record = checker.get_record(file_hash)
    assert failed_record is not None
    assert failed_record["status"] == "failed"
    assert failed_record["error_msg"] == "parse failed"

    checker.mark_success(file_hash, "docs/sample.pdf")

    assert checker.should_skip(file_hash) is True
    success_record = checker.get_record(file_hash)
    assert success_record is not None
    assert success_record["status"] == "success"
    assert success_record["error_msg"] is None


def test_sqlite_database_file_and_table_are_created(tmp_path) -> None:
    db_path = tmp_path / "nested" / "ingestion_history.db"

    SQLiteIntegrityChecker(db_path)

    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name = 'ingestion_history'
            """
        ).fetchone()
    assert row is not None


def test_sqlite_wal_mode_is_enabled(tmp_path) -> None:
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")

    assert checker.journal_mode() == "wal"


def test_concurrent_success_writes_are_supported(tmp_path) -> None:
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")

    def write_record(idx: int) -> bool:
        file_hash = f"hash-{idx}"
        checker.mark_success(file_hash, f"docs/{idx}.pdf")
        return checker.should_skip(file_hash)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(write_record, range(24)))

    assert all(results)


def test_list_processed_and_remove_record_support_document_manager(tmp_path) -> None:
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")
    checker.mark_success("hash-a", "docs/a.pdf", metadata={"collection": "docs"})
    checker.mark_failed("hash-b", "docs/b.pdf")

    records = checker.list_processed()

    assert {record["file_hash"] for record in records} == {"hash-a", "hash-b"}
    assert checker.remove_record("docs/a.pdf") == 1
    assert checker.get_record("hash-a") is None
    assert checker.remove_record("hash-b") == 1
    assert checker.list_processed() == []


def test_mark_success_validates_inputs(tmp_path) -> None:
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")

    with pytest.raises(ValueError, match="file_hash"):
        checker.mark_success("", "docs/sample.pdf")

    with pytest.raises(ValueError, match="file_path"):
        checker.mark_success("hash", "")
