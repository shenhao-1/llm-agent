"""Unit tests for filesystem image storage and SQLite indexing."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.trace import TraceContext
from ingestion.storage import ImageRecord, ImageStorage


def test_save_image_copies_file_and_persists_lookup(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"png-bytes")
    storage = _storage(tmp_path)

    record = storage.save_image(
        image_id="img-1",
        source_path=source,
        collection="docs",
        doc_hash="doc-hash",
        page_num=3,
    )

    assert isinstance(record, ImageRecord)
    assert Path(record.file_path).exists()
    assert Path(record.file_path).read_bytes() == b"png-bytes"
    assert Path(record.file_path).name == "img-1.png"
    assert record.collection == "docs"
    assert record.doc_hash == "doc-hash"
    assert record.page_num == 3

    loaded = _storage(tmp_path).get_record("img-1")
    assert loaded == record
    assert _storage(tmp_path).get_path("img-1") == Path(record.file_path)


def test_save_bytes_writes_file_and_indexes_default_png(tmp_path: Path) -> None:
    storage = _storage(tmp_path)

    record = storage.save_bytes(
        image_id="img-bytes",
        data=b"raw-image",
        collection="figures",
        extension="jpg",
    )

    assert Path(record.file_path).read_bytes() == b"raw-image"
    assert Path(record.file_path).name == "img-bytes.jpg"
    assert _storage(tmp_path).get_record("img-bytes") == record


def test_repeated_save_updates_mapping_without_duplicate_rows(tmp_path: Path) -> None:
    storage = _storage(tmp_path)

    first = storage.save_bytes(
        image_id="img-1",
        data=b"first",
        collection="alpha",
        extension=".png",
    )
    second = storage.save_bytes(
        image_id="img-1",
        data=b"second",
        collection="beta",
        doc_hash="doc-2",
        page_num=1,
        extension=".webp",
    )

    assert first.image_id == second.image_id
    assert Path(second.file_path).name == "img-1.webp"
    assert Path(second.file_path).read_bytes() == b"second"
    assert [record.image_id for record in storage.list_by_collection("alpha")] == []
    assert storage.list_by_collection("beta") == [second]


def test_list_by_collection_returns_deterministic_batch(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    storage.save_bytes(image_id="img-b", data=b"b", collection="docs")
    storage.save_bytes(image_id="img-a", data=b"a", collection="docs")
    storage.save_bytes(image_id="img-c", data=b"c", collection="notes")

    records = storage.list_by_collection("docs")

    assert [record.image_id for record in records] == ["img-a", "img-b"]
    assert all(Path(record.file_path).exists() for record in records)


def test_list_and_delete_by_document_hash(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    first = storage.save_bytes(
        image_id="img-a",
        data=b"a",
        collection="docs",
        doc_hash="doc-1",
    )
    storage.save_bytes(
        image_id="img-b",
        data=b"b",
        collection="notes",
        doc_hash="doc-1",
    )

    assert storage.list_by_document("doc-1", "docs") == [first]
    assert storage.delete_by_document("doc-1", "docs") == 1
    assert not Path(first.file_path).exists()
    assert storage.list_by_document("doc-1", "docs") == []
    assert [record.image_id for record in storage.list_by_document("doc-1")] == ["img-b"]


def test_list_images_supports_collection_and_document_filters(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    storage.save_bytes(
        image_id="img-a",
        data=b"a",
        collection="docs",
        doc_hash="doc-1",
    )
    storage.save_bytes(
        image_id="img-b",
        data=b"b",
        collection="docs",
        doc_hash="doc-2",
    )
    storage.save_bytes(
        image_id="img-c",
        data=b"c",
        collection="notes",
        doc_hash="doc-1",
    )

    assert [record.image_id for record in storage.list_images(collection="docs")] == [
        "img-a",
        "img-b",
    ]
    assert [record.image_id for record in storage.list_images(doc_hash="doc-1")] == [
        "img-a",
        "img-c",
    ]
    assert [
        record.image_id
        for record in storage.list_images(collection="docs", doc_hash="doc-1")
    ] == ["img-a"]


def test_sqlite_uses_wal_mode_and_required_indexes(tmp_path: Path) -> None:
    storage = _storage(tmp_path)

    assert storage.journal_mode() == "wal"
    with storage._connect() as conn:
        indexes = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
    assert {"idx_collection", "idx_doc_hash"}.issubset(indexes)


def test_trace_records_image_storage_stage(tmp_path: Path) -> None:
    trace = TraceContext()

    record = _storage(tmp_path).save_bytes(
        image_id="img-trace",
        data=b"trace",
        collection="docs",
        trace=trace,
    )

    assert trace.stages[-1]["name"] == "store"
    assert trace.stages[-1]["method"] == "image_storage.save_bytes"
    assert trace.stages[-1]["details"] == {
        "image_id": "img-trace",
        "collection": "docs",
        "file_path": record.file_path,
    }


def test_invalid_inputs_are_rejected(tmp_path: Path) -> None:
    storage = _storage(tmp_path)

    with pytest.raises(ValueError, match="image_id"):
        storage.save_bytes(image_id="../bad", data=b"x")
    with pytest.raises(ValueError, match="collection"):
        storage.save_bytes(image_id="img", data=b"x", collection="bad/name")
    with pytest.raises(ValueError, match="data"):
        storage.save_bytes(image_id="img", data=b"")
    with pytest.raises(ValueError, match="page_num"):
        storage.save_bytes(image_id="img", data=b"x", page_num=-1)
    with pytest.raises(FileNotFoundError):
        storage.save_image(image_id="img", source_path=tmp_path / "missing.png")


def _storage(tmp_path: Path) -> ImageStorage:
    return ImageStorage(
        image_root=tmp_path / "images",
        db_path=tmp_path / "db" / "image_index.db",
    )
