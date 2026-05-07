"""End-to-end tests for the offline ingest CLI."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_ingest_cli_runs_and_creates_data_artifacts(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    _write_text_pdf(docs_dir / "simple.pdf", "Offline ingest CLI test document.")
    data_dir = tmp_path / "data"

    result = _run_ingest(docs_dir, data_dir, collection="e2e")

    assert result.returncode == 0, result.stderr
    assert "ingested" in result.stdout
    assert (data_dir / "db" / "ingestion_history.db").exists()
    assert (data_dir / "db" / "bm25" / "index.json").exists()
    assert (data_dir / "db" / "vectors.json").exists()
    assert (data_dir / "db" / "image_index.db").exists()


def test_ingest_cli_skips_unchanged_file_on_second_run(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    _write_text_pdf(docs_dir / "simple.pdf", "Repeat ingest should skip.")
    data_dir = tmp_path / "data"

    first = _run_ingest(docs_dir, data_dir, collection="e2e")
    second = _run_ingest(docs_dir, data_dir, collection="e2e")

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert "ingested" in first.stdout
    assert "skipped" in second.stdout


def test_ingest_cli_force_reprocesses_unchanged_file(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    _write_text_pdf(docs_dir / "simple.pdf", "Force ingest should reprocess.")
    data_dir = tmp_path / "data"

    first = _run_ingest(docs_dir, data_dir, collection="e2e")
    second = _run_ingest(docs_dir, data_dir, collection="e2e", force=True)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert "ingested" in first.stdout
    assert "ingested" in second.stdout


def _run_ingest(
    docs_dir: Path,
    data_dir: Path,
    *,
    collection: str,
    force: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "scripts/ingest.py",
        "--path",
        str(docs_dir),
        "--collection",
        collection,
        "--data-dir",
        str(data_dir),
    ]
    if force:
        command.append("--force")
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60,
    )


def _write_text_pdf(path: Path, text: str) -> None:
    import fitz  # type: ignore[import-not-found]

    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()
