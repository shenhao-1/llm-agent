"""Unit tests for the PDF loader contract."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from core import Document
from libs.loader import BaseLoader, PdfLoader


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def test_base_loader_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseLoader()


def test_pdf_loader_rejects_non_pdf_extension(tmp_path: Path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("content", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file extension"):
        PdfLoader().load(source)


def test_pdf_loader_loads_text_pdf_into_document(tmp_path: Path) -> None:
    pdf_path = tmp_path / "simple.pdf"
    _write_text_pdf(pdf_path, "Hello PDF text")

    document = PdfLoader().load(pdf_path)

    assert isinstance(document, Document)
    assert document.id == document.metadata["doc_hash"]
    assert "Hello PDF text" in document.text
    assert document.metadata["source_path"] == str(pdf_path)
    assert document.metadata["doc_type"] == "pdf"
    assert document.metadata["page_count"] == 1
    assert document.metadata["images"] == []


def test_pdf_loader_extracts_images_and_inserts_placeholders(tmp_path: Path) -> None:
    pdf_path = tmp_path / "with-image.pdf"
    image_source = tmp_path / "pixel.png"
    image_source.write_bytes(PNG_1X1)
    _write_pdf_with_image(pdf_path, image_source)
    output_dir = tmp_path / "images"

    document = PdfLoader(
        {"image_output_dir": str(output_dir), "collection": "default"}
    ).load(pdf_path)

    images = document.metadata["images"]
    assert len(images) == 1
    image_ref = images[0]
    assert image_ref["id"] in document.text
    assert f"[IMAGE: {image_ref['id']}]" in document.text
    assert image_ref["path"].endswith(".png")
    assert Path(image_ref["path"]).is_file()
    assert image_ref["page"] == 0
    assert image_ref["text_length"] == len(f"[IMAGE: {image_ref['id']}]")
    assert document.text[image_ref["text_offset"] : image_ref["text_offset"] + image_ref["text_length"]] == f"[IMAGE: {image_ref['id']}]"
    assert {"x0", "y0", "x1", "y1"}.issubset(image_ref["position"])


def test_pdf_loader_continues_when_image_extraction_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "with-image.pdf"
    image_source = tmp_path / "pixel.png"
    image_source.write_bytes(PNG_1X1)
    _write_pdf_with_image(pdf_path, image_source, text="Text survives")

    def raise_on_save(*args, **kwargs) -> None:
        raise RuntimeError("image backend failed")

    monkeypatch.setattr(PdfLoader, "_save_image", raise_on_save)

    document = PdfLoader({"image_output_dir": str(tmp_path / "images")}).load(pdf_path)

    assert "Text survives" in document.text
    assert len(document.metadata["images"]) == 1
    assert document.metadata["warnings"]
    assert "image backend failed" in document.metadata["warnings"][0]


def _write_text_pdf(path: Path, text: str) -> None:
    import fitz  # type: ignore[import-not-found]

    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()


def _write_pdf_with_image(path: Path, image_path: Path, text: str = "PDF with image") -> None:
    import fitz  # type: ignore[import-not-found]

    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), text)
    page.insert_image(fitz.Rect(72, 96, 112, 136), filename=str(image_path))
    pdf.save(path)
    pdf.close()
