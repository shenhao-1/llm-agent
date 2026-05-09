"""PDF loader implemented with PyMuPDF."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

from core.types import Document, image_placeholder

from .base_loader import BaseLoader

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class PdfLoader(BaseLoader):
    """Load PDF text and image references into the shared Document contract."""

    supported_extensions = (".pdf",)

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self.image_output_dir = Path(self.config.get("image_output_dir", "data/images"))
        self.collection = str(self.config.get("collection", "")).strip()

    def load(self, path: str | Path, trace: TraceContext | None = None) -> Document:
        source_path = self.validate_source_path(path)
        doc_hash = _compute_sha256(source_path)
        collection = self.collection or doc_hash

        try:
            import fitz  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PdfLoader requires PyMuPDF. Install the 'pymupdf' package."
            ) from exc

        text_parts: list[str] = []
        images: list[dict[str, Any]] = []
        warnings: list[str] = []

        with fitz.open(source_path) as pdf:
            for page_index, page in enumerate(pdf):
                page_text = page.get_text("text")
                if page_text:
                    text_parts.append(page_text.rstrip())

                self._append_page_images(
                    pdf=pdf,
                    page=page,
                    page_index=page_index,
                    doc_hash=doc_hash,
                    collection=collection,
                    text_parts=text_parts,
                    images=images,
                    warnings=warnings,
                )

            page_count = pdf.page_count

        text = "\n\n".join(part for part in text_parts if part)
        metadata: dict[str, Any] = {
            "source_path": str(source_path),
            "doc_type": "pdf",
            "doc_hash": doc_hash,
            "collection": collection,
            "page_count": page_count,
            "images": images,
        }
        if warnings:
            metadata["warnings"] = warnings

        return Document(id=doc_hash, text=text, metadata=metadata)

    def _append_page_images(
        self,
        pdf: Any,
        page: Any,
        page_index: int,
        doc_hash: str,
        collection: str,
        text_parts: list[str],
        images: list[dict[str, Any]],
        warnings: list[str],
    ) -> None:
        page_images = page.get_images(full=True)
        for image_index, image_info in enumerate(page_images):
            xref = int(image_info[0])
            image_id = f"{doc_hash}_{page_index}_{image_index}"
            placeholder = image_placeholder(image_id)
            text_offset = _joined_text_length(text_parts)
            text_parts.append(placeholder)

            image_ref: dict[str, Any] = {
                "id": image_id,
                "path": str(self.image_output_dir / collection / f"{image_id}.png"),
                "page": page_index,
                "text_offset": text_offset,
                "text_length": len(placeholder),
                "position": {},
            }
            image_ref["position"] = self._image_position(page, xref)
            images.append(image_ref)

            try:
                self._save_image(pdf, xref, Path(image_ref["path"]))
            except Exception as exc:  # pragma: no cover - error shape depends on PDFs
                warnings.append(f"Failed to extract image {image_id}: {exc}")

    def _save_image(self, pdf: Any, xref: int, image_path: Path) -> None:
        import fitz  # type: ignore[import-not-found]

        image_path.parent.mkdir(parents=True, exist_ok=True)
        pixmap = fitz.Pixmap(pdf, xref)
        try:
            if pixmap.n - pixmap.alpha >= 4:
                converted = fitz.Pixmap(fitz.csRGB, pixmap)
                try:
                    converted.save(image_path)
                finally:
                    converted = None
            else:
                pixmap.save(image_path)
        finally:
            pixmap = None

    @staticmethod
    def _image_position(page: Any, xref: int) -> dict[str, Any]:
        rects = page.get_image_rects(xref)
        if not rects:
            return {}
        rect = rects[0]
        return {
            "x0": float(rect.x0),
            "y0": float(rect.y0),
            "x1": float(rect.x1),
            "y1": float(rect.y1),
        }


def _joined_text_length(parts: list[str]) -> int:
    if not parts:
        return 0
    return sum(len(part) for part in parts) + (len(parts) * 2)


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
