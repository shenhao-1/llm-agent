"""Business adapter from Document objects to Chunk objects."""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from typing import Any, Mapping, TYPE_CHECKING

from core.types import Chunk, Document
from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import SplitterFactory

if TYPE_CHECKING:
    from core.settings import Settings
    from core.trace.trace_context import TraceContext


IMAGE_PLACEHOLDER_RE = re.compile(r"\[IMAGE:\s*([^\]]+?)\s*\]")


class DocumentChunker:
    """Convert parsed Documents into contract-compliant Chunks."""

    def __init__(
        self,
        settings: "Settings | Mapping[str, Any]",
        splitter: BaseSplitter | None = None,
    ) -> None:
        self.splitter = splitter or SplitterFactory.create(settings)

    def split_document(
        self, document: Document, trace: "TraceContext | None" = None
    ) -> list[Chunk]:
        """Split a Document and attach ingestion-specific chunk metadata."""
        if not isinstance(document, Document):
            raise TypeError("document must be a Document")

        chunk_texts = self.splitter.split_text(document.text, trace=trace)
        chunks: list[Chunk] = []
        previous_start = -1

        for index, chunk_text in enumerate(chunk_texts):
            start_offset, end_offset = self._locate_chunk_offsets(
                document.text, chunk_text, previous_start
            )
            previous_start = start_offset
            chunks.append(
                Chunk(
                    id=self._generate_chunk_id(document.id, index, chunk_text),
                    text=chunk_text,
                    metadata=self._inherit_metadata(document, index, chunk_text),
                    start_offset=start_offset,
                    end_offset=end_offset,
                    source_ref=document.id,
                )
            )

        return chunks

    @staticmethod
    def _generate_chunk_id(doc_id: str, index: int, text: str) -> str:
        """Generate a stable chunk ID from document ID, index, and chunk text."""
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError("doc_id must be a non-empty string")
        if not isinstance(index, int) or index < 0:
            raise ValueError("index must be a non-negative integer")
        if not isinstance(text, str):
            raise ValueError("text must be a string")

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"{doc_id}_{index:04d}_{digest}"

    @staticmethod
    def _inherit_metadata(
        document: Document, chunk_index: int, chunk_text: str
    ) -> dict[str, Any]:
        """Copy document metadata and keep only image refs used by the chunk."""
        metadata = deepcopy(document.metadata)
        document_images = metadata.pop("images", [])
        metadata["chunk_index"] = chunk_index

        image_refs = IMAGE_PLACEHOLDER_RE.findall(chunk_text)
        if image_refs:
            image_by_id = {
                image["id"]: image
                for image in document_images
                if isinstance(image, Mapping) and isinstance(image.get("id"), str)
            }
            metadata["image_refs"] = image_refs
            metadata["images"] = [
                deepcopy(image_by_id[image_id])
                for image_id in image_refs
                if image_id in image_by_id
            ]

        return metadata

    @staticmethod
    def _locate_chunk_offsets(
        document_text: str, chunk_text: str, previous_start: int
    ) -> tuple[int, int]:
        search_start = max(previous_start + 1, 0)
        start_offset = document_text.find(chunk_text, search_start)
        if start_offset == -1:
            start_offset = document_text.find(chunk_text)
        if start_offset == -1:
            start_offset = min(search_start, len(document_text))

        end_offset = start_offset + len(chunk_text)
        return start_offset, end_offset


__all__ = ["DocumentChunker", "IMAGE_PLACEHOLDER_RE"]
