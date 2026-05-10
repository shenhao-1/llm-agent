"""Assemble MCP image content from retrieval result metadata."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Mapping

from core.types import RetrievalResult


class MultimodalAssembler:
    """Build MCP ImageContent entries for images attached to retrieved chunks."""

    def assemble_images(
        self,
        retrieval_results: list[RetrievalResult],
    ) -> list[dict[str, str]]:
        if not isinstance(retrieval_results, list):
            raise ValueError("retrieval_results must be a list")

        content: list[dict[str, str]] = []
        seen_paths: set[str] = set()
        for index, result in enumerate(retrieval_results):
            if not isinstance(result, RetrievalResult):
                raise ValueError(f"retrieval_results[{index}] must be a RetrievalResult")
            for image in _selected_images(result.metadata):
                path = image.get("path")
                if not isinstance(path, str) or not path.strip():
                    continue
                image_path = Path(path)
                if not image_path.exists() or not image_path.is_file():
                    continue
                resolved = str(image_path.resolve())
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                content.append(
                    {
                        "type": "image",
                        "data": base64.b64encode(image_path.read_bytes()).decode("ascii"),
                        "mimeType": _mime_type(image_path),
                    }
                )
        return content


def _selected_images(metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    images = metadata.get("images")
    if not isinstance(images, list):
        return []
    image_refs = metadata.get("image_refs")
    refs = {ref for ref in image_refs if isinstance(ref, str)} if isinstance(image_refs, list) else set()
    selected: list[Mapping[str, Any]] = []
    for image in images:
        if not isinstance(image, Mapping):
            continue
        image_id = image.get("id")
        if refs and image_id not in refs:
            continue
        selected.append(image)
    return selected


def _mime_type(path: Path) -> str:
    guessed, _encoding = mimetypes.guess_type(str(path))
    if guessed and guessed.startswith("image/"):
        return guessed
    return "image/png"


__all__ = ["MultimodalAssembler"]
