"""Metadata enrichment for ingestion chunks."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, TYPE_CHECKING

from core.types import Chunk
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory

from .base_transform import BaseTransform

if TYPE_CHECKING:
    from core.settings import Settings
    from core.trace.trace_context import TraceContext


MAX_TITLE_WORDS = 10
MAX_SUMMARY_CHARS = 220
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


class MetadataEnricher(BaseTransform):
    """Add title, summary, and tags to chunk metadata."""

    def __init__(
        self,
        settings: "Settings | Mapping[str, Any] | None" = None,
        llm: BaseLLM | None = None,
    ) -> None:
        self.settings = settings or {}
        self.config = self._extract_config(self.settings)
        self.use_llm = bool(self.config.get("use_llm", False))
        self.llm = llm
        if self.use_llm and self.llm is None:
            self.llm = LLMFactory.create(self.settings)

    def transform(
        self, chunks: list[Chunk], trace: "TraceContext | None" = None
    ) -> list[Chunk]:
        self.validate_chunks(chunks)
        enriched: list[Chunk] = []

        for chunk in chunks:
            try:
                enriched.append(self._enrich_chunk(chunk))
            except Exception as exc:
                metadata = dict(chunk.metadata)
                metadata.setdefault("title", self._fallback_title(chunk.text))
                metadata.setdefault("summary", self._fallback_summary(chunk.text))
                metadata.setdefault("tags", ["untagged"])
                metadata["metadata_enriched_by"] = "fallback"
                metadata["metadata_enrich_error"] = str(exc)
                enriched.append(self._replace_chunk(chunk, metadata))

        if trace is not None:
            trace.record_stage(
                "transform",
                method="metadata_enricher",
                details={"chunk_count": len(chunks), "use_llm": self.use_llm},
            )

        return enriched

    def _enrich_chunk(self, chunk: Chunk) -> Chunk:
        rule_metadata = self._rule_based_enrich(chunk.text)
        metadata = dict(chunk.metadata)
        metadata.update(rule_metadata)
        metadata["metadata_enriched_by"] = "rule"

        if self.use_llm:
            llm_metadata, fallback_reason = self._llm_enrich(chunk.text)
            if llm_metadata is not None:
                metadata.update(llm_metadata)
                metadata["metadata_enriched_by"] = "llm"
            else:
                metadata["metadata_fallback_reason"] = (
                    fallback_reason or "llm returned invalid metadata"
                )

        return self._replace_chunk(chunk, metadata)

    def _rule_based_enrich(self, text: str) -> dict[str, Any]:
        if not isinstance(text, str):
            raise ValueError("text must be a string")

        normalized = _normalize_text(text)
        title = self._fallback_title(normalized)
        summary = self._fallback_summary(normalized)
        tags = self._extract_tags(normalized)
        return {"title": title, "summary": summary, "tags": tags}

    def _llm_enrich(self, text: str) -> tuple[dict[str, Any] | None, str | None]:
        if self.llm is None:
            return None, "llm is not configured"

        try:
            response = self.llm.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "Generate concise JSON metadata for retrieval chunks. "
                            "Return only JSON with title, summary, and tags."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Text:\n"
                            f"{text}\n\n"
                            "Return JSON: {\"title\": string, \"summary\": string, "
                            "\"tags\": [string, ...]}"
                        ),
                    },
                ]
            )
            payload = _extract_json_object(response)
            metadata = json.loads(payload)
            return self._validate_llm_metadata(metadata), None
        except Exception as exc:
            return None, str(exc)

    @staticmethod
    def _fallback_title(text: str) -> str:
        normalized = _normalize_text(text)
        if not normalized:
            return "Untitled Chunk"

        first_sentence = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)[0]
        words = first_sentence.split()
        title = " ".join(words[:MAX_TITLE_WORDS]).strip(" .,:;")
        return title or "Untitled Chunk"

    @staticmethod
    def _fallback_summary(text: str) -> str:
        normalized = _normalize_text(text)
        if not normalized:
            return "No summary available."
        if len(normalized) <= MAX_SUMMARY_CHARS:
            return normalized
        return normalized[: MAX_SUMMARY_CHARS - 1].rstrip() + "..."

    @staticmethod
    def _extract_tags(text: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower())
        tags: list[str] = []
        for token in tokens:
            if token in STOPWORDS or token in tags:
                continue
            tags.append(token)
            if len(tags) == 5:
                break
        return tags or ["untagged"]

    @staticmethod
    def _validate_llm_metadata(metadata: Any) -> dict[str, Any]:
        if not isinstance(metadata, Mapping):
            raise ValueError("LLM metadata must be a mapping")

        title = metadata.get("title")
        summary = metadata.get("summary")
        tags = metadata.get("tags")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("LLM metadata title must be a non-empty string")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("LLM metadata summary must be a non-empty string")
        if not isinstance(tags, list) or not tags:
            raise ValueError("LLM metadata tags must be a non-empty list")
        if not all(isinstance(tag, str) and tag.strip() for tag in tags):
            raise ValueError("LLM metadata tags must contain non-empty strings")

        return {
            "title": title.strip(),
            "summary": summary.strip(),
            "tags": [tag.strip() for tag in tags],
        }

    @staticmethod
    def _extract_config(settings: "Settings | Mapping[str, Any]") -> dict[str, Any]:
        if hasattr(settings, "ingestion"):
            ingestion = getattr(settings, "ingestion")
            if isinstance(ingestion, Mapping):
                enricher = ingestion.get("metadata_enricher")
                if isinstance(enricher, Mapping):
                    return dict(enricher)

        if isinstance(settings, Mapping):
            enricher = settings.get("metadata_enricher")
            if isinstance(enricher, Mapping):
                return dict(enricher)

            ingestion = settings.get("ingestion")
            if isinstance(ingestion, Mapping):
                nested = ingestion.get("metadata_enricher")
                if isinstance(nested, Mapping):
                    return dict(nested)

        return {}

    @staticmethod
    def _replace_chunk(chunk: Chunk, metadata: dict[str, Any]) -> Chunk:
        return Chunk(
            id=chunk.id,
            text=chunk.text,
            metadata=metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref,
        )


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_json_object(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("LLM response must be a string")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("LLM response does not contain a JSON object")
    return text[start : end + 1]


__all__ = ["MetadataEnricher"]
