"""Recursive text splitter provider."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base_splitter import BaseSplitter


class RecursiveSplitter(BaseSplitter):
    """Default splitter backed by LangChain recursive character splitting."""

    def __init__(self, config=None) -> None:
        super().__init__(config)
        chunk_size = self.config.get("chunk_size", 1000)
        chunk_overlap = self.config.get("chunk_overlap", 200)
        keep_separator = self.config.get("keep_separator", True)

        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("recursive splitter chunk_size must be a positive integer")
        if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
            raise ValueError("recursive splitter chunk_overlap must be a non-negative integer")
        if chunk_overlap >= chunk_size:
            raise ValueError("recursive splitter chunk_overlap must be smaller than chunk_size")

        separators = self.config.get(
            "separators",
            ["\n```", "\n#", "\n##", "\n###", "\n\n", "\n", " ", ""],
        )
        if not isinstance(separators, list) or not all(
            isinstance(item, str) for item in separators
        ):
            raise ValueError("recursive splitter separators must be a list of strings")

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_separator=keep_separator,
            separators=separators,
        )

    def split_text(self, text: str, trace=None) -> list[str]:
        self.validate_text(text)
        chunks = self._splitter.split_text(text)
        return [chunk for chunk in chunks if chunk.strip()]


__all__ = ["RecursiveSplitter"]
