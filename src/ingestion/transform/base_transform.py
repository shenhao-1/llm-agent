"""Base contract for ingestion transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from core.types import Chunk

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext

 # 基础LLM输入输出转换器，所有的LLM输入输出转换器都需要继承这个类
 # 也就是不管后面是 Chunk 转换器还是 Message 转换器，或者是图片转换器最终都要实现这个接口
class BaseTransform(ABC):
    """Abstract transform operating on Chunk lists."""

    @abstractmethod
    def transform(
        self, chunks: list[Chunk], trace: "TraceContext | None" = None
    ) -> list[Chunk]:
        """Transform chunks and return a new chunk list."""

    @staticmethod
    def validate_chunks(chunks: list[Chunk]) -> None:
        """Validate transform input."""
        if not isinstance(chunks, list):
            raise ValueError("chunks must be a list")
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise ValueError(f"chunks[{idx}] must be a Chunk")
