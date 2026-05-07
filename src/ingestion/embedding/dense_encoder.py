"""Dense vector encoding for ingestion chunks."""

from __future__ import annotations

from numbers import Real
from typing import Any, Mapping, TYPE_CHECKING

from core.types import Chunk, ChunkRecord
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory

if TYPE_CHECKING:
    from core.settings import Settings
    from core.trace.trace_context import TraceContext

    #
class DenseEncoder:
    """Encode chunk text into dense vectors using the configured embedding provider."""
    #这个类的作用是将文本块（Chunk）编码成密集向量（dense vectors），以便在后续的处理过程中使用。这通常用于自然语言处理任务，例如文本相似度计算、信息检索等。
    def __init__(
        self,
        settings: "Settings | Mapping[str, Any] | None" = None,
        embedding: BaseEmbedding | None = None,
    ) -> None:
        self.settings = settings or {}
        self.embedding = embedding or EmbeddingFactory.create(self.settings)
    #在初始化方法中，DenseEncoder接受两个可选参数：settings和embedding。settings可以是一个Settings对象，也可以是一个字典，或者是None。如果没有提供settings，则默认为一个空字典。embedding参数可以是一个BaseEmbedding对象，如果没有提供，则通过EmbeddingFactory根据settings创建一个默认的嵌入对象。
    def encode(
        self, chunks: list[Chunk], trace: "TraceContext | None" = None
    ) -> list[ChunkRecord]:
        """Return ChunkRecords carrying dense vectors for each input chunk."""
        self._validate_chunks(chunks)
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        vectors = self.embedding.embed(texts, trace=trace)
        self._validate_vectors(vectors, expected_count=len(chunks))

        records = [
            ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=dict(chunk.metadata),
                dense_vector=list(vector),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        if trace is not None:
            trace.record_stage(
                "embed",
                method="dense_encoder",
                details={
                    "chunk_count": len(chunks),
                    "vector_count": len(vectors),
                    "dimension": len(vectors[0]) if vectors else 0,
                },
            )

        return records
        #encode方法是DenseEncoder的核心功能。它接受一个Chunk对象的列表，并返回一个ChunkRecord对象的列表，每个ChunkRecord包含了对应Chunk的文本、元数据以及通过嵌入模型生成的密集向量。方法首先验证输入的chunks是否有效，然后提取文本并调用embedding对象的embed方法生成向量。最后，它将每个Chunk与其对应的向量组合成ChunkRecord，并记录嵌入阶段的信息（如果trace上下文存在）。
    @staticmethod
    def _validate_chunks(chunks: Any) -> None:
        if not isinstance(chunks, list):
            raise ValueError("chunks must be a list")
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise ValueError(f"chunks[{idx}] must be a Chunk")
    #_validate_chunks方法用于验证输入的chunks参数是否是一个列表，并且列表中的每个元素都是Chunk对象。如果验证失败，它会抛出一个ValueError异常，指出具体的问题。
    @staticmethod
    def _validate_vectors(vectors: Any, expected_count: int) -> None:
        if not isinstance(vectors, list):
            raise ValueError("embedding output must be a list")
        if len(vectors) != expected_count:
            raise ValueError(
                "embedding output count must match chunks count "
                f"({len(vectors)} != {expected_count})"
            )
        if not vectors:
            return

        first_dimension = DenseEncoder._validate_vector(vectors[0], "vectors[0]")
        for idx, vector in enumerate(vectors[1:], start=1):
            dimension = DenseEncoder._validate_vector(vector, f"vectors[{idx}]")
            if dimension != first_dimension:
                raise ValueError("embedding vectors must have consistent dimensions")

    @staticmethod
    def _validate_vector(vector: Any, field_name: str) -> int:
        if not isinstance(vector, list) or not vector:
            raise ValueError(f"{field_name} must be a non-empty list")
        for idx, value in enumerate(vector):
            if not isinstance(value, Real):
                raise ValueError(f"{field_name}[{idx}] must be a number")
        return len(vector)


__all__ = ["DenseEncoder"]
