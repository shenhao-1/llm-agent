"""Sparse term-statistics encoding for BM25 indexing."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any, Mapping, TYPE_CHECKING

from core.types import Chunk, ChunkRecord

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

    #这个模块实现了一个SparseEncoder类，用于将文本块（Chunk）编码成稀疏的词频向量，并计算语料库级别的统计信息，以便在BM25索引中使用。SparseEncoder的encode方法接受一个Chunk对象的列表，并返回一个SparseEncodingResult对象，其中包含了每个Chunk对应的ChunkRecord（包括稀疏向量）以及整个语料库的统计信息，如文档数量、平均文档长度、词项的文档频率等。SparseEncoder还包含一个tokenize方法，用于将文本分割成规范化的词项列表，过滤掉长度小于min_token_length的词项。
@dataclass(frozen=True)
class SparseEncodingResult:
    """Sparse records plus corpus statistics needed by BM25 indexing."""

    records: list[ChunkRecord]
    corpus_stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "records": [record.to_dict() for record in self.records],
            "corpus_stats": dict(self.corpus_stats),
        }

#这个是一个数据类（dataclass），用于封装SparseEncoder的输出结果。它包含两个字段：records是一个ChunkRecord对象的列表，每个ChunkRecord包含了对应Chunk的文本、元数据以及稀疏向量；corpus_stats是一个字典，包含了整个语料库的统计信息，如文档数量、平均文档长度、词项的文档频率等。SparseEncodingResult还提供了一个to_dict方法，用于将结果转换成一个JSON可序列化的字典格式，以便在需要时进行存储或传输。
class SparseEncoder:
    """Encode chunks into term-frequency vectors and corpus statistics."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        min_token_length = self.config.get("min_token_length", 2)
        if not isinstance(min_token_length, int) or min_token_length <= 0:
            raise ValueError("min_token_length must be a positive integer")
        self.min_token_length = min_token_length

    def encode(
        self, chunks: list[Chunk], trace: "TraceContext | None" = None
    ) -> SparseEncodingResult:
        """Return sparse ChunkRecords and corpus-level term statistics."""
        self._validate_chunks(chunks)
        records: list[ChunkRecord] = []
        document_frequency: Counter[str] = Counter()
        doc_lengths: dict[str, int] = {}

        for chunk in chunks:
            tokens = self.tokenize(chunk.text)
            term_counts = Counter(tokens)
            sparse_vector = (
                {term: float(count) for term, count in sorted(term_counts.items())}
                if term_counts
                else None
            )
            records.append(
                ChunkRecord(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=dict(chunk.metadata),
                    sparse_vector=sparse_vector,
                )
            )
            doc_lengths[chunk.id] = len(tokens)
            document_frequency.update(term_counts.keys())

        doc_count = len(chunks)
        total_terms = sum(doc_lengths.values())
        corpus_stats: dict[str, Any] = {
            "doc_count": doc_count,
            "avg_doc_length": (total_terms / doc_count) if doc_count else 0.0,
            "document_frequency": dict(sorted(document_frequency.items())),
            "doc_lengths": doc_lengths,
        }

        if trace is not None:
            trace.record_stage(
                "embed",
                method="sparse_encoder",
                details={
                    "chunk_count": doc_count,
                    "vocabulary_size": len(document_frequency),
                    "avg_doc_length": corpus_stats["avg_doc_length"],
                },
            )

        return SparseEncodingResult(records=records, corpus_stats=corpus_stats)

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into normalized terms for sparse retrieval."""
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        return [
            token.lower()
            for token in TOKEN_RE.findall(text)
            if len(token) >= self.min_token_length
        ]

    @staticmethod
    def _validate_chunks(chunks: Any) -> None:
        if not isinstance(chunks, list):
            raise ValueError("chunks must be a list")
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise ValueError(f"chunks[{idx}] must be a Chunk")


__all__ = ["SparseEncoder", "SparseEncodingResult"]
