"""Dense vector retrieval orchestration."""

from __future__ import annotations

from typing import Any, Mapping, TYPE_CHECKING

from core.types import RetrievalResult
from libs.embedding import BaseEmbedding, EmbeddingFactory
from libs.vector_store import BaseVectorStore, VectorStoreFactory

if TYPE_CHECKING:
    from core.settings import Settings
    from core.trace.trace_context import TraceContext

#这是一个稠密检索器类，负责将查询嵌入到向量空间中，并从向量存储中检索语义相似的文本块。它使用一个嵌入客户端来生成查询的向量表示，并使用一个向量存储来执行查询。检索结果被规范化为RetrievalResult对象列表返回。
class DenseRetriever:
    """Embed a query and retrieve semantically similar chunks from a vector store."""

    def __init__(
        self,
        settings: "Settings | Mapping[str, Any] | None" = None,
        embedding_client: BaseEmbedding | None = None,
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        if settings is None and (embedding_client is None or vector_store is None):
            raise ValueError(
                "settings are required unless both embedding_client and vector_store are provided"
            )

        self.settings = settings or {}
        self.embedding_client = embedding_client or EmbeddingFactory.create(self.settings)
        self.vector_store = vector_store or VectorStoreFactory.create(self.settings)
    #这是稠密检索方法
    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: "TraceContext | None" = None,
    ) -> list[RetrievalResult]:
        """Return normalized dense retrieval results."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if filters is not None and not isinstance(filters, Mapping):
            raise ValueError("filters must be a mapping when provided")

        vectors = self.embedding_client.embed([query], trace=trace)
        if not isinstance(vectors, list) or len(vectors) != 1:
            raise ValueError("embedding_client must return exactly one query vector")
        BaseVectorStore.validate_vector(vectors[0], "query_vector")

        raw_results = self.vector_store.query(
            [float(value) for value in vectors[0]],
            top_k,
            filters=filters,
            trace=trace,
        )
        results = [self._normalize_result(item) for item in raw_results]

        if trace is not None:
            trace.record_stage(
                "retrieve",
                method="dense_retriever",
                details={
                    "top_k": top_k,
                    "result_count": len(results),
                    "filters": dict(filters) if filters else {},
                },
            )

        return results

    @staticmethod
    def _normalize_result(item: Mapping[str, Any]) -> RetrievalResult:
        if not isinstance(item, Mapping):
            raise ValueError("vector store result must be a mapping")

        chunk_id = item.get("chunk_id", item.get("id"))
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("vector store result missing chunk id")

        text = item.get("text", item.get("document", ""))
        if not isinstance(text, str):
            text = str(text)

        metadata = item.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise ValueError("vector store result metadata must be a mapping")

        score = DenseRetriever._score_from_item(item)
        return RetrievalResult(
            chunk_id=chunk_id,
            score=score,
            text=text,
            metadata=dict(metadata),
        )
    #这个方法根据向量存储结果中的分数或距离字段计算一个标准化的相似度分数，确保所有结果都具有一致的评分机制。
    @staticmethod
    def _score_from_item(item: Mapping[str, Any]) -> float:
        score = item.get("score")
        if isinstance(score, (int, float)):
            return float(score)

        distance = item.get("distance")
        if isinstance(distance, (int, float)):
            return 1.0 / (1.0 + float(distance))

        return 0.0


__all__ = ["DenseRetriever"]
