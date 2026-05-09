"""Rule-based query preprocessing for retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
FILTER_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*):(\"[^\"]+\"|'[^']+'|[^\s]+)")

DEFAULT_FILTER_FIELDS = {
    "access_level",
    "collection",
    "doc_hash",
    "doc_type",
    "language",
    "source_path",
}
DEFAULT_STOPWORDS = {
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
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
}

#这是输入输出的格式，输入是一个字符串，输出是一个ProcessedQuery对象，包含原始查询、规范化查询、关键词列表和过滤器字典。QueryProcessor类提供了处理查询的方法，包括提取关键词和过滤器。
@dataclass(frozen=True)
class ProcessedQuery:
    """Normalized query payload consumed by retrievers."""

    original_query: str
    normalized_query: str
    keywords: list[str]
    filters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "keywords": list(self.keywords),
            "filters": dict(self.filters),
        }

#这是一个查询处理器类，用于从用户查询中提取关键词和简单的元数据过滤器。它支持配置最小关键词长度、停用词列表和可识别的过滤字段。处理方法会返回一个ProcessedQuery对象，包含原始查询、规范化查询、关键词列表和合并后的过滤器字典。
class QueryProcessor:
    """Extract keywords and simple metadata filters from user queries."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.min_keyword_length = _positive_int(
            self.config.get("min_keyword_length", 2),
            "min_keyword_length",
        )
        configured_stopwords = self.config.get("stopwords", DEFAULT_STOPWORDS)
        if not isinstance(configured_stopwords, (set, list, tuple)):
            raise ValueError("stopwords must be a sequence of strings")
        self.stopwords = {
            str(word).lower()
            for word in configured_stopwords
            if isinstance(word, str) and word.strip()
        }

        configured_fields = self.config.get("filter_fields", DEFAULT_FILTER_FIELDS)
        if not isinstance(configured_fields, (set, list, tuple)):
            raise ValueError("filter_fields must be a sequence of strings")
        self.filter_fields = {
            str(field).lower()
            for field in configured_fields
            if isinstance(field, str) and field.strip()
        }
    #主入口
    def process(
        self,
        query: str,  #原始问题
        filters: Mapping[str, Any] | None = None, #过滤条件
        trace: "TraceContext | None" = None, #链路追踪对象
    ) -> ProcessedQuery: #这是输出 里面四个字段
        """Return normalized keywords and merged metadata filters."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if filters is not None and not isinstance(filters, Mapping):
            raise ValueError("filters must be a mapping when provided")

        original_query = query
        extracted_filters, query_without_filters = self._extract_filters(query)
        merged_filters = dict(extracted_filters)
        if filters:
            merged_filters.update(dict(filters))

        normalized_query = _normalize_query(query_without_filters)
        keywords = self.extract_keywords(normalized_query)
        if not keywords:
            keywords = self.extract_keywords(original_query, ignore_stopwords=True)
        if not keywords:
            raise ValueError("query must contain at least one keyword")

        result = ProcessedQuery(
            original_query=original_query,
            normalized_query=normalized_query,
            keywords=keywords,
            filters=merged_filters,
        )

        if trace is not None:
            trace.record_stage(
                "query_process",
                method="query_processor",
                details={
                    "keyword_count": len(result.keywords),
                    "filters": dict(result.filters),
                },
            )

        return result
#提取关键词 核心是正则 提取英文 数字 下划线 的连续字符串，转换为小写，去除停用词和重复项，并根据最小长度过滤掉过短的关键词。
    def extract_keywords(
        self,
        text: str,
        *,
        ignore_stopwords: bool = False,
    ) -> list[str]:
        """Tokenize, normalize, de-duplicate, and optionally remove stopwords."""
        if not isinstance(text, str):
            raise ValueError("text must be a string")

        keywords: list[str] = []
        seen: set[str] = set()
        for token in TOKEN_RE.findall(text):
            normalized = token.lower()
            if len(normalized) < self.min_keyword_length:
                continue
            if not ignore_stopwords and normalized in self.stopwords:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            keywords.append(normalized)
        return keywords
 # 是从query 里面解析过滤条件
    def _extract_filters(self, query: str) -> tuple[dict[str, str], str]:
        filters: dict[str, str] = {}

        def replace(match: re.Match[str]) -> str:
            field = match.group(1).lower()
            raw_value = match.group(2)
            if field not in self.filter_fields:
                return match.group(0)
            filters[field] = _strip_quotes(raw_value)
            return " "

        return filters, FILTER_RE.sub(replace, query)

    # 多个空格合成一个 去掉首尾空格
def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()

# 是给过滤器用的
def _strip_quotes(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1]
    return stripped


def _positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


__all__ = ["ProcessedQuery", "QueryProcessor"]
