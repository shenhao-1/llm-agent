"""Build MCP-compatible responses from retrieval results."""

from __future__ import annotations

from typing import Any

from core.types import RetrievalResult

from .citation_generator import CitationGenerator
from .multimodal_assembler import MultimodalAssembler


class ResponseBuilder:
    """Format retrieval output as Markdown plus structured citations."""

    def __init__(
        self,
        citation_generator: CitationGenerator | None = None,
        multimodal_assembler: MultimodalAssembler | None = None,
    ) -> None:
        self.citation_generator = citation_generator or CitationGenerator()
        self.multimodal_assembler = multimodal_assembler or MultimodalAssembler()

    def build(
        self,
        retrieval_results: list[RetrievalResult],
        query: str,
    ) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(retrieval_results, list):
            raise ValueError("retrieval_results must be a list")

        citations = self.citation_generator.generate(retrieval_results)
        markdown = self._build_markdown(query.strip(), retrieval_results, citations)
        image_content = self.multimodal_assembler.assemble_images(retrieval_results)
        return {
            "content": [{"type": "text", "text": markdown}, *image_content],
            "structuredContent": {
                "query": query.strip(),
                "citations": citations,
                "image_count": len(image_content),
                "result_count": len(retrieval_results),
            },
        }

    def _build_markdown(
        self,
        query: str,
        retrieval_results: list[RetrievalResult],
        citations: list[dict[str, Any]],
    ) -> str:
        if not retrieval_results:
            return (
                f"No relevant documents found for: {query}\n\n"
                "Please run ingestion first or broaden the query."
            )

        lines = [f"## Query Results for: {query}", ""]
        for index, result in enumerate(retrieval_results, start=1):
            citation = citations[index - 1]
            source = citation["source"]
            page = citation["page"] if citation["page"] is not None else "-"
            lines.extend(
                [
                    f"### [{index}] {source}",
                    f"- Score: {result.score:.4f}",
                    f"- Page: {page}",
                    "",
                    f"{_summarize(result.text)} [{index}]",
                    "",
                ]
            )
        return "\n".join(lines).strip()


def _summarize(text: str, *, limit: int = 500) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


__all__ = ["ResponseBuilder"]
