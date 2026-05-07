"""Rule-based and optional LLM chunk refinement."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping, TYPE_CHECKING

from core.types import Chunk
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory

from .base_transform import BaseTransform

if TYPE_CHECKING:
    from core.settings import Settings
    from core.trace.trace_context import TraceContext


DEFAULT_PROMPT_PATH = Path("config/prompts/chunk_refinement.txt")
DEFAULT_PROMPT_TEMPLATE = (
    "Refine the following chunk to be self-contained. Preserve facts, "
    "remove noisy boilerplate, and return only the refined text.\n\n{text}"
)


class ChunkRefiner(BaseTransform):
    """Clean noisy chunks and optionally rewrite them with an LLM."""
    # 这个类的作用是对输入的 Chunk 列表进行清洗和优化，主要通过规则和可选的 LLM 来实现
    # 第一层：规则去噪 第二层：可选 LLM 增强
    def __init__(
        self,
        settings: "Settings | Mapping[str, Any] | None" = None,
        llm: BaseLLM | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        self.settings = settings or {}
        self.config = self._extract_config(self.settings)
        self.use_llm = bool(self.config.get("use_llm", False))
        self.prompt_template = self._load_prompt(prompt_path or self.config.get("prompt_path"))
        self.llm = llm
        if self.use_llm and self.llm is None:
            self.llm = LLMFactory.create(self.settings)
    # 这个方法的作用是对输入的 Chunk 列表进行处理，首先验证输入格式，然后对每个 Chunk 调用 _refine_chunk 方法进行规则和 LLM 的双重增强，最后记录处理阶段的统计信息并返回新的 Chunk 列表
    def transform(
        self, chunks: list[Chunk], trace: "TraceContext | None" = None
    ) -> list[Chunk]:
        self.validate_chunks(chunks)
        refined_chunks: list[Chunk] = []

        for chunk in chunks:
            try:
                refined_chunks.append(self._refine_chunk(chunk, trace))
            except Exception as exc:
                metadata = dict(chunk.metadata)
                metadata["refined_by"] = "original"
                metadata["refine_error"] = str(exc)
                refined_chunks.append(self._replace_chunk(chunk, chunk.text, metadata))

        if trace is not None:
            trace.record_stage(
                "transform",
                method="chunk_refiner",
                details={"chunk_count": len(chunks), "use_llm": self.use_llm},
            )

        return refined_chunks
    # 这个方法的作用是对单个 Chunk 进行规则和 LLM 的双重增强，首先通过规则去除噪声，然后如果配置了 LLM 就进一步优化文本，并更新元数据以反映处理方式和结果
    def _refine_chunk(self, chunk: Chunk, trace: "TraceContext | None" = None) -> Chunk:
        rule_text = self._rule_based_refine(chunk.text)
        metadata = dict(chunk.metadata)
        metadata["refined_by"] = "rule"

        if self.use_llm:
            llm_text, fallback_reason = self._llm_refine(rule_text, trace)
            if llm_text:
                metadata["refined_by"] = "llm"
                return self._replace_chunk(chunk, llm_text, metadata)
            metadata["fallback_reason"] = fallback_reason or "llm returned empty output"

        return self._replace_chunk(chunk, rule_text, metadata)
    # 规则增强
    def _rule_based_refine(self, text: str) -> str:
        """Remove common boilerplate while preserving code fence content.
        
        多余空白
        连续空行
        页眉页脚
        分隔线
        HTML 注释
        部分格式噪声
        """
        if not isinstance(text, str):
            raise ValueError("text must be a string")

        segments = re.split(r"(```.*?```)", text, flags=re.DOTALL)
        refined_segments: list[str] = []
        for segment in segments:
            if segment.startswith("```") and segment.endswith("```"):
                refined_segments.append(segment.strip("\n"))
            else:
                refined_segments.append(self._clean_non_code_text(segment))

        refined = "\n\n".join(segment for segment in refined_segments if segment)
        return refined.strip()
    # LLM 增强
    # 大意是 请清洗并优化下面的 Chunk 文本，去除噪声，保持原意，不要编造内容：
    def _llm_refine(
        self, text: str, trace: "TraceContext | None" = None
    ) -> tuple[str | None, str | None]:
        """Return LLM-refined text or a fallback reason."""
        if self.llm is None:
            return None, "llm is not configured"

        try:
            prompt = self.prompt_template.format(text=text)
            result = self.llm.chat(
                [
                    {
                        "role": "system",
                        "content": "You refine noisy retrieval chunks without changing facts.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )
        except Exception as exc:
            return None, str(exc)

        if not isinstance(result, str) or not result.strip():
            return None, "llm returned empty output"
        return result.strip(), None
# 下面是一些辅助方法，包括加载提示模板、提取配置、清洗文本和替换 Chunk 内容等，这些方法都是为了支持主流程的实现，确保代码的清晰和可维护性
    @staticmethod
    def _load_prompt(prompt_path: str | Path | None = None) -> str:
        path = Path(prompt_path) if prompt_path else DEFAULT_PROMPT_PATH
        try:
            template = path.read_text(encoding="utf-8")
        except OSError:
            return DEFAULT_PROMPT_TEMPLATE

        if "{text}" not in template:
            return f"{template.rstrip()}\n\n{{text}}"
        return template
    # 从设置中提取配置，支持多层嵌套和不同结构的配置方式，确保能够灵活适配各种配置格式
    @staticmethod
    def _extract_config(settings: "Settings | Mapping[str, Any]") -> dict[str, Any]:
        if hasattr(settings, "ingestion"):
            ingestion = getattr(settings, "ingestion")
            if isinstance(ingestion, Mapping):
                refiner = ingestion.get("chunk_refiner")
                if isinstance(refiner, Mapping):
                    return dict(refiner)

        if isinstance(settings, Mapping):
            refiner = settings.get("chunk_refiner")
            if isinstance(refiner, Mapping):
                return dict(refiner)

            ingestion = settings.get("ingestion")
            if isinstance(ingestion, Mapping):
                nested = ingestion.get("chunk_refiner")
                if isinstance(nested, Mapping):
                    return dict(nested)

        return {}
    # 这个方法的作用是清洗非代码文本，主要通过正则表达式去除一些常见的噪声，比如多余空白、连续空行、页眉页脚、分隔线、HTML 注释等，同时保留代码块内容不变
    @staticmethod
    def _clean_non_code_text(text: str) -> str:
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        lines = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                lines.append("")
                continue
            if re.fullmatch(r"[-=_*]{3,}", line):
                continue
            if re.fullmatch(r"page\s+\d+(\s+of\s+\d+)?", line, flags=re.IGNORECASE):
                continue
            if re.fullmatch(r"\d+\s*/\s*\d+", line):
                continue
            if line.lower() in {"confidential", "draft", "footer"}:
                continue
            line = re.sub(r"[ \t]+", " ", line)
            lines.append(line)

        cleaned = "\n".join(lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
    # 替换 Chunk 内容并更新元数据
    @staticmethod
    def _replace_chunk(chunk: Chunk, text: str, metadata: dict[str, Any]) -> Chunk:
        return Chunk(
            id=chunk.id,
            text=text,
            metadata=metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref,
        )


__all__ = ["ChunkRefiner"]
