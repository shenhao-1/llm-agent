"""Optional image caption generation for multimodal chunks.
输入:List[Chunk]
检查每个 Chunk 的 metadata.image_refs / metadata.images
如果没有图片：原样返回
如果有图片但未启用：标记 has_unprocessed_images=True
如果有图片且启用 Vision LLM:逐张调用 chat_with_image()
成功：写入 image_captions 和 images[].caption
失败：记录 image_caption_errors,不中断流程
输出:List[Chunk]

"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

from core.types import Chunk
from libs.llm.base_vision_llm import BaseVisionLLM
from libs.llm.llm_factory import LLMFactory

from .base_transform import BaseTransform

if TYPE_CHECKING:
    from core.settings import Settings
    from core.trace.trace_context import TraceContext


DEFAULT_PROMPT_PATH = Path("config/prompts/image_captioning.txt") # 提示词路径
DEFAULT_PROMPT_TEMPLATE = (
    "Describe the image with concise, factual details useful for retrieval.\n\n"
    "Chunk context:\n{text}"
)

#产生图像标题的转换器，当启用视觉LLM时使用。它检查块的元数据中的图像引用，并使用视觉LLM生成描述图像内容的标题。生成的标题被添加回块的元数据中，以增强基于文本的检索和理解。
class ImageCaptioner(BaseTransform): # 继承BaseTransform 同C5 C6一样的组件
    """Generate captions for image references when Vision LLM is enabled."""

    def __init__(
        self,
        settings: "Settings | Mapping[str, Any] | None" = None,
        vision_llm: BaseVisionLLM | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        self.settings = settings or {}
        self.config = self._extract_config(self.settings)
        self.enabled = bool(
            self.config.get(
                "enabled",
                self.config.get("use_vision_llm", self.config.get("use_llm", False)),
            )
        )
        self.prompt_template = self._load_prompt(prompt_path or self.config.get("prompt_path"))
        self.vision_llm = vision_llm
        if self.enabled and self.vision_llm is None:
            self.vision_llm = LLMFactory.create_vision_llm(self.settings)
    # 该方法是转换器的核心，负责处理输入的块列表并生成图像标题。它首先验证输入块的格式，然后迭代每个块，调用_caption_chunk方法来处理包含图像引用的块。处理完成后，它记录转换阶段的统计信息，并返回转换后的块列表。
    def transform( # 主入口
        self, chunks: list[Chunk], trace: "TraceContext | None" = None
    ) -> list[Chunk]:
        self.validate_chunks(chunks)
        transformed: list[Chunk] = []
        captioned_images = 0

        for chunk in chunks:
            new_chunk, count = self._caption_chunk(chunk, trace) # 处理每个块，生成图像标题并返回新的块和成功生成的标题数量
            transformed.append(new_chunk)
            captioned_images += count

        if trace is not None:  #记录的
            trace.record_stage(
                "transform",
                method="image_captioner",
                details={
                    "chunk_count": len(chunks),
                    "enabled": self.enabled,
                    "captioned_images": captioned_images,
                },
            )

        return transformed

    def _caption_chunk(
        self, chunk: Chunk, trace: "TraceContext | None" = None
    ) -> tuple[Chunk, int]:
        metadata = deepcopy(chunk.metadata)  # 复制块的元数据以避免修改原始数据
        images = self._normalize_images(metadata) # 从元数据中提取和规范化图像信息，确保它们以一致的格式存在
        image_refs = self._image_refs(metadata, images) # 确定哪些图像需要生成标题，基于块的元数据中的图像引用或图像列表中的ID字段
        if not image_refs:
            return chunk, 0

        if not self.enabled:
            metadata["has_unprocessed_images"] = True
            metadata["image_caption_status"] = "disabled"
            return self._replace_chunk(chunk, metadata), 0

        if self.vision_llm is None:
            metadata["has_unprocessed_images"] = True
            metadata["image_caption_status"] = "vision_llm_unavailable"
            return self._replace_chunk(chunk, metadata), 0

        captioned = 0
        captions: dict[str, str] = {}
        errors: dict[str, str] = {}
        image_by_id = {image.get("id"): image for image in images if isinstance(image, Mapping)}

        for image_id in image_refs:
            image = image_by_id.get(image_id)
            if not isinstance(image, dict):
                errors[image_id] = "image metadata not found"
                continue

            image_path = image.get("path")
            if not isinstance(image_path, str) or not image_path.strip():
                errors[image_id] = "image path missing"
                continue

            try: #调用视觉LLM生成图像标题，传入构建的提示词、图像路径和跟踪上下文。如果生成成功，标题将被清理并添加到块的元数据中；如果发生错误或生成的标题无效，将记录相应的错误信息。
                caption = self.vision_llm.chat_with_image(
                    self._build_prompt(chunk, image),
                    image_path,
                    trace=trace,
                )
            except Exception as exc:
                errors[image_id] = str(exc)
                continue

            if not isinstance(caption, str) or not caption.strip():
                errors[image_id] = "vision llm returned empty caption"
                continue

            cleaned_caption = caption.strip()
            image["caption"] = cleaned_caption
            captions[image_id] = cleaned_caption
            captioned += 1

        metadata["images"] = images
        if captions:
            metadata["image_captions"] = captions
            metadata["image_caption_status"] = "success" if not errors else "partial"
        if errors:
            metadata["has_unprocessed_images"] = True
            metadata["image_caption_errors"] = errors
            metadata.setdefault("image_caption_status", "failed")
        elif image_refs:
            metadata["has_unprocessed_images"] = False

        # Keep the image placeholder, and add searchable text beside it.
        # Re-running the transform must not append the same caption twice.
        searchable_text = chunk.text
        for image_id, caption in captions.items():
            addition = f"[Image description {image_id}: {caption}]"
            if addition not in searchable_text:
                searchable_text += f"\n{addition}"
        return self._replace_chunk(chunk, metadata, text=searchable_text), captioned
    #该方法负责构建传递给视觉LLM的提示词。它使用块的文本、图像ID和源路径等信息填充预定义的提示模板。如果提示模板中没有{text}占位符，它会将块的文本附加到提示词的末尾，以确保视觉LLM有足够的上下文来生成相关的图像标题。
    def _build_prompt(self, chunk: Chunk, image: Mapping[str, Any]) -> str:
        values = {
            "text": chunk.text,
            "image_id": image.get("id", ""),
            "source_path": chunk.metadata.get("source_path", ""),
        }
        try:
            prompt = self.prompt_template.format(**values)
        except KeyError:
            prompt = self.prompt_template
        if "{text}" not in self.prompt_template:
            prompt = f"{prompt.rstrip()}\n\nChunk context:\n{chunk.text}"
        return prompt
    # 该方法从块的元数据中提取图像信息，并确保它们以一致的格式存在。它检查元数据中的"images"字段，如果存在且是一个列表，它会过滤出其中的字典项并返回一个新的图像列表。如果"images"字段不存在或不是一个列表，它将返回一个空列表。
    @staticmethod
    def _normalize_images(metadata: dict[str, Any]) -> list[dict[str, Any]]:
        images = metadata.get("images")
        if images is None:
            return []
        if not isinstance(images, list):
            raise ValueError("metadata.images must be a list")
        return [deepcopy(image) for image in images if isinstance(image, Mapping)]
    # 该方法确定哪些图像需要生成标题，基于块的元数据中的图像引用或图像列表中的ID字段。它首先检查元数据中的"image_refs"字段，如果存在且是一个列表，它会过滤出其中的字符串项并返回它们作为图像引用。如果"image_refs"字段不存在或不是一个列表，它将检查图像列表中的每个图像，提取其中的ID字段，并返回这些ID作为图像引用。
    @staticmethod
    def _image_refs(metadata: Mapping[str, Any], images: list[dict[str, Any]]) -> list[str]:
        refs = metadata.get("image_refs")
        if isinstance(refs, list):
            return [ref for ref in refs if isinstance(ref, str) and ref.strip()]
        return [
            image["id"]
            for image in images
            if isinstance(image.get("id"), str) and image["id"].strip()
        ]

    @staticmethod
    def _load_prompt(prompt_path: str | Path | None = None) -> str:
        path = Path(prompt_path) if prompt_path else DEFAULT_PROMPT_PATH
        try:
            template = path.read_text(encoding="utf-8")
        except OSError:
            return DEFAULT_PROMPT_TEMPLATE
        return template if template.strip() else DEFAULT_PROMPT_TEMPLATE
    # 该方法从设置中提取图像标题生成器的配置。它首先检查设置对象是否具有"ingestion"属性，如果存在且是一个映射类型，它会进一步检查其中是否有"image_captioner"配置，并返回它作为字典。如果在第一层没有找到配置，它会直接检查设置对象是否是一个映射类型，并尝试从中提取"image_captioner"配置或嵌套在"ingestion"中的配置。如果都没有找到有效的配置，它将返回一个空字典。
    @staticmethod
    def _extract_config(settings: "Settings | Mapping[str, Any]") -> dict[str, Any]:
        if hasattr(settings, "ingestion"):
            ingestion = getattr(settings, "ingestion")
            if isinstance(ingestion, Mapping):
                captioner = ingestion.get("image_captioner")
                if isinstance(captioner, Mapping):
                    return dict(captioner)

        if isinstance(settings, Mapping):
            captioner = settings.get("image_captioner")
            if isinstance(captioner, Mapping):
                return dict(captioner)

            ingestion = settings.get("ingestion")
            if isinstance(ingestion, Mapping):
                nested = ingestion.get("image_captioner")
                if isinstance(nested, Mapping):
                    return dict(nested)

        return {}
    # 该方法创建一个新的块，保留原始块的ID、文本、偏移和源引用，但使用更新后的元数据。它通过从原始块复制这些字段并替换元数据来实现这一点，从而确保生成的新块与原始块在结构上保持一致，同时包含新的图像标题信息。
    @staticmethod
    def _replace_chunk(
        chunk: Chunk, metadata: dict[str, Any], *, text: str | None = None
    ) -> Chunk:
        return Chunk(
            id=chunk.id,
            text=chunk.text if text is None else text,
            metadata=metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref,
        )


__all__ = ["ImageCaptioner"]
