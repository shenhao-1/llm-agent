"""Base contract for all vision-capable LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping


class BaseVisionLLM(ABC):
    """Abstract vision LLM contract used by multimodal ingestion steps."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})
    # 该方法定义了一个抽象方法chat_with_image，要求所有实现BaseVisionLLM的子类必须提供这个方法的具体实现。该方法接受一个文本提示和一个图像输入（可以是文件路径或原始字节），以及一个可选的跟踪参数，并返回模型生成的响应字符串。这种设计允许不同的视觉LLM提供商实现自己的图像处理和文本生成逻辑，同时保持接口的一致性。
    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: Any | None = None,
    ) -> str:
        """Return model output for a text prompt with an attached image."""
    
    @staticmethod
    def validate_text(text: str) -> None:
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        if not text.strip():
            raise ValueError("text must be a non-empty string")

    @staticmethod
    def validate_image_input(image_path: str | bytes) -> None:
        if isinstance(image_path, bytes):
            if not image_path:
                raise ValueError("image bytes must not be empty")
            return
        if isinstance(image_path, str):
            if not image_path.strip():
                raise ValueError("image_path must be a non-empty string")
            return
        raise ValueError("image_path must be a file path string or raw bytes")

    @staticmethod
    def load_image_bytes(image_path: str | bytes) -> bytes:
        BaseVisionLLM.validate_image_input(image_path)
        if isinstance(image_path, bytes):
            return image_path
        return Path(image_path).read_bytes()
