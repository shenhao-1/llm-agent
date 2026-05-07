"""Core data contracts shared by ingestion, retrieval, and MCP tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Mapping


Metadata = dict[str, Any]
SparseVector = dict[str, float]


@dataclass(frozen=True)
class Document:
    """Parsed source document before chunking."""
    # 创建时必须提供id和text，metadata可选，默认为空字典
    id: str
    text: str
    metadata: Metadata = field(default_factory=dict)
    # 这个是 创建对象后自动调用 检查输入合不合法
    def __post_init__(self) -> None:
        _validate_non_empty_string(self.id, "Document.id")
        _validate_text(self.text, "Document.text")
        _validate_metadata(self.metadata, "Document.metadata")
    #输出字典 方便存储、测试、传给别的模块
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation with stable fields."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": _copy_mapping(self.metadata),
        }
    # 字典恢复对象 重新得到一个 Document 对象。 输入是payload 就是一个字典
    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Document":
        """Build a Document from a mapping and validate the contract."""
        _validate_mapping(payload, "Document payload")
        return cls(
            id=payload.get("id"),
            text=payload.get("text"),
            metadata=_copy_mapping(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class Chunk:
    """Text fragment derived from a Document."""

    id: str
    text: str
    metadata: Metadata = field(default_factory=dict)
    start_offset: int = 0
    end_offset: int = 0
    source_ref: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.id, "Chunk.id")
        _validate_text(self.text, "Chunk.text")
        _validate_metadata(self.metadata, "Chunk.metadata")
        _validate_offsets(self.start_offset, self.end_offset)
        if self.source_ref is not None:
            _validate_non_empty_string(self.source_ref, "Chunk.source_ref")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation with stable fields."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": _copy_mapping(self.metadata),
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "source_ref": self.source_ref,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Chunk":
        """Build a Chunk from a mapping and validate the contract."""
        _validate_mapping(payload, "Chunk payload")
        return cls(
            id=payload.get("id"),
            text=payload.get("text"),
            metadata=_copy_mapping(payload.get("metadata", {})),
            start_offset=payload.get("start_offset", 0),
            end_offset=payload.get("end_offset", 0),
            source_ref=payload.get("source_ref"),
        )


@dataclass(frozen=True)
class ChunkRecord:
    """Storage and retrieval record for a chunk and its encoded signals."""

    id: str
    text: str
    metadata: Metadata = field(default_factory=dict)
    dense_vector: list[float] | None = None
    sparse_vector: SparseVector | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.id, "ChunkRecord.id")
        _validate_text(self.text, "ChunkRecord.text")
        _validate_metadata(self.metadata, "ChunkRecord.metadata")
        if self.dense_vector is not None:
            _validate_dense_vector(self.dense_vector, "ChunkRecord.dense_vector")
        if self.sparse_vector is not None:
            _validate_sparse_vector(self.sparse_vector, "ChunkRecord.sparse_vector")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation with stable fields."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": _copy_mapping(self.metadata),
            "dense_vector": list(self.dense_vector) if self.dense_vector is not None else None,
            "sparse_vector": dict(self.sparse_vector)
            if self.sparse_vector is not None
            else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ChunkRecord":
        """Build a ChunkRecord from a mapping and validate the contract."""
        _validate_mapping(payload, "ChunkRecord payload")
        dense_vector = payload.get("dense_vector")
        sparse_vector = payload.get("sparse_vector")
        return cls(
            id=payload.get("id"),
            text=payload.get("text"),
            metadata=_copy_mapping(payload.get("metadata", {})),
            dense_vector=list(dense_vector) if dense_vector is not None else None,
            sparse_vector=dict(sparse_vector) if sparse_vector is not None else None,
        )


def image_placeholder(image_id: str) -> str:
    """Return the canonical placeholder used inside Document.text."""
    _validate_non_empty_string(image_id, "image_id")
    return f"[IMAGE: {image_id}]"


def _validate_mapping(value: Any, field_name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")


def _validate_non_empty_string(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_text(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")


def _validate_metadata(metadata: Any, field_name: str) -> None:
    _validate_mapping(metadata, field_name)
    source_path = metadata.get("source_path")
    _validate_non_empty_string(source_path, f"{field_name}.source_path")

    images = metadata.get("images")
    if images is not None:
        _validate_images(images, f"{field_name}.images")


def _validate_images(images: Any, field_name: str) -> None:
    if not isinstance(images, list):
        raise ValueError(f"{field_name} must be a list")

    for idx, image in enumerate(images):
        image_name = f"{field_name}[{idx}]"
        _validate_mapping(image, image_name)
        _validate_non_empty_string(image.get("id"), f"{image_name}.id")
        _validate_non_empty_string(image.get("path"), f"{image_name}.path")
        _validate_non_negative_int(image.get("text_offset"), f"{image_name}.text_offset")
        _validate_non_negative_int(image.get("text_length"), f"{image_name}.text_length")

        page = image.get("page")
        if page is not None:
            _validate_non_negative_int(page, f"{image_name}.page")

        position = image.get("position")
        if position is not None and not isinstance(position, Mapping):
            raise ValueError(f"{image_name}.position must be a mapping")


def _validate_offsets(start_offset: Any, end_offset: Any) -> None:
    _validate_non_negative_int(start_offset, "Chunk.start_offset")
    _validate_non_negative_int(end_offset, "Chunk.end_offset")
    if end_offset < start_offset:
        raise ValueError("Chunk.end_offset must be greater than or equal to start_offset")


def _validate_non_negative_int(value: Any, field_name: str) -> None:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _validate_dense_vector(vector: Any, field_name: str) -> None:
    if not isinstance(vector, list) or not vector:
        raise ValueError(f"{field_name} must be a non-empty list")

    for idx, value in enumerate(vector):
        if not isinstance(value, Real):
            raise ValueError(f"{field_name}[{idx}] must be a number")


def _validate_sparse_vector(vector: Any, field_name: str) -> None:
    _validate_mapping(vector, field_name)
    if not vector:
        raise ValueError(f"{field_name} must be a non-empty mapping")

    for key, value in vector.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name} keys must be non-empty strings")
        if not isinstance(value, Real):
            raise ValueError(f"{field_name}[{key!r}] must be a number")


def _copy_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    _validate_mapping(value, "mapping")
    return dict(value)
