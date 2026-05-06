"""Embedding abstractions and provider factory."""

from .azure_embedding import AzureEmbedding
from .base_embedding import BaseEmbedding
from .embedding_factory import EmbeddingFactory
from .ollama_embedding import OllamaEmbedding
from .openai_embedding import OpenAICompatibleEmbedding, OpenAIEmbedding

__all__ = [
    "AzureEmbedding",
    "BaseEmbedding",
    "EmbeddingFactory",
    "OllamaEmbedding",
    "OpenAICompatibleEmbedding",
    "OpenAIEmbedding",
]
