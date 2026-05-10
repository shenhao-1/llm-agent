"""MCP tool registration helpers."""

from mcp_server.protocol_handler import ToolDefinition

from .get_document_summary import tool_definition as get_document_summary_tool
from .list_collections import tool_definition as list_collections_tool
from .query_knowledge_hub import tool_definition as query_knowledge_hub_tool


def default_tools() -> list[ToolDefinition]:
    return [
        get_document_summary_tool(),
        list_collections_tool(),
        query_knowledge_hub_tool(),
    ]


__all__ = [
    "default_tools",
    "get_document_summary_tool",
    "list_collections_tool",
    "query_knowledge_hub_tool",
]
