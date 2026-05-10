"""JSON-RPC 2.0 protocol handling for the MCP server."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import json
import logging
from typing import Any, Mapping


SERVER_NAME = "modular-rag-mcp-server"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2025-06-18"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

ToolCallable = Callable[[dict[str, Any]], Mapping[str, Any] | str]


class JsonRpcError(Exception):
    """Error that maps directly to a JSON-RPC error response."""

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


@dataclass(frozen=True)
class ToolDefinition:
    """Registered MCP tool schema plus executable handler."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolCallable

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("tool name must be a non-empty string")
        if not isinstance(self.description, str):
            raise ValueError("tool description must be a string")
        if not isinstance(self.input_schema, dict):
            raise ValueError("tool input_schema must be a dict")
        if not callable(self.handler):
            raise ValueError("tool handler must be callable")

    def to_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": dict(self.input_schema),
        }


class ProtocolHandler:
    """Parse JSON-RPC requests and route core MCP methods."""

    def __init__(
        self,
        tools: Iterable[ToolDefinition] | None = None,
        *,
        server_name: str = SERVER_NAME,
        server_version: str = SERVER_VERSION,
        protocol_version: str = PROTOCOL_VERSION,
        logger: logging.Logger | None = None,
    ) -> None:
        self.server_name = _non_empty_string(server_name, "server_name")
        self.server_version = _non_empty_string(server_version, "server_version")
        self.protocol_version = _non_empty_string(protocol_version, "protocol_version")
        self.logger = logger or logging.getLogger(__name__)
        self.tools: dict[str, ToolDefinition] = {}
        for tool in tools or []:
            self.register_tool(tool)

    def register_tool(self, tool: ToolDefinition) -> None:
        if not isinstance(tool, ToolDefinition):
            raise ValueError("tool must be a ToolDefinition")
        if tool.name in self.tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self.tools[tool.name] = tool

    def handle_message(self, raw_message: str | Mapping[str, Any]) -> dict[str, Any] | None:
        """Return a JSON-RPC response for one request, or None for notifications."""
        try:
            payload = self._parse_payload(raw_message)
        except JsonRpcError as exc:
            self.logger.warning("invalid JSON-RPC payload: %s", exc.message)
            return self._error_response(None, exc.code, exc.message)

        request_id = payload.get("id")
        try:
            if payload.get("jsonrpc") != "2.0":
                raise JsonRpcError(INVALID_REQUEST, "Invalid Request")
            method = payload.get("method")
            if not isinstance(method, str) or not method.strip():
                raise JsonRpcError(INVALID_REQUEST, "Invalid Request")

            if method == "notifications/initialized":
                self.logger.info("client initialized notification received")
                return None

            result = self._dispatch(method, payload.get("params"))
            return self._result_response(request_id, result)
        except JsonRpcError as exc:
            self.logger.warning("JSON-RPC request failed: %s", exc.message)
            return self._error_response(request_id, exc.code, exc.message)
        except Exception:
            self.logger.exception("internal JSON-RPC handler failure")
            return self._error_response(request_id, INTERNAL_ERROR, "Internal error")

    def handle_initialize(self, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if params is not None and not isinstance(params, Mapping):
            raise JsonRpcError(INVALID_PARAMS, "Invalid params")
        self.logger.info("initialize request received")
        return {
            "protocolVersion": self.protocol_version,
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": self.server_name,
                "version": self.server_version,
            },
        }

    def handle_tools_list(self) -> dict[str, Any]:
        self.logger.info("tools/list request received")
        return {
            "tools": [
                self.tools[name].to_schema()
                for name in sorted(self.tools)
            ]
        }

    def handle_tools_call(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(name, str) or not name.strip():
            raise JsonRpcError(INVALID_PARAMS, "Invalid params")
        tool = self.tools.get(name)
        if tool is None:
            raise JsonRpcError(INVALID_PARAMS, f"Unknown tool: {name}")
        if arguments is not None and not isinstance(arguments, Mapping):
            raise JsonRpcError(INVALID_PARAMS, "Invalid params")

        self.logger.info("tools/call request received for %s", name)
        result = tool.handler(dict(arguments or {}))
        if isinstance(result, str):
            return {"content": [{"type": "text", "text": result}]}
        if isinstance(result, Mapping):
            return dict(result)
        raise JsonRpcError(INTERNAL_ERROR, "Internal error")

    def _dispatch(self, method: str, params: Any) -> dict[str, Any]:
        if method == "initialize":
            return self.handle_initialize(params)
        if method == "tools/list":
            if params is not None and not isinstance(params, Mapping):
                raise JsonRpcError(INVALID_PARAMS, "Invalid params")
            return self.handle_tools_list()
        if method == "tools/call":
            if not isinstance(params, Mapping):
                raise JsonRpcError(INVALID_PARAMS, "Invalid params")
            name = params.get("name")
            arguments = params.get("arguments", {})
            if not isinstance(name, str):
                raise JsonRpcError(INVALID_PARAMS, "Invalid params")
            return self.handle_tools_call(name, arguments)
        self.logger.warning("unsupported JSON-RPC method: %s", method)
        raise JsonRpcError(METHOD_NOT_FOUND, "Method not found")

    @staticmethod
    def _parse_payload(raw_message: str | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(raw_message, str):
            try:
                parsed = json.loads(raw_message)
            except json.JSONDecodeError as exc:
                raise JsonRpcError(PARSE_ERROR, "Parse error") from exc
        elif isinstance(raw_message, Mapping):
            parsed = dict(raw_message)
        else:
            raise JsonRpcError(INVALID_REQUEST, "Invalid Request")

        if not isinstance(parsed, dict):
            raise JsonRpcError(INVALID_REQUEST, "Invalid Request")
        return parsed

    @staticmethod
    def _result_response(request_id: Any, result: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": dict(result),
        }

    @staticmethod
    def _error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message,
            },
        }


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


__all__ = [
    "INTERNAL_ERROR",
    "INVALID_PARAMS",
    "INVALID_REQUEST",
    "JsonRpcError",
    "METHOD_NOT_FOUND",
    "PARSE_ERROR",
    "PROTOCOL_VERSION",
    "ProtocolHandler",
    "SERVER_NAME",
    "SERVER_VERSION",
    "ToolDefinition",
]
