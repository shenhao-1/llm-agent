"""Unit tests for JSON-RPC protocol handling."""

from __future__ import annotations

import json

from mcp_server.protocol_handler import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    ProtocolHandler,
    ToolDefinition,
)


def test_initialize_returns_server_info_and_capabilities() -> None:
    response = ProtocolHandler().handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        }
    )

    assert response is not None
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert response["result"]["serverInfo"]["name"] == "modular-rag-mcp-server"
    assert response["result"]["capabilities"] == {"tools": {}}


def test_tools_list_returns_registered_tool_schemas_sorted() -> None:
    handler = ProtocolHandler(
        [
            _tool("z_tool", "last"),
            _tool("a_tool", "first"),
        ]
    )

    response = handler.handle_message(
        {"jsonrpc": "2.0", "id": "list", "method": "tools/list", "params": {}}
    )

    assert response is not None
    assert response["result"] == {
        "tools": [
            {
                "name": "a_tool",
                "description": "first",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "z_tool",
                "description": "last",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]
    }


def test_tools_call_routes_to_registered_handler() -> None:
    calls: list[dict[str, object]] = []

    def echo(arguments: dict[str, object]) -> dict[str, object]:
        calls.append(arguments)
        return {"content": [{"type": "text", "text": str(arguments["query"])}]}

    handler = ProtocolHandler([_tool("echo", "Echo query", handler=echo)])

    response = handler.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"query": "hello"}},
        }
    )

    assert calls == [{"query": "hello"}]
    assert response == {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {"content": [{"type": "text", "text": "hello"}]},
    }


def test_tools_call_wraps_string_results_as_text_content() -> None:
    handler = ProtocolHandler(
        [_tool("hello", "Hello", handler=lambda _arguments: "hello world")]
    )

    response = handler.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "hello", "arguments": {}},
        }
    )

    assert response is not None
    assert response["result"] == {
        "content": [{"type": "text", "text": "hello world"}]
    }


def test_notifications_initialized_returns_no_response() -> None:
    response = ProtocolHandler().handle_message(
        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
    )

    assert response is None


def test_unknown_method_returns_method_not_found() -> None:
    response = ProtocolHandler().handle_message(
        {"jsonrpc": "2.0", "id": 4, "method": "unknown"}
    )

    assert response is not None
    assert response["error"]["code"] == METHOD_NOT_FOUND
    assert response["error"]["message"] == "Method not found"


def test_invalid_params_returns_invalid_params() -> None:
    response = ProtocolHandler().handle_message(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"arguments": {}},
        }
    )

    assert response is not None
    assert response["error"]["code"] == INVALID_PARAMS
    assert response["error"]["message"] == "Invalid params"


def test_unknown_tool_returns_invalid_params() -> None:
    response = ProtocolHandler().handle_message(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "missing", "arguments": {}},
        }
    )

    assert response is not None
    assert response["error"]["code"] == INVALID_PARAMS
    assert response["error"]["message"] == "Unknown tool: missing"


def test_internal_handler_error_is_sanitized() -> None:
    def broken(_arguments: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("database password leaked")

    handler = ProtocolHandler([_tool("broken", "Broken", handler=broken)])

    response = handler.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": "broken", "arguments": {}},
        }
    )

    assert response is not None
    assert response["error"] == {"code": INTERNAL_ERROR, "message": "Internal error"}
    assert "password" not in json.dumps(response)


def test_parse_error_and_invalid_request_shape() -> None:
    parse_response = ProtocolHandler().handle_message("{not-json")
    invalid_response = ProtocolHandler().handle_message(["bad"])  # type: ignore[arg-type]

    assert parse_response is not None
    assert parse_response["error"]["code"] == PARSE_ERROR
    assert invalid_response is not None
    assert invalid_response["error"]["code"] == INVALID_REQUEST


def _tool(
    name: str,
    description: str,
    *,
    handler=None,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": {}},
        handler=handler or (lambda _arguments: {"content": []}),
    )
