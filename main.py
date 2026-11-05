"""Application entrypoint for the MCP server runtime."""

from __future__ import annotations

from mcp_server.server import main as server_main


def main() -> int:
    """Start the actual stdio MCP server."""
    return server_main()


if __name__ == "__main__":
    raise SystemExit(main())
