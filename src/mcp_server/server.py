"""Stdio JSON-RPC entrypoint for the MCP server."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, TextIO

from .protocol_handler import (
    PROTOCOL_VERSION,
    SERVER_NAME,
    SERVER_VERSION,
    ProtocolHandler,
)
from .tools import default_tools


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the MCP server over stdio.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Server log level. Logs are written to stderr.",
    )
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("starting %s stdio server", SERVER_NAME)
    handler = ProtocolHandler(tools=default_tools(), logger=logger)
    try:
        run_stdio_server(sys.stdin, sys.stdout, handler=handler)
    except KeyboardInterrupt:
        logger.info("server interrupted")
        return 130
    logger.info("server stopped")
    return 0


def run_stdio_server(
    stdin: TextIO,
    stdout: TextIO,
    *,
    handler: ProtocolHandler | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Read newline-delimited JSON-RPC messages and write responses to stdout."""
    active_handler = handler or ProtocolHandler(tools=default_tools(), logger=logger)
    for raw_line in stdin:
        line = raw_line.strip()
        if not line:
            continue
        response = active_handler.handle_message(line)
        if response is None:
            continue
        stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")))
        stdout.write("\n")
        stdout.flush()


def handle_message(
    raw_message: str,
    *,
    logger: logging.Logger | None = None,
) -> dict[str, Any] | None:
    """Compatibility wrapper for single-message handling."""
    return ProtocolHandler(tools=default_tools(), logger=logger).handle_message(raw_message)


def _json_rpc_error(
    request_id: Any,
    code: int,
    message: str,
) -> dict[str, Any]:
    """Compatibility helper for legacy tests; protocol_handler owns new logic."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": code,
            "message": message,
        },
    }


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(levelname)s:%(name)s:%(message)s",
        stream=sys.stderr,
        force=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
