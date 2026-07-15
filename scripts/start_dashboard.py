"""Launch the Streamlit dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "src" / "observability" / "dashboard" / "app.py"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.port",
        str(args.port),
        "--server.address",
        args.address,
    ]
    if args.headless:
        command.extend(["--server.headless", "true"])
    return subprocess.call(command, cwd=ROOT)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start the Modular RAG dashboard.")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--address", default="localhost")
    parser.add_argument("--headless", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
