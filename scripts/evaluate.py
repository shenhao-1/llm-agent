"""Run golden-set retrieval evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.settings import SettingsValidationError, load_settings  # noqa: E402
from libs.evaluator import EvaluatorFactory  # noqa: E402
from observability.evaluation import EvalRunner  # noqa: E402


DEFAULT_SETTINGS = Path("config/settings.yaml")
DEFAULT_DATA_DIR = Path("data")
DEFAULT_TEST_SET = Path("tests/fixtures/golden_test_set.json")


class QuerySetupError(RuntimeError):
    """Raised when the retrieval indexes required for evaluation are unavailable."""


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.settings)
        search = _build_search(settings, Path(args.data_dir))
        evaluator = EvaluatorFactory.create(settings)
        runner = EvalRunner(
            settings=settings,
            hybrid_search=search,
            evaluator=evaluator,
            default_top_k=args.top_k,
        )
        report = runner.run(args.test_set)
    except QuerySetupError as exc:
        print(str(exc))
        print("No indexed data found. Run scripts/ingest.py before evaluation.")
        return 1
    except FileNotFoundError as exc:
        print(f"evaluation setup failed: {exc}", file=sys.stderr)
        return 2
    except (SettingsValidationError, OSError, ValueError, RuntimeError) as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 2

    _print_report(report.to_dict(), as_json=args.json)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run golden-set evaluation.")
    parser.add_argument(
        "--test-set",
        default=str(DEFAULT_TEST_SET),
        help="Path to golden_test_set.json.",
    )
    parser.add_argument(
        "--settings",
        default=str(DEFAULT_SETTINGS),
        help="Path to YAML settings.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Root directory containing db/vectors.json and db/bm25/index.json.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override retrieval top_k for all cases without an explicit top_k.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full evaluation report as JSON.",
    )
    return parser


def _print_report(report: dict, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return

    print("Evaluation metrics")
    for key, value in sorted(report["metrics"].items()):
        print(f"- {key}: {value:.4f}")
    print(f"\nCases: {len(report['results'])}")
    for result in report["results"]:
        print(
            f"- {result['query']} | "
            f"retrieved={result['retrieved_ids']} | "
            f"expected={result['expected_chunk_ids']}"
        )


def _build_search(settings, data_dir: Path):
    from scripts.query import QuerySetupError as ScriptQuerySetupError
    from scripts.query import _build_search as build_search

    try:
        return build_search(settings, data_dir)
    except ScriptQuerySetupError as exc:
        raise QuerySetupError(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
