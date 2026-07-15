"""Evaluation workflow implementations."""

__all__ = [
    "CompositeEvaluator",
    "DEFAULT_RAGAS_METRICS",
    "EvalCase",
    "EvalCaseResult",
    "EvalReport",
    "EvalRunner",
    "RagasEvaluator",
    "RagasRunner",
    "load_golden_test_set",
]


def __getattr__(name: str):
    if name == "CompositeEvaluator":
        from .composite_evaluator import CompositeEvaluator

        return CompositeEvaluator
    if name in {"DEFAULT_RAGAS_METRICS", "RagasEvaluator", "RagasRunner"}:
        from .ragas_evaluator import (
            DEFAULT_RAGAS_METRICS,
            RagasEvaluator,
            RagasRunner,
        )

        return {
            "DEFAULT_RAGAS_METRICS": DEFAULT_RAGAS_METRICS,
            "RagasEvaluator": RagasEvaluator,
            "RagasRunner": RagasRunner,
        }[name]
    if name in {
        "EvalCase",
        "EvalCaseResult",
        "EvalReport",
        "EvalRunner",
        "load_golden_test_set",
    }:
        from .eval_runner import (
            EvalCase,
            EvalCaseResult,
            EvalReport,
            EvalRunner,
            load_golden_test_set,
        )

        return {
            "EvalCase": EvalCase,
            "EvalCaseResult": EvalCaseResult,
            "EvalReport": EvalReport,
            "EvalRunner": EvalRunner,
            "load_golden_test_set": load_golden_test_set,
        }[name]
    raise AttributeError(name)
