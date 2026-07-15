"""Evaluator abstractions and factory."""

from .base_evaluator import BaseEvaluator
from .custom_evaluator import CustomEvaluator
from .evaluator_factory import EvaluatorFactory
from .ragas_evaluator import RagasEvaluator

__all__ = [
    "BaseEvaluator",
    "CompositeEvaluator",
    "CustomEvaluator",
    "EvaluatorFactory",
    "RagasEvaluator",
]


def __getattr__(name: str):
    if name == "CompositeEvaluator":
        from observability.evaluation.composite_evaluator import CompositeEvaluator

        return CompositeEvaluator
    raise AttributeError(name)
