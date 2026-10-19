"""sil_planners: sampling-based black-box optimizers with a NumPy core."""

from sil_planners.base import BatchEvaluator, SamplingPlanner
from sil_planners.registry import available, get_planner, register
from sil_planners.types import IterationRecord, OptimizeReport, PlannerParams

__all__ = [
    "BatchEvaluator",
    "SamplingPlanner",
    "available",
    "get_planner",
    "register",
    "IterationRecord",
    "OptimizeReport",
    "PlannerParams",
]

from sil_planners import algs  # noqa: E402,F401  (registers built-ins)
