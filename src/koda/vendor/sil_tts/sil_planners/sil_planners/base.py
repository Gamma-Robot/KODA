"""Sampling-planner interface: latent arrays in, scalar costs out.

Stable contract — rationale and extension guide in ../README.md,
executable form in tests/planners/test_base.py:

1. The caller poses a finite-horizon black-box problem: terminal and tail
   structure is already folded into each candidate's scalar cost.
2. Each iteration materializes all `num_candidates` candidates, then calls
   the evaluator once. Evaluator batch capacity (candidate batching) must not
   affect the candidate set or the RNG stream.
3. Costs are lower-is-better and finite; the evaluator maps rollout
   failures to finite penalties.
4. Candidates are normalized latent arrays, never joint commands, torques,
   or tracker actions.
5. `PlannerParams` is planner-owned opaque state that callers never mutate.
   `proposal(params)` is the final answer, in general unevaluated, and
   `optimize` never evaluates it implicitly.
6. One planning update = one `init_params` + one `optimize`;
   `params.iteration` never spans planning updates (DIAL annealing).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from sil_planners.types import IterationRecord, OptimizeReport, PlannerParams

BatchEvaluator = Callable[[np.ndarray], np.ndarray]
"""(num_candidates, num_knots, dim) float64 -> (num_candidates,) finite costs."""


class SamplingPlanner(ABC):
    """Template: iterations x (sample -> evaluate -> update)."""

    def __init__(self, num_candidates: int, num_knots: int, dim: int) -> None:
        if num_candidates < 1 or num_knots < 1 or dim < 1:
            raise ValueError(
                "num_candidates, num_knots, and dim must all be >= 1; got "
                f"({num_candidates}, {num_knots}, {dim})"
            )
        self.num_candidates = num_candidates
        self.num_knots = num_knots
        self.dim = dim

    def init_params(
        self, seed: int, initial_mean: np.ndarray | None = None
    ) -> PlannerParams:
        if initial_mean is None:
            mean = np.zeros((self.num_knots, self.dim), dtype=np.float64)
        else:
            mean = np.asarray(initial_mean, dtype=np.float64).copy()
            if mean.shape != (self.num_knots, self.dim):
                raise ValueError(
                    f"initial_mean must have shape {(self.num_knots, self.dim)}, "
                    f"got {mean.shape}"
                )
            if not np.isfinite(mean).all():
                raise ValueError("initial_mean must be finite")
        return PlannerParams(mean=mean, rng=np.random.default_rng(seed))

    @abstractmethod
    def sample_candidates(
        self, params: PlannerParams
    ) -> tuple[np.ndarray, PlannerParams]:
        """Return (num_candidates, num_knots, dim) candidates and params."""

    @abstractmethod
    def update_params(
        self, params: PlannerParams, candidates: np.ndarray, costs: np.ndarray
    ) -> PlannerParams:
        """Update the sampling distribution from per-candidate costs."""

    def proposal(self, params: PlannerParams) -> np.ndarray:
        """The planner's final answer; in general not an evaluated candidate."""
        return params.mean.copy()

    def optimize(
        self, params: PlannerParams, evaluate: BatchEvaluator, iterations: int
    ) -> tuple[PlannerParams, OptimizeReport]:
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        expected = (self.num_candidates, self.num_knots, self.dim)
        records: list[IterationRecord] = []
        for _ in range(iterations):
            candidates, params = self.sample_candidates(params)
            if not isinstance(candidates, np.ndarray) or candidates.dtype != np.float64:
                raise ValueError(
                    "sample_candidates must return a float64 ndarray "
                    "(float64 is the sole planner dtype); got "
                    f"{type(candidates).__name__}"
                    f"{f' with dtype {candidates.dtype}' if isinstance(candidates, np.ndarray) else ''}"
                )
            if candidates.shape != expected:
                raise ValueError(
                    f"sample_candidates returned shape {candidates.shape}, "
                    f"expected {expected}"
                )
            if not np.isfinite(candidates).all():
                raise ValueError("sample_candidates returned non-finite values")
            costs = np.asarray(evaluate(candidates), dtype=np.float64)
            if costs.shape != (self.num_candidates,):
                raise ValueError(
                    f"evaluator returned shape {costs.shape}, "
                    f"expected {(self.num_candidates,)}"
                )
            if not np.isfinite(costs).all():
                raise ValueError(
                    "evaluator returned non-finite costs; map rollout failures "
                    "to finite penalties in the evaluator"
                )
            best = int(np.argmin(costs))
            record = IterationRecord(
                iteration=params.iteration,
                best_observed_cost=float(costs[best]),
                best_observed_candidate=candidates[best].copy(),
                mean_cost=float(costs.mean()),
            )
            params = self.update_params(params, candidates, costs)
            if (
                not isinstance(params.mean, np.ndarray)
                or params.mean.dtype != np.float64
                or params.mean.shape != (self.num_knots, self.dim)
                or not np.isfinite(params.mean).all()
            ):
                raise ValueError(
                    "update_params must produce a finite float64 ndarray mean "
                    f"with shape {(self.num_knots, self.dim)}"
                )
            records.append(record)
            params.iteration += 1
        return params, OptimizeReport(iterations=tuple(records))
