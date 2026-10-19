"""Shared dataclasses for sampling planners."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PlannerParams:
    """Mutable state of the sampling distribution.

        Planner-owned opaque state: callers (sil_tts) never mutate fields
    directly; warm starts go through ``init_params(initial_mean=...)``.

    Attributes:
        mean: Distribution center, shape (num_knots, dim), float64.
        rng: Stateful numpy generator; consuming it advances the stream.
        iteration: Number of completed optimize iterations within the
            current planning update (never spans planning updates).
    """

    mean: np.ndarray
    rng: np.random.Generator
    iteration: int = 0


# eq=False: generated __eq__/__hash__ are broken with ndarray fields
# (== raises ValueError, hash raises TypeError).
@dataclass(frozen=True, eq=False)
class IterationRecord:
    """Per-iteration record. "Observed" = from an actually evaluated candidate."""

    iteration: int
    best_observed_cost: float
    best_observed_candidate: np.ndarray
    mean_cost: float


@dataclass(frozen=True, eq=False)
class OptimizeReport:
    iterations: tuple[IterationRecord, ...] = field(default_factory=tuple)

    @property
    def best_observed_cost(self) -> float:
        if not self.iterations:
            raise ValueError("report has no iterations")
        return min(record.best_observed_cost for record in self.iterations)

    @property
    def best_observed_candidate(self) -> np.ndarray:
        if not self.iterations:
            raise ValueError("report has no iterations")
        best = min(self.iterations, key=lambda r: r.best_observed_cost)
        return best.best_observed_candidate.copy()
