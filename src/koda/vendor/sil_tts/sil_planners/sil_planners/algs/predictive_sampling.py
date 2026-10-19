"""Predictive sampling."""

from __future__ import annotations

import numpy as np

from sil_planners.base import SamplingPlanner
from sil_planners.registry import register
from sil_planners.types import PlannerParams


@register("predictive_sampling")
class PredictiveSampling(SamplingPlanner):
    """A simple implementation of https://arxiv.org/abs/2212.00541.

    Candidate 0 is the unperturbed mean; the update keeps the lowest-cost
    sample, so the best observed cost never regresses. Null-hypothesis
    baseline for test-time scaling studies.
    """

    def __init__(
        self, num_candidates: int, num_knots: int, dim: int, noise_level: float
    ) -> None:
        super().__init__(num_candidates, num_knots, dim)
        self.noise_level = float(noise_level)

    def sample_candidates(
        self, params: PlannerParams
    ) -> tuple[np.ndarray, PlannerParams]:
        noise = params.rng.standard_normal(
            (self.num_candidates, self.num_knots, self.dim)
        )
        candidates = params.mean + self.noise_level * noise
        candidates[0] = params.mean
        return candidates, params

    def update_params(
        self, params: PlannerParams, candidates: np.ndarray, costs: np.ndarray
    ) -> PlannerParams:
        params.mean = candidates[int(np.argmin(costs))].copy()
        return params
