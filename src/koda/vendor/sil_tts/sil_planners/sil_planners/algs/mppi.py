"""Model-predictive path integral control."""

from __future__ import annotations

import numpy as np

from sil_planners._weights import softmax_weights
from sil_planners.base import SamplingPlanner
from sil_planners.registry import register
from sil_planners.types import PlannerParams


@register("mppi")
class MPPI(SamplingPlanner):
    """Model-predictive path integral control.

    Implements "MPPI-generic" as described in https://arxiv.org/abs/2409.07563.
    Unlike the original MPPI derivation, this does not assume stochastic,
    control-affine dynamics or a separable cost function that is quadratic in
    control.
    """

    def __init__(
        self,
        num_candidates: int,
        num_knots: int,
        dim: int,
        noise_level: float,
        temperature: float,
    ) -> None:
        super().__init__(num_candidates, num_knots, dim)
        self.noise_level = float(noise_level)
        self.temperature = float(temperature)

    def sample_candidates(
        self, params: PlannerParams
    ) -> tuple[np.ndarray, PlannerParams]:
        noise = params.rng.standard_normal(
            (self.num_candidates, self.num_knots, self.dim)
        )
        return params.mean + self.noise_level * noise, params

    def update_params(
        self, params: PlannerParams, candidates: np.ndarray, costs: np.ndarray
    ) -> PlannerParams:
        weights = softmax_weights(costs, self.temperature)
        params.mean = (weights[:, None, None] * candidates).sum(axis=0)
        return params
