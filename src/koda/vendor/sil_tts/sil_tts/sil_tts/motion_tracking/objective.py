"""Finite-horizon motion-tracking objective.

The immutable nominal guidance defines the task. Candidate guidance is the
planner's deformation of that task, and achieved measurements are produced by
closed-loop simulation. Keeping all three explicit prevents candidate tracking
from being mistaken for task fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Mapping

import numpy as np

from sil_trackers import (
    MOTION_GUIDANCE_ARRAY_FIELDS as _GUIDANCE_ARRAYS,
    MotionGuidance,
)
from sil_tts.simulation.measurements import (
    STATE_MEASUREMENT_PRIMITIVES,
    MeasurementPrimitive,
    ControlStepMeasurements,
)

def _readonly_vector(value: np.ndarray, field: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, order="C", copy=True)
    if array.ndim != 1 or array.shape[0] < 1:
        raise ValueError(f"{field} must have shape (B,) with B >= 1, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{field} must be finite")
    array.flags.writeable = False
    return array


@dataclass(frozen=True, eq=False)
class CostResult:
    """Immutable full-horizon costs and weighted per-term contributions."""

    total: np.ndarray
    terms: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        total = _readonly_vector(self.total, "total")
        normalized: dict[str, np.ndarray] = {}
        for name, value in self.terms.items():
            if not isinstance(name, str) or not name:
                raise ValueError("term names must be non-empty strings")
            term = _readonly_vector(value, f"terms[{name!r}]")
            if term.shape != total.shape:
                raise ValueError(
                    f"terms[{name!r}] must have shape {total.shape}, got {term.shape}"
                )
            normalized[name] = term
        if not normalized:
            raise ValueError("terms must contain at least one contribution")
        term_total = np.sum(np.stack(tuple(normalized.values()), axis=0), axis=0)
        if not np.allclose(total, term_total, rtol=1e-12, atol=1e-12):
            raise ValueError("total must equal the sum of all term contributions")
        object.__setattr__(self, "total", total)
        object.__setattr__(self, "terms", MappingProxyType(normalized))


@dataclass(frozen=True)
class MotionTrackingObjective:
    """Weighted task fidelity, candidate deformation, and terminal fidelity.

    Running terms are summed over the fixed planning horizon. Terminal terms
    are evaluated once at its last Control Step. All returned costs are float64 and
    lower-is-better.
    """

    tracking_joint_position_weight: float
    tracking_joint_velocity_weight: float
    tracking_body_position_weight: float
    tracking_body_orientation_weight: float
    deformation_joint_position_weight: float
    deformation_body_position_weight: float
    deformation_body_orientation_weight: float
    joint_speed_weight: float
    terminal_joint_position_weight: float
    terminal_body_position_weight: float
    action_rate_l2_weight: float = 0.0
    body_cost_weights: tuple[float, ...] | None = None
    normalize_body_cost_weights: bool = True
    receding_time_weights: bool = False

    @property
    def requires(self) -> frozenset[MeasurementPrimitive]:
        primitives = set(STATE_MEASUREMENT_PRIMITIVES)
        if self.action_rate_l2_weight > 0.0:
            primitives.add(MeasurementPrimitive.POSITION_TARGET)
        return frozenset(primitives)

    def __post_init__(self) -> None:
        if self.body_cost_weights is not None:
            weights = tuple(float(value) for value in self.body_cost_weights)
            if not weights or any(
                not np.isfinite(value) or value < 0.0 for value in weights
            ):
                raise ValueError("body_cost_weights must be finite and >= 0")
            object.__setattr__(self, "body_cost_weights", weights)
        for name in ("normalize_body_cost_weights", "receding_time_weights"):
            value = getattr(self, name)
            if type(value) is not bool:
                raise TypeError(f"{name} must be bool, got {type(value).__name__}")
        for field in fields(self):
            if field.name in (
                "body_cost_weights",
                "normalize_body_cost_weights",
                "receding_time_weights",
            ):
                continue
            value = float(getattr(self, field.name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{field.name} must be a finite weight >= 0, got {value}"
                )
            object.__setattr__(self, field.name, value)

    def evaluate(
        self,
        measurements: ControlStepMeasurements,
        nominal: MotionGuidance,
        candidate: MotionGuidance,
    ) -> CostResult:
        """Reduce one full rollout per candidate to a cost vector of shape ``(B,)``."""
        self._validate(measurements, nominal, candidate)

        achieved_joint_position = np.asarray(
            measurements.arrays[MeasurementPrimitive.JOINT_POSITION],
            dtype=np.float64,
        )
        achieved_joint_velocity = np.asarray(
            measurements.arrays[MeasurementPrimitive.JOINT_VELOCITY],
            dtype=np.float64,
        )
        achieved_body_position = np.asarray(
            measurements.arrays[MeasurementPrimitive.BODY_POSITION_WORLD],
            dtype=np.float64,
        )
        achieved_body_orientation = np.asarray(
            measurements.arrays[MeasurementPrimitive.BODY_ORIENTATION_WORLD],
            dtype=np.float64,
        )

        deformation = _deformation_terms(self, nominal, candidate)
        time_weights = _horizon_time_weights(
            measurements.num_steps, self.receding_time_weights
        )
        terms = {
            "tracking/joint_position": self.tracking_joint_position_weight
            * _reduce_horizon(
                _joint_error(achieved_joint_position, nominal.joint_position),
                time_weights,
            ),
            "tracking/joint_velocity": self.tracking_joint_velocity_weight
            * _reduce_horizon(
                _joint_error(achieved_joint_velocity, nominal.joint_velocity),
                time_weights,
            ),
            "tracking/body_position": self.tracking_body_position_weight
            * _reduce_horizon(
                _body_position_error(
                    achieved_body_position,
                    nominal.body_position_world,
                    self.body_cost_weights,
                    normalize=self.normalize_body_cost_weights,
                ),
                time_weights,
            ),
            "tracking/body_orientation": self.tracking_body_orientation_weight
            * _reduce_horizon(
                _body_orientation_error(
                    achieved_body_orientation,
                    nominal.body_orientation_world,
                    self.body_cost_weights,
                    normalize=self.normalize_body_cost_weights,
                ),
                time_weights,
            ),
            **deformation,
            "stability/joint_speed": self.joint_speed_weight
            * _reduce_horizon(
                np.mean(achieved_joint_velocity**2, axis=-1), time_weights
            ),
            "stability/action_rate_l2": (
                np.zeros(measurements.batch, dtype=np.float64)
                if self.action_rate_l2_weight == 0.0
                else self.action_rate_l2_weight
                * _reduce_horizon(
                    _action_rate_l2(
                        measurements.arrays[MeasurementPrimitive.POSITION_TARGET]
                    ),
                    time_weights,
                )
            ),
            "terminal/joint_position": self.terminal_joint_position_weight
            * _joint_error(
                achieved_joint_position[:, -1], nominal.joint_position[:, -1]
            ),
            "terminal/body_position": self.terminal_body_position_weight
            * _body_position_error(
                achieved_body_position[:, -1],
                nominal.body_position_world[:, -1],
                self.body_cost_weights,
                normalize=self.normalize_body_cost_weights,
            ),
        }
        total = np.sum(np.stack(tuple(terms.values()), axis=0), axis=0)
        return CostResult(total=total, terms=terms)

    def _validate(
        self,
        measurements: ControlStepMeasurements,
        nominal: MotionGuidance,
        candidate: MotionGuidance,
    ) -> None:
        missing = self.requires.difference(measurements.arrays)
        if missing:
            names = sorted(primitive.value for primitive in missing)
            raise ValueError(f"measurements are missing required primitives: {names}")
        for name, guidance in (("nominal", nominal), ("candidate", candidate)):
            if (
                guidance.joint_names != measurements.joint_names
                or guidance.body_names != measurements.body_names
            ):
                raise ValueError(
                    f"{name} joint/body names must match the measurements"
                )
            if guidance.query_offsets != tuple(range(measurements.num_steps)):
                raise ValueError(
                    f"{name} query_offsets must be dense "
                    f"(0..{measurements.num_steps - 1}), got "
                    f"{guidance.query_offsets}"
                )
            for field in _GUIDANCE_ARRAYS:
                if not np.isfinite(getattr(guidance, field)).all():
                    raise ValueError(f"{name} {field} must be finite")
        if nominal.joint_position.shape[0] != 1:
            raise ValueError(
                f"nominal batch must be 1, got {nominal.joint_position.shape[0]}"
            )
        if candidate.joint_position.shape[0] != measurements.batch:
            raise ValueError(
                f"candidate batch must be {measurements.batch}, "
                f"got {candidate.joint_position.shape[0]}"
            )


def _action_rate_l2(position_target: np.ndarray) -> np.ndarray:
    """Isaac Lab ``action_rate_l2``: ``sum_j (a_t - a_{t-1})^2``, first step 0."""
    values = np.asarray(position_target, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(
            f"position_target must have shape (B, T, J), got {values.shape}"
        )
    per_step = np.zeros(values.shape[:2], dtype=np.float64)
    per_step[:, 1:] = np.sum(np.diff(values, axis=1) ** 2, axis=-1)
    return per_step


def _joint_error(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean(
        (np.asarray(actual, dtype=np.float64) - np.asarray(target, dtype=np.float64))
        ** 2,
        axis=-1,
    )


def _deformation_terms(
    objective: MotionTrackingObjective,
    nominal: MotionGuidance,
    candidate: MotionGuidance,
) -> dict[str, np.ndarray]:
    """Return the achieved-state-independent objective contributions."""
    return {
        "deformation/joint_position": objective.deformation_joint_position_weight
        * _joint_error(candidate.joint_position, nominal.joint_position).sum(axis=1),
        "deformation/body_position": objective.deformation_body_position_weight
        * _body_position_error(
            candidate.body_position_world,
            nominal.body_position_world,
            objective.body_cost_weights,
            normalize=objective.normalize_body_cost_weights,
        ).sum(axis=1),
        "deformation/body_orientation": objective.deformation_body_orientation_weight
        * _body_orientation_error(
            candidate.body_orientation_world,
            nominal.body_orientation_world,
            objective.body_cost_weights,
            normalize=objective.normalize_body_cost_weights,
        ).sum(axis=1),
    }


def _horizon_time_weights(num_steps: int, receding: bool) -> np.ndarray:
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if not receding:
        return np.ones(num_steps, dtype=np.float64)
    weights = np.arange(num_steps, 0, -1, dtype=np.float64)
    return weights / np.sum(weights)


def _reduce_horizon(per_step: np.ndarray, time_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(per_step, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != time_weights.shape[0]:
        raise ValueError(
            f"per-step cost must have shape (B, {time_weights.shape[0]}), "
            f"got {values.shape}"
        )
    return np.sum(values * time_weights, axis=1)


def _body_weights(
    weights: tuple[float, ...] | None,
    num_bodies: int,
    *,
    normalize: bool = True,
) -> np.ndarray:
    if weights is None:
        if normalize:
            return np.full(num_bodies, 1.0 / max(num_bodies, 1), dtype=np.float64)
        return np.ones(num_bodies, dtype=np.float64)
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (num_bodies,):
        raise ValueError(
            f"body_cost_weights must have shape ({num_bodies},), got {values.shape}"
        )
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("body_cost_weights must sum to a positive value")
    if not normalize:
        return values
    return values / total


def _body_position_error(
    actual: np.ndarray,
    target: np.ndarray,
    weights: tuple[float, ...] | None = None,
    *,
    normalize: bool = True,
) -> np.ndarray:
    error = (
        np.asarray(actual, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    )
    per_body = np.sum(error**2, axis=-1)
    scale = _body_weights(weights, per_body.shape[-1], normalize=normalize)
    return np.sum(per_body * scale, axis=-1)


def _body_orientation_error(
    actual: np.ndarray,
    target: np.ndarray,
    weights: tuple[float, ...] | None = None,
    *,
    normalize: bool = True,
) -> np.ndarray:
    dot = np.abs(
        np.sum(
            np.asarray(actual, dtype=np.float64)
            * np.asarray(target, dtype=np.float64),
            axis=-1,
        )
    )
    angle = 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))
    scale = _body_weights(weights, angle.shape[-1], normalize=normalize)
    return np.sum(angle**2 * scale, axis=-1)
