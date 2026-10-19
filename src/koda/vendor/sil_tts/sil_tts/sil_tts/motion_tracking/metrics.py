"""Execution-clip tracking metrics derived from paired execution records."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sil_trackers import MotionGuidance
from sil_tts.records import ExecutionRunArtifact, ExecutionTrace, PairedRunArtifact
from sil_tts.simulation.measurements import MeasurementPrimitive


@dataclass(frozen=True)
class ExecutionClipTrackingMetrics:
    """Unweighted nominal-tracking errors for one complete Execution Clip.

    Joint position is scalar RMSE over Control Steps and joints (radians). Body/root
    position is mean Euclidean distance (metres), and orientation is mean
    sign-invariant quaternion geodesic angle (radians). Body metrics include
    every named body, including the separately reported root.

    Error values are ``None`` after an early failure: a valid-prefix error is
    not mislabeled as a complete Execution Clip result.
    """

    success: bool
    valid_steps: int
    total_steps: int
    failure_step: int | None
    failure_reason: str | None
    joint_position_rmse_rad: float | None
    body_position_mean_error_m: float | None
    body_orientation_mean_error_rad: float | None
    root_position_mean_error_m: float | None
    root_orientation_mean_error_rad: float | None


@dataclass(frozen=True)
class PairedTrackingMetrics:
    """Baseline and optimized metrics computed under validated shared identity."""

    baseline: ExecutionClipTrackingMetrics
    optimized: ExecutionClipTrackingMetrics


def compute_paired_metrics(pair: PairedRunArtifact) -> PairedTrackingMetrics:
    """Validate paired-run parity, then compute nominal Execution Clip errors."""
    if not isinstance(pair, PairedRunArtifact):
        raise TypeError("pair must be PairedRunArtifact")
    _validate_pair(pair.baseline, pair.optimized)
    return PairedTrackingMetrics(
        baseline=_compute_run_metrics(pair.baseline.trace),
        optimized=_compute_run_metrics(pair.optimized.trace),
    )


def _compute_run_metrics(trace: ExecutionTrace) -> ExecutionClipTrackingMetrics:
    common = dict(
        success=trace.success,
        valid_steps=trace.valid_steps,
        total_steps=trace.num_steps,
        failure_step=trace.failure_step,
        failure_reason=trace.failure_reason,
    )
    if not trace.success:
        return ExecutionClipTrackingMetrics(
            **common,
            joint_position_rmse_rad=None,
            body_position_mean_error_m=None,
            body_orientation_mean_error_rad=None,
            root_position_mean_error_m=None,
            root_orientation_mean_error_rad=None,
        )

    achieved = trace.achieved
    if achieved is None:  # Protected by ExecutionTrace; keeps this seam explicit.
        raise ValueError("a successful trace must contain achieved measurements")
    nominal = trace.nominal
    actual_joint = achieved.arrays[MeasurementPrimitive.JOINT_POSITION][0]
    actual_body_position = achieved.arrays[
        MeasurementPrimitive.BODY_POSITION_WORLD
    ][0]
    actual_body_orientation = achieved.arrays[
        MeasurementPrimitive.BODY_ORIENTATION_WORLD
    ][0]
    target_joint = nominal.joint_position[0]
    target_body_position = nominal.body_position_world[0]
    target_body_orientation = nominal.body_orientation_world[0]
    root = nominal.body_names.index(trace.root_body_name)

    return ExecutionClipTrackingMetrics(
        **common,
        joint_position_rmse_rad=float(
            np.sqrt(np.mean((actual_joint - target_joint) ** 2))
        ),
        body_position_mean_error_m=float(
            np.mean(
                np.linalg.norm(
                    actual_body_position - target_body_position, axis=-1
                )
            )
        ),
        body_orientation_mean_error_rad=float(
            np.mean(
                _quaternion_geodesic(
                    actual_body_orientation, target_body_orientation
                )
            )
        ),
        root_position_mean_error_m=float(
            np.mean(
                np.linalg.norm(
                    actual_body_position[:, root] - target_body_position[:, root],
                    axis=-1,
                )
            )
        ),
        root_orientation_mean_error_rad=float(
            np.mean(
                _quaternion_geodesic(
                    actual_body_orientation[:, root],
                    target_body_orientation[:, root],
                )
            )
        ),
    )


def _quaternion_geodesic(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    actual_norm = np.linalg.norm(actual, axis=-1, keepdims=True)
    target_norm = np.linalg.norm(target, axis=-1, keepdims=True)
    if np.any(actual_norm <= 1e-12) or np.any(target_norm <= 1e-12):
        raise ValueError("body orientations must contain non-zero quaternions")
    dots = np.sum((actual / actual_norm) * (target / target_norm), axis=-1)
    return 2.0 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0))


def _validate_pair(
    baseline: ExecutionRunArtifact, optimized: ExecutionRunArtifact
) -> None:
    for field in (
        "target_id",
        "target_digest",
        "tracker_id",
        "checkpoint_digest",
        "robot_model_digest",
        "initial_physical_state_digest",
        "control_period",
    ):
        if getattr(baseline, field) != getattr(optimized, field):
            raise ValueError(f"paired runs must share {field}")

    left = baseline.trace
    right = optimized.trace
    if not np.array_equal(left.timeline, right.timeline):
        raise ValueError("paired runs must share the nominal timeline")
    if left.root_body_name != right.root_body_name:
        raise ValueError("paired runs must share root_body_name")
    if not _same_guidance(left.nominal, right.nominal):
        raise ValueError("paired runs must share the immutable nominal target")


def _same_guidance(left: MotionGuidance, right: MotionGuidance) -> bool:
    if (
        left.query_offsets != right.query_offsets
        or left.joint_names != right.joint_names
        or left.body_names != right.body_names
    ):
        return False
    return all(
        np.array_equal(getattr(left, field), getattr(right, field))
        for field in (
            "joint_position",
            "joint_velocity",
            "body_position_world",
            "body_orientation_world",
            "body_linear_velocity_world",
            "body_angular_velocity_world",
        )
    )
