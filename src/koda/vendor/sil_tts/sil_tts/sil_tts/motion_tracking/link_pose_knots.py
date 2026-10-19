"""True-knot search space over torso + dual-wrist SE(3).

Planner latents have shape ``(N, K, 18)``: three links, each
``[dx, dy, dz, rotvec_x, rotvec_y, rotvec_z]``. Knots sit at uniformly spaced
positions spanning the dense Control-Step grid ``0..H-1``. Decode interpolates
those knots (linear position, rotvec-then-compose orientation) and applies the
deltas only to the tracking links. Other bodies and joint targets stay at the
nominal reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sil_trackers.contracts import MotionGuidance
from sil_tts.motion_tracking.search_space import (
    KnotScales,
    QposKnotsSpace,
    _canonical_unit_quat,
    _quat_multiply,
    _rotvec_to_quat,
)

TRACKING_LINK_NAMES: tuple[str, ...] = (
    "torso_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
NUM_LINKS = len(TRACKING_LINK_NAMES)
POSE_DIM = 6
LINK_POSE_LATENT_DIM = NUM_LINKS * POSE_DIM
ARRAY_FIELDS = (
    "joint_position",
    "joint_velocity",
    "body_position_world",
    "body_orientation_world",
    "body_linear_velocity_world",
    "body_angular_velocity_world",
)


@dataclass(frozen=True)
class LinkPoseScales:
    """Physical magnitude of one normalized latent unit for a single link pose."""

    position_m: float
    rotation_rad: float

    def __post_init__(self) -> None:
        for name in ("position_m", "rotation_rad"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")

    @classmethod
    def from_knot_scales(cls, scales: KnotScales) -> LinkPoseScales:
        if float(scales.root_xy_m) != float(scales.root_z_m):
            raise ValueError(
                "ScaleBFM link-pose scales require root_xy_m == root_z_m; "
                f"got xy={scales.root_xy_m} z={scales.root_z_m}"
            )
        return cls(
            position_m=float(scales.root_xy_m),
            rotation_rad=float(scales.root_rotation_rad),
        )

    def scale_vector(self, num_links: int = NUM_LINKS) -> np.ndarray:
        per_link = np.asarray(
            (
                self.position_m,
                self.position_m,
                self.position_m,
                self.rotation_rad,
                self.rotation_rad,
                self.rotation_rad,
            ),
            dtype=np.float64,
        )
        return np.tile(per_link, int(num_links))


def _copy_guidance(reference: MotionGuidance) -> MotionGuidance:
    return MotionGuidance(
        query_offsets=reference.query_offsets,
        joint_names=reference.joint_names,
        body_names=reference.body_names,
        joint_position=np.array(reference.joint_position, dtype=np.float64),
        joint_velocity=np.array(reference.joint_velocity, dtype=np.float64),
        body_position_world=np.array(reference.body_position_world, dtype=np.float64),
        body_orientation_world=np.array(
            reference.body_orientation_world, dtype=np.float64
        ),
        body_linear_velocity_world=np.array(
            reference.body_linear_velocity_world, dtype=np.float64
        ),
        body_angular_velocity_world=np.array(
            reference.body_angular_velocity_world, dtype=np.float64
        ),
    )


class LinkPoseKnotsSpace:
    """Decode normalized ``(N, K, 18)`` latents onto three tracking links."""

    def __init__(
        self,
        reference: MotionGuidance,
        num_knots: int,
        control_period: float,
        scales: KnotScales | LinkPoseScales,
        tracking_bodies: tuple[str, ...] = TRACKING_LINK_NAMES,
    ) -> None:
        horizon = len(reference.query_offsets)
        if reference.joint_position.shape[0] != 1:
            raise ValueError("reference must have batch size 1")
        if horizon < 2 or reference.query_offsets != tuple(range(horizon)):
            raise ValueError(
                "reference query_offsets must be (0, 1, ..., H-1) with H >= 2"
            )
        num_knots = int(num_knots)
        if num_knots < 2:
            raise ValueError("num_knots must be >= 2")
        control_period = float(control_period)
        if not np.isfinite(control_period) or control_period <= 0.0:
            raise ValueError("control_period must be finite and > 0")
        if len(tracking_bodies) != NUM_LINKS:
            raise ValueError(f"tracking_bodies must contain {NUM_LINKS} names")
        missing = [name for name in tracking_bodies if name not in reference.body_names]
        if missing:
            raise ValueError(f"tracking bodies missing from reference: {missing}")

        if isinstance(scales, KnotScales):
            pose_scales = LinkPoseScales.from_knot_scales(scales)
        elif isinstance(scales, LinkPoseScales):
            pose_scales = scales
        else:
            raise TypeError("scales must be KnotScales or LinkPoseScales")

        self._reference = _copy_guidance(reference)
        self._control_period = control_period
        self._tracking_bodies = tuple(tracking_bodies)
        self._link_indices = np.asarray(
            [reference.body_names.index(name) for name in self._tracking_bodies],
            dtype=np.int64,
        )
        self._scale = pose_scales.scale_vector(NUM_LINKS)
        self._weights = QposKnotsSpace._interpolation_weights(horizon, num_knots)

    @property
    def latent_shape(self) -> tuple[int, int]:
        return (self._weights.shape[1], int(self._scale.shape[0]))

    def zero(self) -> np.ndarray:
        return np.zeros(self.latent_shape, dtype=np.float64)

    def project(self, latent: np.ndarray) -> np.ndarray:
        value = np.asarray(latent, dtype=np.float64)
        if value.shape != self.latent_shape or not np.isfinite(value).all():
            raise ValueError(
                f"latent must be finite with shape {self.latent_shape}, got {value.shape}"
            )
        return np.clip(value, -1.0, 1.0)

    def shift(self, latent: np.ndarray, steps: int) -> np.ndarray:
        value = self.project(latent)
        if not isinstance(steps, (int, np.integer)) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        horizon, knots = self._weights.shape
        if steps >= horizon:
            return np.zeros_like(value)
        positions = np.linspace(0.0, horizon - 1.0, knots)
        shifted = np.stack(
            [
                np.interp(
                    positions + int(steps),
                    positions,
                    value[:, dimension],
                    right=0.0,
                )
                for dimension in range(value.shape[1])
            ],
            axis=-1,
        )
        return np.clip(shifted, -1.0, 1.0)

    def decode(self, latents: np.ndarray) -> MotionGuidance:
        latents = np.asarray(latents, dtype=np.float64)
        if latents.ndim != 3 or latents.shape[1:] != self.latent_shape:
            raise ValueError(
                f"latents must have shape (N, *{self.latent_shape}), got {latents.shape}"
            )
        batch = latents.shape[0]
        horizon = self._weights.shape[0]
        dense = np.einsum(
            "hk,nkd->nhd", self._weights, np.clip(latents, -1.0, 1.0) * self._scale
        )
        poses = dense.reshape(batch, horizon, NUM_LINKS, POSE_DIM)
        position_delta = poses[..., :3]
        rotvec = poses[..., 3:]
        reference = self._reference

        body_position = np.repeat(reference.body_position_world, batch, axis=0)
        body_orientation = np.repeat(reference.body_orientation_world, batch, axis=0)
        link_indices = self._link_indices
        ref_pos = reference.body_position_world[0][:, link_indices]
        ref_quat = reference.body_orientation_world[0][:, link_indices]
        body_position[:, :, link_indices] = ref_pos[None] + position_delta
        body_orientation[:, :, link_indices] = _canonical_unit_quat(
            _quat_multiply(_rotvec_to_quat(rotvec), ref_quat[None])
        )

        lin_vel = np.repeat(reference.body_linear_velocity_world, batch, axis=0)
        ang_vel = np.repeat(reference.body_angular_velocity_world, batch, axis=0)
        deformed = np.any(np.clip(latents, -1.0, 1.0) != 0.0, axis=(1, 2))
        if np.any(deformed):
            computed_lin = np.gradient(
                body_position[:, :, link_indices], self._control_period, axis=1
            )
            computed_ang = np.gradient(
                body_orientation[:, :, link_indices, 1:],
                self._control_period,
                axis=1,
            )
            mask = deformed[:, None, None, None]
            lin_vel[:, :, link_indices] = np.where(
                mask, computed_lin, lin_vel[:, :, link_indices]
            )
            ang_vel[:, :, link_indices] = np.where(
                mask, computed_ang, ang_vel[:, :, link_indices]
            )
        return MotionGuidance(
            query_offsets=reference.query_offsets,
            joint_names=reference.joint_names,
            body_names=reference.body_names,
            joint_position=np.repeat(reference.joint_position, batch, axis=0),
            joint_velocity=np.repeat(reference.joint_velocity, batch, axis=0),
            body_position_world=body_position,
            body_orientation_world=body_orientation,
            body_linear_velocity_world=lin_vel,
            body_angular_velocity_world=ang_vel,
        )
