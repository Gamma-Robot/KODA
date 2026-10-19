"""Device-resident reduction for the motion-tracking rollout objective."""

from __future__ import annotations

from typing import Any

import numpy as np

from sil_trackers import MOTION_GUIDANCE_ARRAY_FIELDS, MotionGuidance
from sil_trackers.batched_tracker import RobotStateView
from sil_tts.motion_tracking.objective import (
    MotionTrackingObjective,
    _body_weights,
    _deformation_terms,
    _horizon_time_weights,
)

try:  # Keep CPU-only imports usable.
    import warp as wp
except ImportError:  # pragma: no cover - runtime construction reports the error
    wp = None


if wp is not None:

    @wp.kernel
    def _accumulate_motion_tracking_cost(
        step_index: int,
        horizon: int,
        joint_count: int,
        body_count: int,
        joint_position: wp.array2d(dtype=wp.float32),
        joint_velocity: wp.array2d(dtype=wp.float32),
        body_position: wp.array2d(dtype=wp.vec3),
        body_orientation: wp.array2d(dtype=wp.quat),
        nominal_joint_position: wp.array2d(dtype=wp.float64),
        nominal_joint_velocity: wp.array2d(dtype=wp.float64),
        nominal_body_position: wp.array3d(dtype=wp.float64),
        nominal_body_orientation: wp.array3d(dtype=wp.float64),
        joint_position_weight: wp.float64,
        joint_velocity_weight: wp.float64,
        body_position_weight: wp.float64,
        body_orientation_weight: wp.float64,
        joint_speed_weight: wp.float64,
        terminal_joint_weight: wp.float64,
        terminal_body_weight: wp.float64,
        action_rate_weight: wp.float64,
        has_prev: int,
        position_target: wp.array2d(dtype=wp.float32),
        prev_position_target: wp.array2d(dtype=wp.float32),
        body_scale: wp.array(dtype=wp.float64),
        time_scale: wp.array(dtype=wp.float64),
        total_cost: wp.array(dtype=wp.float64),
        invalid: wp.array(dtype=wp.int32),
    ):
        lane = wp.tid()
        joint_position_error = wp.float64(0.0)
        joint_velocity_error = wp.float64(0.0)
        joint_speed = wp.float64(0.0)
        body_position_error = wp.float64(0.0)
        body_orientation_error = wp.float64(0.0)
        valid = int(1)

        for joint in range(joint_count):
            joint_position_value = wp.float64(joint_position[lane, joint])
            joint_velocity_value = wp.float64(joint_velocity[lane, joint])
            target_position = nominal_joint_position[step_index, joint]
            target_velocity = nominal_joint_velocity[step_index, joint]
            if (
                not wp.isfinite(joint_position_value)
                or not wp.isfinite(joint_velocity_value)
                or not wp.isfinite(target_position)
                or not wp.isfinite(target_velocity)
            ):
                valid = 0
            position_delta = joint_position_value - target_position
            velocity_delta = joint_velocity_value - target_velocity
            joint_position_error += position_delta * position_delta
            joint_velocity_error += velocity_delta * velocity_delta
            joint_speed += joint_velocity_value * joint_velocity_value

        action_rate = wp.float64(0.0)
        for joint in range(joint_count):
            current = position_target[lane, joint]
            current_target = wp.float64(current)
            if not wp.isfinite(current_target):
                valid = 0
            if has_prev == 1:
                previous_target = wp.float64(prev_position_target[lane, joint])
                if not wp.isfinite(previous_target):
                    valid = 0
                delta = current_target - previous_target
                action_rate += delta * delta
            prev_position_target[lane, joint] = current

        for body in range(body_count):
            body_position_value = body_position[lane, body]
            dx = (
                wp.float64(body_position_value[0])
                - nominal_body_position[step_index, body, 0]
            )
            dy = (
                wp.float64(body_position_value[1])
                - nominal_body_position[step_index, body, 1]
            )
            dz = (
                wp.float64(body_position_value[2])
                - nominal_body_position[step_index, body, 2]
            )
            scale = body_scale[body]
            body_position_error += (dx * dx + dy * dy + dz * dz) * scale

            orientation = body_orientation[lane, body]
            dot = (
                wp.float64(orientation[0])
                * nominal_body_orientation[step_index, body, 0]
                + wp.float64(orientation[1])
                * nominal_body_orientation[step_index, body, 1]
                + wp.float64(orientation[2])
                * nominal_body_orientation[step_index, body, 2]
                + wp.float64(orientation[3])
                * nominal_body_orientation[step_index, body, 3]
            )
            if (
                not wp.isfinite(dx)
                or not wp.isfinite(dy)
                or not wp.isfinite(dz)
                or not wp.isfinite(dot)
            ):
                valid = 0
            angle = wp.float64(2.0) * wp.acos(
                wp.clamp(wp.abs(dot), wp.float64(0.0), wp.float64(1.0))
            )
            body_orientation_error += (angle * angle) * scale

        joints = wp.float64(joint_count)
        running = (
            joint_position_weight * joint_position_error / joints
            + joint_velocity_weight * joint_velocity_error / joints
            + body_position_weight * body_position_error
            + body_orientation_weight * body_orientation_error
            + joint_speed_weight * joint_speed / joints
            + action_rate_weight * action_rate
        )
        if step_index == horizon - 1:
            running += (
                terminal_joint_weight * joint_position_error / joints
                + terminal_body_weight * body_position_error
            )
        running = running * time_scale[step_index]
        updated = total_cost[lane] + running
        if valid == 0 or not wp.isfinite(updated):
            invalid[lane] = 1
        else:
            total_cost[lane] = updated


class MotionTrackingCostReducer:
    """Reusable per-capacity objective state with boundary-only readback."""

    def __init__(
        self,
        objective: MotionTrackingObjective,
        state_view: RobotStateView,
        command_view: Any | None = None,
    ) -> None:
        if wp is None:
            raise RuntimeError("device motion-tracking cost requires warp-lang")
        self._objective = objective
        self._state_view = state_view
        self._joint_names = tuple(state_view.joint_names)
        self._body_names = tuple(state_view.body_names)
        self._batch, self._device = self._validate_state_view(state_view)
        self._horizon: int | None = None
        self._expected_step = 0
        self._prepared = False
        self._has_prev = False
        self._static_cost = np.empty(self._batch, dtype=np.float64)
        body_scale = _body_weights(
            objective.body_cost_weights,
            len(self._body_names),
            normalize=objective.normalize_body_cost_weights,
        )
        with wp.ScopedDevice(self._device):
            self._total = wp.empty(self._batch, dtype=wp.float64, device=self._device)
            self._invalid = wp.empty(self._batch, dtype=wp.int32, device=self._device)
            self._body_scale = wp.array(
                np.ascontiguousarray(body_scale),
                dtype=wp.float64,
                device=self._device,
            )
            self._prev_position_target = wp.zeros(
                (self._batch, len(self._joint_names)),
                dtype=wp.float32,
                device=self._device,
            )
            if command_view is None:
                if objective.action_rate_l2_weight > 0.0:
                    raise ValueError("action_rate_l2 requires a command_view")
                self._position_target = wp.zeros(
                    (self._batch, len(self._joint_names)),
                    dtype=wp.float32,
                    device=self._device,
                )
            else:
                self._position_target = self._validate_position_target(
                    command_view.position_target
                )

    def prepare(
        self,
        nominal: MotionGuidance,
        candidate: MotionGuidance,
    ) -> None:
        """Upload one wave's targets and reset its compact accumulators."""
        horizon = self._validate_guidance(nominal, candidate)
        if self._horizon is None:
            self._allocate_targets(horizon)
        elif horizon != self._horizon:
            raise ValueError(
                f"device cost horizon changed from {self._horizon} to {horizon}"
            )

        deformation = _deformation_terms(self._objective, nominal, candidate)
        self._static_cost = np.sum(
            np.stack(tuple(deformation.values()), axis=0), axis=0
        )
        if not np.isfinite(self._static_cost).all():
            raise ValueError("deformation cost must be finite")

        with wp.ScopedDevice(self._device):
            self._nominal_joint_position.assign(
                np.ascontiguousarray(nominal.joint_position[0], dtype=np.float64)
            )
            self._nominal_joint_velocity.assign(
                np.ascontiguousarray(nominal.joint_velocity[0], dtype=np.float64)
            )
            self._nominal_body_position.assign(
                np.ascontiguousarray(nominal.body_position_world[0], dtype=np.float64)
            )
            self._nominal_body_orientation.assign(
                np.ascontiguousarray(
                    nominal.body_orientation_world[0], dtype=np.float64
                )
            )
            self._time_scale.assign(
                np.ascontiguousarray(
                    _horizon_time_weights(
                        horizon, self._objective.receding_time_weights
                    ),
                    dtype=np.float64,
                )
            )
            self._total.zero_()
            self._invalid.zero_()
            self._prev_position_target.zero_()
        self._expected_step = 0
        self._has_prev = False
        self._prepared = True

    def record(self, step_index: int, state_view: RobotStateView) -> None:
        """Accumulate one post-step frame; steps must be complete and ordered."""
        if not self._prepared or self._horizon is None:
            raise RuntimeError("prepare must be called before record")
        if state_view is not self._state_view:
            raise ValueError("device cost reducer is bound to one RobotStateView")
        if step_index != self._expected_step or step_index >= self._horizon:
            raise ValueError(
                f"expected objective step {self._expected_step}, got {step_index}"
            )
        o = self._objective
        with wp.ScopedDevice(self._device):
            wp.launch(
                _accumulate_motion_tracking_cost,
                dim=self._batch,
                inputs=[
                    step_index,
                    self._horizon,
                    len(self._joint_names),
                    len(self._body_names),
                    state_view.joint_position,
                    state_view.joint_velocity,
                    state_view.body_position_world,
                    state_view.body_orientation_world,
                    self._nominal_joint_position,
                    self._nominal_joint_velocity,
                    self._nominal_body_position,
                    self._nominal_body_orientation,
                    o.tracking_joint_position_weight,
                    o.tracking_joint_velocity_weight,
                    o.tracking_body_position_weight,
                    o.tracking_body_orientation_weight,
                    o.joint_speed_weight,
                    o.terminal_joint_position_weight,
                    o.terminal_body_position_weight,
                    o.action_rate_l2_weight,
                    1 if self._has_prev else 0,
                    self._position_target,
                    self._prev_position_target,
                    self._body_scale,
                    self._time_scale,
                    self._total,
                    self._invalid,
                ],
                device=self._device,
            )
        self._has_prev = True
        self._expected_step += 1

    def finalize(self) -> np.ndarray:
        """Read only the total and invalid vectors after a complete horizon."""
        if not self._prepared or self._horizon is None:
            raise RuntimeError("prepare must be called before finalize")
        if self._expected_step != self._horizon:
            raise RuntimeError(
                f"objective recorded {self._expected_step}/{self._horizon} steps"
            )
        wp.synchronize_device(self._device)
        invalid = np.asarray(self._invalid.numpy(), dtype=np.int32)
        dynamic = np.asarray(self._total.numpy(), dtype=np.float64)
        self._prepared = False
        if invalid.shape != (self._batch,) or dynamic.shape != (self._batch,):
            raise RuntimeError("device objective returned invalid batch shapes")
        result = dynamic + self._static_cost
        bad = (invalid != 0) | ~np.isfinite(dynamic) | ~np.isfinite(result)
        result[bad] = np.nan
        return result

    def _allocate_targets(self, horizon: int) -> None:
        with wp.ScopedDevice(self._device):
            self._nominal_joint_position = wp.empty(
                (horizon, len(self._joint_names)),
                dtype=wp.float64,
                device=self._device,
            )
            self._nominal_joint_velocity = wp.empty(
                (horizon, len(self._joint_names)),
                dtype=wp.float64,
                device=self._device,
            )
            self._nominal_body_position = wp.empty(
                (horizon, len(self._body_names), 3),
                dtype=wp.float64,
                device=self._device,
            )
            self._nominal_body_orientation = wp.empty(
                (horizon, len(self._body_names), 4),
                dtype=wp.float64,
                device=self._device,
            )
            self._time_scale = wp.empty(
                horizon,
                dtype=wp.float64,
                device=self._device,
            )
        self._horizon = horizon

    def _validate_guidance(
        self, nominal: MotionGuidance, candidate: MotionGuidance
    ) -> int:
        horizon = len(nominal.query_offsets)
        dense = tuple(range(horizon))
        if horizon < 1 or nominal.query_offsets != dense or candidate.query_offsets != dense:
            raise ValueError("nominal and candidate guidance must share one dense horizon")
        if nominal.joint_position.shape[0] != 1:
            raise ValueError("nominal guidance must contain one lane")
        if candidate.joint_position.shape[0] != self._batch:
            raise ValueError(
                f"candidate batch must be {self._batch}, got "
                f"{candidate.joint_position.shape[0]}"
            )
        for label, guidance in (("nominal", nominal), ("candidate", candidate)):
            if (
                guidance.joint_names != self._joint_names
                or guidance.body_names != self._body_names
            ):
                raise ValueError(f"{label} names must match the bound RobotStateView")
            for field in MOTION_GUIDANCE_ARRAY_FIELDS:
                if not np.isfinite(getattr(guidance, field)).all():
                    raise ValueError(f"{label} {field} must be finite")
        return horizon

    def _validate_state_view(self, state_view: RobotStateView) -> tuple[int, Any]:
        joint_shape = (int(state_view.joint_position.shape[0]), len(self._joint_names))
        if joint_shape[0] < 1:
            raise ValueError("device state batch must be positive")
        body_shape = (joint_shape[0], len(self._body_names))
        specs = (
            (state_view.joint_position, joint_shape, wp.float32, "joint_position"),
            (state_view.joint_velocity, joint_shape, wp.float32, "joint_velocity"),
            (state_view.body_position_world, body_shape, wp.vec3, "body_position_world"),
            (
                state_view.body_orientation_world,
                body_shape,
                wp.quat,
                "body_orientation_world",
            ),
        )
        device = getattr(state_view.joint_position, "device", None)
        if device is None or not getattr(device, "is_cuda", False):
            raise ValueError("device cost reducer requires CUDA Warp state arrays")
        for value, shape, dtype, field in specs:
            if (
                getattr(value, "shape", None) != shape
                or getattr(value, "dtype", None) != dtype
                or getattr(value, "device", None) != device
                or not getattr(value, "is_contiguous", False)
            ):
                raise ValueError(
                    f"{field} must be contiguous {dtype} with shape {shape} on {device}"
                )
        return joint_shape[0], device

    def _validate_position_target(self, value: Any) -> Any:
        shape = (self._batch, len(self._joint_names))
        if (
            getattr(value, "shape", None) != shape
            or getattr(value, "dtype", None) != wp.float32
            or getattr(value, "device", None) != self._device
            or not getattr(value, "is_contiguous", False)
        ):
            raise ValueError(
                "position_target must be contiguous float32 with shape "
                f"{shape} on {self._device}"
            )
        return value
