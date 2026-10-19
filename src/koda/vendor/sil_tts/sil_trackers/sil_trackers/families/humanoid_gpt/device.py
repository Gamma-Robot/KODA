"""Device-resident Humanoid-GPT observation, policy, lock, and command path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sil_trackers.contracts import DevicePDCommand, MotionGuidance
from sil_trackers.device_abi import (
    normalize_tracker_state,
    validate_bound_state_view,
    validate_device_layout,
)
from sil_trackers.device_policy import PersistentOrtCudaPolicy
from sil_trackers.families.humanoid_gpt.adapter import HumanoidGptAdapter, _order
from sil_trackers.families.humanoid_gpt.observation import (
    ACTION_DIM,
    MOTION_COMMAND_DIM,
    NEXT_OFFSET,
    PROPRIOCEPTION_DIM,
)

try:  # Keep the host family importable without the optional device runtime.
    import warp as wp
except ImportError:  # pragma: no cover - exercised only without GPU extras
    wp = None


if wp is not None:

    @wp.func
    def _safe_quat(q: wp.quat) -> wp.quat:
        norm = wp.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        if norm < 1.0e-8:
            return wp.quat(1.0, 0.0, 0.0, 0.0)
        return wp.quat(q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm)


    @wp.func
    def _wrap_pi(angle: float) -> float:
        return wp.atan2(wp.sin(angle), wp.cos(angle))


    @wp.func
    def _quat_yaw(q_raw: wp.quat) -> float:
        q = _safe_quat(q_raw)
        w = q[0]
        x = q[1]
        y = q[2]
        z = q[3]
        return _wrap_pi(
            wp.atan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))
        )


    @wp.func
    def _quat_from_yaw(yaw: float) -> wp.quat:
        half = 0.5 * yaw
        return wp.quat(wp.cos(half), 0.0, 0.0, wp.sin(half))


    @wp.func
    def _quat_mul(left: wp.quat, right: wp.quat) -> wp.quat:
        w1 = left[0]
        x1 = left[1]
        y1 = left[2]
        z1 = left[3]
        w2 = right[0]
        x2 = right[1]
        y2 = right[2]
        z2 = right[3]
        return wp.quat(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )


    @wp.func
    def _rotate_inverse(q_raw: wp.quat, vector: wp.vec3) -> wp.vec3:
        q = _safe_quat(q_raw)
        scalar = q[0]
        xyz = wp.vec3(q[1], q[2], q[3])
        return (
            vector * (2.0 * scalar * scalar - 1.0)
            - wp.cross(xyz, vector) * (2.0 * scalar)
            + xyz * (2.0 * wp.dot(xyz, vector))
        )


    @wp.func
    def _rotate_xy_x(x: float, y: float, yaw: float) -> float:
        return wp.cos(yaw) * x - wp.sin(yaw) * y


    @wp.func
    def _rotate_xy_y(x: float, y: float, yaw: float) -> float:
        return wp.sin(yaw) * x + wp.cos(yaw) * y


    @wp.func
    def _rotate_vector_z(vector: wp.vec3, yaw: float) -> wp.vec3:
        return wp.vec3(
            _rotate_xy_x(vector[0], vector[1], yaw),
            _rotate_xy_y(vector[0], vector[1], yaw),
            vector[2],
        )


    @wp.func
    def _rebias_xy(
        ref_x: float,
        ref_y: float,
        robot_init_x: float,
        robot_init_y: float,
        ref_init_x: float,
        ref_init_y: float,
        yaw_offset: float,
    ) -> wp.vec3:
        dx = ref_x - ref_init_x
        dy = ref_y - ref_init_y
        return wp.vec3(
            robot_init_x + _rotate_xy_x(dx, dy, yaw_offset),
            robot_init_y + _rotate_xy_y(dx, dy, yaw_offset),
            0.0,
        )


    @wp.func
    def _rebias_quat(
        ref_quat: wp.quat, robot_init_yaw: float, ref_init_yaw: float
    ) -> wp.quat:
        ref_yaw = _quat_yaw(ref_quat)
        yaw_delta = _wrap_pi(
            robot_init_yaw + _wrap_pi(ref_yaw - ref_init_yaw) - ref_yaw
        )
        return _safe_quat(_quat_mul(_quat_from_yaw(yaw_delta), ref_quat))


    @wp.func
    def _to_navi(q_raw: wp.quat, vector: wp.vec3) -> wp.vec3:
        q = _safe_quat(q_raw)
        w = q[0]
        x = q[1]
        y = q[2]
        z = q[3]
        x_proj = wp.vec3(
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        )
        y_axis = wp.vec3(-x_proj[1], x_proj[0], 0.0)
        y_norm = wp.sqrt(y_axis[0] * y_axis[0] + y_axis[1] * y_axis[1])
        if y_norm > 1.0e-8:
            inv = 1.0 / y_norm
            y_axis = wp.vec3(y_axis[0] * inv, y_axis[1] * inv, 0.0)
        else:
            y_axis = wp.vec3(0.0, 1.0, 0.0)
        x_axis = wp.vec3(y_axis[1], -y_axis[0], 0.0)
        return wp.vec3(
            x_axis[0] * vector[0] + x_axis[1] * vector[1],
            y_axis[0] * vector[0] + y_axis[1] * vector[1],
            vector[2],
        )


    @wp.kernel
    def _lock_alignment(
        step_index: int,
        body_orientation: wp.array2d(dtype=wp.quat),
        root_body_index: int,
        guidance_root_position: wp.array2d(dtype=wp.vec3),
        guidance_root_orientation: wp.array2d(dtype=wp.quat),
        alignment_initialized: wp.array(dtype=wp.int32),
        robot_init_yaw: wp.array(dtype=wp.float32),
        robot_init_xy: wp.array2d(dtype=wp.float32),
        ref_init_yaw: wp.array(dtype=wp.float32),
        ref_init_xy: wp.array2d(dtype=wp.float32),
    ):
        lane = wp.tid()
        if alignment_initialized[lane] == 0:
            robot_orientation = body_orientation[lane, root_body_index]
            current = guidance_root_position[lane, step_index]
            robot_init_yaw[lane] = _quat_yaw(robot_orientation)
            robot_init_xy[lane, 0] = 0.0
            robot_init_xy[lane, 1] = 0.0
            ref_init_yaw[lane] = _quat_yaw(
                guidance_root_orientation[lane, step_index]
            )
            ref_init_xy[lane, 0] = current[0]
            ref_init_xy[lane, 1] = current[1]
            alignment_initialized[lane] = 1


    @wp.kernel
    def _pack_proprioception(
        joint_position: wp.array2d(dtype=wp.float32),
        joint_velocity: wp.array2d(dtype=wp.float32),
        body_orientation: wp.array2d(dtype=wp.quat),
        body_angular_velocity: wp.array2d(dtype=wp.vec3),
        state_joint_indices: wp.array(dtype=wp.int32),
        root_body_index: int,
        default_joint_position: wp.array(dtype=wp.float32),
        last_action: wp.array2d(dtype=wp.float32),
        proprioception: wp.array2d(dtype=wp.float32),
    ):
        lane, feature = wp.tid()
        robot_orientation = body_orientation[lane, root_body_index]
        value = float(0.0)
        if feature < 3:
            angular = _rotate_inverse(
                robot_orientation, body_angular_velocity[lane, root_body_index]
            )
            value = angular[feature]
        elif feature < 6:
            gravity = _rotate_inverse(
                robot_orientation, wp.vec3(0.0, 0.0, -1.0)
            )
            value = gravity[feature - 3]
        elif feature < 35:
            joint = feature - 6
            value = (
                joint_position[lane, state_joint_indices[joint]]
                - default_joint_position[joint]
            )
        elif feature < 64:
            joint = feature - 35
            value = joint_velocity[lane, state_joint_indices[joint]]
        else:
            value = last_action[lane, feature - 64]
        proprioception[lane, feature] = value


    @wp.kernel
    def _pack_motion_command(
        step_index: int,
        joint_position: wp.array3d(dtype=wp.float32),
        root_position: wp.array2d(dtype=wp.vec3),
        root_orientation: wp.array2d(dtype=wp.quat),
        root_linear_velocity: wp.array2d(dtype=wp.vec3),
        root_angular_velocity: wp.array2d(dtype=wp.vec3),
        body_position: wp.array2d(dtype=wp.vec3),
        body_orientation: wp.array2d(dtype=wp.quat),
        root_body_index: int,
        default_joint_position: wp.array(dtype=wp.float32),
        robot_init_yaw: wp.array(dtype=wp.float32),
        robot_init_xy: wp.array2d(dtype=wp.float32),
        ref_init_yaw: wp.array(dtype=wp.float32),
        ref_init_xy: wp.array2d(dtype=wp.float32),
        motion_command: wp.array2d(dtype=wp.float32),
    ):
        lane, feature = wp.tid()
        robot_yaw = robot_init_yaw[lane]
        ref_yaw = ref_init_yaw[lane]
        yaw_offset = _wrap_pi(robot_yaw - ref_yaw)
        current_step = step_index
        next_step = step_index + 1
        next_root = root_position[lane, next_step]
        next_quat = _rebias_quat(
            root_orientation[lane, next_step], robot_yaw, ref_yaw
        )
        value = float(0.0)
        if feature < ACTION_DIM:
            value = (
                joint_position[lane, next_step, feature]
                - default_joint_position[feature]
            )
        elif feature == ACTION_DIM:
            value = next_root[2]
        elif feature < 33:
            gravity = _rotate_inverse(next_quat, wp.vec3(0.0, 0.0, -1.0))
            value = gravity[feature - 30]
        elif feature < 39:
            angular = _to_navi(
                next_quat,
                _rotate_vector_z(root_angular_velocity[lane, next_step], yaw_offset),
            )
            linear = _to_navi(
                next_quat,
                _rotate_vector_z(root_linear_velocity[lane, next_step], yaw_offset),
            )
            local = feature - 33
            if local < 3:
                value = angular[local]
            else:
                value = linear[local - 3]
        else:
            current_root = root_position[lane, current_step]
            current_xy = _rebias_xy(
                current_root[0],
                current_root[1],
                robot_init_xy[lane, 0],
                robot_init_xy[lane, 1],
                ref_init_xy[lane, 0],
                ref_init_xy[lane, 1],
                yaw_offset,
            )
            current_quat = _rebias_quat(
                root_orientation[lane, current_step], robot_yaw, ref_yaw
            )
            robot_orientation = body_orientation[lane, root_body_index]
            robot_position = body_position[lane, root_body_index]
            yaw_error = _wrap_pi(
                _quat_yaw(current_quat) - _quat_yaw(robot_orientation)
            )
            robot_heading = _quat_yaw(robot_orientation)
            dx = current_xy[0] - robot_position[0]
            dy = current_xy[1] - robot_position[1]
            xy_x = _rotate_xy_x(dx, dy, -robot_heading)
            xy_y = _rotate_xy_y(dx, dy, -robot_heading)
            if feature == 39:
                value = wp.cos(yaw_error)
            elif feature == 40:
                value = wp.sin(yaw_error)
            elif feature == 41:
                value = xy_x
            else:
                value = xy_y
        motion_command[lane, feature] = value


    @wp.kernel
    def _map_action(
        action: wp.array2d(dtype=wp.float32),
        command_to_policy: wp.array(dtype=wp.int32),
        default_joint_position: wp.array(dtype=wp.float32),
        action_scale: wp.array(dtype=wp.float32),
        proportional_gain: wp.array(dtype=wp.float32),
        derivative_gain: wp.array(dtype=wp.float32),
        position_target: wp.array2d(dtype=wp.float32),
        velocity_target: wp.array2d(dtype=wp.float32),
        feedforward_torque: wp.array2d(dtype=wp.float32),
        command_kp: wp.array2d(dtype=wp.float32),
        command_kd: wp.array2d(dtype=wp.float32),
        last_action: wp.array2d(dtype=wp.float32),
    ):
        lane, command_joint = wp.tid()
        policy_joint = command_to_policy[command_joint]
        value = action[lane, policy_joint]
        position_target[lane, command_joint] = (
            default_joint_position[policy_joint]
            + value * action_scale[policy_joint]
        )
        velocity_target[lane, command_joint] = 0.0
        feedforward_torque[lane, command_joint] = 0.0
        command_kp[lane, command_joint] = proportional_gain[policy_joint]
        command_kd[lane, command_joint] = derivative_gain[policy_joint]
        last_action[lane, policy_joint] = value


@dataclass(frozen=True)
class _GuidanceBuffers:
    joint_position: Any
    root_position: Any
    root_orientation: Any
    root_linear_velocity: Any
    root_angular_velocity: Any


class HumanoidGptDeviceTracker:
    """Prepared Humanoid-GPT tracker bound to one CUDA simulation batch."""

    def __init__(
        self,
        adapter: HumanoidGptAdapter,
        policy_path: str | Path,
        state_view: object,
        command_view: object,
    ) -> None:
        if wp is None:
            raise RuntimeError("Humanoid-GPT device execution requires warp-lang")
        self._config = adapter._config
        self._family = adapter._family
        self._state_view = state_view
        self._command = command_view
        self._batch_size, self._device = validate_device_layout(
            state_view,
            command_view,
            family="Humanoid-GPT",
            command_joint_names=self._config.command_joint_names,
            float_dtype=wp.float32,
            body_dtypes={
                "body_position_world": wp.vec3,
                "body_orientation_world": wp.quat,
                "body_angular_velocity_world": wp.vec3,
            },
        )
        state_joint_names = tuple(getattr(state_view, "joint_names"))
        state_body_names = tuple(getattr(state_view, "body_names"))
        if self._family.root_body not in state_body_names:
            raise ValueError("state_view does not contain the Humanoid-GPT root body")
        self._root_body_index = state_body_names.index(self._family.root_body)
        state_joint_indices = np.asarray(
            _order(state_joint_names, self._config.policy_joint_names), np.int32
        )
        command_to_policy = np.asarray(
            _order(self._config.policy_joint_names, self._config.command_joint_names),
            np.int32,
        )
        self._stream = wp.get_stream(self._device)

        with wp.ScopedStream(self._stream):
            self._state_joint_indices = self._array(state_joint_indices, wp.int32)
            self._command_to_policy = self._array(command_to_policy, wp.int32)
            self._default_joint_position = self._array(
                self._family.default_joint_position, wp.float32
            )
            self._action_scale = self._array(self._family.action_scale, wp.float32)
            self._proportional_gain = self._array(
                self._family.proportional_gain, wp.float32
            )
            self._derivative_gain = self._array(
                self._family.derivative_gain, wp.float32
            )
            self._last_action = wp.empty(
                (self._batch_size, ACTION_DIM),
                dtype=wp.float32,
                device=self._device,
            )
            self._alignment_initialized = wp.empty(
                self._batch_size, dtype=wp.int32, device=self._device
            )
            self._robot_init_yaw = wp.empty(
                self._batch_size, dtype=wp.float32, device=self._device
            )
            self._robot_init_xy = wp.empty(
                (self._batch_size, 2), dtype=wp.float32, device=self._device
            )
            self._ref_init_yaw = wp.empty(
                self._batch_size, dtype=wp.float32, device=self._device
            )
            self._ref_init_xy = wp.empty(
                (self._batch_size, 2), dtype=wp.float32, device=self._device
            )
            self._policy_inputs = {
                "proprioception": wp.empty(
                    (self._batch_size, PROPRIOCEPTION_DIM),
                    dtype=wp.float32,
                    device=self._device,
                ),
                "motion_command": wp.empty(
                    (self._batch_size, MOTION_COMMAND_DIM),
                    dtype=wp.float32,
                    device=self._device,
                ),
            }
            self._policy_outputs = {
                "action": wp.empty(
                    (self._batch_size, ACTION_DIM),
                    dtype=wp.float32,
                    device=self._device,
                )
            }
        self._policy = PersistentOrtCudaPolicy(
            policy_path,
            self._config,
            input_buffers=self._policy_inputs,
            output_buffers=self._policy_outputs,
            stream=self._stream,
        )
        self._guidance_buffers: dict[int, _GuidanceBuffers] = {}
        self._prepared: _GuidanceBuffers | None = None
        self._horizon = 0
        self._execution_steps = 0
        self._total_steps = 0
        self._state_template = adapter.initial_state(self._batch_size)

    def reset(self, state: object) -> None:
        values = normalize_tracker_state(
            state, self._state_template, family="Humanoid-GPT"
        )
        self._policy.drain()
        self._prepared = None
        self._horizon = 0
        self._execution_steps = 0
        self._total_steps = 0
        with wp.ScopedStream(self._stream):
            self._last_action.assign(values["last_action"])
            self._alignment_initialized.assign(
                values["alignment_initialized"].astype(np.int32)
            )
            self._robot_init_yaw.assign(values["robot_init_yaw"])
            self._robot_init_xy.assign(values["robot_init_xy"])
            self._ref_init_yaw.assign(values["ref_init_yaw"])
            self._ref_init_xy.assign(values["ref_init_xy"])

    def capture_state(self) -> dict[str, np.ndarray]:
        wp.synchronize_stream(self._stream)
        return {
            "last_action": np.asarray(self._last_action.numpy()).copy(),
            "alignment_initialized": np.asarray(
                self._alignment_initialized.numpy(), dtype=bool
            ),
            "robot_init_yaw": np.asarray(self._robot_init_yaw.numpy()).copy(),
            "robot_init_xy": np.asarray(self._robot_init_xy.numpy()).copy(),
            "ref_init_yaw": np.asarray(self._ref_init_yaw.numpy()).copy(),
            "ref_init_xy": np.asarray(self._ref_init_xy.numpy()).copy(),
        }

    def prepare(
        self,
        history: MotionGuidance | None,
        guidance: MotionGuidance,
        right_context: MotionGuidance | None,
        *,
        num_steps: int,
    ) -> None:
        del history
        horizon = len(guidance.query_offsets)
        if not isinstance(num_steps, (int, np.integer)) or not (
            1 <= int(num_steps) <= horizon
        ):
            raise ValueError("num_steps must be within prepared guidance")
        self._validate_guidance(guidance, "guidance")
        if right_context is not None:
            self._validate_guidance(right_context, "right_context")
            if (
                right_context.joint_names != guidance.joint_names
                or right_context.body_names != guidance.body_names
            ):
                raise ValueError("right_context names differ from guidance")

        def combined(field: str) -> np.ndarray:
            candidate = getattr(guidance, field)
            if right_context is None:
                return candidate
            return np.concatenate((candidate, getattr(right_context, field)), axis=1)

        total_steps = horizon + (
            0 if right_context is None else len(right_context.query_offsets)
        )
        if total_steps <= horizon + NEXT_OFFSET - 1:
            raise ValueError(
                "Humanoid-GPT prepared guidance requires one right-context step"
            )
        joint_order = _order(guidance.joint_names, self._config.policy_joint_names)
        root_index = guidance.body_names.index(self._family.root_body)
        arrays = {
            "joint_position": combined("joint_position")[:, :, joint_order],
            "root_position": combined("body_position_world")[:, :, root_index],
            "root_orientation": combined("body_orientation_world")[:, :, root_index],
            "root_linear_velocity": combined("body_linear_velocity_world")[
                :, :, root_index
            ],
            "root_angular_velocity": combined("body_angular_velocity_world")[
                :, :, root_index
            ],
        }

        self._prepared = None
        self._horizon = 0
        self._execution_steps = 0
        self._total_steps = 0
        buffers = self._guidance_buffers.get(total_steps)
        if buffers is None:
            with wp.ScopedStream(self._stream):
                buffers = _GuidanceBuffers(
                    joint_position=wp.empty(
                        (self._batch_size, total_steps, ACTION_DIM),
                        dtype=wp.float32,
                        device=self._device,
                    ),
                    root_position=wp.empty(
                        (self._batch_size, total_steps),
                        dtype=wp.vec3,
                        device=self._device,
                    ),
                    root_orientation=wp.empty(
                        (self._batch_size, total_steps),
                        dtype=wp.quat,
                        device=self._device,
                    ),
                    root_linear_velocity=wp.empty(
                        (self._batch_size, total_steps),
                        dtype=wp.vec3,
                        device=self._device,
                    ),
                    root_angular_velocity=wp.empty(
                        (self._batch_size, total_steps),
                        dtype=wp.vec3,
                        device=self._device,
                    ),
                )
            self._guidance_buffers[total_steps] = buffers
        with wp.ScopedStream(self._stream):
            for name, value in arrays.items():
                getattr(buffers, name).assign(
                    np.ascontiguousarray(value, dtype=np.float32)
                )
        self._prepared = buffers
        self._horizon = horizon
        self._execution_steps = int(num_steps)
        self._total_steps = total_steps

    def step_prepared(
        self, state_view: object, step_index: int
    ) -> DevicePDCommand:
        if self._prepared is None:
            raise RuntimeError("prepare() must be called before step_prepared()")
        if not isinstance(step_index, (int, np.integer)) or not (
            0 <= int(step_index) < self._horizon
        ):
            raise ValueError("step_index is outside prepared guidance")
        if int(step_index) + NEXT_OFFSET >= self._total_steps:
            raise ValueError("prepared guidance lacks Humanoid-GPT next-frame context")
        validate_bound_state_view(state_view, self._state_view, family="Humanoid-GPT")
        buffers = self._prepared
        wp.launch(
            _lock_alignment,
            dim=self._batch_size,
            inputs=[
                int(step_index),
                state_view.body_orientation_world,
                self._root_body_index,
                buffers.root_position,
                buffers.root_orientation,
                self._alignment_initialized,
                self._robot_init_yaw,
                self._robot_init_xy,
                self._ref_init_yaw,
                self._ref_init_xy,
            ],
            device=self._device,
            stream=self._stream,
        )
        wp.launch(
            _pack_proprioception,
            dim=(self._batch_size, PROPRIOCEPTION_DIM),
            inputs=[
                state_view.joint_position,
                state_view.joint_velocity,
                state_view.body_orientation_world,
                state_view.body_angular_velocity_world,
                self._state_joint_indices,
                self._root_body_index,
                self._default_joint_position,
                self._last_action,
                self._policy_inputs["proprioception"],
            ],
            device=self._device,
            stream=self._stream,
        )
        wp.launch(
            _pack_motion_command,
            dim=(self._batch_size, MOTION_COMMAND_DIM),
            inputs=[
                int(step_index),
                buffers.joint_position,
                buffers.root_position,
                buffers.root_orientation,
                buffers.root_linear_velocity,
                buffers.root_angular_velocity,
                state_view.body_position_world,
                state_view.body_orientation_world,
                self._root_body_index,
                self._default_joint_position,
                self._robot_init_yaw,
                self._robot_init_xy,
                self._ref_init_yaw,
                self._ref_init_xy,
                self._policy_inputs["motion_command"],
            ],
            device=self._device,
            stream=self._stream,
        )
        self._policy.enqueue(
            synchronize=int(step_index) == self._execution_steps - 1
        )
        wp.launch(
            _map_action,
            dim=(self._batch_size, ACTION_DIM),
            inputs=[
                self._policy_outputs["action"],
                self._command_to_policy,
                self._default_joint_position,
                self._action_scale,
                self._proportional_gain,
                self._derivative_gain,
                self._command.position_target,
                self._command.velocity_target,
                self._command.feedforward_torque,
                self._command.proportional_gain,
                self._command.derivative_gain,
                self._last_action,
            ],
            device=self._device,
            stream=self._stream,
        )
        return self._command

    def close(self) -> None:
        self._policy.close()

    def _array(self, value: np.ndarray, dtype: Any) -> Any:
        return wp.array(np.ascontiguousarray(value), dtype=dtype, device=self._device)

    def _validate_guidance(self, guidance: MotionGuidance, label: str) -> None:
        steps = len(guidance.query_offsets)
        if guidance.query_offsets != tuple(range(steps)):
            raise ValueError(f"{label} must be dense and zero-based")
        if guidance.joint_position.shape[0] != self._batch_size:
            raise ValueError(f"{label} batch size differs from the device tracker")


__all__ = ["HumanoidGptDeviceTracker"]
