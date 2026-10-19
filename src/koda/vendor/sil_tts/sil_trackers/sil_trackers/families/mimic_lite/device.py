"""Device-resident Mimic-Lite observation, history, policy, and command path."""

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
from sil_trackers.families.mimic_lite.adapter import MimicLiteAdapter, _order
from sil_trackers.families.mimic_lite.observation import (
    ACTION_DIM,
    COMMAND_DIM,
    HISTORY_LAGS,
    HISTORY_LENGTH,
    POLICY_DIM,
    PREVIOUS_ACTIONS,
    QUERY_OFFSETS,
)

try:  # Keep the host family importable without the optional device runtime.
    import warp as wp
except ImportError:  # pragma: no cover - exercised only without GPU extras
    wp = None


_ANGULAR_VELOCITY = 0
_GRAVITY = 3
_JOINT_POSITION = 6
_JOINT_VELOCITY = 35
_HISTORY_DIM = 64
_CURRENT_QUERY = QUERY_OFFSETS.index(0)
_PAST_STEPS = max(0, -min(QUERY_OFFSETS))
_FUTURE_STEPS = max(0, max(QUERY_OFFSETS))


if wp is not None:

    @wp.func
    def _safe_quat(q: wp.quat) -> wp.quat:
        norm = wp.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        if norm < 1.0e-8:
            return wp.quat(1.0, 0.0, 0.0, 0.0)
        return wp.quat(q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm)


    @wp.func
    def _quat_conjugate(q: wp.quat) -> wp.quat:
        return wp.quat(q[0], -q[1], -q[2], -q[3])


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
    def _rotate(q_raw: wp.quat, vector: wp.vec3) -> wp.vec3:
        q = _safe_quat(q_raw)
        scalar = q[0]
        xyz = wp.vec3(q[1], q[2], q[3])
        return (
            vector * (2.0 * scalar * scalar - 1.0)
            + wp.cross(xyz, vector) * (2.0 * scalar)
            + xyz * (2.0 * wp.dot(xyz, vector))
        )


    @wp.func
    def _rotate_inverse(q_raw: wp.quat, vector: wp.vec3) -> wp.vec3:
        return _rotate(_quat_conjugate(_safe_quat(q_raw)), vector)


    @wp.func
    def _projected_yaw(q_raw: wp.quat) -> wp.quat:
        q = _safe_quat(q_raw)
        x_world = _rotate(q, wp.vec3(1.0, 0.0, 0.0))
        z_world = _rotate(q, wp.vec3(0.0, 0.0, 1.0))
        heading_x = x_world[0]
        heading_y = x_world[1]
        if wp.sqrt(heading_x * heading_x + heading_y * heading_y) <= 0.1:
            if x_world[2] < 0.0:
                heading_x = z_world[0]
                heading_y = z_world[1]
            else:
                heading_x = -z_world[0]
                heading_y = -z_world[1]
        yaw = wp.atan2(heading_y, heading_x)
        return wp.quat(wp.cos(0.5 * yaw), 0.0, 0.0, wp.sin(0.5 * yaw))


    @wp.func
    def _rotation_row_6d(q_raw: wp.quat, component: int) -> float:
        q = _safe_quat(q_raw)
        w = q[0]
        x = q[1]
        y = q[2]
        z = q[3]
        if component == 0:
            return 1.0 - 2.0 * (y * y + z * z)
        if component == 1:
            return 2.0 * (x * y - z * w)
        if component == 2:
            return 2.0 * (x * z + y * w)
        if component == 3:
            return 2.0 * (x * y + z * w)
        if component == 4:
            return 1.0 - 2.0 * (x * x + z * z)
        return 2.0 * (y * z - x * w)


    @wp.kernel
    def _observe(
        step_index: int,
        timeline_start: int,
        joint_position: wp.array2d(dtype=wp.float32),
        joint_velocity: wp.array2d(dtype=wp.float32),
        body_orientation: wp.array2d(dtype=wp.quat),
        body_angular_velocity: wp.array2d(dtype=wp.vec3),
        state_joint_indices: wp.array(dtype=wp.int32),
        root_body_index: int,
        guidance_root_orientation: wp.array2d(dtype=wp.quat),
        last_action: wp.array2d(dtype=wp.float32),
        history: wp.array3d(dtype=wp.float32),
        previous_actions: wp.array3d(dtype=wp.float32),
        heading_offset: wp.array(dtype=wp.quat),
        heading_initialized: wp.array(dtype=wp.int32),
    ):
        lane, feature = wp.tid()
        root_orientation = body_orientation[lane, root_body_index]
        value = float(0.0)
        if feature < _GRAVITY:
            angular = _rotate_inverse(
                root_orientation, body_angular_velocity[lane, root_body_index]
            )
            value = angular[feature]
        elif feature < _JOINT_POSITION:
            gravity = _rotate_inverse(
                root_orientation, wp.vec3(0.0, 0.0, -1.0)
            )
            value = gravity[feature - _GRAVITY]
        elif feature < _JOINT_VELOCITY:
            joint = feature - _JOINT_POSITION
            value = joint_position[lane, state_joint_indices[joint]]
        else:
            joint = feature - _JOINT_VELOCITY
            value = joint_velocity[lane, state_joint_indices[joint]]

        for index in range(HISTORY_LENGTH - 1):
            slot = HISTORY_LENGTH - 1 - index
            history[lane, slot, feature] = history[lane, slot - 1, feature]
        history[lane, 0, feature] = value

        if feature < ACTION_DIM:
            for index in range(PREVIOUS_ACTIONS - 1):
                slot = PREVIOUS_ACTIONS - 1 - index
                previous_actions[lane, slot, feature] = previous_actions[
                    lane, slot - 1, feature
                ]
            previous_actions[lane, 0, feature] = last_action[lane, feature]

        if feature == 0 and heading_initialized[lane] == 0:
            robot_yaw = _projected_yaw(root_orientation)
            reference_yaw = _projected_yaw(
                guidance_root_orientation[lane, timeline_start + step_index]
            )
            heading_offset[lane] = _quat_mul(
                reference_yaw, _quat_conjugate(robot_yaw)
            )
            heading_initialized[lane] = 1


    @wp.kernel
    def _pack_policy(
        history_lags: wp.array(dtype=wp.int32),
        history: wp.array3d(dtype=wp.float32),
        previous_actions: wp.array3d(dtype=wp.float32),
        policy: wp.array2d(dtype=wp.float32),
    ):
        lane, feature = wp.tid()
        value = float(0.0)
        if feature < 7 * 3:
            lag = feature // 3
            component = feature - lag * 3
            value = history[
                lane, history_lags[lag], _ANGULAR_VELOCITY + component
            ]
        elif feature < 14 * 3:
            local = feature - 7 * 3
            lag = local // 3
            component = local - lag * 3
            value = history[lane, history_lags[lag], _GRAVITY + component]
        elif feature < 14 * 3 + 7 * ACTION_DIM:
            local = feature - 14 * 3
            lag = local // ACTION_DIM
            joint = local - lag * ACTION_DIM
            value = history[
                lane, history_lags[lag], _JOINT_POSITION + joint
            ]
        elif feature < 14 * 3 + 14 * ACTION_DIM:
            local = feature - (14 * 3 + 7 * ACTION_DIM)
            lag = local // ACTION_DIM
            joint = local - lag * ACTION_DIM
            value = history[
                lane, history_lags[lag], _JOINT_VELOCITY + joint
            ]
        else:
            local = feature - (14 * 3 + 14 * ACTION_DIM)
            action_slot = local // ACTION_DIM
            joint = local - action_slot * ACTION_DIM
            value = previous_actions[lane, action_slot, joint]
        policy[lane, feature] = value


    @wp.kernel
    def _pack_command(
        step_index: int,
        timeline_start: int,
        query_offsets: wp.array(dtype=wp.int32),
        guidance_joint_position: wp.array3d(dtype=wp.float32),
        guidance_root_position: wp.array2d(dtype=wp.vec3),
        guidance_root_orientation: wp.array2d(dtype=wp.quat),
        guidance_anchor_position: wp.array2d(dtype=wp.vec3),
        guidance_anchor_orientation: wp.array2d(dtype=wp.quat),
        body_orientation: wp.array2d(dtype=wp.quat),
        root_body_index: int,
        heading_offset: wp.array(dtype=wp.quat),
        command: wp.array2d(dtype=wp.float32),
    ):
        lane, feature = wp.tid()
        origin = timeline_start + step_index
        value = float(0.0)
        if feature < 8 * 3:
            query = feature // 3
            component = feature - query * 3
            root = guidance_root_position[lane, origin + query_offsets[query]]
            anchor = guidance_anchor_position[lane, origin]
            delta = wp.vec3(root[0] - anchor[0], root[1] - anchor[1], root[2])
            anchor_yaw = _projected_yaw(guidance_anchor_orientation[lane, origin])
            position = _rotate_inverse(anchor_yaw, delta)
            value = position[component]
        elif feature < 8 * 9:
            local = feature - 8 * 3
            query = local // 6
            component = local - query * 6
            aligned_robot = _quat_mul(
                heading_offset[lane], body_orientation[lane, root_body_index]
            )
            relative = _quat_mul(
                _quat_conjugate(_safe_quat(aligned_robot)),
                _safe_quat(
                    guidance_root_orientation[lane, origin + query_offsets[query]]
                ),
            )
            value = _rotation_row_6d(relative, component)
        else:
            local = feature - 8 * 9
            query = local // ACTION_DIM
            joint = local - query * ACTION_DIM
            value = guidance_joint_position[
                lane, origin + query_offsets[query], joint
            ]
        command[lane, feature] = value


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
    anchor_position: Any
    anchor_orientation: Any
    length: int


class MimicLiteDeviceTracker:
    """Prepared Mimic-Lite tracker bound to one CUDA simulation batch."""

    def __init__(
        self,
        adapter: MimicLiteAdapter,
        policy_path: str | Path,
        state_view: object,
        command_view: object,
    ) -> None:
        if wp is None:
            raise RuntimeError("Mimic-Lite device execution requires warp-lang")
        self._config = adapter._config
        self._family = adapter._family
        self._state_view = state_view
        self._command = command_view
        self._batch_size, self._device = validate_device_layout(
            state_view,
            command_view,
            family="Mimic-Lite",
            command_joint_names=self._config.command_joint_names,
            float_dtype=wp.float32,
            body_dtypes={
                "body_orientation_world": wp.quat,
                "body_angular_velocity_world": wp.vec3,
            },
        )
        state_joint_names = tuple(getattr(state_view, "joint_names"))
        state_body_names = tuple(getattr(state_view, "body_names"))
        if self._family.root_body not in state_body_names:
            raise ValueError("state_view does not contain the Mimic-Lite root body")
        self._root_body_index = state_body_names.index(self._family.root_body)
        self._stream = wp.get_stream(self._device)

        state_joint_indices = np.asarray(
            _order(state_joint_names, self._config.policy_joint_names), np.int32
        )
        command_to_policy = np.asarray(
            _order(self._config.policy_joint_names, self._config.command_joint_names),
            np.int32,
        )
        with wp.ScopedStream(self._stream):
            self._state_joint_indices = self._array(state_joint_indices, wp.int32)
            self._command_to_policy = self._array(command_to_policy, wp.int32)
            self._query_offsets = self._array(
                np.asarray(QUERY_OFFSETS, np.int32), wp.int32
            )
            self._history_lags = self._array(
                np.asarray(HISTORY_LAGS, np.int32), wp.int32
            )
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
            self._history = wp.empty(
                (self._batch_size, HISTORY_LENGTH, _HISTORY_DIM),
                dtype=wp.float32,
                device=self._device,
            )
            self._previous_actions = wp.empty(
                (self._batch_size, PREVIOUS_ACTIONS, ACTION_DIM),
                dtype=wp.float32,
                device=self._device,
            )
            self._last_action = wp.empty(
                (self._batch_size, ACTION_DIM),
                dtype=wp.float32,
                device=self._device,
            )
            self._heading_offset = wp.empty(
                self._batch_size, dtype=wp.quat, device=self._device
            )
            self._heading_initialized = wp.empty(
                self._batch_size, dtype=wp.int32, device=self._device
            )
            self._policy_inputs = {
                "policy": wp.empty(
                    (self._batch_size, POLICY_DIM),
                    dtype=wp.float32,
                    device=self._device,
                ),
                "command": wp.empty(
                    (self._batch_size, COMMAND_DIM),
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
        self._state_template = adapter.initial_state(self._batch_size)

    def reset(self, state: object) -> None:
        values = normalize_tracker_state(
            state, self._state_template, family="Mimic-Lite"
        )
        history = np.concatenate(
            (
                values["root_angular_velocity"],
                values["projected_gravity"],
                values["joint_position"],
                values["joint_velocity"],
            ),
            axis=2,
        )
        self._policy.drain()
        self._prepared = None
        self._horizon = 0
        self._execution_steps = 0
        with wp.ScopedStream(self._stream):
            self._history.assign(np.ascontiguousarray(history, dtype=np.float32))
            self._previous_actions.assign(values["previous_actions"])
            self._last_action.assign(values["last_action"])
            self._heading_offset.assign(values["heading_offset"])
            self._heading_initialized.assign(
                values["heading_initialized"].astype(np.int32)
            )

    def capture_state(self) -> dict[str, np.ndarray]:
        wp.synchronize_stream(self._stream)
        history = np.asarray(self._history.numpy()).copy()
        return {
            "root_angular_velocity": history[
                :, :, _ANGULAR_VELOCITY:_GRAVITY
            ],
            "projected_gravity": history[:, :, _GRAVITY:_JOINT_POSITION],
            "joint_position": history[:, :, _JOINT_POSITION:_JOINT_VELOCITY],
            "joint_velocity": history[:, :, _JOINT_VELOCITY:],
            "previous_actions": np.asarray(
                self._previous_actions.numpy()
            ).copy(),
            "last_action": np.asarray(self._last_action.numpy()).copy(),
            "heading_offset": np.asarray(self._heading_offset.numpy()).copy(),
            "heading_initialized": np.asarray(
                self._heading_initialized.numpy(), dtype=bool
            ),
        }

    def prepare(
        self,
        history: MotionGuidance | None,
        guidance: MotionGuidance,
        right_context: MotionGuidance | None,
        *,
        num_steps: int,
    ) -> None:
        horizon = len(guidance.query_offsets)
        if not isinstance(num_steps, (int, np.integer)) or not (
            1 <= int(num_steps) <= horizon
        ):
            raise ValueError("num_steps must be within prepared guidance")
        if guidance.query_offsets != tuple(range(horizon)):
            raise ValueError("Mimic-Lite prepared guidance must be dense and zero-based")
        if history is None or len(history.query_offsets) < _PAST_STEPS:
            raise ValueError(
                f"Mimic-Lite prepared guidance requires {_PAST_STEPS} history steps"
            )
        history_steps = len(history.query_offsets)
        if history.query_offsets != tuple(range(-history_steps, 0)):
            raise ValueError("Mimic-Lite history must use a dense negative query grid")
        self._validate_guidance(history, guidance, "history")
        if right_context is not None:
            context_steps = len(right_context.query_offsets)
            if right_context.query_offsets != tuple(range(context_steps)):
                raise ValueError(
                    "Mimic-Lite right_context must be dense and zero-based"
                )
            self._validate_guidance(right_context, guidance, "right_context")

        future = 0 if right_context is None else min(
            _FUTURE_STEPS, len(right_context.query_offsets)
        )
        parts: list[tuple[MotionGuidance, slice]] = [
            (history, slice(history_steps - _PAST_STEPS, history_steps)),
            (guidance, slice(0, horizon)),
        ]
        if future:
            assert right_context is not None
            parts.append((right_context, slice(0, future)))
        length = _PAST_STEPS + horizon + future
        if _PAST_STEPS + int(num_steps) - 1 + _FUTURE_STEPS >= length:
            raise ValueError(
                "Mimic-Lite prepared guidance lacks four right-context steps"
            )
        joint_order = _order(guidance.joint_names, self._config.policy_joint_names)
        root_column = guidance.body_names.index(self._family.root_body)
        anchor_column = guidance.body_names.index(self._family.anchor_body)

        def joined(field: str, column: int | None = None) -> np.ndarray:
            arrays = []
            for source, section in parts:
                value = getattr(source, field)[:, section]
                if column is not None:
                    value = value[:, :, column]
                arrays.append(value)
            return np.concatenate(arrays, axis=1)

        arrays = {
            "joint_position": joined("joint_position")[:, :, joint_order],
            "root_position": joined("body_position_world", root_column),
            "root_orientation": joined("body_orientation_world", root_column),
            "anchor_position": joined("body_position_world", anchor_column),
            "anchor_orientation": joined("body_orientation_world", anchor_column),
        }

        self._prepared = None
        self._horizon = 0
        self._execution_steps = 0
        buffers = self._guidance_buffers.get(length)
        if buffers is None:
            with wp.ScopedStream(self._stream):
                buffers = _GuidanceBuffers(
                    joint_position=wp.empty(
                        (self._batch_size, length, ACTION_DIM),
                        dtype=wp.float32,
                        device=self._device,
                    ),
                    root_position=wp.empty(
                        (self._batch_size, length),
                        dtype=wp.vec3,
                        device=self._device,
                    ),
                    root_orientation=wp.empty(
                        (self._batch_size, length),
                        dtype=wp.quat,
                        device=self._device,
                    ),
                    anchor_position=wp.empty(
                        (self._batch_size, length),
                        dtype=wp.vec3,
                        device=self._device,
                    ),
                    anchor_orientation=wp.empty(
                        (self._batch_size, length),
                        dtype=wp.quat,
                        device=self._device,
                    ),
                    length=length,
                )
            self._guidance_buffers[length] = buffers
        with wp.ScopedStream(self._stream):
            for name, value in arrays.items():
                getattr(buffers, name).assign(
                    np.ascontiguousarray(value, dtype=np.float32)
                )
        self._prepared = buffers
        self._horizon = horizon
        self._execution_steps = int(num_steps)

    def step_prepared(
        self, state_view: object, step_index: int
    ) -> DevicePDCommand:
        if self._prepared is None:
            raise RuntimeError("prepare() must be called before step_prepared()")
        if not isinstance(step_index, (int, np.integer)) or not (
            0 <= int(step_index) < self._horizon
        ):
            raise ValueError("step_index is outside prepared guidance")
        if _PAST_STEPS + int(step_index) + _FUTURE_STEPS >= self._prepared.length:
            raise ValueError(
                "prepared guidance lacks Mimic-Lite future context for this step"
            )
        validate_bound_state_view(
            state_view, self._state_view, family="Mimic-Lite"
        )
        buffers = self._prepared
        wp.launch(
            _observe,
            dim=(self._batch_size, _HISTORY_DIM),
            inputs=[
                int(step_index),
                _PAST_STEPS,
                state_view.joint_position,
                state_view.joint_velocity,
                state_view.body_orientation_world,
                state_view.body_angular_velocity_world,
                self._state_joint_indices,
                self._root_body_index,
                buffers.root_orientation,
                self._last_action,
                self._history,
                self._previous_actions,
                self._heading_offset,
                self._heading_initialized,
            ],
            device=self._device,
            stream=self._stream,
        )
        wp.launch(
            _pack_policy,
            dim=(self._batch_size, POLICY_DIM),
            inputs=[
                self._history_lags,
                self._history,
                self._previous_actions,
                self._policy_inputs["policy"],
            ],
            device=self._device,
            stream=self._stream,
        )
        wp.launch(
            _pack_command,
            dim=(self._batch_size, COMMAND_DIM),
            inputs=[
                int(step_index),
                _PAST_STEPS,
                self._query_offsets,
                buffers.joint_position,
                buffers.root_position,
                buffers.root_orientation,
                buffers.anchor_position,
                buffers.anchor_orientation,
                state_view.body_orientation_world,
                self._root_body_index,
                self._heading_offset,
                self._policy_inputs["command"],
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

    def _validate_guidance(
        self, value: MotionGuidance, reference: MotionGuidance, label: str
    ) -> None:
        if (
            value.joint_position.shape[0] != self._batch_size
            or value.joint_names != reference.joint_names
            or value.body_names != reference.body_names
        ):
            raise ValueError(
                f"Mimic-Lite {label} must match guidance batch and names"
            )

    def _array(self, value: np.ndarray, dtype: Any) -> Any:
        return wp.array(np.ascontiguousarray(value), dtype=dtype, device=self._device)


__all__ = ["MimicLiteDeviceTracker"]
