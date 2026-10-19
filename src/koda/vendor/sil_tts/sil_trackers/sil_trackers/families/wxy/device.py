"""Device-resident WXY observation, policy, history, and command execution."""

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
from sil_trackers.families.wxy.adapter import WxyAdapter, _column_order
from sil_trackers.families.wxy.observation import (
    ACTION_DIM,
    HISTORY_LENGTH,
    MOTION_COMMAND_DIM,
    PROPRIOCEPTION_DIM,
)

try:  # Keep the serial family importable without the optional device runtime.
    import warp as wp
except ImportError:  # pragma: no cover - exercised only without runtime extras
    wp = None


_REF_LIMB = 0
_ROBOT_LIMB = 36
_GRAVITY = 72
_ANGULAR_VELOCITY = 75
_JOINT_POSITION = 78
_JOINT_VELOCITY = 107
_ACTION = 136
_HISTORY_DIM = 165


if wp is not None:

    @wp.func
    def _safe_quat(q: wp.quat) -> wp.quat:
        norm = wp.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        if norm < 1.0e-8:
            return wp.quat(1.0, 0.0, 0.0, 0.0)
        return wp.quat(q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm)


    @wp.func
    def _rotate_inverse(q_raw: wp.quat, v: wp.vec3) -> wp.vec3:
        q = _safe_quat(q_raw)
        w = q[0]
        qv = wp.vec3(q[1], q[2], q[3])
        return (
            v * (2.0 * w * w - 1.0)
            - wp.cross(qv, v) * (2.0 * w)
            + qv * (2.0 * wp.dot(qv, v))
        )


    @wp.func
    def _relative_quat(anchor_raw: wp.quat, limb: wp.quat) -> wp.quat:
        anchor = _safe_quat(anchor_raw)
        w1 = anchor[0]
        x1 = -anchor[1]
        y1 = -anchor[2]
        z1 = -anchor[3]
        w2 = limb[0]
        x2 = limb[1]
        y2 = limb[2]
        z2 = limb[3]
        return wp.quat(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )


    @wp.func
    def _rotation_6d_value(q: wp.quat, component: int) -> float:
        w = q[0]
        x = q[1]
        y = q[2]
        z = q[3]
        two_s = 2.0 / (w * w + x * x + y * y + z * z)
        if component == 0:
            return 1.0 - two_s * (y * y + z * z)
        if component == 1:
            return two_s * (x * y - z * w)
        if component == 2:
            return two_s * (x * y + z * w)
        if component == 3:
            return 1.0 - two_s * (x * x + z * z)
        if component == 4:
            return two_s * (x * z - y * w)
        return two_s * (y * z + x * w)


    @wp.func
    def _limb_value(
        positions: wp.array2d(dtype=wp.vec3),
        orientations: wp.array2d(dtype=wp.quat),
        lane: int,
        limb: int,
        anchor: int,
        component: int,
    ) -> float:
        anchor_q = orientations[lane, anchor]
        if component < 3:
            relative = _rotate_inverse(
                anchor_q, positions[lane, limb] - positions[lane, anchor]
            )
            return relative[component]
        relative_q = _relative_quat(anchor_q, orientations[lane, limb])
        return _rotation_6d_value(relative_q, component - 3)


    @wp.func
    def _guided_limb_value(
        positions: wp.array3d(dtype=wp.vec3),
        orientations: wp.array3d(dtype=wp.quat),
        lane: int,
        step_index: int,
        limb: int,
        anchor: int,
        component: int,
    ) -> float:
        anchor_q = orientations[lane, step_index, anchor]
        if component < 3:
            relative = _rotate_inverse(
                anchor_q,
                positions[lane, step_index, limb]
                - positions[lane, step_index, anchor],
            )
            return relative[component]
        relative_q = _relative_quat(
            anchor_q, orientations[lane, step_index, limb]
        )
        return _rotation_6d_value(relative_q, component - 3)


    @wp.kernel
    def _observe(
        step_index: int,
        joint_position: wp.array2d(dtype=wp.float32),
        joint_velocity: wp.array2d(dtype=wp.float32),
        body_position: wp.array2d(dtype=wp.vec3),
        body_orientation: wp.array2d(dtype=wp.quat),
        body_angular_velocity: wp.array2d(dtype=wp.vec3),
        state_joint_indices: wp.array(dtype=wp.int32),
        state_body_indices: wp.array(dtype=wp.int32),
        limb_indices: wp.array(dtype=wp.int32),
        anchor_index: int,
        guidance_body_position: wp.array3d(dtype=wp.vec3),
        guidance_body_orientation: wp.array3d(dtype=wp.quat),
        default_joint_position: wp.array(dtype=wp.float32),
        last_action: wp.array2d(dtype=wp.float32),
        initialized: wp.array(dtype=wp.int32),
        history: wp.array3d(dtype=wp.float32),
    ):
        lane, feature = wp.tid()
        value = float(0.0)
        if feature < _ROBOT_LIMB:
            local = feature - _REF_LIMB
            limb = local // 9
            component = local - limb * 9
            value = _guided_limb_value(
                guidance_body_position,
                guidance_body_orientation,
                lane,
                step_index,
                limb_indices[limb],
                anchor_index,
                component,
            )
        elif feature < _GRAVITY:
            local = feature - _ROBOT_LIMB
            limb = local // 9
            component = local - limb * 9
            value = _limb_value(
                body_position,
                body_orientation,
                lane,
                state_body_indices[limb_indices[limb]],
                state_body_indices[anchor_index],
                component,
            )
        elif feature < _ANGULAR_VELOCITY:
            component = feature - _GRAVITY
            root = state_body_indices[anchor_index]
            gravity = _rotate_inverse(
                body_orientation[lane, root], wp.vec3(0.0, 0.0, -1.0)
            )
            value = gravity[component]
        elif feature < _JOINT_POSITION:
            component = feature - _ANGULAR_VELOCITY
            root = state_body_indices[anchor_index]
            angular_velocity = _rotate_inverse(
                body_orientation[lane, root],
                body_angular_velocity[lane, root],
            )
            value = angular_velocity[component]
        elif feature < _JOINT_VELOCITY:
            joint = feature - _JOINT_POSITION
            value = (
                joint_position[lane, state_joint_indices[joint]]
                - default_joint_position[joint]
            )
        elif feature < _ACTION:
            joint = feature - _JOINT_VELOCITY
            value = joint_velocity[lane, state_joint_indices[joint]]
        else:
            value = last_action[lane, feature - _ACTION]

        if initialized[lane] == 0:
            for slot in range(HISTORY_LENGTH):
                history[lane, slot, feature] = value
        else:
            for slot in range(HISTORY_LENGTH - 1):
                history[lane, slot, feature] = history[lane, slot + 1, feature]
            history[lane, HISTORY_LENGTH - 1, feature] = value


    @wp.kernel
    def _pack_inputs(
        step_index: int,
        guidance_joint_position: wp.array3d(dtype=wp.float32),
        guidance_joint_velocity: wp.array3d(dtype=wp.float32),
        guidance_body_angular_velocity: wp.array3d(dtype=wp.vec3),
        tracking_anchor_index: int,
        history: wp.array3d(dtype=wp.float32),
        initialized: wp.array(dtype=wp.int32),
        motion_command: wp.array2d(dtype=wp.float32),
        proprioception: wp.array2d(dtype=wp.float32),
    ):
        lane, feature = wp.tid()
        if feature < MOTION_COMMAND_DIM:
            if feature < ACTION_DIM:
                motion_command[lane, feature] = guidance_joint_position[
                    lane, step_index, feature
                ]
            elif feature < 2 * ACTION_DIM:
                motion_command[lane, feature] = guidance_joint_velocity[
                    lane, step_index, feature - ACTION_DIM
                ]
            elif feature < 2 * ACTION_DIM + HISTORY_LENGTH * 36:
                local = feature - 2 * ACTION_DIM
                slot = local // 36
                item = local - slot * 36
                motion_command[lane, feature] = history[
                    lane, slot, _REF_LIMB + item
                ]
            else:
                component = feature - (2 * ACTION_DIM + HISTORY_LENGTH * 36)
                motion_command[lane, feature] = guidance_body_angular_velocity[
                    lane, step_index, tracking_anchor_index
                ][component]

        if feature < PROPRIOCEPTION_DIM:
            history_offset = 0
            item_dim = 36
            local = feature
            if feature < 180:
                history_offset = _ROBOT_LIMB
            elif feature < 195:
                history_offset = _GRAVITY
                item_dim = 3
                local = feature - 180
            elif feature < 210:
                history_offset = _ANGULAR_VELOCITY
                item_dim = 3
                local = feature - 195
            elif feature < 355:
                history_offset = _JOINT_POSITION
                item_dim = ACTION_DIM
                local = feature - 210
            elif feature < 500:
                history_offset = _JOINT_VELOCITY
                item_dim = ACTION_DIM
                local = feature - 355
            else:
                history_offset = _ACTION
                item_dim = ACTION_DIM
                local = feature - 500
            slot = local // item_dim
            item = local - slot * item_dim
            proprioception[lane, feature] = history[
                lane, slot, history_offset + item
            ]

        if feature == 0:
            initialized[lane] = 1


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
    joint_velocity: Any
    body_position: Any
    body_orientation: Any
    body_angular_velocity: Any


class WxyDeviceTracker:
    """Prepared WXY tracker whose per-Control-Step path stays on one CUDA stream."""

    def __init__(
        self,
        adapter: WxyAdapter,
        policy_path: str | Path,
        state_view: object,
        command_view: object,
    ) -> None:
        if wp is None:
            raise RuntimeError("WXY device execution requires warp-lang")
        self._adapter = adapter
        self._config = adapter._config
        self._family = adapter._family
        self._state_view = state_view
        self._command = command_view
        self._batch_size, self._device = validate_device_layout(
            state_view,
            command_view,
            family="WXY",
            command_joint_names=self._config.command_joint_names,
            float_dtype=wp.float32,
            body_dtypes={
                "body_position_world": wp.vec3,
                "body_orientation_world": wp.quat,
                "body_angular_velocity_world": wp.vec3,
            },
        )
        self._stream = wp.get_stream(self._device)

        state_joints = tuple(getattr(state_view, "joint_names"))
        state_bodies = tuple(getattr(state_view, "body_names"))
        state_joint_indices = np.asarray(
            _column_order(state_joints, self._config.policy_joint_names), dtype=np.int32
        )
        state_body_indices = np.asarray(
            _column_order(state_bodies, self._config.guidance_body_names),
            dtype=np.int32,
        )
        command_to_policy = np.asarray(
            _column_order(
                self._config.policy_joint_names, self._config.command_joint_names
            ),
            dtype=np.int32,
        )

        with wp.ScopedStream(self._stream):
            self._state_joint_indices = self._array(state_joint_indices, wp.int32)
            self._state_body_indices = self._array(state_body_indices, wp.int32)
            self._limb_indices = self._array(
                adapter._limb_indices.astype(np.int32), wp.int32
            )
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
            self._history = wp.empty(
                (self._batch_size, HISTORY_LENGTH, _HISTORY_DIM),
                dtype=wp.float32,
                device=self._device,
            )
            self._last_action = wp.empty(
                (self._batch_size, ACTION_DIM), dtype=wp.float32, device=self._device
            )
            self._initialized = wp.empty(
                self._batch_size, dtype=wp.int32, device=self._device
            )
            self._policy_inputs = {
                "motion_command": wp.empty(
                    (self._batch_size, MOTION_COMMAND_DIM),
                    dtype=wp.float32,
                    device=self._device,
                ),
                "proprioception": wp.empty(
                    (self._batch_size, PROPRIOCEPTION_DIM),
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
        values = normalize_tracker_state(state, self._state_template, family="WXY")
        history = np.concatenate(
            [
                values["ref_limb_ee_pose_b"],
                values["robot_limb_ee_pose_b"],
                values["projected_gravity"],
                values["base_ang_vel"],
                values["joint_pos"],
                values["joint_vel"],
                values["actions"],
            ],
            axis=2,
        )
        self._policy.drain()
        self._prepared = None
        self._horizon = 0
        self._execution_steps = 0
        with wp.ScopedStream(self._stream):
            self._history.assign(np.ascontiguousarray(history, dtype=np.float32))
            self._last_action.assign(values["last_action"])
            self._initialized.assign(values["history_initialized"].astype(np.int32))

    def capture_state(self) -> dict[str, np.ndarray]:
        wp.synchronize_stream(self._stream)
        history = np.asarray(self._history.numpy()).copy()
        return {
            "ref_limb_ee_pose_b": history[:, :, _REF_LIMB:_ROBOT_LIMB],
            "robot_limb_ee_pose_b": history[:, :, _ROBOT_LIMB:_GRAVITY],
            "projected_gravity": history[:, :, _GRAVITY:_ANGULAR_VELOCITY],
            "base_ang_vel": history[:, :, _ANGULAR_VELOCITY:_JOINT_POSITION],
            "joint_pos": history[:, :, _JOINT_POSITION:_JOINT_VELOCITY],
            "joint_vel": history[:, :, _JOINT_VELOCITY:_ACTION],
            "actions": history[:, :, _ACTION:],
            "last_action": np.asarray(self._last_action.numpy()).copy(),
            "history_initialized": np.asarray(self._initialized.numpy(), dtype=bool),
        }

    def prepare(
        self,
        history: MotionGuidance | None,
        guidance: MotionGuidance,
        right_context: MotionGuidance | None,
        *,
        num_steps: int,
    ) -> None:
        del history, right_context
        horizon = len(guidance.query_offsets)
        if not isinstance(num_steps, (int, np.integer)) or not (
            1 <= int(num_steps) <= horizon
        ):
            raise ValueError("num_steps must be within prepared guidance")
        if guidance.query_offsets != tuple(range(horizon)):
            raise ValueError("WXY prepared guidance must be dense and zero-based")
        if guidance.joint_position.shape[0] != self._batch_size:
            raise ValueError("guidance batch size differs from the device tracker")
        joint_order = _column_order(
            guidance.joint_names, self._config.policy_joint_names
        )
        body_order = _column_order(
            guidance.body_names, self._config.guidance_body_names
        )
        # Fail closed if an upload below fails; a partially overwritten cache
        # must never remain executable as the previous prepared segment.
        self._prepared = None
        self._horizon = 0
        self._execution_steps = 0
        buffers = self._guidance_buffers.get(horizon)
        if buffers is None:
            with wp.ScopedStream(self._stream):
                buffers = _GuidanceBuffers(
                    joint_position=wp.empty(
                        (self._batch_size, horizon, ACTION_DIM),
                        dtype=wp.float32,
                        device=self._device,
                    ),
                    joint_velocity=wp.empty(
                        (self._batch_size, horizon, ACTION_DIM),
                        dtype=wp.float32,
                        device=self._device,
                    ),
                    body_position=wp.empty(
                        (self._batch_size, horizon, len(body_order)),
                        dtype=wp.vec3,
                        device=self._device,
                    ),
                    body_orientation=wp.empty(
                        (self._batch_size, horizon, len(body_order)),
                        dtype=wp.quat,
                        device=self._device,
                    ),
                    body_angular_velocity=wp.empty(
                        (self._batch_size, horizon, len(body_order)),
                        dtype=wp.vec3,
                        device=self._device,
                    ),
                )
            self._guidance_buffers[horizon] = buffers
        with wp.ScopedStream(self._stream):
            buffers.joint_position.assign(
                np.ascontiguousarray(
                    guidance.joint_position[:, :, joint_order], dtype=np.float32
                )
            )
            buffers.joint_velocity.assign(
                np.ascontiguousarray(
                    guidance.joint_velocity[:, :, joint_order], dtype=np.float32
                )
            )
            buffers.body_position.assign(
                np.ascontiguousarray(
                    guidance.body_position_world[:, :, body_order], dtype=np.float32
                )
            )
            buffers.body_orientation.assign(
                np.ascontiguousarray(
                    guidance.body_orientation_world[:, :, body_order], dtype=np.float32
                )
            )
            buffers.body_angular_velocity.assign(
                np.ascontiguousarray(
                    guidance.body_angular_velocity_world[:, :, body_order],
                    dtype=np.float32,
                )
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
        validate_bound_state_view(state_view, self._state_view, family="WXY")
        buffers = self._prepared
        wp.launch(
            _observe,
            dim=(self._batch_size, _HISTORY_DIM),
            inputs=[
                int(step_index),
                state_view.joint_position,
                state_view.joint_velocity,
                state_view.body_position_world,
                state_view.body_orientation_world,
                state_view.body_angular_velocity_world,
                self._state_joint_indices,
                self._state_body_indices,
                self._limb_indices,
                self._adapter._anchor_index,
                buffers.body_position,
                buffers.body_orientation,
                self._default_joint_position,
                self._last_action,
                self._initialized,
                self._history,
            ],
            device=self._device,
            stream=self._stream,
        )
        wp.launch(
            _pack_inputs,
            dim=(self._batch_size, PROPRIOCEPTION_DIM),
            inputs=[
                int(step_index),
                buffers.joint_position,
                buffers.joint_velocity,
                buffers.body_angular_velocity,
                self._adapter._tracking_anchor_index,
                self._history,
                self._initialized,
                self._policy_inputs["motion_command"],
                self._policy_inputs["proprioception"],
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
        return wp.array(
            np.ascontiguousarray(value), dtype=dtype, device=self._device
        )

__all__ = ["WxyDeviceTracker"]
