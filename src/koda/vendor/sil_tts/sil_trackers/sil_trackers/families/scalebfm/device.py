"""Device-resident ScaleBFM observation, policy, history, and PD command path.

Policy ``action`` stays on device only to form PD targets and the next
proprioceptive history. This family does not capture or persist action /
motion-token traces; the SIL product is executed kinematics.
"""

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
from sil_trackers.families.scalebfm.adapter import ScaleBfmAdapter, _order
from sil_trackers.families.scalebfm.observation import (
    ACTION_DIM,
    HISTORY_LENGTH,
    NUM_FUTURE,
    NUM_SELECTED,
    TRACKING_BODY_NAMES,
    TRACKING_SELECTED_SLOTS,
)

try:  # Keep the serial family importable without the optional device runtime.
    import warp as wp
except ImportError:  # pragma: no cover - exercised only without runtime extras
    wp = None


_TORSO_SLOT = TRACKING_SELECTED_SLOTS[0]
_LEFT_WRIST_SLOT = TRACKING_SELECTED_SLOTS[1]
_RIGHT_WRIST_SLOT = TRACKING_SELECTED_SLOTS[2]


if wp is not None:

    @wp.func
    def _rotate_inverse(q: wp.quat, vector: wp.vec3) -> wp.vec3:
        inv_xyz = wp.vec3(-q[1], -q[2], -q[3])
        tangent = wp.cross(inv_xyz, vector) * 2.0
        return vector + q[0] * tangent + wp.cross(inv_xyz, tangent)


    @wp.func
    def _quat_mul_inverse_left(anchor: wp.quat, target: wp.quat) -> wp.quat:
        w1 = anchor[0]
        x1 = -anchor[1]
        y1 = -anchor[2]
        z1 = -anchor[3]
        w2 = target[0]
        x2 = target[1]
        y2 = target[2]
        z2 = target[3]
        return wp.quat(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )


    @wp.kernel
    def _observe(
        joint_position: wp.array2d(dtype=wp.float32),
        joint_velocity: wp.array2d(dtype=wp.float32),
        body_orientation: wp.array2d(dtype=wp.quat),
        body_angular_velocity: wp.array2d(dtype=wp.vec3),
        state_joint_indices: wp.array(dtype=wp.int32),
        root_body_index: int,
        last_action: wp.array2d(dtype=wp.float32),
        history_initialized: wp.array(dtype=wp.int32),
        root_quat_buffer: wp.array3d(dtype=wp.float32),
        base_ang_vel_buffer: wp.array3d(dtype=wp.float32),
        dof_pos_buffer: wp.array3d(dtype=wp.float32),
        dof_vel_buffer: wp.array3d(dtype=wp.float32),
        last_action_buffer: wp.array3d(dtype=wp.float32),
    ):
        lane = wp.tid()
        root_q = body_orientation[lane, root_body_index]
        ang = _rotate_inverse(root_q, body_angular_velocity[lane, root_body_index])
        if history_initialized[lane] == 0:
            for slot in range(HISTORY_LENGTH):
                root_quat_buffer[lane, slot, 0] = root_q[0]
                root_quat_buffer[lane, slot, 1] = root_q[1]
                root_quat_buffer[lane, slot, 2] = root_q[2]
                root_quat_buffer[lane, slot, 3] = root_q[3]
                base_ang_vel_buffer[lane, slot, 0] = ang[0]
                base_ang_vel_buffer[lane, slot, 1] = ang[1]
                base_ang_vel_buffer[lane, slot, 2] = ang[2]
                for joint in range(ACTION_DIM):
                    source = state_joint_indices[joint]
                    dof_pos_buffer[lane, slot, joint] = joint_position[lane, source]
                    dof_vel_buffer[lane, slot, joint] = joint_velocity[lane, source]
                    last_action_buffer[lane, slot, joint] = last_action[lane, joint]
        else:
            for slot in range(HISTORY_LENGTH - 1):
                nxt = slot + 1
                for component in range(4):
                    root_quat_buffer[lane, slot, component] = root_quat_buffer[
                        lane, nxt, component
                    ]
                for component in range(3):
                    base_ang_vel_buffer[lane, slot, component] = base_ang_vel_buffer[
                        lane, nxt, component
                    ]
                for joint in range(ACTION_DIM):
                    dof_pos_buffer[lane, slot, joint] = dof_pos_buffer[
                        lane, nxt, joint
                    ]
                    dof_vel_buffer[lane, slot, joint] = dof_vel_buffer[
                        lane, nxt, joint
                    ]
                    last_action_buffer[lane, slot, joint] = last_action_buffer[
                        lane, nxt, joint
                    ]
            last = HISTORY_LENGTH - 1
            root_quat_buffer[lane, last, 0] = root_q[0]
            root_quat_buffer[lane, last, 1] = root_q[1]
            root_quat_buffer[lane, last, 2] = root_q[2]
            root_quat_buffer[lane, last, 3] = root_q[3]
            base_ang_vel_buffer[lane, last, 0] = ang[0]
            base_ang_vel_buffer[lane, last, 1] = ang[1]
            base_ang_vel_buffer[lane, last, 2] = ang[2]
            for joint in range(ACTION_DIM):
                source = state_joint_indices[joint]
                dof_pos_buffer[lane, last, joint] = joint_position[lane, source]
                dof_vel_buffer[lane, last, joint] = joint_velocity[lane, source]
                last_action_buffer[lane, last, joint] = last_action[lane, joint]
        history_initialized[lane] = 1


    @wp.kernel
    def _pack_targets(
        step_index: int,
        query_offsets: wp.array(dtype=wp.int32),
        body_position: wp.array2d(dtype=wp.vec3),
        body_orientation: wp.array2d(dtype=wp.quat),
        root_body_index: int,
        guide_position: wp.array3d(dtype=wp.vec3),
        guide_orientation: wp.array3d(dtype=wp.quat),
        target_pos: wp.array4d(dtype=wp.float32),
        target_rot: wp.array4d(dtype=wp.float32),
    ):
        lane, query, body = wp.tid()
        target_pos[lane, query, body, 0] = 0.0
        target_pos[lane, query, body, 1] = 0.0
        target_pos[lane, query, body, 2] = 0.0
        target_rot[lane, query, body, 0] = 1.0
        target_rot[lane, query, body, 1] = 0.0
        target_rot[lane, query, body, 2] = 0.0
        target_rot[lane, query, body, 3] = 0.0
        link = int(-1)
        if body == _TORSO_SLOT:
            link = 0
        elif body == _LEFT_WRIST_SLOT:
            link = 1
        elif body == _RIGHT_WRIST_SLOT:
            link = 2
        if link >= 0:
            root_pos = body_position[lane, root_body_index]
            root_q = body_orientation[lane, root_body_index]
            reference_step = step_index + query_offsets[query]
            local_pos = _rotate_inverse(
                root_q,
                guide_position[lane, reference_step, link] - root_pos,
            )
            local_rot = _quat_mul_inverse_left(
                root_q, guide_orientation[lane, reference_step, link]
            )
            target_pos[lane, query, body, 0] = local_pos[0]
            target_pos[lane, query, body, 1] = local_pos[1]
            target_pos[lane, query, body, 2] = local_pos[2]
            target_rot[lane, query, body, 0] = local_rot[0]
            target_rot[lane, query, body, 1] = local_rot[1]
            target_rot[lane, query, body, 2] = local_rot[2]
            target_rot[lane, query, body, 3] = local_rot[3]


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
    link_position: Any
    link_orientation: Any


class ScaleBfmDeviceTracker:
    """Prepared ScaleBFM tracker bound to one CUDA simulation batch."""

    def __init__(
        self,
        adapter: ScaleBfmAdapter,
        policy_path: str | Path,
        state_view: object,
        command_view: object,
    ) -> None:
        if wp is None:
            raise RuntimeError("ScaleBFM device execution requires warp-lang")
        self._config = adapter._config
        self._family = adapter._family
        self._state_view = state_view
        self._command = command_view
        self._batch_size, self._device = validate_device_layout(
            state_view,
            command_view,
            family="ScaleBFM",
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
            _order(state_joints, self._config.policy_joint_names), dtype=np.int32
        )
        self._root_body_index = state_bodies.index(self._family.root_body)
        command_to_policy = np.asarray(
            _order(self._config.policy_joint_names, self._config.command_joint_names),
            dtype=np.int32,
        )
        offsets = self._config.guidance_query_offsets
        self._max_query_offset = max(offsets)

        with wp.ScopedStream(self._stream):
            self._state_joint_indices = self._array(state_joint_indices, wp.int32)
            self._command_to_policy = self._array(command_to_policy, wp.int32)
            self._query_offsets = self._array(
                np.asarray(offsets, dtype=np.int32), wp.int32
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
            self._last_action = wp.empty(
                (self._batch_size, ACTION_DIM), dtype=wp.float32, device=self._device
            )
            self._history_initialized = wp.empty(
                self._batch_size, dtype=wp.int32, device=self._device
            )
            self._policy_inputs = {
                "root_quat_buffer": wp.empty(
                    (self._batch_size, HISTORY_LENGTH, 4),
                    dtype=wp.float32,
                    device=self._device,
                ),
                "base_ang_vel_buffer": wp.empty(
                    (self._batch_size, HISTORY_LENGTH, 3),
                    dtype=wp.float32,
                    device=self._device,
                ),
                "dof_pos_buffer": wp.empty(
                    (self._batch_size, HISTORY_LENGTH, ACTION_DIM),
                    dtype=wp.float32,
                    device=self._device,
                ),
                "dof_vel_buffer": wp.empty(
                    (self._batch_size, HISTORY_LENGTH, ACTION_DIM),
                    dtype=wp.float32,
                    device=self._device,
                ),
                "last_action_buffer": wp.empty(
                    (self._batch_size, HISTORY_LENGTH, ACTION_DIM),
                    dtype=wp.float32,
                    device=self._device,
                ),
                "target_body_pos_future_to_robot_base": wp.empty(
                    (self._batch_size, NUM_FUTURE, NUM_SELECTED, 3),
                    dtype=wp.float32,
                    device=self._device,
                ),
                "target_body_rot_future_to_robot_base": wp.empty(
                    (self._batch_size, NUM_FUTURE, NUM_SELECTED, 4),
                    dtype=wp.float32,
                    device=self._device,
                ),
            }
            self._policy_outputs = {
                "action": wp.empty(
                    (self._batch_size, ACTION_DIM),
                    dtype=wp.float32,
                    device=self._device,
                ),
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
        values = normalize_tracker_state(state, self._state_template, family="ScaleBFM")
        self._policy.drain()
        self._prepared = None
        self._horizon = 0
        self._execution_steps = 0
        self._total_steps = 0
        with wp.ScopedStream(self._stream):
            self._policy_inputs["root_quat_buffer"].assign(
                np.ascontiguousarray(values["root_quat"], dtype=np.float32)
            )
            self._policy_inputs["base_ang_vel_buffer"].assign(
                np.ascontiguousarray(values["base_ang_vel"], dtype=np.float32)
            )
            self._policy_inputs["dof_pos_buffer"].assign(
                np.ascontiguousarray(values["joint_position"], dtype=np.float32)
            )
            self._policy_inputs["dof_vel_buffer"].assign(
                np.ascontiguousarray(values["joint_velocity"], dtype=np.float32)
            )
            self._policy_inputs["last_action_buffer"].assign(
                np.ascontiguousarray(values["last_action_history"], dtype=np.float32)
            )
            self._last_action.assign(values["last_action"])
            self._history_initialized.assign(
                values["history_initialized"].astype(np.int32)
            )

    def capture_state(self) -> dict[str, np.ndarray]:
        wp.synchronize_stream(self._stream)
        return {
            "root_quat": np.asarray(
                self._policy_inputs["root_quat_buffer"].numpy()
            ).copy(),
            "base_ang_vel": np.asarray(
                self._policy_inputs["base_ang_vel_buffer"].numpy()
            ).copy(),
            "joint_position": np.asarray(
                self._policy_inputs["dof_pos_buffer"].numpy()
            ).copy(),
            "joint_velocity": np.asarray(
                self._policy_inputs["dof_vel_buffer"].numpy()
            ).copy(),
            "last_action_history": np.asarray(
                self._policy_inputs["last_action_buffer"].numpy()
            ).copy(),
            "last_action": np.asarray(self._last_action.numpy()).copy(),
            "history_initialized": np.asarray(
                self._history_initialized.numpy(), dtype=bool
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
        context_steps = (
            0 if right_context is None else len(right_context.query_offsets)
        )
        total_steps = horizon + context_steps
        if total_steps <= self._max_query_offset:
            raise ValueError(
                f"ScaleBFM prepared guidance must cover offset {self._max_query_offset}"
            )

        link_order = _order(guidance.body_names, TRACKING_BODY_NAMES)

        def combined(field: str) -> np.ndarray:
            candidate = getattr(guidance, field)
            if right_context is None:
                return candidate
            return np.concatenate((candidate, getattr(right_context, field)), axis=1)

        link_position = combined("body_position_world")[:, :, link_order]
        link_orientation = combined("body_orientation_world")[:, :, link_order]

        self._prepared = None
        self._horizon = 0
        self._execution_steps = 0
        self._total_steps = 0
        buffers = self._guidance_buffers.get(total_steps)
        if buffers is None:
            with wp.ScopedStream(self._stream):
                buffers = _GuidanceBuffers(
                    link_position=wp.empty(
                        (self._batch_size, total_steps, len(TRACKING_BODY_NAMES)),
                        dtype=wp.vec3,
                        device=self._device,
                    ),
                    link_orientation=wp.empty(
                        (self._batch_size, total_steps, len(TRACKING_BODY_NAMES)),
                        dtype=wp.quat,
                        device=self._device,
                    ),
                )
            self._guidance_buffers[total_steps] = buffers
        with wp.ScopedStream(self._stream):
            buffers.link_position.assign(
                np.ascontiguousarray(link_position, dtype=np.float32)
            )
            buffers.link_orientation.assign(
                np.ascontiguousarray(link_orientation, dtype=np.float32)
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
        if int(step_index) + self._max_query_offset >= self._total_steps:
            raise ValueError("prepared guidance does not cover ScaleBFM future queries")
        validate_bound_state_view(state_view, self._state_view, family="ScaleBFM")
        buffers = self._prepared
        wp.launch(
            _observe,
            dim=self._batch_size,
            inputs=[
                state_view.joint_position,
                state_view.joint_velocity,
                state_view.body_orientation_world,
                state_view.body_angular_velocity_world,
                self._state_joint_indices,
                self._root_body_index,
                self._last_action,
                self._history_initialized,
                self._policy_inputs["root_quat_buffer"],
                self._policy_inputs["base_ang_vel_buffer"],
                self._policy_inputs["dof_pos_buffer"],
                self._policy_inputs["dof_vel_buffer"],
                self._policy_inputs["last_action_buffer"],
            ],
            device=self._device,
            stream=self._stream,
        )
        wp.launch(
            _pack_targets,
            dim=(self._batch_size, NUM_FUTURE, NUM_SELECTED),
            inputs=[
                int(step_index),
                self._query_offsets,
                state_view.body_position_world,
                state_view.body_orientation_world,
                self._root_body_index,
                buffers.link_position,
                buffers.link_orientation,
                self._policy_inputs["target_body_pos_future_to_robot_base"],
                self._policy_inputs["target_body_rot_future_to_robot_base"],
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


__all__ = ["ScaleBfmDeviceTracker"]
