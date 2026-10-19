"""WXY-WBC tracker adapter over RobotState / MotionGuidance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from sil_trackers.contracts import MotionGuidance, NamedTensors, PDCommand, RobotState
from sil_trackers.families.wxy.command import action_to_pd
from sil_trackers.families.wxy.observation import (
    ACTION_DIM,
    HISTORY_LENGTH,
    LIMB_POSE_DIM,
    MOTION_COMMAND_DIM,
    PROPRIOCEPTION_DIM,
    append_history,
    base_ang_vel_body,
    limb_pose_in_anchor_frame,
    pack_motion_command,
    pack_proprioception,
    projected_gravity,
)
from sil_trackers.registry import TrackerInstanceConfig, available, register

if TYPE_CHECKING:
    from sil_trackers.batched_tracker import (
        PreparedDeviceTracker,
        RobotStateView,
    )
    from sil_trackers.contracts import DevicePDCommand

FAMILY = "wxy"


@dataclass(frozen=True)
class _WxyConfig:
    history_length: int
    anchor_body: str
    tracking_anchor_body: str
    limb_body_names: tuple[str, ...]
    default_joint_position: np.ndarray
    action_scale: np.ndarray
    proportional_gain: np.ndarray
    derivative_gain: np.ndarray

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> _WxyConfig:
        required = {
            "history_length",
            "anchor_body",
            "tracking_anchor_body",
            "limb_body_names",
            "default_joint_position",
            "action_scale",
            "proportional_gain",
            "derivative_gain",
        }
        if set(raw) != required:
            raise ValueError(
                f"wxy family_config keys must be {sorted(required)}, got {sorted(raw)}"
            )
        history_length = raw["history_length"]
        if not isinstance(history_length, int) or history_length != HISTORY_LENGTH:
            raise ValueError(f"wxy history_length must be {HISTORY_LENGTH}")

        def body_name(field: str) -> str:
            value = raw[field]
            if not isinstance(value, str) or not value:
                raise ValueError(f"wxy {field} must be a non-empty string")
            return value

        raw_limbs = raw["limb_body_names"]
        if not isinstance(raw_limbs, (list, tuple)):
            raise ValueError("wxy limb_body_names must be a sequence")
        limbs = tuple(raw_limbs)
        if (
            len(limbs) != 4
            or len(set(limbs)) != 4
            or any(not isinstance(name, str) or not name for name in limbs)
        ):
            raise ValueError("wxy limb_body_names must contain four non-empty names")

        def vector(field: str) -> np.ndarray:
            value = np.asarray(raw[field], dtype=np.float32)
            if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
                raise ValueError(
                    f"wxy {field} must contain {ACTION_DIM} finite numbers"
                )
            return value.copy()

        return cls(
            history_length=history_length,
            anchor_body=body_name("anchor_body"),
            tracking_anchor_body=body_name("tracking_anchor_body"),
            limb_body_names=limbs,
            default_joint_position=vector("default_joint_position"),
            action_scale=vector("action_scale"),
            proportional_gain=vector("proportional_gain"),
            derivative_gain=vector("derivative_gain"),
        )


def _column_order(names: tuple[str, ...], target: tuple[str, ...]) -> list[int]:
    try:
        return [names.index(name) for name in target]
    except ValueError as exc:
        raise ValueError(f"missing names for {target}: {exc}") from exc


class WxyAdapter:
    """Serial/batched observation + PD mapping; no MuJoCo, no ORT."""

    def __init__(self, config: TrackerInstanceConfig) -> None:
        if config.family != FAMILY:
            raise ValueError(f"WxyAdapter requires family {FAMILY!r}, got {config.family!r}")
        input_shapes = {spec.name: spec.non_batch_shape for spec in config.policy_inputs}
        output_shapes = {spec.name: spec.non_batch_shape for spec in config.policy_outputs}
        expected_inputs = {
            "motion_command": (MOTION_COMMAND_DIM,),
            "proprioception": (PROPRIOCEPTION_DIM,),
        }
        expected_outputs = {"action": (ACTION_DIM,)}
        if input_shapes != expected_inputs:
            raise ValueError(f"wxy policy_inputs must be {expected_inputs}, got {input_shapes}")
        if output_shapes != expected_outputs:
            raise ValueError(
                f"wxy policy_outputs must be {expected_outputs}, got {output_shapes}"
            )
        if len(config.policy_joint_names) != ACTION_DIM:
            raise ValueError(f"wxy requires {ACTION_DIM} policy joints")
        if config.guidance_query_offsets != (0,):
            raise ValueError("wxy guidance_query_offsets must be (0,)")
        family = _WxyConfig.from_mapping(config.family_config)
        bodies = config.guidance_body_names
        for required in (
            family.anchor_body,
            family.tracking_anchor_body,
            *family.limb_body_names,
        ):
            if required not in bodies:
                raise ValueError(f"wxy guidance bodies must include {required!r}")
        self._config = config
        self._family = family
        self._limb_indices = np.asarray(
            [bodies.index(name) for name in family.limb_body_names], dtype=np.int64
        )
        self._anchor_index = bodies.index(family.anchor_body)
        self._tracking_anchor_index = bodies.index(family.tracking_anchor_body)
        self._input_dtypes = {
            spec.name: np.dtype(spec.dtype) for spec in config.policy_inputs
        }

    def initial_state(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return {
            "ref_limb_ee_pose_b": np.zeros(
                (batch_size, self._family.history_length, LIMB_POSE_DIM),
                dtype=np.float32,
            ),
            "robot_limb_ee_pose_b": np.zeros(
                (batch_size, self._family.history_length, LIMB_POSE_DIM),
                dtype=np.float32,
            ),
            "projected_gravity": np.zeros(
                (batch_size, self._family.history_length, 3), dtype=np.float32
            ),
            "base_ang_vel": np.zeros(
                (batch_size, self._family.history_length, 3), dtype=np.float32
            ),
            "joint_pos": np.zeros(
                (batch_size, self._family.history_length, ACTION_DIM),
                dtype=np.float32,
            ),
            "joint_vel": np.zeros(
                (batch_size, self._family.history_length, ACTION_DIM),
                dtype=np.float32,
            ),
            "actions": np.zeros(
                (batch_size, self._family.history_length, ACTION_DIM),
                dtype=np.float32,
            ),
            "last_action": np.zeros((batch_size, ACTION_DIM), dtype=np.float32),
            "history_initialized": np.zeros(batch_size, dtype=bool),
        }

    def clone_state(self, state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in state.items()}

    def expand_state(
        self, state: dict[str, np.ndarray], batch_size: int
    ) -> dict[str, np.ndarray]:
        if batch_size < 1 or any(value.shape[0] != 1 for value in state.values()):
            raise ValueError("wxy expand_state requires one lane and positive batch_size")
        return {
            name: np.repeat(value, batch_size, axis=0)
            for name, value in state.items()
        }

    def build_policy_inputs(
        self,
        state: RobotState,
        guidance: MotionGuidance,
        tracker_state: dict[str, np.ndarray],
    ) -> tuple[NamedTensors, dict[str, np.ndarray]]:
        if guidance.query_offsets != self._config.guidance_query_offsets:
            raise ValueError(
                "wxy guidance offsets differ from the tracker instance config"
            )
        joints = _column_order(state.joint_names, self._config.policy_joint_names)
        guide_joints = _column_order(
            guidance.joint_names, self._config.policy_joint_names
        )
        bodies = _column_order(guidance.body_names, self._config.guidance_body_names)
        state_bodies = _column_order(state.body_names, self._config.guidance_body_names)
        batch = state.joint_position.shape[0]
        if guidance.joint_position.shape[0] != batch:
            raise ValueError("RobotState and MotionGuidance batch sizes differ")

        ref_joint_pos = guidance.joint_position[:, 0, :][:, guide_joints].astype(
            np.float32, copy=False
        )
        ref_joint_vel = guidance.joint_velocity[:, 0, :][:, guide_joints].astype(
            np.float32, copy=False
        )
        ref_body_pos = guidance.body_position_world[:, 0, bodies].astype(
            np.float32, copy=False
        )
        ref_body_quat = guidance.body_orientation_world[:, 0, bodies].astype(
            np.float32, copy=False
        )
        motion_ref_ang_vel = guidance.body_angular_velocity_world[
            :, 0, bodies[self._tracking_anchor_index]
        ].astype(np.float32, copy=False)

        joint_pos = state.joint_position[:, joints].astype(np.float32, copy=False)
        joint_vel = state.joint_velocity[:, joints].astype(np.float32, copy=False)
        robot_body_pos = state.body_position_world[:, state_bodies].astype(
            np.float32, copy=False
        )
        robot_body_quat = state.body_orientation_world[:, state_bodies].astype(
            np.float32, copy=False
        )
        root_quat = robot_body_quat[:, self._anchor_index]
        root_ang_vel_w = state.body_angular_velocity_world[
            :, state_bodies[self._anchor_index]
        ].astype(np.float32, copy=False)

        values = {
            "ref_limb_ee_pose_b": limb_pose_in_anchor_frame(
                ref_body_pos,
                ref_body_quat,
                limb_indices=self._limb_indices,
                anchor_index=self._anchor_index,
            ),
            "robot_limb_ee_pose_b": limb_pose_in_anchor_frame(
                robot_body_pos,
                robot_body_quat,
                limb_indices=self._limb_indices,
                anchor_index=self._anchor_index,
            ),
            "projected_gravity": projected_gravity(root_quat),
            "base_ang_vel": base_ang_vel_body(root_quat, root_ang_vel_w),
            "joint_pos": joint_pos - self._family.default_joint_position[None, :],
            "joint_vel": joint_vel,
            "actions": tracker_state["last_action"].astype(np.float32, copy=False),
        }
        next_state = {
            name: append_history(
                tracker_state[name], values[name], tracker_state["history_initialized"]
            )
            for name in (
                "ref_limb_ee_pose_b",
                "robot_limb_ee_pose_b",
                "projected_gravity",
                "base_ang_vel",
                "joint_pos",
                "joint_vel",
                "actions",
            )
        }
        next_state["last_action"] = tracker_state["last_action"].copy()
        next_state["history_initialized"] = np.ones(batch, dtype=bool)

        inputs = {
            "motion_command": pack_motion_command(
                ref_joint_pos,
                ref_joint_vel,
                next_state["ref_limb_ee_pose_b"],
                motion_ref_ang_vel,
            ).astype(self._input_dtypes["motion_command"]),
            "proprioception": pack_proprioception(
                next_state["robot_limb_ee_pose_b"],
                next_state["projected_gravity"],
                next_state["base_ang_vel"],
                next_state["joint_pos"],
                next_state["joint_vel"],
                next_state["actions"],
            ).astype(self._input_dtypes["proprioception"]),
        }
        return inputs, next_state

    def map_policy_outputs(
        self,
        outputs: NamedTensors,
        tracker_state: dict[str, np.ndarray],
    ) -> tuple[PDCommand, dict[str, np.ndarray]]:
        action = np.asarray(outputs["action"], dtype=np.float32)
        if action.shape != tracker_state["last_action"].shape:
            raise ValueError(
                f"action shape {action.shape} != {tracker_state['last_action'].shape}"
            )
        command = action_to_pd(
            action,
            policy_joint_names=self._config.policy_joint_names,
            command_joint_names=self._config.command_joint_names,
            default_joint_pos=self._family.default_joint_position,
            action_scale=self._family.action_scale,
            joint_kp=self._family.proportional_gain,
            joint_kd=self._family.derivative_gain,
        )
        next_state = self.clone_state(tracker_state)
        next_state["last_action"] = action.copy()
        return command, next_state

    def build_device_tracker(
        self,
        policy_path: str | Path,
        state_view: RobotStateView,
        command_view: DevicePDCommand,
    ) -> PreparedDeviceTracker:
        """Build the optional CUDA runtime without importing it on the host path."""

        from sil_trackers.families.wxy.device import WxyDeviceTracker

        return WxyDeviceTracker(self, policy_path, state_view, command_view)


@register(FAMILY)
def build_wxy(config: TrackerInstanceConfig) -> WxyAdapter:
    return WxyAdapter(config)


# Re-export for callers that import the module for its side effect only.
assert FAMILY in available()
