"""Adapter for the officially released Humanoid-GPT G1 policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from sil_trackers.contracts import MotionGuidance, NamedTensors, PDCommand, RobotState
from sil_trackers.families.humanoid_gpt.command import action_to_pd
from sil_trackers.families.humanoid_gpt.observation import (
    ACTION_DIM,
    CURRENT_OFFSET,
    MOTION_COMMAND_DIM,
    NEXT_OFFSET,
    PROPRIOCEPTION_DIM,
    QUERY_OFFSETS,
    lock_root_alignment,
    pack_motion_command,
    pack_proprioception,
    rebias_root_pose,
)
from sil_trackers.registry import TrackerInstanceConfig, available, register

if TYPE_CHECKING:
    from sil_trackers.batched_tracker import PreparedDeviceTracker, RobotStateView
    from sil_trackers.contracts import DevicePDCommand

FAMILY = "humanoid_gpt"


def _order(names: tuple[str, ...], target: tuple[str, ...]) -> list[int]:
    try:
        return [names.index(name) for name in target]
    except ValueError as exc:
        raise ValueError(f"missing names for {target}: {exc}") from exc


@dataclass(frozen=True)
class _HumanoidGptConfig:
    root_body: str
    default_joint_position: np.ndarray
    action_scale: np.ndarray
    proportional_gain: np.ndarray
    derivative_gain: np.ndarray

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> _HumanoidGptConfig:
        required = {
            "root_body",
            "default_joint_position",
            "action_scale",
            "proportional_gain",
            "derivative_gain",
        }
        if set(raw) != required:
            raise ValueError(
                "humanoid_gpt family_config keys must be "
                f"{sorted(required)}, got {sorted(raw)}"
            )
        root = raw["root_body"]
        if not isinstance(root, str) or not root:
            raise ValueError("humanoid_gpt root_body must be non-empty")

        def vector(name: str) -> np.ndarray:
            value = np.asarray(raw[name], dtype=np.float32)
            if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
                raise ValueError(
                    f"humanoid_gpt {name} must contain {ACTION_DIM} finite values"
                )
            return value.copy()

        return cls(
            root_body=root,
            default_joint_position=vector("default_joint_position"),
            action_scale=vector("action_scale"),
            proportional_gain=vector("proportional_gain"),
            derivative_gain=vector("derivative_gain"),
        )


class HumanoidGptAdapter:
    """Humanoid-GPT observation and action semantics on host NumPy arrays."""

    def __init__(self, config: TrackerInstanceConfig) -> None:
        if config.family != FAMILY:
            raise ValueError(f"HumanoidGptAdapter requires family {FAMILY!r}")
        expected_inputs = {
            "proprioception": (PROPRIOCEPTION_DIM,),
            "motion_command": (MOTION_COMMAND_DIM,),
        }
        actual_inputs = {
            spec.name: spec.non_batch_shape for spec in config.policy_inputs
        }
        if actual_inputs != expected_inputs:
            raise ValueError(f"humanoid_gpt policy_inputs must be {expected_inputs}")
        actual_outputs = {
            spec.name: spec.non_batch_shape for spec in config.policy_outputs
        }
        if actual_outputs != {"action": (ACTION_DIM,)}:
            raise ValueError(
                "humanoid_gpt runtime policy_outputs must contain only action[29]"
            )
        if config.guidance_query_offsets != QUERY_OFFSETS:
            raise ValueError(
                f"humanoid_gpt guidance_query_offsets must be {QUERY_OFFSETS}"
            )
        if len(config.policy_joint_names) != ACTION_DIM:
            raise ValueError(f"humanoid_gpt requires {ACTION_DIM} policy joints")
        family = _HumanoidGptConfig.from_mapping(config.family_config)
        if family.root_body not in config.guidance_body_names:
            raise ValueError("humanoid_gpt guidance bodies must contain root_body")
        self._config = config
        self._family = family
        self._root_index = config.guidance_body_names.index(family.root_body)
        self._input_dtypes = {
            spec.name: np.dtype(spec.dtype) for spec in config.policy_inputs
        }

    def initial_state(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return {
            "last_action": np.zeros((batch_size, ACTION_DIM), np.float32),
            "alignment_initialized": np.zeros(batch_size, dtype=bool),
            "robot_init_yaw": np.zeros(batch_size, np.float32),
            "robot_init_xy": np.zeros((batch_size, 2), np.float32),
            "ref_init_yaw": np.zeros(batch_size, np.float32),
            "ref_init_xy": np.zeros((batch_size, 2), np.float32),
        }

    def clone_state(self, state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in state.items()}

    def expand_state(
        self, state: dict[str, np.ndarray], batch_size: int
    ) -> dict[str, np.ndarray]:
        if batch_size < 1 or any(value.shape[0] != 1 for value in state.values()):
            raise ValueError(
                "humanoid_gpt expand_state requires one lane and positive batch_size"
            )
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
        if guidance.query_offsets != QUERY_OFFSETS:
            raise ValueError(
                "humanoid_gpt guidance offsets differ from the tracker config"
            )
        batch = state.joint_position.shape[0]
        if (
            guidance.joint_position.shape[0] != batch
            or tracker_state["last_action"].shape != (batch, ACTION_DIM)
            or tracker_state["alignment_initialized"].shape != (batch,)
        ):
            raise ValueError(
                "Humanoid-GPT state/guidance/tracker-state batch sizes differ"
            )
        state_joints = _order(state.joint_names, self._config.policy_joint_names)
        guide_joints = _order(guidance.joint_names, self._config.policy_joint_names)
        state_bodies = _order(state.body_names, self._config.guidance_body_names)
        guide_bodies = _order(guidance.body_names, self._config.guidance_body_names)
        state_root = state_bodies[self._root_index]
        guide_root = guide_bodies[self._root_index]
        current_query = guidance.query_offsets.index(CURRENT_OFFSET)
        next_query = guidance.query_offsets.index(NEXT_OFFSET)

        robot_orientation = state.body_orientation_world[:, state_root]
        (
            initialized,
            robot_init_yaw,
            robot_init_xy,
            ref_init_yaw,
            ref_init_xy,
        ) = lock_root_alignment(
            tracker_state["alignment_initialized"],
            tracker_state["robot_init_yaw"],
            tracker_state["robot_init_xy"],
            tracker_state["ref_init_yaw"],
            tracker_state["ref_init_xy"],
            robot_orientation,
            guidance.body_position_world[:, current_query, guide_root],
            guidance.body_orientation_world[:, current_query, guide_root],
        )
        current_position, current_orientation, yaw_offset = rebias_root_pose(
            guidance.body_position_world[:, current_query, guide_root],
            guidance.body_orientation_world[:, current_query, guide_root],
            robot_init_yaw,
            robot_init_xy,
            ref_init_yaw,
            ref_init_xy,
        )
        next_position, next_orientation, _ = rebias_root_pose(
            guidance.body_position_world[:, next_query, guide_root],
            guidance.body_orientation_world[:, next_query, guide_root],
            robot_init_yaw,
            robot_init_xy,
            ref_init_yaw,
            ref_init_xy,
        )
        inputs = {
            "proprioception": pack_proprioception(
                robot_orientation,
                state.body_angular_velocity_world[:, state_root],
                state.joint_position[:, state_joints],
                state.joint_velocity[:, state_joints],
                tracker_state["last_action"],
                self._family.default_joint_position,
            ),
            "motion_command": pack_motion_command(
                guidance.joint_position[:, next_query, guide_joints],
                next_position,
                next_orientation,
                guidance.body_angular_velocity_world[:, next_query, guide_root],
                guidance.body_linear_velocity_world[:, next_query, guide_root],
                current_position,
                current_orientation,
                state.body_position_world[:, state_root],
                robot_orientation,
                yaw_offset,
                self._family.default_joint_position,
            ),
        }
        next_state = self.clone_state(tracker_state)
        next_state["alignment_initialized"] = initialized
        next_state["robot_init_yaw"] = robot_init_yaw
        next_state["robot_init_xy"] = robot_init_xy
        next_state["ref_init_yaw"] = ref_init_yaw
        next_state["ref_init_xy"] = ref_init_xy
        return (
            {
                name: value.astype(self._input_dtypes[name], copy=False)
                for name, value in inputs.items()
            },
            next_state,
        )

    def map_policy_outputs(
        self,
        outputs: NamedTensors,
        tracker_state: dict[str, np.ndarray],
    ) -> tuple[PDCommand, dict[str, np.ndarray]]:
        action = np.asarray(outputs["action"], dtype=np.float32)
        if action.shape != tracker_state["last_action"].shape:
            raise ValueError("Humanoid-GPT action batch differs from tracker state")
        command = action_to_pd(
            action,
            policy_joint_names=self._config.policy_joint_names,
            command_joint_names=self._config.command_joint_names,
            default_joint_position=self._family.default_joint_position,
            action_scale=self._family.action_scale,
            proportional_gain=self._family.proportional_gain,
            derivative_gain=self._family.derivative_gain,
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

        from sil_trackers.families.humanoid_gpt.device import HumanoidGptDeviceTracker

        return HumanoidGptDeviceTracker(self, policy_path, state_view, command_view)


@register(FAMILY)
def build_humanoid_gpt(config: TrackerInstanceConfig) -> HumanoidGptAdapter:
    return HumanoidGptAdapter(config)


assert FAMILY in available()
