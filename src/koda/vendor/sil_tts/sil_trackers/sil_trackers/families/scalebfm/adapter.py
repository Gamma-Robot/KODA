"""Torso-UMI-3 ScaleBFM adapter over common tracker currencies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sil_trackers.batched_tracker import PreparedDeviceTracker, RobotStateView
from sil_trackers.contracts import (
    DevicePDCommand,
    MotionGuidance,
    NamedTensors,
    PDCommand,
    RobotState,
)
from sil_trackers.families.scalebfm.command import action_to_pd
from sil_trackers.families.scalebfm.observation import (
    ACTION_DIM,
    HISTORY_LENGTH,
    NUM_FUTURE,
    NUM_SELECTED,
    QUERY_OFFSETS,
    SELECTED_BODY_NAMES,
    TRACKING_BODY_NAMES,
    TRACKING_SELECTED_SLOTS,
    append_history,
    pack_three_point_targets,
    quat_apply_inverse,
)
from sil_trackers.registry import TrackerInstanceConfig, available, register

FAMILY = "scalebfm"
MODE_NAME = "Torso-UMI-3"


def _order(names: tuple[str, ...], target: tuple[str, ...]) -> list[int]:
    try:
        return [names.index(name) for name in target]
    except ValueError as exc:
        raise ValueError(f"missing names for {target}: {exc}") from exc


@dataclass(frozen=True)
class _ScaleBfmConfig:
    root_body: str
    history_length: int
    mode_index: int
    mode_name: str
    selected_body_names: tuple[str, ...]
    tracking_body_names: tuple[str, ...]
    tracking_selected_slots: tuple[int, ...]
    default_joint_position: np.ndarray
    action_scale: np.ndarray
    proportional_gain: np.ndarray
    derivative_gain: np.ndarray

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> _ScaleBfmConfig:
        required = {
            "root_body",
            "history_length",
            "mode_index",
            "mode_name",
            "selected_body_names",
            "tracking_body_names",
            "tracking_selected_slots",
            "default_joint_position",
            "action_scale",
            "proportional_gain",
            "derivative_gain",
        }
        if set(raw) != required:
            raise ValueError(
                "scalebfm family_config keys must be "
                f"{sorted(required)}, got {sorted(raw)}"
            )
        root = raw["root_body"]
        if not isinstance(root, str) or not root:
            raise ValueError("scalebfm root_body must be non-empty")
        history = raw["history_length"]
        if not isinstance(history, int) or history != HISTORY_LENGTH:
            raise ValueError(f"scalebfm history_length must be {HISTORY_LENGTH}")
        mode_index = raw["mode_index"]
        if not isinstance(mode_index, int) or mode_index < 0:
            raise ValueError("scalebfm mode_index must be a non-negative integer")
        mode_name = raw["mode_name"]
        if mode_name != MODE_NAME:
            raise ValueError(f"scalebfm mode_name must be {MODE_NAME!r}")
        selected = tuple(raw["selected_body_names"])  # type: ignore[arg-type]
        if selected != SELECTED_BODY_NAMES:
            raise ValueError("scalebfm selected_body_names must match Torso-UMI-3")
        tracking = tuple(raw["tracking_body_names"])  # type: ignore[arg-type]
        if tracking != TRACKING_BODY_NAMES:
            raise ValueError(f"scalebfm tracking_body_names must be {TRACKING_BODY_NAMES}")
        slots = tuple(int(value) for value in raw["tracking_selected_slots"])  # type: ignore[arg-type]
        if slots != TRACKING_SELECTED_SLOTS:
            raise ValueError(
                f"scalebfm tracking_selected_slots must be {TRACKING_SELECTED_SLOTS}"
            )

        def vector(name: str) -> np.ndarray:
            value = np.asarray(raw[name], dtype=np.float32)
            if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
                raise ValueError(
                    f"scalebfm {name} must contain {ACTION_DIM} finite values"
                )
            return value.copy()

        return cls(
            root_body=root,
            history_length=history,
            mode_index=mode_index,
            mode_name=str(mode_name),
            selected_body_names=selected,
            tracking_body_names=tracking,
            tracking_selected_slots=slots,
            default_joint_position=vector("default_joint_position"),
            action_scale=vector("action_scale"),
            proportional_gain=vector("proportional_gain"),
            derivative_gain=vector("derivative_gain"),
        )


class ScaleBfmAdapter:
    """ScaleBFM observation/state semantics without simulator or policy execution."""

    def __init__(self, config: TrackerInstanceConfig) -> None:
        if config.family != FAMILY:
            raise ValueError(f"ScaleBfmAdapter requires family {FAMILY!r}")
        family = _ScaleBfmConfig.from_mapping(config.family_config)
        expected_inputs = {
            "root_quat_buffer": (HISTORY_LENGTH, 4),
            "base_ang_vel_buffer": (HISTORY_LENGTH, 3),
            "dof_pos_buffer": (HISTORY_LENGTH, ACTION_DIM),
            "dof_vel_buffer": (HISTORY_LENGTH, ACTION_DIM),
            "last_action_buffer": (HISTORY_LENGTH, ACTION_DIM),
            "target_body_pos_future_to_robot_base": (NUM_FUTURE, NUM_SELECTED, 3),
            "target_body_rot_future_to_robot_base": (NUM_FUTURE, NUM_SELECTED, 4),
        }
        actual_inputs = {
            spec.name: spec.non_batch_shape for spec in config.policy_inputs
        }
        if actual_inputs != expected_inputs:
            raise ValueError(f"scalebfm policy_inputs must be {expected_inputs}")
        actual_outputs = {
            spec.name: spec.non_batch_shape for spec in config.policy_outputs
        }
        if actual_outputs != {"action": (ACTION_DIM,)}:
            raise ValueError("scalebfm policy_outputs must be {'action': (29,)}")
        if config.guidance_query_offsets != QUERY_OFFSETS:
            raise ValueError(
                f"scalebfm guidance_query_offsets must be {QUERY_OFFSETS}"
            )
        if len(config.policy_joint_names) != ACTION_DIM:
            raise ValueError(f"scalebfm requires {ACTION_DIM} policy joints")
        if family.root_body not in config.guidance_body_names:
            raise ValueError("scalebfm guidance bodies must contain root_body")
        missing = [
            name
            for name in family.tracking_body_names
            if name not in config.guidance_body_names
        ]
        if missing:
            raise ValueError(f"scalebfm guidance bodies missing tracking links: {missing}")
        self._config = config
        self._family = family
        self._input_dtypes = {
            spec.name: np.dtype(spec.dtype) for spec in config.policy_inputs
        }

    def initial_state(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return {
            "root_quat": np.zeros((batch_size, HISTORY_LENGTH, 4), np.float32),
            "base_ang_vel": np.zeros((batch_size, HISTORY_LENGTH, 3), np.float32),
            "joint_position": np.zeros(
                (batch_size, HISTORY_LENGTH, ACTION_DIM), np.float32
            ),
            "joint_velocity": np.zeros(
                (batch_size, HISTORY_LENGTH, ACTION_DIM), np.float32
            ),
            "last_action_history": np.zeros(
                (batch_size, HISTORY_LENGTH, ACTION_DIM), np.float32
            ),
            "last_action": np.zeros((batch_size, ACTION_DIM), np.float32),
            "history_initialized": np.zeros(batch_size, dtype=bool),
        }

    def clone_state(self, state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in state.items()}

    def expand_state(
        self, state: dict[str, np.ndarray], batch_size: int
    ) -> dict[str, np.ndarray]:
        if batch_size < 1 or any(value.shape[0] != 1 for value in state.values()):
            raise ValueError(
                "scalebfm expand_state requires one lane and positive batch_size"
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
            raise ValueError("scalebfm guidance offsets differ from the tracker config")
        batch = state.joint_position.shape[0]
        if (
            guidance.joint_position.shape[0] != batch
            or tracker_state["history_initialized"].shape != (batch,)
        ):
            raise ValueError("ScaleBFM state/guidance/tracker-state batch sizes differ")
        state_joints = _order(state.joint_names, self._config.policy_joint_names)
        state_root = state.body_names.index(self._family.root_body)
        guide_links = _order(guidance.body_names, self._family.tracking_body_names)

        root_quat = state.body_orientation_world[:, state_root].astype(
            np.float32, copy=False
        )
        root_pos = state.body_position_world[:, state_root]
        base_ang = quat_apply_inverse(
            root_quat, state.body_angular_velocity_world[:, state_root]
        ).astype(np.float32)
        joint_position = state.joint_position[:, state_joints].astype(
            np.float32, copy=False
        )
        joint_velocity = state.joint_velocity[:, state_joints].astype(
            np.float32, copy=False
        )
        last_action = tracker_state["last_action"].astype(np.float32, copy=False)

        next_state = self.clone_state(tracker_state)
        next_state["root_quat"] = append_history(
            tracker_state["root_quat"], root_quat, tracker_state["history_initialized"]
        )
        next_state["base_ang_vel"] = append_history(
            tracker_state["base_ang_vel"],
            base_ang,
            tracker_state["history_initialized"],
        )
        next_state["joint_position"] = append_history(
            tracker_state["joint_position"],
            joint_position,
            tracker_state["history_initialized"],
        )
        next_state["joint_velocity"] = append_history(
            tracker_state["joint_velocity"],
            joint_velocity,
            tracker_state["history_initialized"],
        )
        next_state["last_action_history"] = append_history(
            tracker_state["last_action_history"],
            last_action,
            tracker_state["history_initialized"],
        )
        next_state["history_initialized"] = np.ones(batch, dtype=bool)

        target_pos, target_rot = pack_three_point_targets(
            root_position=root_pos,
            root_orientation=root_quat,
            link_position_world=guidance.body_position_world[:, :, guide_links],
            link_orientation_world=guidance.body_orientation_world[:, :, guide_links],
            tracking_slots=self._family.tracking_selected_slots,
        )
        inputs = {
            "root_quat_buffer": next_state["root_quat"],
            "base_ang_vel_buffer": next_state["base_ang_vel"],
            "dof_pos_buffer": next_state["joint_position"],
            "dof_vel_buffer": next_state["joint_velocity"],
            "last_action_buffer": next_state["last_action_history"],
            "target_body_pos_future_to_robot_base": target_pos,
            "target_body_rot_future_to_robot_base": target_rot,
        }
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
            raise ValueError("ScaleBFM action batch differs from tracker state")
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

        from sil_trackers.families.scalebfm.device import ScaleBfmDeviceTracker

        return ScaleBfmDeviceTracker(self, policy_path, state_view, command_view)


@register(FAMILY)
def build_scalebfm(config: TrackerInstanceConfig) -> ScaleBfmAdapter:
    return ScaleBfmAdapter(config)


assert FAMILY in available()
