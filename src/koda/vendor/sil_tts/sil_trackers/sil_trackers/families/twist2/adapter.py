"""Adapter for the officially released TWIST2 G1 policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from sil_trackers.contracts import MotionGuidance, NamedTensors, PDCommand, RobotState
from sil_trackers.families.twist2.command import action_to_pd
from sil_trackers.families.twist2.observation import (
    ACTION_DIM,
    CURRENT_OFFSET,
    FUTURE_OFFSET,
    HISTORY_LENGTH,
    INPUT_DIM,
    QUERY_OFFSETS,
    RESET_FILL_STEPS,
    pack_input,
    pack_mimic,
    pack_proprioception,
)
from sil_trackers.registry import TrackerInstanceConfig, available, register

if TYPE_CHECKING:
    from sil_trackers.batched_tracker import PreparedDeviceTracker, RobotStateView
    from sil_trackers.contracts import DevicePDCommand

FAMILY = "twist2"


def _order(names: tuple[str, ...], target: tuple[str, ...]) -> list[int]:
    try:
        return [names.index(name) for name in target]
    except ValueError as exc:
        raise ValueError(f"missing names for {target}: {exc}") from exc


@dataclass(frozen=True)
class _Twist2Config:
    root_body: str
    history_length: int
    ankle_joint_names: tuple[str, ...]
    angular_velocity_scale: float
    joint_position_scale: float
    joint_velocity_scale: float
    action_clip: float
    default_joint_position: np.ndarray
    action_scale: np.ndarray
    proportional_gain: np.ndarray
    derivative_gain: np.ndarray

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> _Twist2Config:
        required = {
            "root_body",
            "history_length",
            "ankle_joint_names",
            "angular_velocity_scale",
            "joint_position_scale",
            "joint_velocity_scale",
            "action_clip",
            "default_joint_position",
            "action_scale",
            "proportional_gain",
            "derivative_gain",
        }
        if set(raw) != required:
            raise ValueError(
                "twist2 family_config keys must be "
                f"{sorted(required)}, got {sorted(raw)}"
            )
        root = raw["root_body"]
        if not isinstance(root, str) or not root:
            raise ValueError("twist2 root_body must be non-empty")
        history_length = raw["history_length"]
        if history_length != HISTORY_LENGTH:
            raise ValueError(f"twist2 history_length must be {HISTORY_LENGTH}")
        ankles = tuple(raw["ankle_joint_names"])  # type: ignore[arg-type]
        if len(ankles) != 4 or len(set(ankles)) != 4:
            raise ValueError("twist2 ankle_joint_names must contain four unique names")

        def scalar(name: str) -> float:
            value = float(raw[name])  # type: ignore[arg-type]
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"twist2 {name} must be finite and positive")
            return value

        def vector(name: str) -> np.ndarray:
            value = np.asarray(raw[name], dtype=np.float32)
            if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
                raise ValueError(
                    f"twist2 {name} must contain {ACTION_DIM} finite values"
                )
            return value.copy()

        return cls(
            root_body=root,
            history_length=history_length,
            ankle_joint_names=ankles,
            angular_velocity_scale=scalar("angular_velocity_scale"),
            joint_position_scale=scalar("joint_position_scale"),
            joint_velocity_scale=scalar("joint_velocity_scale"),
            action_clip=scalar("action_clip"),
            default_joint_position=vector("default_joint_position"),
            action_scale=vector("action_scale"),
            proportional_gain=vector("proportional_gain"),
            derivative_gain=vector("derivative_gain"),
        )


class Twist2Adapter:
    """TWIST2 observation, history, and action semantics on host NumPy arrays."""

    def __init__(self, config: TrackerInstanceConfig) -> None:
        if config.family != FAMILY:
            raise ValueError(f"Twist2Adapter requires family {FAMILY!r}")
        inputs = {spec.name: spec.non_batch_shape for spec in config.policy_inputs}
        outputs = {spec.name: spec.non_batch_shape for spec in config.policy_outputs}
        if inputs != {"input": (INPUT_DIM,)}:
            raise ValueError(f"twist2 policy_inputs must be input[{INPUT_DIM}]")
        if outputs != {"output": (ACTION_DIM,)}:
            raise ValueError(f"twist2 policy_outputs must be output[{ACTION_DIM}]")
        if config.guidance_query_offsets != QUERY_OFFSETS:
            raise ValueError(f"twist2 guidance_query_offsets must be {QUERY_OFFSETS}")
        if len(config.policy_joint_names) != ACTION_DIM:
            raise ValueError(f"twist2 requires {ACTION_DIM} policy joints")
        family = _Twist2Config.from_mapping(config.family_config)
        if family.root_body not in config.guidance_body_names:
            raise ValueError("twist2 guidance bodies must contain root_body")
        if any(name not in config.policy_joint_names for name in family.ankle_joint_names):
            raise ValueError("twist2 ankle joints must be policy joints")
        self._config = config
        self._family = family
        self._root_index = config.guidance_body_names.index(family.root_body)
        self._ankle_indices = np.asarray(
            [config.policy_joint_names.index(name) for name in family.ankle_joint_names],
            dtype=np.int64,
        )
        self._input_dtype = np.dtype(config.policy_inputs[0].dtype)

    def initial_state(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return {
            "history": np.zeros(
                (batch_size, HISTORY_LENGTH, 127), dtype=np.float32
            ),
            "history_fill_steps_remaining": np.full(
                batch_size, RESET_FILL_STEPS, dtype=np.int32
            ),
            "last_action": np.zeros((batch_size, ACTION_DIM), dtype=np.float32),
        }

    def clone_state(self, state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in state.items()}

    def expand_state(
        self, state: dict[str, np.ndarray], batch_size: int
    ) -> dict[str, np.ndarray]:
        if batch_size < 1 or any(value.shape[0] != 1 for value in state.values()):
            raise ValueError(
                "twist2 expand_state requires one lane and positive batch_size"
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
            raise ValueError("twist2 guidance offsets differ from the tracker config")
        batch = state.joint_position.shape[0]
        if (
            guidance.joint_position.shape[0] != batch
            or tracker_state["last_action"].shape != (batch, ACTION_DIM)
            or tracker_state["history_fill_steps_remaining"].shape != (batch,)
        ):
            raise ValueError("TWIST2 state/guidance/tracker-state batch sizes differ")
        state_joints = _order(state.joint_names, self._config.policy_joint_names)
        guide_joints = _order(guidance.joint_names, self._config.policy_joint_names)
        state_bodies = _order(state.body_names, self._config.guidance_body_names)
        guide_bodies = _order(guidance.body_names, self._config.guidance_body_names)
        state_root = state_bodies[self._root_index]
        guide_root = guide_bodies[self._root_index]
        current_query = guidance.query_offsets.index(CURRENT_OFFSET)
        future_query = guidance.query_offsets.index(FUTURE_OFFSET)

        current_mimic = pack_mimic(
            guidance.body_position_world[:, current_query, guide_root],
            guidance.body_orientation_world[:, current_query, guide_root],
            guidance.body_linear_velocity_world[:, current_query, guide_root],
            guidance.body_angular_velocity_world[:, current_query, guide_root],
            guidance.joint_position[:, current_query, guide_joints],
        )
        future_mimic = pack_mimic(
            guidance.body_position_world[:, future_query, guide_root],
            guidance.body_orientation_world[:, future_query, guide_root],
            guidance.body_linear_velocity_world[:, future_query, guide_root],
            guidance.body_angular_velocity_world[:, future_query, guide_root],
            guidance.joint_position[:, future_query, guide_joints],
        )
        proprioception = pack_proprioception(
            state.body_orientation_world[:, state_root],
            state.body_angular_velocity_world[:, state_root],
            state.joint_position[:, state_joints],
            state.joint_velocity[:, state_joints],
            tracker_state["last_action"],
            default_joint_position=self._family.default_joint_position,
            ankle_indices=self._ankle_indices,
            angular_velocity_scale=self._family.angular_velocity_scale,
            joint_position_scale=self._family.joint_position_scale,
            joint_velocity_scale=self._family.joint_velocity_scale,
        )
        policy_input, history = pack_input(
            current_mimic,
            proprioception,
            tracker_state["history"],
            tracker_state["history_fill_steps_remaining"],
            future_mimic,
        )
        next_state = self.clone_state(tracker_state)
        next_state["history"] = history
        next_state["history_fill_steps_remaining"] = np.maximum(
            tracker_state["history_fill_steps_remaining"] - 1, 0
        ).astype(np.int32)
        return {"input": policy_input.astype(self._input_dtype, copy=False)}, next_state

    def map_policy_outputs(
        self,
        outputs: NamedTensors,
        tracker_state: dict[str, np.ndarray],
    ) -> tuple[PDCommand, dict[str, np.ndarray]]:
        action = np.asarray(outputs["output"], dtype=np.float32)
        if action.shape != tracker_state["last_action"].shape:
            raise ValueError("TWIST2 action batch differs from tracker state")
        command = action_to_pd(
            action,
            clip=self._family.action_clip,
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

        from sil_trackers.families.twist2.device import Twist2DeviceTracker

        return Twist2DeviceTracker(self, policy_path, state_view, command_view)


@register(FAMILY)
def build_twist2(config: TrackerInstanceConfig) -> Twist2Adapter:
    return Twist2Adapter(config)


assert FAMILY in available()
