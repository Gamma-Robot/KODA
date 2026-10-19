"""Mimic-Lite tracker adapter over common tracker currencies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from sil_trackers.contracts import MotionGuidance, NamedTensors, PDCommand, RobotState
from sil_trackers.families.mimic_lite.command import action_to_pd
from sil_trackers.families.mimic_lite.observation import (
    ACTION_DIM,
    COMMAND_DIM,
    HISTORY_LAGS,
    HISTORY_LENGTH,
    POLICY_DIM,
    PREVIOUS_ACTIONS,
    QUERY_OFFSETS,
    append_newest,
    heading_offset,
    pack_command,
    pack_policy,
    projected_gravity,
    root_angular_velocity_body,
)
from sil_trackers.registry import TrackerInstanceConfig, available, register

if TYPE_CHECKING:
    from sil_trackers.batched_tracker import PreparedDeviceTracker, RobotStateView
    from sil_trackers.contracts import DevicePDCommand

FAMILY = "mimic_lite"


def _order(names: tuple[str, ...], target: tuple[str, ...]) -> list[int]:
    try:
        return [names.index(name) for name in target]
    except ValueError as exc:
        raise ValueError(f"missing names for {target}: {exc}") from exc


@dataclass(frozen=True)
class _MimicLiteConfig:
    root_body: str
    anchor_body: str
    history_lags: tuple[int, ...]
    previous_action_count: int
    default_joint_position: np.ndarray
    action_scale: np.ndarray
    proportional_gain: np.ndarray
    derivative_gain: np.ndarray

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> _MimicLiteConfig:
        required = {
            "root_body",
            "anchor_body",
            "history_lags",
            "previous_action_count",
            "default_joint_position",
            "action_scale",
            "proportional_gain",
            "derivative_gain",
        }
        if set(raw) != required:
            raise ValueError(
                "mimic_lite family_config keys must be "
                f"{sorted(required)}, got {sorted(raw)}"
            )
        root = raw["root_body"]
        if not isinstance(root, str) or not root:
            raise ValueError("mimic_lite root_body must be non-empty")
        anchor = raw["anchor_body"]
        if not isinstance(anchor, str) or not anchor:
            raise ValueError("mimic_lite anchor_body must be non-empty")
        lags = tuple(raw["history_lags"])  # type: ignore[arg-type]
        if lags != HISTORY_LAGS:
            raise ValueError(f"mimic_lite history_lags must be {HISTORY_LAGS}")
        previous = raw["previous_action_count"]
        if not isinstance(previous, int) or previous != PREVIOUS_ACTIONS:
            raise ValueError(
                f"mimic_lite previous_action_count must be {PREVIOUS_ACTIONS}"
            )

        def vector(name: str) -> np.ndarray:
            value = np.asarray(raw[name], dtype=np.float32)
            if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
                raise ValueError(
                    f"mimic_lite {name} must contain {ACTION_DIM} finite values"
                )
            return value.copy()

        return cls(
            root_body=root,
            anchor_body=anchor,
            history_lags=lags,
            previous_action_count=previous,
            default_joint_position=vector("default_joint_position"),
            action_scale=vector("action_scale"),
            proportional_gain=vector("proportional_gain"),
            derivative_gain=vector("derivative_gain"),
        )


class MimicLiteAdapter:
    """Legacy 32x8192-huge observation/state semantics."""

    def __init__(self, config: TrackerInstanceConfig) -> None:
        if config.family != FAMILY:
            raise ValueError(f"MimicLiteAdapter requires family {FAMILY!r}")
        actual_inputs = {
            spec.name: spec.non_batch_shape for spec in config.policy_inputs
        }
        if actual_inputs != {"policy": (POLICY_DIM,), "command": (COMMAND_DIM,)}:
            raise ValueError(
                "mimic_lite policy inputs must be policy[535], command[304]"
            )
        actual_outputs = {
            spec.name: spec.non_batch_shape for spec in config.policy_outputs
        }
        if actual_outputs != {"action": (ACTION_DIM,)}:
            raise ValueError("mimic_lite policy output must be action[29]")
        if config.guidance_query_offsets != QUERY_OFFSETS:
            raise ValueError(
                f"mimic_lite guidance_query_offsets must be {QUERY_OFFSETS}"
            )
        if len(config.policy_joint_names) != ACTION_DIM:
            raise ValueError(f"mimic_lite requires {ACTION_DIM} policy joints")
        family = _MimicLiteConfig.from_mapping(config.family_config)
        if not {family.root_body, family.anchor_body}.issubset(
            config.guidance_body_names
        ):
            raise ValueError(
                "mimic_lite guidance bodies must contain root_body and anchor_body"
            )
        self._config = config
        self._family = family
        self._input_dtypes = {
            spec.name: np.dtype(spec.dtype) for spec in config.policy_inputs
        }

    def initial_state(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return {
            "root_angular_velocity": np.zeros(
                (batch_size, HISTORY_LENGTH, 3), np.float32
            ),
            "projected_gravity": np.zeros(
                (batch_size, HISTORY_LENGTH, 3), np.float32
            ),
            "joint_position": np.zeros(
                (batch_size, HISTORY_LENGTH, ACTION_DIM), np.float32
            ),
            "joint_velocity": np.zeros(
                (batch_size, HISTORY_LENGTH, ACTION_DIM), np.float32
            ),
            "previous_actions": np.zeros(
                (batch_size, PREVIOUS_ACTIONS, ACTION_DIM), np.float32
            ),
            "last_action": np.zeros((batch_size, ACTION_DIM), np.float32),
            "heading_offset": np.zeros((batch_size, 4), np.float32),
            "heading_initialized": np.zeros(batch_size, dtype=np.bool_),
        }

    def clone_state(self, state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in state.items()}

    def expand_state(
        self, state: dict[str, np.ndarray], batch_size: int
    ) -> dict[str, np.ndarray]:
        if batch_size < 1 or any(value.shape[0] != 1 for value in state.values()):
            raise ValueError(
                "mimic_lite expand_state requires one lane and positive batch_size"
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
            raise ValueError("mimic_lite guidance offsets differ from tracker config")
        batch = state.joint_position.shape[0]
        if (
            guidance.joint_position.shape[0] != batch
            or tracker_state["heading_initialized"].shape != (batch,)
        ):
            raise ValueError(
                "Mimic-Lite state/guidance/tracker-state batch sizes differ"
            )
        state_joints = _order(state.joint_names, self._config.policy_joint_names)
        guide_joints = _order(
            guidance.joint_names, self._config.policy_joint_names
        )
        try:
            state_root = state.body_names.index(self._family.root_body)
            guide_root = guidance.body_names.index(self._family.root_body)
            guide_anchor = guidance.body_names.index(self._family.anchor_body)
        except ValueError as exc:
            raise ValueError("Mimic-Lite state/guidance is missing root or anchor") from exc

        joint_position = state.joint_position[:, state_joints].astype(
            np.float32, copy=False
        )
        joint_velocity = state.joint_velocity[:, state_joints].astype(
            np.float32, copy=False
        )
        robot_root_orientation = state.body_orientation_world[
            :, state_root
        ].astype(np.float32, copy=False)
        robot_root_angular_velocity = state.body_angular_velocity_world[
            :, state_root
        ].astype(np.float32, copy=False)
        reference_joint_position = guidance.joint_position[
            :, :, guide_joints
        ].astype(np.float32, copy=False)
        reference_root_position = guidance.body_position_world[
            :, :, guide_root
        ].astype(np.float32, copy=False)
        reference_root_orientation = guidance.body_orientation_world[
            :, :, guide_root
        ].astype(np.float32, copy=False)
        reference_anchor_position = guidance.body_position_world[
            :, :, guide_anchor
        ].astype(np.float32, copy=False)
        reference_anchor_orientation = guidance.body_orientation_world[
            :, :, guide_anchor
        ].astype(np.float32, copy=False)

        next_state = self.clone_state(tracker_state)
        next_state["root_angular_velocity"] = append_newest(
            tracker_state["root_angular_velocity"],
            root_angular_velocity_body(
                robot_root_orientation, robot_root_angular_velocity
            ),
        )
        next_state["projected_gravity"] = append_newest(
            tracker_state["projected_gravity"],
            projected_gravity(robot_root_orientation),
        )
        next_state["joint_position"] = append_newest(
            tracker_state["joint_position"], joint_position
        )
        next_state["joint_velocity"] = append_newest(
            tracker_state["joint_velocity"], joint_velocity
        )
        next_state["previous_actions"] = append_newest(
            tracker_state["previous_actions"], tracker_state["last_action"]
        )

        current_reference = reference_root_orientation[:, QUERY_OFFSETS.index(0)]
        initial_heading = heading_offset(
            robot_root_orientation, current_reference
        )
        next_state["heading_offset"] = np.where(
            tracker_state["heading_initialized"][:, None],
            tracker_state["heading_offset"],
            initial_heading,
        ).astype(np.float32, copy=False)
        next_state["heading_initialized"] = np.ones(batch, dtype=np.bool_)

        inputs = {
            "policy": pack_policy(
                next_state["root_angular_velocity"],
                next_state["projected_gravity"],
                next_state["joint_position"],
                next_state["joint_velocity"],
                next_state["previous_actions"],
            ),
            "command": pack_command(
                reference_root_position,
                reference_root_orientation,
                reference_anchor_position,
                reference_anchor_orientation,
                reference_joint_position,
                robot_root_orientation,
                next_state["heading_offset"],
            ),
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
            raise ValueError("Mimic-Lite action batch differs from tracker state")
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

        from sil_trackers.families.mimic_lite.device import MimicLiteDeviceTracker

        return MimicLiteDeviceTracker(self, policy_path, state_view, command_view)


@register(FAMILY)
def build_mimic_lite(config: TrackerInstanceConfig) -> MimicLiteAdapter:
    return MimicLiteAdapter(config)


assert FAMILY in available()


__all__ = ["MimicLiteAdapter", "build_mimic_lite"]
