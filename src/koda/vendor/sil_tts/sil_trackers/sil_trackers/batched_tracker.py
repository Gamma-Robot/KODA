"""Stateful tracker execution over host observations and one batched policy call."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from sil_trackers.adapter import TrackerAdapter
from sil_trackers.contracts import (
    DevicePDCommand,
    MotionGuidance,
    NamedTensors,
    PDCommand,
    RobotState,
    TrackerStateBatch,
)


@runtime_checkable
class RobotStateView(Protocol):
    """Canonical physical arrays exposed by a simulation world.

    Arrays may live on a device, but must either support ``numpy()`` or be
    convertible with ``numpy.asarray``. Quaternion fields use ``(w, x, y, z)``.
    """

    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    joint_position: object
    joint_velocity: object
    body_position_world: object
    body_orientation_world: object
    body_linear_velocity_world: object
    body_angular_velocity_world: object


@runtime_checkable
class BatchedTracker(Protocol):
    """Stateful pretrained tracker; the caller owns physics and applies commands."""

    def reset(self, state: TrackerStateBatch) -> None:
        """Restore an independently owned tracker-state copy."""

    def capture_state(self) -> TrackerStateBatch:
        """Return an independent copy of the current tracker state."""

    def step(
        self, state_view: RobotStateView, guidance: MotionGuidance
    ) -> PDCommand:
        """Advance tracker state by one control step and return a PD command."""


@runtime_checkable
class PreparedDeviceTracker(Protocol):
    """Optional device runtime that binds a complete guidance segment once."""

    def reset(self, state: TrackerStateBatch) -> None:
        """Complete queued policy work, then restore an owned tracker-state copy."""

    def capture_state(self) -> TrackerStateBatch:
        """Return an independent copy of the current tracker state."""

    def prepare(
        self,
        history: MotionGuidance | None,
        guidance: MotionGuidance,
        right_context: MotionGuidance | None,
        *,
        num_steps: int,
    ) -> None:
        """Bind guidance and the exact number of control steps to execute."""

    def step_prepared(
        self, state_view: RobotStateView, step_index: int
    ) -> DevicePDCommand:
        """Advance one control step using the guidance bound by ``prepare``."""

    def close(self) -> None:
        """Release policy runtime resources after all queued work completes."""


@runtime_checkable
class DeviceTrackerProvider(Protocol):
    """Optional adapter capability for one persistent device tracker."""

    def build_device_tracker(
        self,
        policy_path: str | Path,
        state_view: RobotStateView,
        command_view: DevicePDCommand,
    ) -> PreparedDeviceTracker:
        """Bind family semantics to one fixed simulation batch."""


class _BatchedPolicy(Protocol):
    def run(self, inputs: NamedTensors) -> NamedTensors: ...


class SerialReferenceTracker:
    """Host-NumPy tracker path used as the reference implementation.

    The family adapter owns observation, action, and opaque-state semantics.
    Only policy inference is delegated, once per control step for the complete batch.
    State advances atomically after policy execution and output mapping succeed.
    """

    def __init__(
        self,
        adapter: TrackerAdapter,
        policy: _BatchedPolicy,
        *,
        batch_size: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._adapter = adapter
        self._policy = policy
        self._state = adapter.initial_state(batch_size)

    def reset(self, state: TrackerStateBatch) -> None:
        self._state = self._adapter.clone_state(state)

    def capture_state(self) -> TrackerStateBatch:
        return self._adapter.clone_state(self._state)

    def step(
        self, state_view: RobotStateView, guidance: MotionGuidance
    ) -> PDCommand:
        state = _copy_to_host(state_view)
        batch_size = state.joint_position.shape[0]
        if guidance.joint_position.shape[0] != batch_size:
            raise ValueError("RobotStateView and MotionGuidance batch sizes differ")

        inputs, observed_state = self._adapter.build_policy_inputs(
            state, guidance, self._state
        )
        outputs = self._policy.run(inputs)
        command, next_state = self._adapter.map_policy_outputs(
            outputs, observed_state
        )
        if command.position_target.shape[0] != batch_size:
            raise ValueError("tracker command batch size differs from RobotStateView")
        self._state = next_state
        return command


def _copy_to_host(view: RobotStateView) -> RobotState:
    if isinstance(view, RobotState):
        return view

    def array(field: str) -> np.ndarray:
        value = getattr(view, field)
        to_numpy = getattr(value, "numpy", None)
        host = to_numpy() if callable(to_numpy) else value
        return np.asarray(host).copy()

    return RobotState(
        joint_names=tuple(view.joint_names),
        body_names=tuple(view.body_names),
        joint_position=array("joint_position"),
        joint_velocity=array("joint_velocity"),
        body_position_world=array("body_position_world"),
        body_orientation_world=array("body_orientation_world"),
        body_linear_velocity_world=array("body_linear_velocity_world"),
        body_angular_velocity_world=array("body_angular_velocity_world"),
    )
