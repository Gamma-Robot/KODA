"""Interface implemented by each tracker family."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sil_trackers.contracts import (
    MotionGuidance,
    NamedTensors,
    PDCommand,
    RobotState,
    TrackerStateBatch,
)


@runtime_checkable
class TrackerAdapter(Protocol):
    """Pure tracker semantics; policy execution and physics live elsewhere."""

    def initial_state(self, batch_size: int) -> TrackerStateBatch:
        """Create family-owned state for independent rollout lanes."""

    def clone_state(self, state: TrackerStateBatch) -> TrackerStateBatch:
        """Return an independent copy of family-owned state."""

    def expand_state(
        self, state: TrackerStateBatch, batch_size: int
    ) -> TrackerStateBatch:
        """Copy one execution-lane state into independent rollout lanes."""

    def build_policy_inputs(
        self,
        state: RobotState,
        guidance: MotionGuidance,
        tracker_state: TrackerStateBatch,
    ) -> tuple[NamedTensors, TrackerStateBatch]:
        """Build named inputs and return state advanced by observation history."""

    def map_policy_outputs(
        self,
        outputs: NamedTensors,
        tracker_state: TrackerStateBatch,
    ) -> tuple[PDCommand, TrackerStateBatch]:
        """Map named outputs to a command and the next family-owned state."""
