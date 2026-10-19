"""The single guidance -> tracker -> PD -> simulation execution path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np

from sil_trackers import (
    DevicePDCommand,
    MOTION_GUIDANCE_ARRAY_FIELDS as _GUIDANCE_ARRAYS,
    MotionGuidance,
    PDCommand,
    PreparedDeviceTracker,
)
from sil_trackers.batched_tracker import BatchedTracker, RobotStateView
from sil_tts.simulation.measurements import (
    STATE_MEASUREMENT_PRIMITIVES,
    MeasurementBuffer,
    MeasurementPrimitive,
    ControlStepMeasurements,
)
from sil_tts.simulation.mjwarp import MjwarpWorld
from sil_tts.simulation.execution_state import ExecutionState


def _owned(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True, order="C")
    result.flags.writeable = False
    return result


def _capture_policy_outputs(tracker: object) -> dict[str, np.ndarray] | None:
    capture = getattr(tracker, "capture_policy_outputs", None)
    if not callable(capture):
        return None
    outputs = capture()
    if not isinstance(outputs, Mapping):
        raise TypeError("capture_policy_outputs must return a mapping")
    return {str(name): np.asarray(value) for name, value in outputs.items()}


def _stack_policy_outputs(
    steps: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray] | None:
    if not steps:
        return None
    names = tuple(steps[0])
    if any(tuple(step) != names for step in steps):
        raise RuntimeError("policy output names changed during execution")
    if not names:
        return None
    stacked = {
        name: _owned(np.stack([step[name] for step in steps], axis=1))
        for name in names
    }
    return stacked


@dataclass(frozen=True, eq=False)
class ExecutionResult:
    """Measurements, physical trace, and restartable state after one segment."""

    measurements: ControlStepMeasurements
    qpos: np.ndarray
    qvel: np.ndarray
    final_execution_state: ExecutionState
    policy_outputs: Mapping[str, np.ndarray] | None = None


class RolloutReducer(Protocol):
    """Consume post-step device state and return one compact rollout result."""

    def record(self, step_index: int, state_view: RobotStateView) -> None: ...

    def finalize(self) -> np.ndarray: ...


class GuidanceExecutor:
    """Execute a dense candidate window through one tracker and one world.

    Tracker queries may be signed and ordered arbitrarily. Past queries read
    the execution history carried by the execution state; queries beyond the
    candidate window read an explicit nominal right context. Neither context
    extends the number of simulated or evaluated Candidate Rollout Control Steps.
    """

    def __init__(
        self,
        world: MjwarpWorld,
        tracker: BatchedTracker | PreparedDeviceTracker,
        *,
        control_period: float,
        guidance_query_offsets: tuple[int, ...],
    ) -> None:
        period = float(control_period)
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("control_period must be finite and positive")
        ratio = period / float(world.physics_timestep)
        substeps = int(round(ratio))
        if substeps < 1 or not np.isclose(ratio, substeps, rtol=0.0, atol=1e-9):
            raise ValueError(
                "control_period must be an integer multiple of physics_timestep"
            )
        offsets = tuple(guidance_query_offsets)
        if (
            not offsets
            or any(not isinstance(offset, int) for offset in offsets)
            or len(set(offsets)) != len(offsets)
        ):
            raise ValueError("guidance_query_offsets must be unique integers")
        self._world = world
        self._tracker = tracker
        self._control_period = period
        self._substeps = substeps
        self._query_offsets = offsets
        self._past_context_steps = max(0, -min(offsets))
        self._future_context_steps = max(0, max(offsets))

    @property
    def past_context_steps(self) -> int:
        """Executed guidance frames required before a candidate window."""
        return self._past_context_steps

    @property
    def future_context_steps(self) -> int:
        """Nominal guidance frames required after a full candidate window."""
        return self._future_context_steps

    def execute(
        self,
        execution_state: ExecutionState,
        guidance: MotionGuidance,
        *,
        num_steps: int | None = None,
        right_context: MotionGuidance | None = None,
        requires: frozenset[MeasurementPrimitive] = STATE_MEASUREMENT_PRIMITIVES,
    ) -> ExecutionResult:
        """Execute and retain the restart state and physical trace."""
        measurements, qpos_trace, qvel_trace, command, output_trace = self._run(
            execution_state,
            guidance,
            num_steps=num_steps,
            right_context=right_context,
            requires=requires,
            capture_trace=True,
        )
        assert measurements is not None
        steps = measurements.num_steps
        qpos, qvel, times = self._world.physical_state()
        if not np.allclose(times, times[0], rtol=0.0, atol=1e-6):
            raise RuntimeError("simulation lanes disagree on time")
        host_command = (
            self._world.host_command()
            if isinstance(command, DevicePDCommand)
            else command
        )
        final_execution_state = ExecutionState(
            qpos=_owned(qpos),
            qvel=_owned(qvel),
            control_step=execution_state.control_step + steps,
            time=float(times[0]),
            guidance_index=execution_state.guidance_index + steps,
            command=host_command,
            tracker_state=self._tracker.capture_state(),
            guidance_history=self._advance_history(
                execution_state.guidance_history, guidance, steps
            ),
        )
        return ExecutionResult(
            measurements=measurements,
            qpos=_owned(np.stack(qpos_trace, axis=1)),
            qvel=_owned(np.stack(qvel_trace, axis=1)),
            final_execution_state=final_execution_state,
            policy_outputs=_stack_policy_outputs(output_trace),
        )

    def rollout(
        self,
        execution_state: ExecutionState,
        guidance: MotionGuidance,
        *,
        num_steps: int | None = None,
        right_context: MotionGuidance | None = None,
        requires: frozenset[MeasurementPrimitive] = STATE_MEASUREMENT_PRIMITIVES,
    ) -> ControlStepMeasurements:
        """Evaluate a Candidate Rollout without downloading restart or traces."""
        measurements, _, _, _, _ = self._run(
            execution_state,
            guidance,
            num_steps=num_steps,
            right_context=right_context,
            requires=requires,
            capture_trace=False,
        )
        assert measurements is not None
        return measurements

    def rollout_reduced(
        self,
        execution_state: ExecutionState,
        guidance: MotionGuidance,
        reducer: RolloutReducer,
        *,
        num_steps: int | None = None,
        right_context: MotionGuidance | None = None,
    ) -> np.ndarray:
        """Roll out into a compact device reducer without collecting a host trace."""
        measurements, _, _, _, _ = self._run(
            execution_state,
            guidance,
            num_steps=num_steps,
            right_context=right_context,
            requires=frozenset(),
            capture_trace=False,
            reducer=reducer,
        )
        assert measurements is None
        return reducer.finalize()

    def _run(
        self,
        execution_state: ExecutionState,
        guidance: MotionGuidance,
        *,
        num_steps: int | None,
        right_context: MotionGuidance | None,
        requires: frozenset[MeasurementPrimitive],
        capture_trace: bool,
        reducer: RolloutReducer | None = None,
    ) -> tuple[
        ControlStepMeasurements | None,
        list[np.ndarray],
        list[np.ndarray],
        PDCommand | DevicePDCommand | None,
        list[dict[str, np.ndarray]],
    ]:
        horizon = len(guidance.query_offsets)
        if num_steps is not None and (
            not isinstance(num_steps, (int, np.integer)) or num_steps < 1
        ):
            raise ValueError("num_steps must be a positive integer")
        steps = horizon if num_steps is None else int(num_steps)
        if steps > horizon:
            raise ValueError("num_steps cannot exceed the candidate guidance horizon")
        if guidance.query_offsets != tuple(range(horizon)):
            raise ValueError("guidance must use a dense zero-based query grid")
        if guidance.joint_position.shape[0] != self._world.batch_size:
            raise ValueError("guidance batch size must match the world")
        if (
            guidance.joint_names != self._world.joint_names
            or guidance.body_names != self._world.body_names
        ):
            raise ValueError("guidance joint/body names must match the world")
        self._validate_history(execution_state, guidance)
        self._validate_right_context(right_context, guidance, steps)

        self._world.reset(execution_state)
        self._tracker.reset(execution_state.tracker_state)
        prepared = isinstance(self._tracker, PreparedDeviceTracker)
        if prepared:
            self._tracker.prepare(
                execution_state.guidance_history,
                guidance,
                right_context,
                num_steps=steps,
            )
        buffer = (
            None
            if reducer is not None
            else MeasurementBuffer(
                frozenset(requires),
                self._world.joint_names,
                self._world.body_names,
                self._world.batch_size,
                steps,
            )
        )
        qpos_trace: list[np.ndarray] = []
        qvel_trace: list[np.ndarray] = []
        output_trace: list[dict[str, np.ndarray]] = []
        command = execution_state.command
        state_view = None if prepared else self._world.state_view
        for step_index in range(steps):
            if prepared:
                command = self._tracker.step_prepared(
                    self._world.state_view, step_index
                )
            else:
                assert state_view is not None
                control_step_guidance = self._slice_guidance(
                    execution_state.guidance_history,
                    guidance,
                    right_context,
                    step_index,
                )
                command = self._tracker.step(state_view, control_step_guidance)
            self._world.apply_pd(command)
            if self._world.kinematic_joint_names:
                self._world.apply_kinematic_frame(
                    execution_state.control_step + step_index
                )
            self._world.step(self._substeps)
            if reducer is not None:
                reducer.record(step_index, self._world.state_view)
            if buffer is not None or not prepared:
                host_state = self._world.robot_state()
                if buffer is not None:
                    buffer.record(
                        step_index,
                        host_state,
                        command=_command_for_buffer(buffer, command, self._world),
                    )
                if not prepared:
                    state_view = host_state
            if capture_trace:
                qpos, qvel, _ = self._world.physical_state()
                qpos_trace.append(qpos)
                qvel_trace.append(qvel)
                captured = _capture_policy_outputs(self._tracker)
                if captured is not None:
                    output_trace.append(captured)

        measurements = None if buffer is None else buffer.finalize()
        return measurements, qpos_trace, qvel_trace, command, output_trace

    def _validate_history(
        self, execution_state: ExecutionState, guidance: MotionGuidance
    ) -> None:
        if self._past_context_steps == 0:
            return
        history = execution_state.guidance_history
        if history is None or len(history.query_offsets) < self._past_context_steps:
            raise ValueError(
                "execution_state guidance_history must contain at least "
                f"{self._past_context_steps} steps"
            )
        if (
            history.joint_position.shape[0] != self._world.batch_size
            or history.joint_names != guidance.joint_names
            or history.body_names != guidance.body_names
        ):
            raise ValueError("guidance_history must match guidance batch and names")

    def _validate_right_context(
        self,
        right_context: MotionGuidance | None,
        guidance: MotionGuidance,
        steps: int,
    ) -> None:
        required = max(
            0,
            steps + self._future_context_steps - len(guidance.query_offsets),
        )
        if required == 0:
            return
        if right_context is None or len(right_context.query_offsets) < required:
            raise ValueError(
                f"right_context must contain at least {required} nominal steps"
            )
        context_steps = len(right_context.query_offsets)
        if right_context.query_offsets != tuple(range(context_steps)):
            raise ValueError("right_context must use a dense zero-based query grid")
        if (
            right_context.joint_position.shape[0] != self._world.batch_size
            or right_context.joint_names != guidance.joint_names
            or right_context.body_names != guidance.body_names
        ):
            raise ValueError("right_context must match guidance batch and names")

    def _slice_guidance(
        self,
        history: MotionGuidance | None,
        guidance: MotionGuidance,
        right_context: MotionGuidance | None,
        step_index: int,
    ) -> MotionGuidance:
        sources: list[tuple[MotionGuidance, int]] = []
        horizon = len(guidance.query_offsets)
        for offset in self._query_offsets:
            index = step_index + offset
            if index < 0:
                assert history is not None
                sources.append((history, len(history.query_offsets) + index))
            elif index < horizon:
                sources.append((guidance, index))
            else:
                assert right_context is not None
                sources.append((right_context, index - horizon))
        return MotionGuidance(
            query_offsets=self._query_offsets,
            joint_names=guidance.joint_names,
            body_names=guidance.body_names,
            **{
                name: np.stack(
                    [getattr(source, name)[:, index] for source, index in sources],
                    axis=1,
                )
                for name in _GUIDANCE_ARRAYS
            },
        )

    def _advance_history(
        self,
        history: MotionGuidance | None,
        guidance: MotionGuidance,
        steps: int,
    ) -> MotionGuidance | None:
        if self._past_context_steps == 0:
            return None
        assert history is not None
        arrays = {}
        for name in _GUIDANCE_ARRAYS:
            combined = np.concatenate(
                (getattr(history, name), getattr(guidance, name)[:, :steps]), axis=1
            )
            arrays[name] = _owned(combined[:, -self._past_context_steps :])
        return MotionGuidance(
            query_offsets=tuple(range(-self._past_context_steps, 0)),
            joint_names=guidance.joint_names,
            body_names=guidance.body_names,
            **arrays,
        )


def _command_for_buffer(
    buffer: MeasurementBuffer,
    command: PDCommand | DevicePDCommand | None,
    world: MjwarpWorld,
) -> PDCommand | DevicePDCommand | None:
    if not buffer.requires_command or command is None:
        return None
    if isinstance(command, DevicePDCommand):
        return world.host_command()
    return command
