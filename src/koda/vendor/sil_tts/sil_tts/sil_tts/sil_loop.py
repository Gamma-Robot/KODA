"""Synchronous simulation-in-the-loop receding-horizon execution."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from sil_planners import SamplingPlanner
from sil_trackers import (
    MOTION_GUIDANCE_ARRAY_FIELDS as _GUIDANCE_ARRAYS,
    MotionGuidance,
)
from sil_tts.motion_tracking.search_space import KnotSearchSpace
from sil_tts.planning.evaluation import EvaluationLedger, RolloutEvaluator
from sil_tts.planning.update import PlanningUpdateResult, run_planning_update
from sil_tts.records import ExecutionTrace, PlanningDiagnostics
from sil_tts.simulation.guided_execution import GuidanceExecutor, ExecutionResult
from sil_tts.simulation.measurements import (
    STATE_MEASUREMENT_PRIMITIVES,
    ControlStepMeasurements,
)
from sil_tts.simulation.execution_state import ExecutionState


SearchSpaceFactory = Callable[[MotionGuidance], KnotSearchSpace]
EvaluatorFactory = Callable[
    [KnotSearchSpace, MotionGuidance, MotionGuidance | None, ExecutionState],
    RolloutEvaluator,
]


@dataclass(frozen=True)
class SilLoopResult:
    """Minimal optimized-run payload consumed by ``ExecutionRunArtifact``."""

    trace: ExecutionTrace
    planning: tuple[PlanningDiagnostics, ...]


def run_sil_loop(
    *,
    nominal_future: MotionGuidance,
    initial_execution_state: ExecutionState,
    planner: SamplingPlanner,
    search_space_factory: SearchSpaceFactory,
    evaluator_factory: EvaluatorFactory,
    executor: GuidanceExecutor,
    horizon_steps: int,
    execution_chunk_steps: int,
    control_period: float,
    seed: int,
    iterations: int,
    root_body_name: str,
    execution_steps: int | None = None,
    future_context_steps: int = 0,
    evaluate_proposal: bool = True,
) -> SilLoopResult:
    """Plan against the supplied nominal future and execute its declared prefix.

    ``nominal_future`` is batch-one guidance aligned with achieved post-step
    samples. For a target containing the reset pose at frame zero, callers pass
    ``target[1:]`` reindexed to a dense zero-based query grid. Every Planning
    Update receives a fixed-size edge-padded window, fresh search-space and
    evaluator instances, and a fresh planner initialization. The first update
    uses the planner's zero default; later updates warm-start from the previous
    selected latent shifted by the number of executed chunk steps.

    ``execution_steps`` limits executed/evaluated Control Steps while later frames in
    ``nominal_future`` remain immutable planning and tracker context. When it
    is omitted, the complete supplied future is executed. ``future_context_steps``
    appends immutable nominal frames only at the tracker observation boundary.
    Candidate rollout and objective horizons remain exactly
    ``horizon_steps`` Control Steps.

    The returned candidate is the best finite-budget evaluated choice; this
    function makes no global-optimality claim.
    """
    total_steps = _validate_inputs(
        nominal_future=nominal_future,
        initial_execution_state=initial_execution_state,
        horizon_steps=horizon_steps,
        execution_chunk_steps=execution_chunk_steps,
        control_period=control_period,
        seed=seed,
        iterations=iterations,
        evaluate_proposal=evaluate_proposal,
        root_body_name=root_body_name,
        execution_steps=execution_steps,
        future_context_steps=future_context_steps,
    )
    period = float(control_period)
    horizon = int(horizon_steps)
    chunk_step_limit = int(execution_chunk_steps)
    context_steps = int(future_context_steps)

    execution_state = initial_execution_state
    cursor = 0
    update_index = 0
    previous_latent: np.ndarray | None = None
    previous_chunk_steps: int | None = None
    executed_guidance_segments: list[MotionGuidance] = []
    measurement_segments: list[ControlStepMeasurements] = []
    qpos_segments: list[np.ndarray] = []
    qvel_segments: list[np.ndarray] = []
    output_segments: dict[str, list[np.ndarray]] | None = None
    persisted: list[PlanningDiagnostics] = []

    while cursor < total_steps:
        nominal_window = _window(nominal_future, cursor, horizon)
        right_context = _right_context(
            nominal_future, cursor + horizon, context_steps
        )
        search_space = search_space_factory(nominal_window)
        expected_shape = (planner.num_knots, planner.dim)
        if tuple(search_space.latent_shape) != expected_shape:
            raise ValueError(
                "search-space latent shape must match planner shape "
                f"{expected_shape}, got {search_space.latent_shape}"
            )
        evaluator = evaluator_factory(
            search_space, nominal_window, right_context, execution_state
        )
        if previous_latent is None:
            initial_mean = None
        else:
            assert previous_chunk_steps is not None
            initial_mean = search_space.shift(
                previous_latent, previous_chunk_steps
            )
        # run_planning_update returns host-resident costs, so all work behind
        # them has synchronized before the clock stops.
        update_started = time.perf_counter()
        update = run_planning_update(
            planner,
            evaluator,
            seed=int(seed) + update_index,
            iterations=int(iterations),
            initial_mean=initial_mean,
            evaluate_proposal=evaluate_proposal,
        )
        update_wall_seconds = time.perf_counter() - update_started
        if _all_candidate_evaluations_penalized(update.diagnostics.evaluations):
            raise RuntimeError(
                f"planning update at step {cursor} penalized every candidate evaluation"
            )

        executed_chunk_steps = min(chunk_step_limit, total_steps - cursor)
        execution = executor.execute(
            execution_state,
            update.selected_guidance,
            num_steps=executed_chunk_steps,
            right_context=right_context,
            requires=STATE_MEASUREMENT_PRIMITIVES,
        )
        _validate_execution(
            execution, execution_state, executed_chunk_steps, period
        )

        executed_guidance_segments.append(
            _guidance_prefix(update.selected_guidance, executed_chunk_steps)
        )
        measurement_segments.append(execution.measurements)
        qpos_segments.append(np.asarray(execution.qpos)[0])
        qvel_segments.append(np.asarray(execution.qvel)[0])
        output_segments = _append_policy_outputs(
            output_segments,
            execution.policy_outputs,
            later_chunk=cursor > 0,
        )
        persisted.append(
            _planning_diagnostics(cursor, update, nominal_window, update_wall_seconds)
        )

        execution_state = execution.final_execution_state
        previous_latent = update.selected_latent
        previous_chunk_steps = executed_chunk_steps
        cursor += executed_chunk_steps
        update_index += 1

    executed_guidance = _concatenate_guidance(executed_guidance_segments)
    achieved = _concatenate_measurements(measurement_segments)
    timeline = initial_execution_state.time + period * np.arange(
        1, total_steps + 1, dtype=np.float64
    )
    trace = ExecutionTrace(
        timeline=timeline,
        nominal=_guidance_prefix(nominal_future, total_steps),
        executed_guidance=executed_guidance,
        achieved=achieved,
        qpos=np.concatenate(qpos_segments, axis=0),
        qvel=np.concatenate(qvel_segments, axis=0),
        valid=np.ones(total_steps, dtype=np.bool_),
        root_body_name=root_body_name,
        policy_outputs=_concatenate_policy_outputs(output_segments),
    )
    return SilLoopResult(trace=trace, planning=tuple(persisted))


def _validate_inputs(
    *,
    nominal_future: MotionGuidance,
    initial_execution_state: ExecutionState,
    horizon_steps: int,
    execution_chunk_steps: int,
    control_period: float,
    seed: int,
    iterations: int,
    evaluate_proposal: bool,
    root_body_name: str,
    execution_steps: int | None,
    future_context_steps: int,
) -> int:
    if not isinstance(nominal_future, MotionGuidance):
        raise TypeError("nominal_future must be MotionGuidance")
    available_steps = len(nominal_future.query_offsets)
    if available_steps < 1:
        raise ValueError("nominal_future must contain at least one future frame")
    if nominal_future.query_offsets != tuple(range(available_steps)):
        raise ValueError("nominal_future must use a dense zero-based query grid")
    if nominal_future.joint_position.shape[0] != 1:
        raise ValueError("nominal_future must contain exactly one execution lane")
    for field in _GUIDANCE_ARRAYS:
        if not np.isfinite(getattr(nominal_future, field)).all():
            raise ValueError(f"nominal_future {field} must be finite")
    if initial_execution_state.qpos.shape[0] != 1:
        raise ValueError("initial_execution_state must contain exactly one execution lane")
    if (
        not isinstance(root_body_name, str)
        or root_body_name not in nominal_future.body_names
    ):
        raise ValueError("root_body_name must name one nominal body")
    _positive_integer(horizon_steps, "horizon_steps", minimum=2)
    _positive_integer(execution_chunk_steps, "execution_chunk_steps")
    if execution_chunk_steps > horizon_steps:
        raise ValueError("execution_chunk_steps cannot exceed horizon_steps")
    period = float(control_period)
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("control_period must be finite and positive")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be a non-negative integer")
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    _positive_integer(iterations, "iterations")
    _positive_integer(future_context_steps, "future_context_steps", minimum=0)
    if execution_steps is None:
        total_steps = available_steps
    else:
        _positive_integer(execution_steps, "execution_steps")
        total_steps = int(execution_steps)
        if total_steps > available_steps:
            raise ValueError("execution_steps cannot exceed the supplied nominal future")
    if not isinstance(evaluate_proposal, bool):
        raise TypeError("evaluate_proposal must be a bool")
    return total_steps


def _positive_integer(value: int, field: str, *, minimum: int = 1) -> None:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{field} must be an integer >= {minimum}")


def _window(guidance: MotionGuidance, start: int, horizon: int) -> MotionGuidance:
    indices = np.minimum(np.arange(start, start + horizon), len(guidance.query_offsets) - 1)
    return _take_guidance(guidance, indices)


def _right_context(
    guidance: MotionGuidance, start: int, num_steps: int
) -> MotionGuidance | None:
    if num_steps == 0:
        return None
    context = _window(guidance, start, num_steps)
    arrays = {}
    for name in _GUIDANCE_ARRAYS:
        value = np.array(getattr(context, name), copy=True, order="C")
        value.flags.writeable = False
        arrays[name] = value
    return MotionGuidance(
        query_offsets=context.query_offsets,
        joint_names=context.joint_names,
        body_names=context.body_names,
        **arrays,
    )


def _guidance_prefix(guidance: MotionGuidance, num_steps: int) -> MotionGuidance:
    return _take_guidance(guidance, np.arange(num_steps))


def _take_guidance(guidance: MotionGuidance, indices: np.ndarray) -> MotionGuidance:
    return MotionGuidance(
        query_offsets=tuple(range(len(indices))),
        joint_names=guidance.joint_names,
        body_names=guidance.body_names,
        **{name: np.asarray(getattr(guidance, name))[:, indices] for name in _GUIDANCE_ARRAYS},
    )


def _all_candidate_evaluations_penalized(
    evaluations: tuple[EvaluationLedger, ...],
) -> bool:
    candidates = sum(record.candidate_evaluations for record in evaluations)
    penalized = sum(record.penalized_candidate_evaluations for record in evaluations)
    return candidates > 0 and penalized == candidates


def _validate_execution(
    execution: ExecutionResult,
    before: ExecutionState,
    num_steps: int,
    control_period: float,
) -> None:
    if (
        execution.measurements.batch != 1
        or execution.measurements.num_steps != num_steps
    ):
        raise RuntimeError("execution returned an incompatible measurement trace")
    if execution.qpos.ndim != 3 or execution.qpos.shape[:2] != (1, num_steps):
        raise RuntimeError("execution returned an incompatible qpos trace")
    if execution.qvel.ndim != 3 or execution.qvel.shape[:2] != (1, num_steps):
        raise RuntimeError("execution returned an incompatible qvel trace")
    if execution.policy_outputs is not None:
        for name, value in execution.policy_outputs.items():
            if np.asarray(value).shape[:2] != (1, num_steps):
                raise RuntimeError(
                    f"execution returned an incompatible policy output {name!r}"
                )
    after = execution.final_execution_state
    expected_time = before.time + num_steps * control_period
    clock_magnitude = np.float32(max(1.0, abs(after.time), abs(expected_time)))
    clock_tolerance = max(
        1e-4,
        2.0 * num_steps * float(np.spacing(clock_magnitude)),
    )
    time_matches = np.isclose(
        after.time,
        expected_time,
        rtol=0.0,
        # Four float32 physics increments per control step can accumulate up
        # to roughly two clock ULPs; control_step and guidance_index stay exact.
        atol=clock_tolerance,
    )
    if (
        after.qpos.shape[0] != 1
        or after.control_step != before.control_step + num_steps
        or after.guidance_index != before.guidance_index + num_steps
        or not time_matches
    ):
        raise RuntimeError("execution did not advance the complete execution state")


def _planning_diagnostics(
    update_step: int,
    update: PlanningUpdateResult,
    nominal: MotionGuidance,
    update_wall_seconds: float,
) -> PlanningDiagnostics:
    diagnostics = update.diagnostics
    quantiles = np.asarray(
        [record.cost_quantiles for record in diagnostics.evaluations],
        dtype=np.float64,
    )
    return PlanningDiagnostics(
        update_step=update_step,
        selected_source=diagnostics.selected_source,
        selected_candidate_id=_candidate_id(update.selected_latent),
        selected_cost=update.selected_cost,
        best_observed_candidate_id=_candidate_id(
            diagnostics.best_observed_candidate
        ),
        best_observed_cost=diagnostics.best_observed_cost,
        proposal_candidate_id=_candidate_id(diagnostics.proposal_candidate),
        proposal_cost=diagnostics.proposal_cost,
        cost_quantiles=quantiles,
        candidate_deformation=_candidate_deformation(
            update.selected_guidance, nominal
        ),
        candidate_evaluations=diagnostics.candidate_evaluations,
        scenario_count=diagnostics.scenario_count,
        scenario_digest=diagnostics.scenario_digest,
        logical_rollouts=diagnostics.logical_rollouts,
        physical_rollouts=diagnostics.physical_rollouts,
        penalized_candidate_evaluations=(
            diagnostics.penalized_candidate_evaluations
        ),
        invalid_scenario_rollouts=diagnostics.invalid_scenario_rollouts,
        update_wall_seconds=update_wall_seconds,
    )


def _candidate_id(candidate: np.ndarray) -> str:
    value = np.ascontiguousarray(candidate, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes())
    return f"sha256:{digest.hexdigest()}"


def _candidate_deformation(
    candidate: MotionGuidance, nominal: MotionGuidance
) -> dict[str, float]:
    joint_delta = candidate.joint_position - nominal.joint_position
    body_delta = candidate.body_position_world - nominal.body_position_world
    dot = np.abs(
        np.sum(
            candidate.body_orientation_world * nominal.body_orientation_world,
            axis=-1,
        )
    )
    angle = 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))
    return {
        "joint_position_rmse_rad": float(np.sqrt(np.mean(joint_delta**2))),
        "body_position_rmse_m": float(
            np.sqrt(np.mean(np.sum(body_delta**2, axis=-1)))
        ),
        "body_orientation_rmse_rad": float(np.sqrt(np.mean(angle**2))),
    }


def _concatenate_guidance(segments: list[MotionGuidance]) -> MotionGuidance:
    first = segments[0]
    return MotionGuidance(
        query_offsets=tuple(range(sum(len(segment.query_offsets) for segment in segments))),
        joint_names=first.joint_names,
        body_names=first.body_names,
        **{
            name: np.concatenate([getattr(segment, name) for segment in segments], axis=1)
            for name in _GUIDANCE_ARRAYS
        },
    )


def _concatenate_measurements(
    segments: list[ControlStepMeasurements],
) -> ControlStepMeasurements:
    first = segments[0]
    return ControlStepMeasurements(
        joint_names=first.joint_names,
        body_names=first.body_names,
        arrays={
            primitive: np.concatenate(
                [segment.arrays[primitive] for segment in segments], axis=1
            )
            for primitive in first.arrays
        },
    )


def _append_policy_outputs(
    segments: dict[str, list[np.ndarray]] | None,
    outputs: Mapping[str, np.ndarray] | None,
    *,
    later_chunk: bool,
) -> dict[str, list[np.ndarray]] | None:
    if outputs is None or not outputs:
        if segments is not None:
            raise RuntimeError("executed policy outputs missing from a later chunk")
        return None
    lane = {str(name): np.asarray(value)[0] for name, value in outputs.items()}
    if segments is None:
        if later_chunk:
            raise RuntimeError(
                "executed policy outputs missing from an earlier chunk"
            )
        return {name: [array] for name, array in lane.items()}
    if set(segments) != set(lane):
        raise RuntimeError("executed policy output names changed across chunks")
    for name, array in lane.items():
        segments[name].append(array)
    return segments


def _concatenate_policy_outputs(
    segments: dict[str, list[np.ndarray]] | None,
) -> dict[str, np.ndarray] | None:
    if segments is None:
        return None
    return {
        name: np.concatenate(chunks, axis=0) for name, chunks in segments.items()
    }
