"""Run one configured SIL experiment and expose the package console commands."""

from __future__ import annotations

import argparse
import copy
import dataclasses
from contextlib import ExitStack
from datetime import datetime, timezone
import inspect
import json
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sil_planners import SamplingPlanner, get_planner
from sil_trackers import (
    DeviceTrackerProvider,
    MOTION_GUIDANCE_ARRAY_FIELDS as _GUIDANCE_ARRAYS,
    MotionGuidance,
    PreparedDeviceTracker,
    SerialReferenceTracker,
    TrackerAdapter,
    TrackerInstanceConfig,
    get_family,
    instance_config_from_mapping,
)
from sil_trackers.onnx_policy import BatchedOrtCudaPolicy
from sil_tts.artifacts import (
    LOCK_RELATIVE,
    ArtifactEntry,
    default_repo_root,
    ensure_artifact,
    load_lock,
    resolve,
    sha256_file,
)
from sil_tts.config import (
    RESOLVED_EXPERIMENT_FILE,
    dataclass_from_dict,
    resolve_config,
    write_resolved_experiment,
)
from sil_tts.motion_tracking.metrics import compute_paired_metrics
from sil_tts.motion_tracking.device_cost import MotionTrackingCostReducer
from sil_tts.motion_tracking.objective import MotionTrackingObjective
from sil_tts.motion_tracking.link_pose_knots import LinkPoseKnotsSpace
from sil_tts.motion_tracking.search_space import (
    KnotScales,
    KnotSearchSpace,
    QposKnotsSpace,
    SCALEBFM_FAMILY,
    planner_latent_dim,
)
from sil_tts.motion_tracking.scaletrack_kinematics import (
    SCALETRACK_KINEMATICS_FILE,
    SCALETRACK_META_FILE,
    write_executed_scaletrack_kinematics,
)
from sil_tts.motion_tracking.scaletrack_reference import (
    apply_init_state_npz,
    load_scaletrack_guidance,
    scaletrack_clip_duration_seconds,
)
from sil_tts.motion_tracking.any4hdmi import (
    load_any4hdmi_target,
    resolve_any4hdmi_motion,
)
from sil_tts.motion_tracking.kinematics_npz import (
    executed_kinematics_arrays,
    load_hand_sidecar,
    write_optimized_kinematics,
)
from sil_tts.planning.evaluation import RolloutEvaluator
from sil_tts.records import (
    ExecutionRunArtifact,
    ExecutionTrace,
    PairedRunArtifact,
    PlanningDiagnostics,
)
from sil_tts.sil_loop import run_sil_loop
from sil_tts.simulation.guided_execution import GuidanceExecutor
from sil_tts.simulation.measurements import STATE_MEASUREMENT_PRIMITIVES
from sil_tts.simulation.mjwarp import MjwarpWorld
from sil_tts.simulation.randomization import (
    RobustnessConfig,
    ScenarioBatch,
    sample_scenarios,
)
from sil_tts.simulation.execution_state import ExecutionState, physical_state_digest
from sil_tts.tracker_preparation import prepare_tracker_artifact
from sil_tts.visualization import (
    ComparisonView,
    RenderConfig,
    RenderBindings,
    RenderTrack,
    render_mp4,
)


TRACE_FILE = "paired_traces.npz"
PLANNING_FILE = "planning.json"
REPORT_FILE = "report.json"
PROVENANCE_FILE = "provenance.json"
RENDER_FILE = "comparison.mp4"
RENDER_PROVENANCE_FILE = "render_provenance.json"
RANDOMIZATION_FILE = "randomization.json"
OPTIMIZED_KINEMATICS_FILE = "optimized_kinematics.npz"
LIVE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class TargetConfig:
    manifest_artifact: str
    motion_artifact: str
    motion_id: str
    clip_duration_seconds: float
    dataset_root: str | None = None

    def __post_init__(self) -> None:
        if self.dataset_root is not None:
            if not isinstance(self.dataset_root, str) or not self.dataset_root:
                raise ValueError("target.dataset_root must be a non-empty string")
        else:
            for field in ("manifest_artifact", "motion_artifact"):
                if not isinstance(getattr(self, field), str) or not getattr(self, field):
                    raise ValueError(f"target.{field} must be a non-empty string")
        if not isinstance(self.motion_id, str) or not self.motion_id:
            raise ValueError("target.motion_id must be a non-empty string")
        if not np.isfinite(self.clip_duration_seconds) or self.clip_duration_seconds <= 0.0:
            raise ValueError("target.clip_duration_seconds must be finite and positive")


@dataclass(frozen=True)
class PlanningConfig:
    horizon_steps: int
    execution_chunk_steps: int
    num_knots: int
    max_worlds: int
    failure_penalty: float
    seed: int
    evaluate_proposal: bool
    root_body_name: str
    scales: KnotScales

    def __post_init__(self) -> None:
        for field, minimum in (
            ("horizon_steps", 2),
            ("execution_chunk_steps", 1),
            ("num_knots", 2),
            ("max_worlds", 1),
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"planning.{field} must be an integer >= {minimum}")
        if self.execution_chunk_steps > self.horizon_steps:
            raise ValueError("planning.execution_chunk_steps cannot exceed horizon_steps")
        if not np.isfinite(self.failure_penalty) or self.failure_penalty <= 0.0:
            raise ValueError("planning.failure_penalty must be finite and positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("planning.seed must be a non-negative integer")
        if not isinstance(self.evaluate_proposal, bool):
            raise TypeError("planning.evaluate_proposal must be bool")
        if not isinstance(self.root_body_name, str) or not self.root_body_name:
            raise ValueError("planning.root_body_name must be non-empty")


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    name: str
    tracker: dict[str, Any]
    robot: dict[str, Any]
    planner: dict[str, Any]
    motion_tracking: dict[str, Any]
    target: TargetConfig
    planning: PlanningConfig
    visualization: RenderConfig
    robustness: RobustnessConfig = dataclasses.field(
        default_factory=RobustnessConfig
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != LIVE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"schema_version must be exactly {LIVE_SCHEMA_VERSION}"
            )
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("experiment name must be non-empty")
        for field in ("tracker", "robot", "planner", "motion_tracking"):
            if not isinstance(getattr(self, field), dict):
                raise TypeError(f"{field} config must be a mapping")


@dataclass(frozen=True)
class ExperimentResult:
    output_dir: Path
    pair: PairedRunArtifact


@dataclass(frozen=True)
class _TargetAssets:
    dataset_root: Path
    target_id: str
    target_digest: str


class _RolloutRuntimePool:
    """Own persistent execution resources for every requested batch capacity."""

    def __init__(
        self,
        *,
        world_factory: Callable[[int], MjwarpWorld],
        adapter: TrackerAdapter,
        tracker_config: TrackerInstanceConfig,
        policy_path: Path,
        device_index: int,
        objective: MotionTrackingObjective,
        control_period: float,
        guidance_query_offsets: tuple[int, ...],
    ) -> None:
        self._world_factory = world_factory
        self._adapter = adapter
        self._policy_path = policy_path
        self._objective = objective
        self._control_period = control_period
        self._guidance_query_offsets = guidance_query_offsets
        self._uses_device_tracker = isinstance(adapter, DeviceTrackerProvider)
        self._host_policy = (
            None
            if self._uses_device_tracker
            else _build_host_policy(policy_path, tracker_config, device_index)
        )
        self._worlds: dict[int, MjwarpWorld] = {}
        self._trackers: dict[int, PreparedDeviceTracker] = {}
        self._executors: dict[int, GuidanceExecutor] = {}
        self._reducers: dict[int, MotionTrackingCostReducer] = {}

    def world(self, batch: int) -> MjwarpWorld:
        if batch not in self._worlds:
            self._worlds[batch] = self._world_factory(batch)
        return self._worlds[batch]

    @property
    def implementation(self) -> str:
        return "device" if self._uses_device_tracker else "host-reference"

    @property
    def reducer_factory(self) -> Callable[[int], MotionTrackingCostReducer] | None:
        return self.reducer if self._uses_device_tracker else None

    def executor(self, batch: int) -> GuidanceExecutor:
        if batch not in self._executors:
            world = self.world(batch)
            if self._uses_device_tracker:
                tracker = self._adapter.build_device_tracker(
                    self._policy_path, world.state_view, world.command_view
                )
                if not isinstance(tracker, PreparedDeviceTracker):
                    raise TypeError(
                        "build_device_tracker must return a PreparedDeviceTracker"
                    )
                self._trackers[batch] = tracker
            else:
                assert self._host_policy is not None
                tracker = SerialReferenceTracker(
                    self._adapter, self._host_policy, batch_size=batch
                )
            self._executors[batch] = GuidanceExecutor(
                world,
                tracker,
                control_period=self._control_period,
                guidance_query_offsets=self._guidance_query_offsets,
            )
        return self._executors[batch]

    def reducer(self, batch: int) -> MotionTrackingCostReducer:
        if not self._uses_device_tracker:
            raise RuntimeError("host-reference runtime has no device cost reducer")
        if batch not in self._reducers:
            self.executor(batch)
            self._reducers[batch] = MotionTrackingCostReducer(
                self._objective,
                self._worlds[batch].state_view,
                self._worlds[batch].command_view,
            )
        return self._reducers[batch]

    def close(self) -> None:
        """Release policies before the world-owned CUDA buffers disappear."""
        error: Exception | None = None
        for tracker in reversed(tuple(self._trackers.values())):
            try:
                tracker.close()
            except Exception as exc:  # Preserve teardown of remaining capacities.
                if error is None:
                    error = exc
        self._trackers.clear()
        self._executors.clear()
        self._reducers.clear()
        self._worlds.clear()
        self._host_policy = None
        if error is not None:
            raise error


def load_experiment(
    path: str | Path, *, configs_root: str | Path | None = None
) -> tuple[dict[str, Any], ExperimentConfig]:
    repo = default_repo_root()
    root = repo / "configs" if configs_root is None else Path(configs_root)
    resolved = resolve_config(Path(path), root)
    config = dataclass_from_dict(ExperimentConfig, resolved)
    return resolved, config


def apply_experiment_overrides(
    resolved: Mapping[str, Any],
    *,
    dataset_root: str | Path | None = None,
    motion_id: str | None = None,
    clip_duration_seconds: float | None = None,
    visualization_enabled: bool | None = None,
) -> tuple[dict[str, Any], ExperimentConfig]:
    """Overlay local dataset / clip fields onto a resolved experiment mapping."""

    patched = copy.deepcopy(dict(resolved))
    target = dict(patched["target"])
    if dataset_root is not None:
        target["dataset_root"] = str(Path(dataset_root).expanduser().resolve())
    if motion_id is not None:
        if not isinstance(motion_id, str) or not motion_id:
            raise ValueError("motion_id must be a non-empty string")
        target["motion_id"] = motion_id
    if clip_duration_seconds is not None:
        target["clip_duration_seconds"] = float(clip_duration_seconds)
    patched["target"] = target
    if visualization_enabled is not None:
        visualization = dict(patched.get("visualization", {}))
        visualization["enabled"] = bool(visualization_enabled)
        patched["visualization"] = visualization
    return patched, dataclass_from_dict(ExperimentConfig, patched)


def build_planner(
    config: Mapping[str, Any], *, num_knots: int, dim: int
) -> tuple[SamplingPlanner, int]:
    """Construct any registered planner from a strict constructor mapping."""
    algorithm = config.get("algorithm")
    if not isinstance(algorithm, str) or not algorithm:
        raise ValueError("planner.algorithm must be a non-empty string")
    iterations = config.get("num_iterations")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("planner.num_iterations must be a positive integer")
    planner_type = get_planner(algorithm)
    accepted = inspect.signature(planner_type).parameters
    reserved = {"num_knots", "dim"}.intersection(config)
    if reserved:
        raise ValueError(
            "planner config cannot set search-space fields: "
            f"{sorted(reserved)}"
        )
    unknown = set(config).difference({"algorithm", "num_iterations", *accepted})
    if unknown:
        raise ValueError(
            f"unknown parameters for planner {algorithm!r}: {sorted(unknown)}"
        )
    values = {**dict(config), "num_knots": num_knots, "dim": dim}
    values.pop("algorithm", None)
    kwargs = {key: value for key, value in values.items() if key in accepted}
    return planner_type(**kwargs), iterations


def _build_host_policy(
    policy_path: Path,
    tracker_config: TrackerInstanceConfig,
    device_index: int,
):
    """Construct the host policy runtime for ONNX or TorchScript artifacts."""

    if policy_path.suffix == ".pt":
        from sil_trackers.torchscript_policy import BatchedTorchScriptPolicy

        return BatchedTorchScriptPolicy(
            policy_path, tracker_config, device_id=device_index
        )
    return BatchedOrtCudaPolicy(
        policy_path, tracker_config, device_id=device_index
    )


def run_experiment(
    experiment_path: str | Path,
    *,
    device_index: int = 0,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    configs_root: str | Path | None = None,
    dataset_root: str | Path | None = None,
    motion_id: str | None = None,
    clip_duration_seconds: float | None = None,
    visualization_enabled: bool | None = None,
    reference_npz: str | Path | None = None,
    init_state_npz: str | Path | None = None,
) -> ExperimentResult | None:
    """Execute one paired baseline/optimized experiment from committed YAML."""
    with ExitStack() as resources:
        return _execute_experiment(
            experiment_path,
            device_index=device_index,
            output_dir=output_dir,
            dry_run=dry_run,
            configs_root=configs_root,
            dataset_root=dataset_root,
            motion_id=motion_id,
            clip_duration_seconds=clip_duration_seconds,
            visualization_enabled=visualization_enabled,
            reference_npz=reference_npz,
            init_state_npz=init_state_npz,
            resources=resources,
        )


def _execute_experiment(
    experiment_path: str | Path,
    *,
    device_index: int,
    output_dir: str | Path | None,
    dry_run: bool,
    configs_root: str | Path | None,
    resources: ExitStack,
    dataset_root: str | Path | None = None,
    motion_id: str | None = None,
    clip_duration_seconds: float | None = None,
    visualization_enabled: bool | None = None,
    reference_npz: str | Path | None = None,
    init_state_npz: str | Path | None = None,
) -> ExperimentResult | None:
    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
        raise ValueError("device_index must be a non-negative integer")
    session_reference = (
        Path(reference_npz).expanduser().resolve() if reference_npz is not None else None
    )
    session_init = (
        Path(init_state_npz).expanduser().resolve() if init_state_npz is not None else None
    )
    if session_init is not None and session_reference is None:
        raise ValueError("init_state_npz requires reference_npz")
    if session_reference is not None:
        if not session_reference.is_file():
            raise FileNotFoundError(f"reference npz does not exist: {session_reference}")
        if session_init is not None and not session_init.is_file():
            raise FileNotFoundError(f"init-state npz does not exist: {session_init}")
        if clip_duration_seconds is None:
            clip_duration_seconds = scaletrack_clip_duration_seconds(session_reference)
    resolved, config = load_experiment(experiment_path, configs_root=configs_root)
    if any(
        value is not None
        for value in (
            dataset_root,
            motion_id,
            clip_duration_seconds,
            visualization_enabled,
        )
    ):
        resolved, config = apply_experiment_overrides(
            resolved,
            dataset_root=dataset_root,
            motion_id=motion_id,
            clip_duration_seconds=clip_duration_seconds,
            visualization_enabled=visualization_enabled,
        )
    tracker_config = instance_config_from_mapping(config.tracker)
    objective = dataclass_from_dict(MotionTrackingObjective, config.motion_tracking)
    latent_dim = planner_latent_dim(
        tracker_config.family, len(tracker_config.command_joint_names)
    )
    planner, iterations = build_planner(
        config.planner, num_knots=config.planning.num_knots, dim=latent_dim
    )
    _validate_cross_config(config, tracker_config)
    if dry_run:
        return None

    repo = default_repo_root()
    source_commit = _source_commit(repo)
    destination = (
        _default_output_dir(repo, config.name, source_commit)
        if output_dir is None
        else Path(output_dir)
    ).resolve()
    _prepare_output_dir(destination)
    write_resolved_experiment(resolved, destination)

    entries = load_lock(repo / LOCK_RELATIVE)
    policy_entry = resolve(tracker_config.artifacts["policy"], lock=entries)
    robot_entry = resolve(str(config.robot["artifact"]), lock=entries)
    policy_path = ensure_artifact(policy_entry.id, root=repo, lock=entries)
    robot_root = ensure_artifact(robot_entry.id, root=repo, lock=entries)
    if session_reference is None:
        target_assets = _resolve_target_assets(config.target, entries, repo)
    else:
        target_assets = _TargetAssets(
            session_reference.parent,
            f"scaletrack:{session_reference.name}",
            sha256_file(session_reference),
        )
    model_path = (robot_root / str(config.robot["entrypoint"])).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"robot entrypoint does not exist: {model_path}")

    adapter = get_family(tracker_config.family)(tracker_config)
    kinematic_names = (
        ()
        if session_reference is not None
        else _kinematic_joint_names(config.robot)
    )
    hand_qpos = None
    hand_qvel = None
    if kinematic_names:
        hand_qpos, hand_qvel = load_hand_sidecar(
            target_assets.dataset_root,
            config.target.motion_id,
            joint_names=kinematic_names,
        )

    def world(batch: int) -> MjwarpWorld:
        built = _build_world(
            model_path,
            config.robot,
            tracker_config,
            batch=batch,
            device_index=device_index,
            scenario_count=(
                config.robustness.num_scenarios
                if config.robustness.enabled and batch != 1
                else 1
            ),
            kinematic_joint_names=kinematic_names,
        )
        if kinematic_names:
            assert hand_qpos is not None
            assert hand_qvel is not None
            built.bind_kinematic_trajectory(hand_qpos, hand_qvel)
        return built

    runtimes = _RolloutRuntimePool(
        world_factory=world,
        adapter=adapter,
        tracker_config=tracker_config,
        policy_path=policy_path,
        device_index=device_index,
        objective=objective,
        control_period=tracker_config.control_period,
        guidance_query_offsets=tracker_config.guidance_query_offsets,
    )
    resources.callback(runtimes.close)
    execution_world = runtimes.world(1)
    if session_reference is not None:
        target = load_scaletrack_guidance(
            session_reference,
            joint_names=tracker_config.command_joint_names,
            body_names=tracker_config.guidance_body_names,
            control_period=tracker_config.control_period,
        )
    else:
        target = load_any4hdmi_target(
            target_assets.dataset_root,
            config.target.motion_id,
            joint_names=tracker_config.command_joint_names,
            body_names=tracker_config.guidance_body_names,
            control_period=tracker_config.control_period,
            projector=execution_world,
            clip_duration_seconds=None,
        )
    if len(target.query_offsets) < 2:
        raise ValueError("target must contain a reset frame and at least one future frame")
    target_qpos, target_qvel = execution_world.physical_trajectory(
        target, tracker_config.control_period
    )
    if kinematic_names:
        target_qpos, target_qvel = execution_world.overlay_kinematic_trajectory(
            target_qpos, target_qvel
        )
    if session_init is not None:
        init_qpos, init_qvel = apply_init_state_npz(
            session_init,
            qpos_template=target_qpos[0, 0],
            qvel_template=target_qvel[0, 0],
            joint_names=tracker_config.command_joint_names,
            joint_qpos_addresses=execution_world.command_qpos_addresses,
            joint_dof_addresses=execution_world.command_dof_addresses,
            root_qpos_address=execution_world.root_qpos_address,
            root_dof_address=execution_world.root_dof_address,
        )
        target_qpos = np.array(target_qpos, copy=True)
        target_qvel = np.array(target_qvel, copy=True)
        target_qpos[:, 0] = init_qpos
        target_qvel[:, 0] = init_qvel
    nominal_future = _take_guidance(target, np.arange(1, len(target.query_offsets)))
    execution_steps = round(
        config.target.clip_duration_seconds / tracker_config.control_period
    )
    if execution_steps > len(nominal_future.query_offsets):
        raise ValueError("target duration exceeds the available motion")
    execution_nominal = _take_guidance(
        nominal_future, np.arange(execution_steps)
    )
    past_context_steps = max(0, -min(tracker_config.guidance_query_offsets))
    future_context_steps = max(0, max(tracker_config.guidance_query_offsets))
    initial_execution_state = ExecutionState.capture(
        target_qpos[:, 0],
        target_qvel[:, 0],
        control_step=0,
        time=0.0,
        guidance_index=0,
        command=None,
        tracker_state=adapter.initial_state(1),
        adapter=adapter,
        guidance_history=_initial_guidance_history(target, past_context_steps),
    )

    execution = runtimes.executor(1)
    baseline_raw = execution.execute(
        initial_execution_state,
        execution_nominal,
        num_steps=execution_steps,
        right_context=_nominal_right_context(
            nominal_future, execution_steps, future_context_steps
        ),
        requires=STATE_MEASUREMENT_PRIMITIVES,
    )
    baseline_outputs = None
    if baseline_raw.policy_outputs:
        baseline_outputs = {
            name: np.asarray(value)[0]
            for name, value in baseline_raw.policy_outputs.items()
        }
    baseline_trace = ExecutionTrace(
        timeline=tracker_config.control_period
        * np.arange(1, execution_steps + 1, dtype=np.float64),
        nominal=execution_nominal,
        executed_guidance=execution_nominal,
        achieved=baseline_raw.measurements,
        qpos=baseline_raw.qpos[0],
        qvel=baseline_raw.qvel[0],
        valid=np.ones(execution_steps, dtype=np.bool_),
        root_body_name=config.planning.root_body_name,
        policy_outputs=baseline_outputs,
    )

    def search_space(reference: MotionGuidance) -> KnotSearchSpace:
        if tracker_config.family == SCALEBFM_FAMILY:
            return LinkPoseKnotsSpace(
                reference,
                num_knots=config.planning.num_knots,
                control_period=tracker_config.control_period,
                scales=config.planning.scales,
            )
        return QposKnotsSpace(
            reference,
            num_knots=config.planning.num_knots,
            control_period=tracker_config.control_period,
            scales=config.planning.scales,
            projector=execution_world,
            pelvis_body=config.planning.root_body_name,
        )

    def evaluator(
        space: KnotSearchSpace,
        nominal: MotionGuidance,
        right_context: MotionGuidance | None,
        execution_state: ExecutionState,
    ) -> RolloutEvaluator:
        scenarios = None
        scenario_configurer = None
        if config.robustness.enabled:
            num_joints = execution_state.qvel.shape[1] - 6
            if num_joints < 1:
                raise ValueError("robust evaluation requires one free root and joint DOFs")
            scenarios = sample_scenarios(
                config.robustness.ranges,
                count=config.robustness.num_scenarios,
                seed=config.robustness.scenario_seed + execution_state.guidance_index,
                num_joints=num_joints,
            )

            def scenario_configurer(batch: int, values: ScenarioBatch) -> None:
                runtimes.world(batch).configure_scenarios(values)

        return RolloutEvaluator(
            search_space=space,
            objective=objective,
            nominal=nominal,
            execution_state=execution_state,
            tracker_adapter=adapter,
            executor_factory=runtimes.executor,
            max_worlds=config.planning.max_worlds,
            failure_penalty=config.planning.failure_penalty,
            right_context=right_context,
            reducer_factory=runtimes.reducer_factory,
            scenarios=scenarios,
            risk_strategy=config.robustness.risk,
            scenario_configurer=scenario_configurer,
        )

    optimized = run_sil_loop(
        nominal_future=nominal_future,
        initial_execution_state=initial_execution_state,
        planner=planner,
        search_space_factory=search_space,
        evaluator_factory=evaluator,
        executor=execution,
        horizon_steps=config.planning.horizon_steps,
        execution_chunk_steps=config.planning.execution_chunk_steps,
        control_period=tracker_config.control_period,
        seed=config.planning.seed,
        iterations=iterations,
        root_body_name=config.planning.root_body_name,
        execution_steps=execution_steps,
        future_context_steps=future_context_steps,
        evaluate_proposal=config.planning.evaluate_proposal,
    )

    identities = dict(
        target_id=target_assets.target_id,
        target_digest=target_assets.target_digest,
        tracker_id=tracker_config.instance,
        checkpoint_digest=policy_entry.sha256,
        robot_model_digest=robot_entry.tree_sha256 or robot_entry.sha256,
        initial_physical_state_digest=physical_state_digest(
            initial_execution_state.qpos, initial_execution_state.qvel
        ),
        control_period=tracker_config.control_period,
    )
    pair = PairedRunArtifact(
        baseline=ExecutionRunArtifact(
            kind="baseline", trace=baseline_trace, planning=(), **identities
        ),
        optimized=ExecutionRunArtifact(
            kind="optimized",
            trace=optimized.trace,
            planning=optimized.planning,
            **identities,
        ),
    )
    metrics = compute_paired_metrics(pair)
    nominal_qpos = target_qpos[0, 1 : execution_steps + 1]
    _persist_experiment(
        destination,
        pair=pair,
        metrics=metrics,
        nominal_qpos=nominal_qpos,
        source_commit=source_commit,
        source_dirty=_source_dirty(repo),
        model_path=model_path,
        target_sha256=target_assets.target_digest,
        config=config,
        device_index=device_index,
        tracker_runtime=runtimes.implementation,
        world=execution_world,
        initial_qpos=np.asarray(initial_execution_state.qpos)[0],
        initial_qvel=np.asarray(initial_execution_state.qvel)[0],
    )
    if config.visualization.enabled:
        _render_experiment(destination, model_path=model_path)
    return ExperimentResult(destination, pair)


def _validate_cross_config(config: ExperimentConfig, tracker: Any) -> None:
    joints = tuple(config.robot.get("joint_names", ()))
    if joints != tracker.command_joint_names:
        raise ValueError("robot and tracker command joint order must match exactly")
    if config.planning.root_body_name not in tracker.guidance_body_names:
        raise ValueError("planning root body must be part of tracker guidance bodies")
    if (
        config.robustness.enabled
        and config.planning.max_worlds < config.robustness.num_scenarios
    ):
        raise ValueError("planning.max_worlds must fit one complete scenario group")
    duration_steps = round(config.target.clip_duration_seconds / tracker.control_period)
    if duration_steps < 1 or not np.isclose(
        config.target.clip_duration_seconds,
        duration_steps * tracker.control_period,
        rtol=1e-9,
        atol=1e-12,
    ):
        raise ValueError("target.clip_duration_seconds must align with tracker control_period")


def _resolve_target_assets(
    config: TargetConfig,
    entries: Mapping[str, ArtifactEntry],
    repo: Path,
) -> _TargetAssets:
    if config.dataset_root is not None:
        dataset_root = Path(config.dataset_root).expanduser().resolve()
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"target dataset_root does not exist: {dataset_root}")
        motion_path = resolve_any4hdmi_motion(dataset_root, config.motion_id)
        return _TargetAssets(
            dataset_root,
            f"local:{config.motion_id}",
            sha256_file(motion_path),
        )
    manifest_entry = resolve(config.manifest_artifact, lock=entries)
    motion_entry = resolve(config.motion_artifact, lock=entries)
    manifest_path = ensure_artifact(manifest_entry.id, root=repo, lock=entries)
    motion_path = ensure_artifact(motion_entry.id, root=repo, lock=entries)

    if manifest_entry.kind == motion_entry.kind == "file":
        dataset_root = manifest_path.parent.resolve()
        if manifest_path.resolve() != dataset_root / "manifest.json":
            raise ValueError("target manifest artifact must be manifest.json")
        expected_motion = resolve_any4hdmi_motion(dataset_root, config.motion_id)
        if motion_path.resolve() != expected_motion:
            raise ValueError("target artifact path does not match target.motion_id")
        return _TargetAssets(dataset_root, motion_entry.id, motion_entry.sha256)

    if manifest_entry.kind == motion_entry.kind == "tar.gz":
        manifest_root = manifest_path.resolve()
        motion_root = motion_path.resolve()
        if manifest_root != motion_root:
            raise ValueError("target tree artifacts must resolve to the same dataset root")
        resolve_any4hdmi_motion(motion_root, config.motion_id)
        if motion_entry.tree_sha256 is None:
            raise ValueError("target tree artifact requires tree_sha256")
        return _TargetAssets(
            motion_root,
            f"{motion_entry.id}:{config.motion_id}",
            motion_entry.tree_sha256,
        )

    raise ValueError("target manifest and motion artifacts must both be files or trees")


def _build_world(
    model_path: Path,
    robot: Mapping[str, Any],
    tracker: Any,
    *,
    batch: int,
    device_index: int,
    scenario_count: int = 1,
    kinematic_joint_names: tuple[str, ...] | None = None,
) -> MjwarpWorld:
    return MjwarpWorld(
        model_path,
        batch_size=batch,
        joint_names=tracker.command_joint_names,
        body_names=tracker.guidance_body_names,
        physics_timestep=float(robot["physics_timestep"]),
        joint_effort_limit=np.asarray(robot["joint_effort_limit"], dtype=np.float64),
        floor_friction=tuple(robot["floor_friction"]),
        joint_armature=np.asarray(robot["joint_armature"], dtype=np.float64),
        joint_frictionloss=np.asarray(robot["joint_frictionloss"], dtype=np.float64),
        kinematic_joint_names=(
            _kinematic_joint_names(robot)
            if kinematic_joint_names is None
            else kinematic_joint_names
        ),
        guidance_velocity_stencil=tracker.guidance_velocity_stencil,
        scenario_count=scenario_count,
        device=f"cuda:{device_index}",
        nconmax=int(robot["nconmax"]),
        njmax=int(robot["njmax"]),
    )


def _kinematic_joint_names(robot: Mapping[str, Any]) -> tuple[str, ...]:
    names = robot.get("kinematic_joint_names", ())
    if names in (None, ()):
        return ()
    if not isinstance(names, (list, tuple)) or any(
        not isinstance(name, str) or not name for name in names
    ):
        raise ValueError("robot.kinematic_joint_names must be a list of non-empty strings")
    return tuple(names)


def _declared_policy_output_names(tracker: Mapping[str, Any]) -> tuple[str, ...]:
    specs = tracker.get("policy_outputs", ())
    if specs in (None, ()):
        return ()
    if not isinstance(specs, (list, tuple)):
        raise ValueError("tracker.policy_outputs must be a list")
    names: list[str] = []
    for spec in specs:
        if not isinstance(spec, Mapping) or not spec.get("name"):
            raise ValueError("tracker.policy_outputs entries must have a name")
        names.append(str(spec["name"]))
    return tuple(names)


def _take_guidance(guidance: MotionGuidance, indices: np.ndarray) -> MotionGuidance:
    return MotionGuidance(
        query_offsets=tuple(range(len(indices))),
        joint_names=guidance.joint_names,
        body_names=guidance.body_names,
        **{field: getattr(guidance, field)[:, indices] for field in _GUIDANCE_ARRAYS},
    )


def _initial_guidance_history(
    target: MotionGuidance, num_steps: int
) -> MotionGuidance | None:
    if num_steps == 0:
        return None
    reset = _take_guidance(target, np.zeros(num_steps, dtype=np.int64))
    return MotionGuidance(
        query_offsets=tuple(range(-num_steps, 0)),
        joint_names=reset.joint_names,
        body_names=reset.body_names,
        **{field: getattr(reset, field) for field in _GUIDANCE_ARRAYS},
    )


def _nominal_right_context(
    guidance: MotionGuidance, start: int, num_steps: int
) -> MotionGuidance | None:
    if num_steps == 0:
        return None
    indices = np.minimum(
        np.arange(start, start + num_steps),
        len(guidance.query_offsets) - 1,
    )
    context = _take_guidance(guidance, indices)
    arrays = {}
    for field in _GUIDANCE_ARRAYS:
        value = np.array(getattr(context, field), copy=True, order="C")
        value.flags.writeable = False
        arrays[field] = value
    return MotionGuidance(
        query_offsets=context.query_offsets,
        joint_names=context.joint_names,
        body_names=context.body_names,
        **arrays,
    )


def _prepare_output_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(
                f"experiment output path is not a directory: {path}"
            )
        if any(path.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite non-empty experiment directory: {path}"
            )
    path.mkdir(parents=True, exist_ok=True)


def _default_output_dir(repo: Path, experiment: str, source_commit: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return repo / "outputs" / experiment / f"{timestamp}-{source_commit[:8]}"


def _persist_experiment(
    output: Path,
    *,
    pair: PairedRunArtifact,
    metrics: Any,
    nominal_qpos: np.ndarray,
    source_commit: str,
    source_dirty: bool,
    model_path: Path,
    target_sha256: str,
    config: ExperimentConfig,
    device_index: int,
    tracker_runtime: str,
    world: MjwarpWorld,
    initial_qpos: np.ndarray,
    initial_qvel: np.ndarray,
) -> None:
    declared_outputs = _declared_policy_output_names(config.tracker)
    if "motion_token" in declared_outputs:
        outputs = pair.optimized.trace.policy_outputs
        if outputs is None or "motion_token" not in outputs:
            raise ValueError(
                "tracker declares motion_token but the executed optimized "
                "trace does not contain it"
            )
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.int64(LIVE_SCHEMA_VERSION),
        "nominal_qpos": nominal_qpos,
    }
    nominal = pair.baseline.trace.nominal
    for field in _GUIDANCE_ARRAYS:
        arrays[f"nominal_{field}"] = getattr(nominal, field)
    for label, run in (("baseline", pair.baseline), ("optimized", pair.optimized)):
        trace = run.trace
        arrays[f"{label}_timeline"] = trace.timeline
        arrays[f"{label}_valid"] = trace.valid
        arrays[f"{label}_qpos"] = trace.qpos
        arrays[f"{label}_qvel"] = trace.qvel
        if (
            trace.policy_outputs is not None
            and str(config.tracker.get("family")) != SCALEBFM_FAMILY
        ):
            for name, value in trace.policy_outputs.items():
                arrays[f"{label}_policy_output_{name}"] = value
        for field in _GUIDANCE_ARRAYS:
            arrays[f"{label}_executed_guidance_{field}"] = getattr(
                trace.executed_guidance, field
            )
        if trace.achieved is not None:
            for primitive, value in trace.achieved.arrays.items():
                arrays[f"{label}_achieved_{primitive.value}"] = value
    np.savez_compressed(output / TRACE_FILE, **arrays)
    kinematics_path = None
    if "motion_token" in declared_outputs:
        assert pair.optimized.trace.policy_outputs is not None
        kinematics_path = write_optimized_kinematics(
            output / OPTIMIZED_KINEMATICS_FILE,
            executed_kinematics_arrays(
                qpos=pair.optimized.trace.qpos,
                qvel=pair.optimized.trace.qvel,
                timeline=pair.optimized.trace.timeline,
                initial_qpos=initial_qpos,
                initial_qvel=initial_qvel,
                joint_names=world.joint_names,
                joint_qpos_addresses=world.command_qpos_addresses,
                joint_dof_addresses=world.command_dof_addresses,
                hand_joint_names=world.kinematic_joint_names,
                hand_qpos_addresses=world.kinematic_qpos_addresses,
                hand_dof_addresses=world.kinematic_dof_addresses,
                root_qpos_address=world.root_qpos_address,
                root_dof_address=world.root_dof_address,
                motion_token=pair.optimized.trace.policy_outputs["motion_token"],
                fps=int(round(1.0 / pair.optimized.control_period)),
            ),
        )
    elif str(config.tracker.get("family")) == SCALEBFM_FAMILY:
        kinematics_path = write_executed_scaletrack_kinematics(
            output / SCALETRACK_KINEMATICS_FILE,
            meta_path=output / SCALETRACK_META_FILE,
            initial_qpos=initial_qpos,
            initial_qvel=initial_qvel,
            post_qpos=pair.optimized.trace.qpos,
            post_qvel=pair.optimized.trace.qvel,
            source_joint_names=world.joint_names,
            source_joint_qpos_addresses=world.command_qpos_addresses,
            source_joint_dof_addresses=world.command_dof_addresses,
            source_root_qpos_address=world.root_qpos_address,
            source_root_dof_address=world.root_dof_address,
            extra_meta={"mpc_backend": "sil-tts"},
        )
    planning_path = output / PLANNING_FILE
    _write_json(
        planning_path,
        {
            "schema_version": LIVE_SCHEMA_VERSION,
            "updates": [
                _planning_json(item) for item in pair.optimized.planning
            ],
        },
    )
    randomization_path = output / RANDOMIZATION_FILE
    _write_json(
        randomization_path,
        _randomization_json(config.robustness, pair.optimized.planning, config.robot),
    )
    report = {
        "schema_version": LIVE_SCHEMA_VERSION,
        "claim_scope": "paired configured-clip tracking; no global-optimality claim",
        "baseline": dataclasses.asdict(metrics.baseline),
        "optimized": dataclasses.asdict(metrics.optimized),
        "identities": {
            "target_id": pair.baseline.target_id,
            "target_digest": pair.baseline.target_digest,
            "tracker_id": pair.baseline.tracker_id,
            "checkpoint_digest": pair.baseline.checkpoint_digest,
            "robot_model_digest": pair.baseline.robot_model_digest,
            "initial_physical_state_digest": (
                pair.baseline.initial_physical_state_digest
            ),
            "control_period": pair.baseline.control_period,
        },
    }
    _write_json(output / REPORT_FILE, report)
    provenance = {
        "schema_version": LIVE_SCHEMA_VERSION,
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "target_sha256": target_sha256,
        "mjcf_path": str(model_path),
        "mjcf_sha256": sha256_file(model_path),
        "trace_sha256": sha256_file(output / TRACE_FILE),
        "report_sha256": sha256_file(output / REPORT_FILE),
        "planning_sha256": sha256_file(planning_path),
        "randomization_sha256": sha256_file(randomization_path),
        "resolved_experiment_sha256": sha256_file(
            output / RESOLVED_EXPERIMENT_FILE
        ),
        "runtime_versions": {
            package: version(package)
            for package in ("mujoco", "mujoco-warp", "warp-lang", "onnxruntime-gpu")
        },
        "tracker_runtime": tracker_runtime,
        "mpc_backend": (
            "sil-tts"
            if str(config.tracker.get("family")) == SCALEBFM_FAMILY
            else "sil"
        ),
        "hardware": _hardware_identity(device_index),
        "visualization": dataclasses.asdict(config.visualization),
    }
    if kinematics_path is not None:
        provenance["optimized_kinematics_sha256"] = sha256_file(kinematics_path)
    _write_json(output / PROVENANCE_FILE, provenance)


def _hardware_identity(device_index: int) -> dict[str, Any]:
    """Identify the selected logical CUDA device through ``nvidia-smi``."""
    visible_raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    base: dict[str, Any] = {
        "logical_device_index": device_index,
        "cuda_visible_devices": visible_raw,
    }
    if visible_raw is None:
        selector = str(device_index)
    else:
        visible = tuple(
            item.strip() for item in visible_raw.split(",") if item.strip()
        )
        if device_index >= len(visible):
            return {
                **base,
                "available": False,
                "error": "logical device is absent from CUDA_VISIBLE_DEVICES",
            }
        selector = visible[device_index]
    base["selector"] = selector
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                f"--id={selector}",
                "--query-gpu=index,uuid,name,compute_cap,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {
            **base,
            "available": False,
            "error": "nvidia-smi query failed",
        }
    rows = [line for line in output.splitlines() if line.strip()]
    fields = [item.strip() for item in rows[0].split(",")] if len(rows) == 1 else []
    if len(fields) != 5:
        return {
            **base,
            "available": False,
            "error": "unexpected nvidia-smi output",
        }
    physical_index, gpu_uuid, gpu_name, compute_capability, driver_version = fields
    return {
        **base,
        "available": True,
        "physical_device_index": int(physical_index),
        "gpu_uuid": gpu_uuid,
        "gpu_name": gpu_name,
        "compute_capability": compute_capability,
        "driver_version": driver_version,
    }


def _planning_json(item: PlanningDiagnostics) -> dict[str, Any]:
    return {
        "update_step": item.update_step,
        "selected_source": item.selected_source,
        "selected_candidate_id": item.selected_candidate_id,
        "selected_cost": item.selected_cost,
        "best_observed_candidate_id": item.best_observed_candidate_id,
        "best_observed_cost": item.best_observed_cost,
        "proposal_candidate_id": item.proposal_candidate_id,
        "proposal_cost": item.proposal_cost,
        "cost_quantiles": item.cost_quantiles.tolist(),
        "candidate_deformation": dict(item.candidate_deformation),
        "candidate_evaluations": item.candidate_evaluations,
        "scenario_count": item.scenario_count,
        "scenario_digest": item.scenario_digest,
        "logical_rollouts": item.logical_rollouts,
        "physical_rollouts": item.physical_rollouts,
        "penalized_candidate_evaluations": (
            item.penalized_candidate_evaluations
        ),
        "invalid_scenario_rollouts": item.invalid_scenario_rollouts,
        "update_wall_seconds": item.update_wall_seconds,
    }


def _randomization_json(
    config: RobustnessConfig,
    planning: tuple[PlanningDiagnostics, ...],
    robot: Mapping[str, Any],
) -> dict[str, Any]:
    updates: list[dict[str, Any]] = []
    for item in planning:
        expected = None
        if config.enabled:
            expected = sample_scenarios(
                config.ranges,
                count=config.num_scenarios,
                seed=config.scenario_seed + item.update_step,
                num_joints=len(tuple(robot["joint_names"])),
            ).digest
        if item.scenario_digest != expected:
            raise RuntimeError("persisted planning scenario identity is inconsistent")
        updates.append(
            {
                "update_step": item.update_step,
                "seed": (
                    None
                    if not config.enabled
                    else config.scenario_seed + item.update_step
                ),
                "scenario_digest": item.scenario_digest,
            }
        )
    return {
        "schema_version": LIVE_SCHEMA_VERSION,
        "enabled": config.enabled,
        "num_scenarios": config.num_scenarios,
        "scenario_seed": config.scenario_seed,
        "risk": config.risk,
        "ranges": dataclasses.asdict(config.ranges),
        "updates": updates,
    }


def _render_experiment(
    experiment_dir: Path, *, model_path: Path | None = None
) -> Path:
    resolved = yaml.safe_load(
        (experiment_dir / RESOLVED_EXPERIMENT_FILE).read_text()
    )
    experiment = dataclass_from_dict(ExperimentConfig, resolved)
    for name in (REPORT_FILE, PLANNING_FILE, RANDOMIZATION_FILE):
        payload = json.loads((experiment_dir / name).read_text())
        _require_schema_version(payload, name)
    provenance = json.loads((experiment_dir / PROVENANCE_FILE).read_text())
    _require_schema_version(provenance, PROVENANCE_FILE)
    if model_path is None:
        robot = experiment.robot
        root = ensure_artifact(str(robot["artifact"]), root=default_repo_root())
        model_path = root / str(robot["entrypoint"])
    with np.load(experiment_dir / TRACE_FILE, allow_pickle=False) as traces:
        trace_schema_version = np.asarray(traces["schema_version"])
        if (
            trace_schema_version.shape != ()
            or not np.issubdtype(trace_schema_version.dtype, np.integer)
            or int(trace_schema_version) != LIVE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"{TRACE_FILE} schema_version must be exactly {LIVE_SCHEMA_VERSION}"
            )
        render = render_mp4(
            mjcf_path=model_path,
            timeline=traces["baseline_timeline"],
            nominal_qpos=traces["nominal_qpos"],
            view=ComparisonView(
                left=RenderTrack("baseline", traces["baseline_qpos"]),
                right=RenderTrack("optimized", traces["optimized_qpos"]),
            ),
            output_path=experiment_dir / RENDER_FILE,
            config=experiment.visualization,
            bindings=RenderBindings(
                source_commit=provenance["source_commit"],
                target_sha256=provenance["target_sha256"],
                trace_sha256=sha256_file(experiment_dir / TRACE_FILE),
                report_sha256=sha256_file(experiment_dir / REPORT_FILE),
                mjcf_sha256=sha256_file(model_path),
            ),
        )
    _write_json(
        experiment_dir / RENDER_PROVENANCE_FILE, dataclasses.asdict(render)
    )
    return experiment_dir / RENDER_FILE


def _require_schema_version(payload: object, name: str) -> None:
    if (
        not isinstance(payload, Mapping)
        or isinstance(payload.get("schema_version"), bool)
        or not isinstance(payload.get("schema_version"), int)
        or payload.get("schema_version") != LIVE_SCHEMA_VERSION
    ):
        raise ValueError(
            f"{name} schema_version must be exactly {LIVE_SCHEMA_VERSION}"
        )


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable evidence: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _source_commit(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()


def _source_dirty(repo: Path) -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo,
            check=True,
            text=True,
            capture_output=True,
        ).stdout
    )


def prepare_tracker(
    artifact_id: str = "wxy-wbc-policy.batched",
    *,
    root: str | Path | None = None,
    lock_path: str | Path | None = None,
) -> Path:
    """Rebuild one lock-declared derived tracker artifact."""
    return prepare_tracker_artifact(artifact_id, root=root, lock_path=lock_path)


def ingest_kinematics_dataset(
    source_root: str | Path,
    output_root: str | Path,
    *,
    robot_config: str | Path | None = None,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """Ingest any ``session_*/final_50hz/kinematics.npz`` tree into one dataset."""

    from sil_tts.motion_tracking.kinematics_npz import (
        DEFAULT_ROBOT_RELATIVE,
        ingest_session_kinematics,
        load_robot_joint_layout,
    )

    robot_path = (
        Path(robot_config).expanduser().resolve()
        if robot_config is not None
        else default_repo_root() / DEFAULT_ROBOT_RELATIVE
    )
    joints, hands, mjcf = load_robot_joint_layout(robot_path)
    output = Path(output_root).expanduser().resolve()
    name = dataset_name if dataset_name else output.name
    if not isinstance(name, str) or not name:
        raise ValueError("dataset_name must be a non-empty string")
    return ingest_session_kinematics(
        source_root,
        output,
        joint_names=joints,
        hand_joint_names=hands,
        dataset_name=name,
        mjcf=mjcf,
    )


def _add_ingest_parser(
    commands: Any, name: str, help_text: str
) -> argparse.ArgumentParser:
    ingest = commands.add_parser(name, help=help_text)
    ingest.add_argument(
        "source",
        type=Path,
        help="root directory containing session_*/final_50hz/kinematics.npz",
    )
    ingest.add_argument(
        "--output",
        type=Path,
        required=True,
        help="destination Any4HDMI dataset directory",
    )
    ingest.add_argument(
        "--robot",
        type=Path,
        help="robot YAML providing command and Dex3 joint names",
    )
    ingest.add_argument("--dataset-name", help="Any4HDMI dataset_name (default: output folder name)")
    return ingest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sil-tts")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="execute one experiment YAML")
    run.add_argument("experiment", type=Path)
    run.add_argument("--device-index", type=int, default=0)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--dataset-root", type=Path)
    run.add_argument("--motion-id")
    run.add_argument("--clip-duration-seconds", type=float)
    run.add_argument("--disable-visualization", action="store_true")
    batch = commands.add_parser("batch", help="execute an Experiment Batch YAML")
    batch.add_argument("experiment_batch", type=Path)
    batch.add_argument("--device-index", type=int, default=0)
    batch.add_argument("--output-dir", type=Path)
    batch.add_argument("--dry-run", action="store_true")
    render = commands.add_parser("render", help="render persisted paired traces")
    render.add_argument("experiment_dir", type=Path)
    prepare = commands.add_parser("prepare-tracker", help="prepare batched tracker ONNX")
    prepare.add_argument(
        "artifact_id",
        nargs="?",
        default="wxy-wbc-policy.batched",
        help="derived artifact identity from artifacts/lock.v1.json",
    )
    low_latency = commands.add_parser(
        "prepare-sonic-low-latency",
        help="import and fuse official low-latency SONIC ONNX models",
    )
    low_latency.add_argument("--encoder", type=Path, required=True)
    low_latency.add_argument("--decoder", type=Path, required=True)
    low_latency.add_argument("--observation-config", type=Path, required=True)
    commands.add_parser(
        "prepare-scalebfm-torso-umi3",
        help="export Torso-UMI-3 ScaleBFM TorchScript to batched float32 ONNX",
    )
    _add_ingest_parser(
        commands,
        "ingest-kinematics",
        "ingest session_*/final_50hz/kinematics.npz trees into one Any4HDMI dataset",
    )
    _add_ingest_parser(
        commands,
        "ingest-walk-and-pick",
        "alias of ingest-kinematics for the same session directory layout",
    )
    clips = commands.add_parser(
        "run-clips",
        help="run ingested clips on a multi-GPU flock queue",
    )
    clips.add_argument("experiment", type=Path)
    clips.add_argument(
        "--source-root",
        type=Path,
        help="root with session_*/final_50hz/kinematics.npz; ingested under output-root/_ingested",
    )
    clips.add_argument(
        "--dataset-root",
        type=Path,
        help="already-ingested Any4HDMI dataset; do not pass with --source-root",
    )
    clips.add_argument("--gpus", required=True, help="comma-separated GPU indices, e.g. 0,1,2,3")
    clips.add_argument("--output-root", type=Path)
    clips.add_argument(
        "--robot",
        type=Path,
        help="robot YAML used when ingesting --source-root (default: experiment robot.ref)",
    )
    clips.add_argument("--dry-run", action="store_true")
    clips.add_argument("--continue-on-error", action="store_true")
    clips.add_argument("--motion-ids", nargs="*")
    session = commands.add_parser(
        "run-scalebfm-session",
        help="run ScaleBFM true-knot SIL from a ScaleTrack reference and Kimodo init state",
    )
    session.add_argument("experiment", type=Path)
    session.add_argument(
        "--reference-npz",
        type=Path,
        required=True,
        help="ScaleTrack torso_umi3_reference.npz with body_pos_w / body_quat_w / joint_pos",
    )
    session.add_argument(
        "--init-state-npz",
        type=Path,
        required=True,
        help="Kimodo init_state.npz mapped onto the G1 free root and 29 command joints",
    )
    session.add_argument("--output-dir", type=Path)
    session.add_argument("--device-index", type=int, default=0)
    session.add_argument("--dry-run", action="store_true")
    session.add_argument("--disable-visualization", action="store_true")
    encode = commands.add_parser(
        "encode-kinematics-tokens",
        help=(
            "encode SONIC motion tokens from session_*/final_50hz/kinematics.npz "
            "with state=kinematics (no sim rollout)"
        ),
    )
    encode.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="root with session_*/final_50hz/kinematics.npz",
    )
    encode.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="destination clips root (writes <session>/optimized_kinematics.npz)",
    )
    encode.add_argument(
        "--encoder",
        type=Path,
        help="low-latency G1 model_encoder.onnx "
        "(default: artifacts/store/sonic/low_latency/g1/model_encoder.onnx)",
    )
    encode.add_argument(
        "--gpus",
        default="0",
        help="comma-separated GPU indices, e.g. 0,4,5,6 (default: 0)",
    )
    encode.add_argument("--dry-run", action="store_true")
    encode.add_argument("--continue-on-error", action="store_true")
    encode.add_argument("--motion-ids", nargs="*")
    encode.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="frames packed per encode chunk (still runs ONNX at batch=1)",
    )
    encode.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="re-encode even when optimized_kinematics.npz already looks complete",
    )
    groot = commands.add_parser(
        "export-groot",
        help="export SIL clips plus source kinematics to an Isaac-GR00T SONIC LeRobot dataset",
    )
    groot.add_argument("--source-root", type=Path, required=True)
    groot.add_argument("--clips-root", type=Path, required=True)
    groot.add_argument("--output", type=Path, required=True)
    groot.add_argument("--task", required=True)
    groot.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        result = run_experiment(
            args.experiment,
            device_index=args.device_index,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            dataset_root=args.dataset_root,
            motion_id=args.motion_id,
            clip_duration_seconds=args.clip_duration_seconds,
            visualization_enabled=(
                False if args.disable_visualization else None
            ),
        )
        if result is not None:
            print(result.output_dir)
    elif args.command == "batch":
        from sil_tts.batch_run_sil import run_experiment_batch

        result = run_experiment_batch(
            args.experiment_batch,
            execute_experiment=run_experiment,
            device_index=args.device_index,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
        if result is not None:
            print(result.output_dir)
            if not result.success:
                return 1
    elif args.command == "render":
        print(_render_experiment(args.experiment_dir.resolve()))
    elif args.command == "prepare-tracker":
        print(prepare_tracker(args.artifact_id))
    elif args.command in {"ingest-kinematics", "ingest-walk-and-pick"}:
        summary = ingest_kinematics_dataset(
            args.source,
            args.output,
            robot_config=args.robot,
            dataset_name=args.dataset_name,
        )
        print(json.dumps(summary, indent=2))
        if summary["num_ingested"] == 0:
            return 1
    elif args.command == "run-clips":
        from sil_tts.clip_queue import parse_gpu_list, run_clip_queue

        if (args.dataset_root is None) == (args.source_root is None):
            _parser().error("run-clips requires exactly one of --source-root or --dataset-root")
        summary = run_clip_queue(
            args.experiment,
            dataset_root=args.dataset_root,
            source_root=args.source_root,
            gpus=parse_gpu_list(args.gpus),
            output_root=args.output_root,
            dry_run=args.dry_run,
            continue_on_error=True if args.continue_on_error else None,
            motion_ids=args.motion_ids,
            robot_config=args.robot,
        )
        print(summary["output_root"])
        if not summary.get("dry_run") and not summary.get("success", True):
            return 1
    elif args.command == "encode-kinematics-tokens":
        from sil_tts.encode_kinematics_tokens import (
            encode_kinematics_dataset,
            parse_gpu_list,
        )

        summary = encode_kinematics_dataset(
            args.source_root,
            args.output_root,
            gpus=parse_gpu_list(args.gpus),
            encoder_path=args.encoder,
            dry_run=args.dry_run,
            continue_on_error=True if args.continue_on_error else None,
            motion_ids=args.motion_ids,
            skip_existing=not args.no_skip_existing,
            chunk_size=args.chunk_size,
        )
        print(json.dumps(summary, indent=2))
        if not summary.get("dry_run") and not summary.get("success", True):
            return 1
    elif args.command == "export-groot":
        from sil_tts.groot_export import export_groot_dataset

        summary = export_groot_dataset(
            source_root=args.source_root,
            clips_root=args.clips_root,
            output_root=args.output,
            task=args.task,
            dry_run=args.dry_run,
        )
        print(json.dumps(summary, indent=2))
        if not summary.get("dry_run"):
            print(
                "Generate Isaac-GR00T stats with:\n"
                "python gr00t/data/stats.py "
                f"--dataset-path {summary['output_root']} "
                f"--embodiment-tag {summary.get('embodiment_tag', 'UNITREE_G1_SONIC')}"
            )
        if summary.get("failures"):
            return 1
    elif args.command == "run-scalebfm-session":
        result = run_experiment(
            args.experiment,
            device_index=args.device_index,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            visualization_enabled=(
                False if args.disable_visualization else None
            ),
            reference_npz=args.reference_npz,
            init_state_npz=args.init_state_npz,
        )
        if result is not None:
            print(result.output_dir)
    elif args.command == "prepare-sonic-low-latency":
        from sil_tts.sonic_low_latency_preparation import prepare_sonic_low_latency

        print(
            prepare_sonic_low_latency(
                args.encoder, args.decoder, args.observation_config
            )
        )
    elif args.command == "prepare-scalebfm-torso-umi3":
        from sil_tts.scalebfm_onnx_preparation import prepare_scalebfm_torso_umi3

        print(prepare_scalebfm_torso_umi3())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
