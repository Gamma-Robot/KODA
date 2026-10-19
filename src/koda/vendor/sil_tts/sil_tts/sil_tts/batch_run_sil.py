"""Run and aggregate a strict YAML-declared batch of SIL experiments."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import math
import re
import statistics
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sil_trackers import instance_config_from_mapping
from sil_tts.artifacts import default_repo_root, sha256_file
from sil_tts.config import (
    RESOLVED_EXPERIMENT_FILE,
    ConfigError,
    dataclass_from_dict,
    resolve_config,
)


RESOLVED_EXPERIMENT_BATCH_FILE = "resolved_experiment_batch.yaml"
EXPERIMENT_BATCH_SUMMARY_FILE = "experiment_batch_summary.json"
_EXPERIMENT_FILES = (
    RESOLVED_EXPERIMENT_FILE,
    "report.json",
    "planning.json",
    "randomization.json",
    "provenance.json",
    "paired_traces.npz",
)
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
LIVE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class BudgetPoint:
    id: str
    num_candidates: int
    num_iterations: int

    def __post_init__(self) -> None:
        _require_id(self.id, "budget id")
        _require_int(self.num_candidates, "num_candidates", minimum=1)
        _require_int(self.num_iterations, "num_iterations", minimum=1)


@dataclass(frozen=True)
class PlannerVariant:
    id: str
    ref: str
    overrides: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_id(self.id, "planner variant id")
        if not isinstance(self.ref, str) or not self.ref:
            raise ConfigError("planner variant ref must be a non-empty string")
        if not isinstance(self.overrides, Mapping):
            raise ConfigError("planner variant overrides must be a mapping")
        forbidden = {"algorithm", "num_candidates", "num_iterations"}.intersection(
            self.overrides
        )
        if forbidden:
            raise ConfigError(
                "planner variant overrides cannot set fields owned by ref/budget: "
                f"{sorted(forbidden)}"
            )
        object.__setattr__(self, "overrides", dict(self.overrides))


@dataclass(frozen=True)
class ExperimentBatchConfig:
    schema_version: int
    name: str
    experiment_refs: Sequence[str]
    clip_duration_seconds: float
    budgets: Sequence[BudgetPoint | Mapping[str, Any]]
    planner_variants: Sequence[PlannerVariant | Mapping[str, Any]]
    seeds: Sequence[int]
    rollout_batch_ceiling: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != LIVE_SCHEMA_VERSION
        ):
            raise ConfigError(
                f"schema_version must be exactly {LIVE_SCHEMA_VERSION}"
            )
        if not isinstance(self.name, str) or not self.name:
            raise ConfigError("experiment batch name must be a non-empty string")
        _require_id(self.name, "experiment batch name")
        refs = _sequence(self.experiment_refs, "experiment_refs")
        budgets = _bind_sequence(BudgetPoint, self.budgets, "budgets")
        variants = _bind_sequence(
            PlannerVariant, self.planner_variants, "planner_variants"
        )
        seeds = _sequence(self.seeds, "seeds")
        if not refs or any(not isinstance(ref, str) or not ref for ref in refs):
            raise ConfigError("experiment_refs must contain non-empty strings")
        if len(set(refs)) != len(refs):
            raise ConfigError("experiment_refs must be unique")
        _unique_ids(budgets, "budgets")
        _unique_ids(variants, "planner_variants")
        _unique_nonnegative_ints(seeds, "seeds", allow_zero=True)
        if isinstance(self.clip_duration_seconds, bool):
            raise ConfigError("clip_duration_seconds must be finite and > 0")
        try:
            seconds = float(self.clip_duration_seconds)
        except (TypeError, ValueError) as exc:
            raise ConfigError("clip_duration_seconds must be finite and > 0") from exc
        if not math.isfinite(seconds) or seconds <= 0.0:
            raise ConfigError("clip_duration_seconds must be finite and > 0")
        _require_int(
            self.rollout_batch_ceiling,
            "rollout_batch_ceiling",
            minimum=1,
        )
        object.__setattr__(self, "experiment_refs", refs)
        object.__setattr__(self, "clip_duration_seconds", seconds)
        object.__setattr__(self, "budgets", budgets)
        object.__setattr__(self, "planner_variants", variants)
        object.__setattr__(self, "seeds", seeds)


@dataclass(frozen=True)
class ExperimentBatchCell:
    cell_id: str
    tracker_id: str
    planner_variant_id: str
    planner_algorithm: str
    budget_id: str
    candidate_count: int
    iteration_count: int
    seed: int
    clip_duration_seconds: float
    rollout_batch_size: int
    candidate_latent_bytes_per_iteration: int
    resolved: dict[str, Any]


@dataclass(frozen=True)
class ExperimentBatchResult:
    output_dir: Path
    success: bool
    cells: int


@dataclass(frozen=True)
class SourceQualification:
    local_upstream_tracking_ref: str
    local_upstream_tracking_tip: str
    checked_at_utc: str


ExperimentExecutor = Callable[..., Any]


def load_experiment_batch(
    batch_path: str | Path, *, configs_root: str | Path | None = None
) -> tuple[dict[str, Any], ExperimentBatchConfig, tuple[ExperimentBatchCell, ...]]:
    """Load one strict Experiment Batch YAML and expand its cells."""

    repo = default_repo_root()
    root = (repo / "configs" if configs_root is None else Path(configs_root)).resolve()
    path = Path(batch_path)
    try:
        resolved = yaml.safe_load(path.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"experiment batch config file not found: {path}") from exc
    if not isinstance(resolved, dict):
        raise ConfigError(
            f"top-level experiment batch config must be a mapping: {path}"
        )
    config = dataclass_from_dict(ExperimentBatchConfig, resolved)
    cells = _expand_cells(config, configs_root=root)
    return resolved, config, cells


def run_experiment_batch(
    batch_path: str | Path,
    *,
    execute_experiment: ExperimentExecutor,
    device_index: int = 0,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    configs_root: str | Path | None = None,
) -> ExperimentBatchResult | None:
    """Preflight and execute one immutable tracker × planner × budget matrix."""

    if (
        isinstance(device_index, bool)
        or not isinstance(device_index, int)
        or device_index < 0
    ):
        raise ValueError("device_index must be a non-negative integer")
    if not callable(execute_experiment):
        raise TypeError("execute_experiment must be callable")
    batch_path = Path(batch_path)
    resolved, config, cells = load_experiment_batch(batch_path, configs_root=configs_root)
    repo = default_repo_root()
    root = (repo / "configs" if configs_root is None else Path(configs_root)).resolve()

    _preflight(cells, execute_experiment, device_index=device_index, configs_root=root)
    if dry_run:
        return None

    # Keep the Experiment Batch module independent of the composition root at
    # import time while reusing its established output and Git discipline.
    from sil_tts.run_sil import (
        _default_output_dir,
        _prepare_output_dir,
        _source_commit,
        _source_dirty,
    )

    source_commit = _source_commit(repo)
    source_dirty = _source_dirty(repo)
    if source_dirty:
        raise RuntimeError(
            "experiment batch execution requires a clean tracked source commit"
        )
    source_qualification = _validate_formal_source(
        batch_path,
        config=config,
        configs_root=root,
        repo=repo,
        source_commit=source_commit,
    )
    destination = (
        _default_output_dir(repo, f"experiment_batches/{config.name}", source_commit)
        if output_dir is None
        else Path(output_dir)
    ).resolve()
    _prepare_output_dir(destination)
    _write_yaml(destination / RESOLVED_EXPERIMENT_BATCH_FILE, resolved)
    resolved_dir = destination / "resolved_experiments"
    resolved_dir.mkdir()
    cells_dir = destination / "cells"
    cells_dir.mkdir()

    rows: list[dict[str, Any]] = []
    for cell in cells:
        cell_config = resolved_dir / f"{cell.cell_id}.yaml"
        cell_output = cells_dir / cell.cell_id
        _write_yaml(cell_config, cell.resolved)
        try:
            execute_experiment(
                cell_config,
                device_index=device_index,
                output_dir=cell_output,
                dry_run=False,
                configs_root=root,
            )
            row = _summarize_cell(
                cell,
                cell_output,
                batch_root=destination,
                source_commit=source_commit,
            )
        except Exception as exc:
            cell_output.mkdir(parents=True, exist_ok=True)
            failure = f"{type(exc).__name__}: {exc}"
            failure_path = cell_output / "failure.json"
            if not failure_path.exists():
                _write_json(failure_path, {"cell_id": cell.cell_id, "failure": failure})
            row = _failed_row(cell, cell_output, destination, failure)
        rows.append(row)

    successful_digests = {
        row["initial_physical_state_digest"]
        for row in rows
        if row["status"] == "success"
    }
    invariant_failures: list[str] = []
    if len(successful_digests) != 1:
        invariant_failures.append(
            "successful cells do not share exactly one initial physical-state digest"
        )
    successful_hardware = {
        row["hardware"]["gpu_uuid"]
        for row in rows
        if row["status"] == "success"
    }
    if len(successful_hardware) != 1:
        invariant_failures.append("successful cells do not share one physical GPU")
    success = all(row["status"] == "success" for row in rows) and not invariant_failures
    summary = {
        "schema_version": LIVE_SCHEMA_VERSION,
        "experiment_batch": {
            "name": config.name,
            "source_commit": source_commit,
            "source_dirty": source_dirty,
            "source_qualification": {
                "local_upstream_tracking_ref": (
                    source_qualification.local_upstream_tracking_ref
                ),
                "local_upstream_tracking_tip": (
                    source_qualification.local_upstream_tracking_tip
                ),
                "checked_at_utc": source_qualification.checked_at_utc,
            },
            "experiment_batch_yaml_sha256": sha256_file(batch_path),
            "resolved_experiment_batch_sha256": sha256_file(
                destination / RESOLVED_EXPERIMENT_BATCH_FILE
            ),
            "experiment_refs": list(config.experiment_refs),
            "clip_duration_seconds": config.clip_duration_seconds,
            "budgets": [
                {
                    "id": budget.id,
                    "num_candidates": budget.num_candidates,
                    "num_iterations": budget.num_iterations,
                }
                for budget in config.budgets
            ],
            "planner_variants": [
                {
                    "id": variant.id,
                    "ref": variant.ref,
                    "overrides": dict(variant.overrides),
                }
                for variant in config.planner_variants
            ],
            "seeds": list(config.seeds),
            "rollout_batch_ceiling": config.rollout_batch_ceiling,
            "initial_physical_state_digest": (
                next(iter(successful_digests)) if len(successful_digests) == 1 else None
            ),
            "invariant_failures": invariant_failures,
            "success": success,
        },
        "cells": rows,
        "groups": _group_rows(rows),
    }
    _write_json(destination / EXPERIMENT_BATCH_SUMMARY_FILE, summary)
    return ExperimentBatchResult(destination, success, len(rows))


def _expand_cells(
    batch_config: ExperimentBatchConfig, *, configs_root: Path
) -> tuple[ExperimentBatchCell, ...]:
    from sil_tts.run_sil import load_experiment

    planners: list[tuple[PlannerVariant, dict[str, Any]]] = []
    for variant in batch_config.planner_variants:
        path = _safe_planner_ref(variant.ref, configs_root)
        planner = resolve_config(path, configs_root)
        if not isinstance(planner, dict):
            raise ConfigError(f"planner ref must resolve to a mapping: {variant.ref!r}")
        planner = copy.deepcopy(planner)
        planner.update(variant.overrides)
        algorithm = planner.get("algorithm")
        if not isinstance(algorithm, str) or not algorithm:
            raise ConfigError(
                f"planner variant {variant.id!r} has no non-empty algorithm"
            )
        planners.append((variant, planner))

    experiments: list[tuple[dict[str, Any], Any, Any]] = []
    common: dict[str, Any] | None = None
    common_control_period: float | None = None
    common_joints: tuple[str, ...] | None = None
    trackers: set[str] = set()
    for ref in batch_config.experiment_refs:
        path = _safe_experiment_ref(ref, configs_root)
        resolved, config = load_experiment(path, configs_root=configs_root)
        tracker = instance_config_from_mapping(config.tracker)
        if tracker.instance in trackers:
            raise ConfigError(
                "duplicate tracker instance in experiment batch: "
                f"{tracker.instance}"
            )
        trackers.add(tracker.instance)
        comparable = _common_experiment_fields(resolved)
        if common is None:
            common = comparable
            common_control_period = tracker.control_period
            common_joints = tracker.command_joint_names
        elif comparable != common:
            raise ConfigError(
                f"experiment {ref!r} differs outside fields owned by the "
                "Experiment Batch"
            )
        if tracker.control_period != common_control_period:
            raise ConfigError(
                "experiment batch trackers must share one control period"
            )
        if tracker.command_joint_names != common_joints:
            raise ConfigError(
                "experiment batch trackers must share canonical command joints"
            )
        experiments.append((resolved, config, tracker))

    cells: list[ExperimentBatchCell] = []
    seen: set[str] = set()
    for resolved, config, tracker in experiments:
        for variant, planner in planners:
            algorithm = str(planner["algorithm"])
            for budget in batch_config.budgets:
                for seed in batch_config.seeds:
                    cell_id = (
                        f"{tracker.instance}-{variant.id}-{budget.id}-s{seed}"
                    )
                    if cell_id in seen:
                        raise ConfigError(
                            f"duplicate experiment batch cell id: {cell_id}"
                        )
                    seen.add(cell_id)
                    value = copy.deepcopy(resolved)
                    value["name"] = f"{batch_config.name}-{cell_id}"
                    value["planner"] = copy.deepcopy(planner)
                    value["planner"]["num_candidates"] = budget.num_candidates
                    value["planner"]["num_iterations"] = budget.num_iterations
                    value["target"]["clip_duration_seconds"] = (
                        batch_config.clip_duration_seconds
                    )
                    value["planning"]["seed"] = seed
                    scenario_count = (
                        config.robustness.num_scenarios
                        if config.robustness.enabled
                        else 1
                    )
                    candidate_capacity = (
                        batch_config.rollout_batch_ceiling // scenario_count
                    )
                    if candidate_capacity < 1:
                        raise ConfigError(
                            "rollout_batch_ceiling must fit one complete "
                            "scenario group"
                        )
                    rollout_batch_size = min(
                        budget.num_candidates, candidate_capacity
                    ) * scenario_count
                    value["planning"]["max_worlds"] = rollout_batch_size
                    from sil_tts.motion_tracking.search_space import planner_latent_dim

                    latent_dim = planner_latent_dim(
                        tracker.family, len(tracker.command_joint_names)
                    )
                    cells.append(
                        ExperimentBatchCell(
                            cell_id=cell_id,
                            tracker_id=tracker.instance,
                            planner_variant_id=variant.id,
                            planner_algorithm=algorithm,
                            budget_id=budget.id,
                            candidate_count=budget.num_candidates,
                            iteration_count=budget.num_iterations,
                            seed=seed,
                            clip_duration_seconds=(
                                batch_config.clip_duration_seconds
                            ),
                            rollout_batch_size=rollout_batch_size,
                            candidate_latent_bytes_per_iteration=(
                                budget.num_candidates
                                * config.planning.num_knots
                                * latent_dim
                                * 8
                            ),
                            resolved=value,
                        )
                    )
    return tuple(cells)


def _safe_experiment_ref(ref: str, configs_root: Path) -> Path:
    relative = Path(ref)
    if relative.is_absolute() or relative.suffix not in (".yaml", ".yml"):
        raise ConfigError(f"unsafe experiment ref: {ref!r}")
    candidate = (configs_root / relative).resolve()
    if not candidate.is_relative_to(configs_root) or not candidate.is_file():
        raise ConfigError(f"unsafe or missing experiment ref: {ref!r}")
    return candidate


def _safe_planner_ref(ref: str, configs_root: Path) -> Path:
    relative = Path(ref)
    planners_root = (configs_root / "planners").resolve()
    if relative.is_absolute() or relative.suffix not in (".yaml", ".yml"):
        raise ConfigError(f"unsafe planner ref: {ref!r}")
    candidate = (configs_root / relative).resolve()
    if not candidate.is_relative_to(planners_root) or not candidate.is_file():
        raise ConfigError(f"unsafe or missing planner ref: {ref!r}")
    return candidate


def _common_experiment_fields(resolved: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(resolved))
    value.pop("name", None)
    value.pop("tracker", None)
    value.pop("planner", None)
    planning = value.get("planning")
    target = value.get("target")
    if not isinstance(planning, dict) or not isinstance(target, dict):
        raise ConfigError(
            "Experiment Batch inputs require planning and target mappings"
        )
    target.pop("clip_duration_seconds", None)
    planning.pop("seed", None)
    planning.pop("max_worlds", None)
    return value


def _preflight(
    cells: Sequence[ExperimentBatchCell],
    executor: ExperimentExecutor,
    *,
    device_index: int,
    configs_root: Path,
) -> None:
    with tempfile.TemporaryDirectory(
        prefix="sil-tts-experiment-batch-preflight-"
    ) as temp:
        root = Path(temp)
        for cell in cells:
            path = root / f"{cell.cell_id}.yaml"
            _write_yaml(path, cell.resolved)
            executor(
                path,
                device_index=device_index,
                output_dir=None,
                dry_run=True,
                configs_root=configs_root,
            )


def _summarize_cell(
    cell: ExperimentBatchCell,
    output: Path,
    *,
    batch_root: Path,
    source_commit: str,
) -> dict[str, Any]:
    paths = {name: output / name for name in _EXPERIMENT_FILES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"cell output is missing required files: {missing}")
    report = _read_json(paths["report.json"])
    planning_payload = _read_json(paths["planning.json"])
    randomization = _read_json(paths["randomization.json"])
    provenance = _read_json(paths["provenance.json"])
    resolved_experiment = yaml.safe_load(
        paths[RESOLVED_EXPERIMENT_FILE].read_text()
    )
    if resolved_experiment != cell.resolved:
        raise ValueError(
            "persisted resolved experiment differs from the experiment batch cell"
        )
    if (
        not isinstance(report, dict)
        or not isinstance(planning_payload, dict)
        or not isinstance(randomization, dict)
        or not isinstance(provenance, dict)
    ):
        raise ValueError(
            "cell report/planning/randomization/provenance schema is invalid"
        )
    for name, payload in (
        ("report.json", report),
        ("planning.json", planning_payload),
        ("randomization.json", randomization),
        ("provenance.json", provenance),
    ):
        if (
            isinstance(payload.get("schema_version"), bool)
            or not isinstance(payload.get("schema_version"), int)
            or payload.get("schema_version") != LIVE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"cell {name} schema_version must be exactly {LIVE_SCHEMA_VERSION}"
            )
    with np.load(paths["paired_traces.npz"], allow_pickle=False) as traces:
        trace_version = np.asarray(traces["schema_version"])
        if (
            trace_version.shape != ()
            or not np.issubdtype(trace_version.dtype, np.integer)
            or int(trace_version) != LIVE_SCHEMA_VERSION
        ):
            raise ValueError(
                "cell paired_traces.npz schema_version must be exactly "
                f"{LIVE_SCHEMA_VERSION}"
            )
    planning = planning_payload.get("updates")
    if not isinstance(planning, list):
        raise ValueError("planning.json updates must be a list")
    if provenance.get("source_commit") != source_commit or provenance.get("source_dirty"):
        raise ValueError(
            "cell provenance is not bound to the clean Experiment Batch source commit"
        )
    hardware = provenance.get("hardware")
    if not isinstance(hardware, dict) or not hardware.get("available") or not hardware.get(
        "gpu_uuid"
    ):
        raise ValueError("cell hardware provenance is incomplete")
    tracker_runtime = provenance.get("tracker_runtime")
    if tracker_runtime not in ("device", "host-reference"):
        raise ValueError("cell tracker runtime provenance is invalid")
    file_digests = {name: sha256_file(path) for name, path in paths.items()}
    for name, key in (
        ("report.json", "report_sha256"),
        ("planning.json", "planning_sha256"),
        ("randomization.json", "randomization_sha256"),
        (RESOLVED_EXPERIMENT_FILE, "resolved_experiment_sha256"),
        ("paired_traces.npz", "trace_sha256"),
    ):
        if provenance.get(key) != file_digests[name]:
            raise ValueError(f"cell provenance {key} does not match {name}")
    randomization_updates = randomization.get("updates")
    if not isinstance(randomization_updates, list) or len(
        randomization_updates
    ) != len(planning):
        raise ValueError("randomization updates disagree with planning ledger")
    for randomization_update, planning_update in zip(
        randomization_updates, planning, strict=True
    ):
        if not isinstance(randomization_update, dict) or (
            randomization_update.get("update_step")
            != planning_update.get("update_step")
            or randomization_update.get("scenario_digest")
            != planning_update.get("scenario_digest")
        ):
            raise ValueError("randomization identity disagrees with planning ledger")

    identities = report.get("identities")
    baseline = report.get("baseline")
    optimized = report.get("optimized")
    if not all(isinstance(value, dict) for value in (identities, baseline, optimized)):
        raise ValueError("cell report identities/metrics are invalid")
    assert isinstance(identities, dict)
    assert isinstance(baseline, dict)
    assert isinstance(optimized, dict)
    if identities.get("tracker_id") != cell.tracker_id:
        raise ValueError("cell tracker identity differs from Experiment Batch expansion")
    if provenance.get("target_sha256") != identities.get("target_digest"):
        raise ValueError("target digest differs between report and provenance")
    execution_failures: list[str] = []
    expected_execution_steps = round(
        cell.clip_duration_seconds / float(identities["control_period"])
    )
    for label, metrics in (("baseline", baseline), ("optimized", optimized)):
        if not metrics.get("success") or metrics.get("valid_steps") != metrics.get(
            "total_steps"
        ):
            execution_failures.append(
                f"{label} execution did not complete all valid steps"
            )
        elif int(metrics["total_steps"]) != expected_execution_steps:
            execution_failures.append(
                f"{label} execution did not contain the declared clip steps"
            )

    if not planning:
        raise ValueError("cell planning ledger must not be empty")
    iterations = cell.iteration_count
    if int(cell.resolved["planner"]["num_iterations"]) != iterations:
        raise ValueError("resolved planner iterations differ from Experiment Batch budget")
    evaluate_proposal = bool(cell.resolved["planning"]["evaluate_proposal"])
    all_penalty = 0
    for record in planning:
        if not isinstance(record, dict):
            raise ValueError("planning entries must be mappings")
        selected = float(record["selected_cost"])
        if not math.isfinite(selected):
            raise ValueError("planning selected cost must be finite")
        base_candidates = cell.candidate_count * iterations
        candidate_value = record.get("candidate_evaluations")
        if isinstance(candidate_value, bool) or not isinstance(candidate_value, int):
            raise ValueError("planning candidate ledger must be an integer")
        candidate_evaluations = candidate_value
        extra_evaluation = candidate_evaluations - base_candidates
        if extra_evaluation not in (0, 1):
            raise ValueError(
                "planning logical rollout ledger disagrees with Experiment Batch budget"
            )
        if not evaluate_proposal and extra_evaluation:
            raise ValueError("planning ledger contains a disabled proposal evaluation")
        if (
            record.get("proposal_candidate_id")
            == record.get("best_observed_candidate_id")
            and extra_evaluation
        ):
            raise ValueError("best-observed proposal must not be evaluated twice")
        scenario_count = record.get("scenario_count")
        logical_rollouts = record.get("logical_rollouts")
        if (
            isinstance(scenario_count, bool)
            or not isinstance(scenario_count, int)
            or scenario_count < 1
        ):
            raise ValueError("planning scenario_count ledger must be a positive integer")
        scenario_digest = record.get("scenario_digest")
        if scenario_digest is not None and (
            not isinstance(scenario_digest, str)
            or not scenario_digest.startswith("sha256:")
        ):
            raise ValueError("planning scenario_digest ledger is invalid")
        if isinstance(logical_rollouts, bool) or not isinstance(logical_rollouts, int):
            raise ValueError("planning logical rollout ledger must be an integer")
        if logical_rollouts != candidate_evaluations * scenario_count:
            raise ValueError("planning scenario rollout ledger is inconsistent")
        physical_rollouts = record.get("physical_rollouts")
        if (
            isinstance(physical_rollouts, bool)
            or not isinstance(physical_rollouts, int)
            or physical_rollouts < logical_rollouts
        ):
            raise ValueError("planning physical rollout ledger is invalid")
        penalized_candidates = record.get("penalized_candidate_evaluations")
        if (
            isinstance(penalized_candidates, bool)
            or not isinstance(penalized_candidates, int)
            or not 0 <= penalized_candidates <= candidate_evaluations
        ):
            raise ValueError("planning penalized candidate ledger is invalid")
        invalid_scenarios = record.get("invalid_scenario_rollouts")
        if (
            isinstance(invalid_scenarios, bool)
            or not isinstance(invalid_scenarios, int)
            or not 0 <= invalid_scenarios <= logical_rollouts
        ):
            raise ValueError("planning invalid scenario ledger is invalid")
        rows = record.get("cost_quantiles")
        expected_evaluations = iterations + extra_evaluation
        if not isinstance(rows, list) or len(rows) != expected_evaluations:
            raise ValueError("planning cost_quantiles disagree with evaluator calls")
        if any(not isinstance(row, list) or len(row) != 5 for row in rows):
            raise ValueError("planning cost quantile rows are invalid")
        if penalized_candidates == candidate_evaluations:
            all_penalty += 1
    if all_penalty:
        execution_failures.append(
            f"cell contains {all_penalty} all-penalty planning updates"
        )

    deformations = _mean_deformation(planning)
    optimized_minus_baseline = {
        key: float(optimized[key]) - float(baseline[key])
        for key in _metric_keys(baseline, optimized)
    }
    candidate_evaluations = sum(
        int(record["candidate_evaluations"]) for record in planning
    )
    logical_rollouts = sum(int(record["logical_rollouts"]) for record in planning)
    physical_rollouts = sum(int(record["physical_rollouts"]) for record in planning)
    penalized_candidate_evaluations = sum(
        int(record["penalized_candidate_evaluations"]) for record in planning
    )
    invalid_scenario_rollouts = sum(
        int(record["invalid_scenario_rollouts"]) for record in planning
    )
    horizon_steps = int(cell.resolved["planning"]["horizon_steps"])
    return {
        **_cell_summary_fields(cell, output, batch_root),
        "status": "failed" if execution_failures else "success",
        "file_sha256": file_digests,
        "hardware": hardware,
        "tracker_runtime": tracker_runtime,
        "initial_physical_state_digest": identities[
            "initial_physical_state_digest"
        ],
        "target_id": identities["target_id"],
        "target_digest": identities["target_digest"],
        "checkpoint_digest": identities["checkpoint_digest"],
        "robot_model_digest": identities["robot_model_digest"],
        "control_period": identities["control_period"],
        "baseline": baseline,
        "optimized": optimized,
        "optimized_minus_baseline": optimized_minus_baseline,
        "mean_candidate_deformation": deformations,
        "candidate_evaluations": candidate_evaluations,
        "logical_rollouts": logical_rollouts,
        "physical_rollouts": physical_rollouts,
        "penalized_candidate_evaluations": penalized_candidate_evaluations,
        "invalid_scenario_rollouts": invalid_scenario_rollouts,
        "logical_rollout_steps": logical_rollouts * horizon_steps,
        "physical_rollout_steps": physical_rollouts * horizon_steps,
        "planning_wall_seconds": sum(
            float(record["update_wall_seconds"]) for record in planning
        ),
        "all_penalty_updates": all_penalty,
        "failure": "; ".join(execution_failures) if execution_failures else None,
    }


def _failed_row(
    cell: ExperimentBatchCell, output: Path, batch_root: Path, failure: str
) -> dict[str, Any]:
    return {
        **_cell_summary_fields(cell, output, batch_root),
        "status": "failed",
        "file_sha256": {
            name: sha256_file(output / name) if (output / name).is_file() else None
            for name in _EXPERIMENT_FILES
        },
        "initial_physical_state_digest": None,
        "hardware": None,
        "tracker_runtime": None,
        "target_id": None,
        "target_digest": None,
        "checkpoint_digest": None,
        "robot_model_digest": None,
        "control_period": None,
        "baseline": None,
        "optimized": None,
        "optimized_minus_baseline": None,
        "mean_candidate_deformation": None,
        "candidate_evaluations": None,
        "logical_rollouts": None,
        "physical_rollouts": None,
        "penalized_candidate_evaluations": None,
        "invalid_scenario_rollouts": None,
        "logical_rollout_steps": None,
        "physical_rollout_steps": None,
        "planning_wall_seconds": None,
        "all_penalty_updates": None,
        "failure": failure,
    }


def _cell_summary_fields(
    cell: ExperimentBatchCell, output: Path, batch_root: Path
) -> dict[str, Any]:
    return {
        "cell_id": cell.cell_id,
        "tracker_id": cell.tracker_id,
        "planner_variant_id": cell.planner_variant_id,
        "planner_algorithm": cell.planner_algorithm,
        "budget_id": cell.budget_id,
        "candidate_count": cell.candidate_count,
        "iteration_count": cell.iteration_count,
        "seed": cell.seed,
        "clip_duration_seconds": cell.clip_duration_seconds,
        "rollout_batch_size": cell.rollout_batch_size,
        "output": str(output.relative_to(batch_root)),
        "candidate_latent_bytes_per_iteration": (
            cell.candidate_latent_bytes_per_iteration
        ),
    }


def _validate_formal_source(
    batch_path: Path,
    *,
    config: ExperimentBatchConfig,
    configs_root: Path,
    repo: Path,
    source_commit: str,
) -> SourceQualification:
    repo = repo.resolve()
    expected_configs = (repo / "configs").resolve()
    expected_experiments = expected_configs / "experiments"
    if configs_root != expected_configs:
        raise RuntimeError(
            "formal Experiment Batch execution requires the repository configs root"
        )
    batch_path = batch_path.resolve()
    if (
        batch_path.parent != expected_experiments
        or not batch_path.name.endswith(".batch.yaml")
    ):
        raise RuntimeError(
            "formal Experiment Batch YAML must be a *.batch.yaml file directly under "
            "configs/experiments"
        )
    paths = [
        batch_path,
        *(_safe_experiment_ref(ref, configs_root) for ref in config.experiment_refs),
        *(_safe_planner_ref(variant.ref, configs_root) for variant in config.planner_variants),
    ]
    for path in paths:
        relative = path.relative_to(repo).as_posix()
        committed = subprocess.run(
            ["git", "show", f"{source_commit}:{relative}"],
            cwd=repo,
            text=False,
            capture_output=True,
        )
        if committed.returncode != 0 or committed.stdout != path.read_bytes():
            raise RuntimeError(
                "formal Experiment Batch input is not byte-identical to "
                f"{source_commit}: {relative}"
            )
    upstream = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
        cwd=repo,
        text=True,
        capture_output=True,
    )
    if upstream.returncode != 0 or not upstream.stdout.strip():
        raise RuntimeError("formal Experiment Batch source branch has no upstream")
    upstream_ref = upstream.stdout.strip()
    tip = subprocess.run(
        ["git", "rev-parse", upstream_ref],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    pushed = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            source_commit,
            upstream_ref,
        ],
        cwd=repo,
        capture_output=True,
    )
    if pushed.returncode != 0:
        raise RuntimeError(
            "formal Experiment Batch source commit is not present on its upstream"
        )
    return SourceQualification(
        local_upstream_tracking_ref=upstream_ref,
        local_upstream_tracking_tip=tip,
        checked_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def _mean_deformation(planning: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    names = sorted(
        {
            name
            for record in planning
            for name in record.get("candidate_deformation", {})
        }
    )
    if not names:
        raise ValueError("planning ledger must contain candidate deformation")
    return {
        name: statistics.fmean(
            float(record["candidate_deformation"][name]) for record in planning
        )
        for name in names
    }


def _sequence(value: Any, field: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError(f"{field} must be a sequence")
    return tuple(value)


def _metric_keys(
    baseline: Mapping[str, Any], optimized: Mapping[str, Any]
) -> tuple[str, ...]:
    excluded = {"total_steps", "valid_steps", "failure_step"}
    return tuple(
        key
        for key in sorted(set(baseline).intersection(optimized).difference(excluded))
        if isinstance(baseline[key], (int, float))
        and not isinstance(baseline[key], bool)
        and isinstance(optimized[key], (int, float))
        and not isinstance(optimized[key], bool)
    )


def _group_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys: list[tuple[str, str, str]] = []
    for row in rows:
        key = (
            str(row["tracker_id"]),
            str(row["planner_variant_id"]),
            str(row["budget_id"]),
        )
        if key not in keys:
            keys.append(key)
    groups: list[dict[str, Any]] = []
    for tracker_id, planner_variant_id, budget_id in keys:
        members = [
            row
            for row in rows
            if row["tracker_id"] == tracker_id
            and row["planner_variant_id"] == planner_variant_id
            and row["budget_id"] == budget_id
        ]
        successes = [row for row in members if row["status"] == "success"]
        scalar_names = sorted(
            {name for row in successes for name in _group_scalars(row)}
        )
        statistics_by_name = {}
        for name in scalar_names:
            values = [_group_scalars(row)[name] for row in successes]
            statistics_by_name[name] = {
                "mean": statistics.fmean(values),
                "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        groups.append(
            {
                "tracker_id": tracker_id,
                "planner_variant_id": planner_variant_id,
                "planner_algorithm": members[0]["planner_algorithm"],
                "budget_id": budget_id,
                "candidate_count": members[0]["candidate_count"],
                "iteration_count": members[0]["iteration_count"],
                "n": len(members),
                "successes": len(successes),
                "statistics": statistics_by_name,
            }
        )
    return groups


def _group_scalars(row: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for section in ("baseline", "optimized", "optimized_minus_baseline"):
        for name, value in row[section].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if name not in ("total_steps", "valid_steps", "failure_step"):
                    values[f"{section}.{name}"] = float(value)
    for name, value in row["mean_candidate_deformation"].items():
        values[f"deformation.{name}"] = float(value)
    for name in (
        "candidate_evaluations",
        "logical_rollouts",
        "physical_rollouts",
        "penalized_candidate_evaluations",
        "invalid_scenario_rollouts",
        "logical_rollout_steps",
        "physical_rollout_steps",
        "candidate_latent_bytes_per_iteration",
        "planning_wall_seconds",
    ):
        values[f"compute.{name}"] = float(row[name])
    return values


def _unique_nonnegative_ints(
    values: Sequence[int], field: str, *, allow_zero: bool
) -> None:
    if not values:
        raise ConfigError(f"{field} must not be empty")
    minimum = 0 if allow_zero else 1
    for value in values:
        _require_int(value, field, minimum=minimum)
    if len(set(values)) != len(values):
        raise ConfigError(f"{field} must contain unique values")


def _bind_sequence(
    cls: type[Any], value: Any, field: str
) -> tuple[Any, ...]:
    rows = _sequence(value, field)
    if not rows:
        raise ConfigError(f"{field} must not be empty")
    return tuple(
        row if isinstance(row, cls) else dataclass_from_dict(cls, row)
        for row in rows
    )


def _unique_ids(values: Sequence[Any], field: str) -> None:
    ids = [value.id for value in values]
    if len(set(ids)) != len(ids):
        raise ConfigError(f"{field} must contain unique ids")


def _require_id(value: Any, field: str) -> None:
    if not isinstance(value, str) or _ID_PATTERN.fullmatch(value) is None:
        raise ConfigError(
            f"{field} must match {_ID_PATTERN.pattern!r}, got {value!r}"
        )


def _require_int(value: Any, field: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{field} values must be integers >= {minimum}")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_yaml(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable evidence: {path}")
    path.write_text(yaml.safe_dump(value, sort_keys=True))


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable evidence: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
