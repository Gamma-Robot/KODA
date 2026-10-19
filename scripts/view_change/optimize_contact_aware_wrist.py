#!/usr/bin/env python3
"""Refine an experimental Dex3 wrist candidate using observed cup contact.

The script produces two experiment-only candidates without changing the
production retargeting pipeline:

1. one static local 6-DoF residual shared by the entire interval;
2. a causal per-frame residual regularized to the static calibration and the
   preceding frame.

Both retain the selected finger-retargeting output unchanged. Object contact
is therefore used only to test and refine the meaning of ``wrist_yaw_link``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.transform import Rotation

from cup_eval_fast import acquire_eval_buffers, evaluate_packed_configuration
from evaluate_hand_object_interaction import (
    CupModel,
    Dex3MeshSampler,
    evaluate_configuration,
    finite_stats,
    max_true_run,
    plot_keyframe,
    summarize_records,
)


FINGER_GROUPS = ("thumb", "index", "middle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--interaction-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument(
        "--hand-mode",
        choices=("dex3", "g1_brainco"),
        default="dex3",
    )
    parser.add_argument("--wrist-method", default="mcp_arc_palm_site")
    parser.add_argument("--finger-method", default="position")
    parser.add_argument("--start-frame", type=int, default=300)
    parser.add_argument("--end-frame", type=int, default=360)
    parser.add_argument("--keyframe", type=int, default=331)
    parser.add_argument("--static-stride", type=int, default=4)
    parser.add_argument("--mesh-samples-per-geom", type=int, default=400)
    parser.add_argument("--contact-threshold-m", type=float, default=0.008)
    parser.add_argument("--penetration-tolerance-m", type=float, default=0.005)
    parser.add_argument("--penetration-ratio-tolerance", type=float, default=0.02)
    parser.add_argument("--optimization-contact-target-m", type=float, default=0.004)
    parser.add_argument(
        "--optimization-penetration-target-m", type=float, default=0.004
    )
    parser.add_argument(
        "--optimization-penetration-ratio-target", type=float, default=0.015
    )
    parser.add_argument("--robust-contact-m", type=float, default=0.007)
    parser.add_argument("--robust-penetration-m", type=float, default=0.0045)
    parser.add_argument("--robust-penetration-ratio", type=float, default=0.018)
    parser.add_argument("--translation-bound-m", type=float, default=0.04)
    parser.add_argument("--rotation-component-bound-deg", type=float, default=25.0)
    parser.add_argument("--static-max-iterations", type=int, default=50)
    parser.add_argument("--causal-max-iterations", type=int, default=180)
    parser.add_argument("--rescue-max-iterations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--require-all-finger-contact",
        action="store_true",
        help=(
            "Experiment-only constraint requiring the Dex3 thumb, index, and "
            "middle finger to contact the object simultaneously."
        ),
    )
    return parser.parse_args()


def load_cup_models(path: Path) -> dict[int, CupModel]:
    models: dict[int, CupModel] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record["center_xy"] = tuple(record["center_xy"])
        values = {
            field: record[field] for field in CupModel.__dataclass_fields__
        }
        model = CupModel(**values)
        models[model.frame] = model
    return models


def apply_residual(
    position: np.ndarray, rotation: np.ndarray, residual: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(residual, dtype=np.float64)
    refined_position = position + rotation @ value[:3]
    refined_rotation = rotation @ Rotation.from_rotvec(value[3:]).as_matrix()
    return refined_position, refined_rotation


def physical_loss(metric: dict[str, Any], args: argparse.Namespace) -> float:
    if args.require_all_finger_contact:
        contact_distances = [
            float(metric[f"{group}_surface_distance_m"])
            for group in FINGER_GROUPS
        ]
    else:
        contact_distances = [
            float(metric["thumb_surface_distance_m"]),
            float(metric["opposition_surface_distance_m"]),
        ]
    contact = 2.0 * sum(
        max(distance - float(args.optimization_contact_target_m), 0.0)
        for distance in contact_distances
    )
    opposition = 0.03 * max(float(metric["opposition_normal_dot"]) + 0.30, 0.0)
    penetration = (
        15.0
        * max(
            float(metric["hand_penetration_p95_m"])
            - float(args.optimization_penetration_target_m),
            0.0,
        )
        + 0.20
        * max(
            float(metric["hand_penetration_ratio"])
            - float(args.optimization_penetration_ratio_target),
            0.0,
        )
    )
    return float(contact + opposition + penetration)


def anchor_loss(residual: np.ndarray) -> float:
    value = np.asarray(residual, dtype=np.float64)
    return float(
        0.08 * np.linalg.norm(value[:3])
        + 0.004 * np.linalg.norm(value[3:])
    )


def temporal_loss(current: np.ndarray, previous: np.ndarray) -> float:
    return float(
        0.80 * np.linalg.norm(current[:3] - previous[:3])
        + 0.02 * np.linalg.norm(current[3:] - previous[3:])
    )


def robust_grasp(metric: dict[str, Any], args: argparse.Namespace) -> bool:
    """Use margins inside the reporting thresholds for causal hold decisions."""
    if args.require_all_finger_contact:
        contact_valid = bool(metric["all_finger_contact"]) and all(
            float(metric[f"{group}_surface_distance_m"])
            <= float(args.robust_contact_m)
            for group in FINGER_GROUPS
        )
    else:
        contact_valid = bool(metric["bilateral_contact"]) and (
            float(metric["thumb_surface_distance_m"])
            <= float(args.robust_contact_m)
            and float(metric["opposition_surface_distance_m"])
            <= float(args.robust_contact_m)
        )
    return bool(
        contact_valid
        and metric["bilateral_contact"]
        and float(metric["hand_penetration_p95_m"])
        <= float(args.robust_penetration_m)
        and float(metric["hand_penetration_ratio"])
        <= float(args.robust_penetration_ratio)
    )


def annotate_finger_contact_metrics(
    metric: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    for group in FINGER_GROUPS:
        metric[f"{group}_surface_distance_m"] = float(
            metric["groups"][group]["surface_distance_min_m"]
        )
        metric[f"{group}_contact"] = bool(metric["groups"][group]["contact"])
    metric["all_finger_contact"] = all(
        bool(metric[f"{group}_contact"]) for group in FINGER_GROUPS
    )
    metric["all_finger_valid_grasp"] = bool(
        metric["all_finger_contact"]
        and metric["bilateral_contact"]
        and float(metric["hand_penetration_p95_m"])
        <= float(args.penetration_tolerance_m)
        and float(metric["hand_penetration_ratio"])
        <= float(args.penetration_ratio_tolerance)
    )
    return metric


def selected_grasp_valid(metric: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.require_all_finger_contact:
        return bool(metric["all_finger_valid_grasp"])
    return bool(metric["valid_grasp"])


def summarize_finger_contacts(
    records: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    summary = summarize_records(records)
    all_contact = [bool(record["all_finger_contact"]) for record in records]
    all_valid = [bool(record["all_finger_valid_grasp"]) for record in records]
    summary.update(
        {
            "contact_constraint": (
                "thumb_index_middle"
                if args.require_all_finger_contact
                else "thumb_and_best_opposition"
            ),
            "all_finger_contact_rate": float(np.mean(all_contact)),
            "all_finger_contact_frames": int(np.count_nonzero(all_contact)),
            "all_finger_contact_longest_run": max_true_run(all_contact),
            "all_finger_valid_grasp_rate": float(np.mean(all_valid)),
            "all_finger_valid_grasp_frames": int(np.count_nonzero(all_valid)),
            "all_finger_valid_grasp_longest_run": max_true_run(all_valid),
        }
    )
    for group in FINGER_GROUPS:
        contacts = [bool(record[f"{group}_contact"]) for record in records]
        summary[f"{group}_contact_rate"] = float(np.mean(contacts))
        summary[f"{group}_contact_frames"] = int(np.count_nonzero(contacts))
        summary[f"{group}_surface_distance_m"] = finite_stats(
            [record[f"{group}_surface_distance_m"] for record in records]
        )
    return summary


def residual_bounds(
    translation_bound_m: float, rotation_component_bound_deg: float
) -> list[tuple[float, float]]:
    translation = abs(float(translation_bound_m))
    rotation = math.radians(abs(float(rotation_component_bound_deg)))
    return [(-translation, translation)] * 3 + [(-rotation, rotation)] * 3


def metric_for(
    packed_local: np.ndarray,
    offsets: np.ndarray,
    names: tuple[str, ...],
    base_position: np.ndarray,
    base_rotation: np.ndarray,
    residual: np.ndarray,
    cup: CupModel,
    args: argparse.Namespace,
    *,
    world_buf: np.ndarray | None = None,
    stats_buf: np.ndarray | None = None,
    scratch_buf: np.ndarray | None = None,
) -> dict[str, Any]:
    position, rotation = apply_residual(base_position, base_rotation, residual)
    metric, _ = evaluate_packed_configuration(
        packed_local,
        offsets,
        names,
        position,
        rotation,
        cup,
        float(args.contact_threshold_m),
        float(args.penetration_tolerance_m),
        float(args.penetration_ratio_tolerance),
        return_world_samples=False,
        world_buf=world_buf,
        stats_buf=stats_buf,
        scratch_buf=scratch_buf,
    )
    return annotate_finger_contact_metrics(metric, args)


def plot_refinement_timeline(
    output: Path,
    frames: list[int],
    base_records: list[dict[str, Any]],
    static_records: list[dict[str, Any]],
    causal_records: list[dict[str, Any]],
    residuals: np.ndarray,
    args: argparse.Namespace,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    for label, records in (
        ("base", base_records),
        ("static", static_records),
        ("causal", causal_records),
    ):
        axes[0].plot(
            frames,
            [record["hand_penetration_ratio"] for record in records],
            label=label,
        )
        axes[1].plot(
            frames,
            [record["hand_penetration_p95_m"] for record in records],
            label=label,
        )
        axes[2].plot(
            frames,
            [
                max(
                    [record[f"{group}_surface_distance_m"] for group in FINGER_GROUPS]
                    if args.require_all_finger_contact
                    else [
                        record["thumb_surface_distance_m"],
                        record["opposition_surface_distance_m"],
                    ]
                )
                for record in records
            ],
            label=label,
        )
    axes[0].set_ylabel("penetrating fraction")
    axes[1].set_ylabel("penetration p95 (m)")
    axes[2].set_ylabel("worst contact distance (m)")
    axes[0].axhline(0.02, c="black", ls="--", lw=0.8)
    axes[1].axhline(0.005, c="black", ls="--", lw=0.8)
    axes[2].axhline(0.008, c="black", ls="--", lw=0.8)
    axes[3].plot(
        frames, 1000.0 * np.linalg.norm(residuals[:, :3], axis=1),
        label="translation",
    )
    axes[3].plot(
        frames, np.degrees(np.linalg.norm(residuals[:, 3:], axis=1)),
        label="rotation",
    )
    axes[3].set_ylabel("mm / deg")
    axes[3].set_xlabel("source frame")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    experiment = args.experiment_dir.expanduser().resolve()
    interaction = args.interaction_dir.expanduser().resolve()
    output = args.out_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    wrist_path = experiment / "wrist_candidates_native.npz"
    finger_path = experiment / "finger_candidates_native.npz"
    object_path = interaction / "object_models.jsonl"
    for path in (wrist_path, finger_path, object_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(wrist_path, allow_pickle=False) as archive:
        wrist_data = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(finger_path, allow_pickle=False) as archive:
        finger_data = {key: np.asarray(archive[key]) for key in archive.files}
    position_key = f"{args.wrist_method}__{args.side}_position"
    rotation_key = f"{args.wrist_method}__{args.side}_rotation"
    qpos_key = f"{args.finger_method}__qpos"
    for key, archive in (
        (position_key, wrist_data),
        (rotation_key, wrist_data),
        (qpos_key, finger_data),
    ):
        if key not in archive:
            raise KeyError(key)
    base_position = np.asarray(wrist_data[position_key], dtype=np.float64)
    base_rotation = np.asarray(wrist_data[rotation_key], dtype=np.float64)
    finger_qpos = np.asarray(finger_data[qpos_key], dtype=np.float64)
    frame_count = min(len(base_position), len(finger_qpos))
    start = max(0, int(args.start_frame))
    end = min(int(args.end_frame), frame_count - 1)
    frames = list(range(start, end + 1))
    if not frames:
        raise ValueError(f"empty frame interval {start}:{end}")
    cups = load_cup_models(object_path)
    missing = sorted(set(frames) - set(cups))
    if missing:
        raise KeyError(f"cup models missing frames {missing}")

    sampler = Dex3MeshSampler(
        args.side,
        int(args.mesh_samples_per_geom),
        int(args.seed),
        hand_mode=str(args.hand_mode),
    )
    packed_by_frame: dict[int, np.ndarray] = {}
    local_samples: dict[int, dict[str, np.ndarray]] = {}
    for frame in frames:
        packed = np.empty((sampler.packed_count, 3), dtype=np.float64)
        sampler.fill_packed_local_samples(finger_qpos[frame], packed)
        packed_by_frame[frame] = packed
        local_samples[frame] = sampler.unpack_packed_samples(packed)
    packed_names = sampler.packed_names
    packed_offsets = sampler.packed_offsets
    world_buf, stats_buf, scratch_buf = acquire_eval_buffers(
        sampler.packed_count, len(packed_names)
    )

    def metric_for_frame(frame: int, residual: np.ndarray) -> dict[str, Any]:
        return metric_for(
            packed_by_frame[frame],
            packed_offsets,
            packed_names,
            base_position[frame],
            base_rotation[frame],
            residual,
            cups[frame],
            args,
            world_buf=world_buf,
            stats_buf=stats_buf,
            scratch_buf=scratch_buf,
        )

    bounds = residual_bounds(
        float(args.translation_bound_m),
        float(args.rotation_component_bound_deg),
    )
    train_frames = frames[:: max(1, int(args.static_stride))]
    if train_frames[-1] != frames[-1]:
        train_frames.append(frames[-1])

    def static_objective(residual: np.ndarray) -> float:
        losses = [
            physical_loss(metric_for_frame(frame, residual), args)
            for frame in train_frames
        ]
        return float(np.mean(losses) + anchor_loss(residual))

    static_result = differential_evolution(
        static_objective,
        bounds,
        seed=int(args.seed),
        popsize=8,
        maxiter=int(args.static_max_iterations),
        polish=True,
        updating="immediate",
        workers=1,
        tol=1e-5,
    )
    static_residual = np.asarray(static_result.x, dtype=np.float64)

    zero_residual = np.zeros(6, dtype=np.float64)
    base_records: list[dict[str, Any]] = []
    static_records: list[dict[str, Any]] = []
    causal_records: list[dict[str, Any]] = []
    causal_residuals = np.empty((len(frames), 6), dtype=np.float64)
    optimizer_records: list[dict[str, Any]] = []
    previous = static_residual.copy()

    for output_index, frame in enumerate(frames):
        previous_before = previous.copy()

        def causal_objective(residual: np.ndarray) -> float:
            metric = metric_for_frame(frame, residual)
            return float(
                physical_loss(metric, args)
                + anchor_loss(residual)
                + temporal_loss(residual, previous)
            )

        candidates: list[tuple[str, np.ndarray, float, dict[str, Any], int]] = []
        previous_metric = metric_for_frame(frame, previous_before)
        if robust_grasp(previous_metric, args):
            candidates.append(
                (
                    "previous_hold",
                    previous_before,
                    float(causal_objective(previous_before)),
                    previous_metric,
                    1,
                )
            )
        else:
            powell = minimize(
                causal_objective,
                previous_before,
                method="Powell",
                bounds=bounds,
                options={
                    "maxiter": int(args.causal_max_iterations),
                    "xtol": 2e-4,
                    "ftol": 1e-5,
                },
            )
            powell_residual = np.asarray(powell.x, dtype=np.float64)
            powell_metric = metric_for_frame(frame, powell_residual)
            candidates.append(
                (
                    "powell",
                    powell_residual,
                    float(powell.fun),
                    powell_metric,
                    int(powell.nfev),
                )
            )
            if not robust_grasp(powell_metric, args):
                rescue = differential_evolution(
                    causal_objective,
                    bounds,
                    x0=previous_before,
                    seed=int(args.seed) + frame,
                    popsize=6,
                    maxiter=int(args.rescue_max_iterations),
                    polish=True,
                    updating="immediate",
                    workers=1,
                    tol=1e-5,
                )
                rescue_residual = np.asarray(rescue.x, dtype=np.float64)
                rescue_metric = metric_for_frame(frame, rescue_residual)
                candidates.append(
                    (
                        "differential_evolution_rescue",
                        rescue_residual,
                        float(rescue.fun),
                        rescue_metric,
                        int(rescue.nfev),
                    )
                )
        candidates.sort(
            key=lambda item: (
                not robust_grasp(item[3], args),
                not selected_grasp_valid(item[3], args),
                item[2],
            )
        )
        solver, residual, objective, causal_metric, evaluations = candidates[0]
        if robust_grasp(causal_metric, args) and not robust_grasp(
            previous_metric, args
        ):
            low = 0.0
            high = 1.0
            first_valid: tuple[float, dict[str, Any]] | None = None
            for alpha in np.linspace(0.05, 1.0, 20):
                trial = previous_before + alpha * (residual - previous_before)
                trial_metric = metric_for_frame(frame, trial)
                if robust_grasp(trial_metric, args):
                    first_valid = (float(alpha), trial_metric)
                    high = float(alpha)
                    low = max(0.0, high - 0.05)
                    break
            if first_valid is not None:
                for _ in range(8):
                    alpha = 0.5 * (low + high)
                    trial = previous_before + alpha * (residual - previous_before)
                    trial_metric = metric_for_frame(frame, trial)
                    if robust_grasp(trial_metric, args):
                        high = alpha
                        first_valid = (alpha, trial_metric)
                    else:
                        low = alpha
                alpha, causal_metric = first_valid
                residual = previous_before + alpha * (residual - previous_before)
                objective = float(causal_objective(residual))
                solver += "_minimal_step"
        causal_residuals[output_index] = residual
        previous = residual.copy()

        base_metric = metric_for_frame(frame, zero_residual)
        static_metric = metric_for_frame(frame, static_residual)
        for variant, metric in (
            ("base", base_metric),
            ("static", static_metric),
            ("causal", causal_metric),
        ):
            metric.update(
                {
                    "frame": frame,
                    "variant": variant,
                    "wrist_method": args.wrist_method,
                    "finger_method": args.finger_method,
                }
            )
        base_records.append(base_metric)
        static_records.append(static_metric)
        causal_records.append(causal_metric)
        optimizer_records.append(
            {
                "frame": frame,
                "solver": solver,
                "objective": objective,
                "function_evaluations": evaluations,
                "residual_translation_local_m": residual[:3].tolist(),
                "residual_rotation_vector_rad": residual[3:].tolist(),
                "translation_norm_m": float(np.linalg.norm(residual[:3])),
                "rotation_norm_deg": float(
                    np.degrees(np.linalg.norm(residual[3:]))
                ),
                "valid_grasp": bool(causal_metric["valid_grasp"]),
                "all_finger_contact": bool(causal_metric["all_finger_contact"]),
                "all_finger_valid_grasp": bool(
                    causal_metric["all_finger_valid_grasp"]
                ),
                "selected_grasp_valid": selected_grasp_valid(causal_metric, args),
            }
        )

    refined_position = base_position.copy()
    refined_rotation = base_rotation.copy()
    static_position = base_position.copy()
    static_rotation = base_rotation.copy()
    for output_index, frame in enumerate(frames):
        static_position[frame], static_rotation[frame] = apply_residual(
            base_position[frame], base_rotation[frame], static_residual
        )
        refined_position[frame], refined_rotation[frame] = apply_residual(
            base_position[frame], base_rotation[frame], causal_residuals[output_index]
        )

    np.savez_compressed(
        output / "contact_aware_wrist_candidates_native.npz",
        timestamps=wrist_data["timestamps"],
        frame_indices=np.asarray(frames, dtype=np.int64),
        static_residual_local=static_residual,
        causal_residual_local=causal_residuals,
        base_position=base_position,
        base_rotation=base_rotation,
        static_position=static_position,
        static_rotation=static_rotation,
        causal_position=refined_position,
        causal_rotation=refined_rotation,
    )
    with (output / "contact_aware_frame_metrics.jsonl").open("w") as stream:
        for variant_records in (base_records, static_records, causal_records):
            for record in variant_records:
                stream.write(json.dumps(record) + "\n")
    with (output / "contact_aware_optimizer.jsonl").open("w") as stream:
        for record in optimizer_records:
            stream.write(json.dumps(record) + "\n")

    summaries = {
        "base": summarize_finger_contacts(base_records, args),
        "static": summarize_finger_contacts(static_records, args),
        "causal": summarize_finger_contacts(causal_records, args),
    }
    summaries["static"]["residual_translation_local_m"] = static_residual[:3].tolist()
    summaries["static"]["residual_rotation_vector_deg"] = np.degrees(
        static_residual[3:]
    ).tolist()
    summaries["static"]["optimizer_objective"] = float(static_result.fun)
    summaries["static"]["optimizer_success"] = bool(static_result.success)
    summaries["causal"]["translation_norm_m"] = finite_stats(
        np.linalg.norm(causal_residuals[:, :3], axis=1)
    )
    summaries["causal"]["rotation_norm_deg"] = finite_stats(
        np.degrees(np.linalg.norm(causal_residuals[:, 3:], axis=1))
    )
    summaries["causal"]["rescue_frames"] = [
        record["frame"]
        for record in optimizer_records
        if record["solver"].startswith("differential_evolution_rescue")
    ]
    summaries["causal"]["hold_frames"] = [
        record["frame"]
        for record in optimizer_records
        if record["solver"] == "previous_hold"
    ]
    summaries["causal"]["minimal_step_frames"] = [
        record["frame"]
        for record in optimizer_records
        if record["solver"].endswith("_minimal_step")
    ]
    summary_document = {
        "schema_version": 2,
        "experiment_only": True,
        "production_pipeline_modified": False,
        "require_all_finger_contact": bool(args.require_all_finger_contact),
        "contact_constraint": (
            "thumb_index_middle"
            if args.require_all_finger_contact
            else "thumb_and_best_opposition"
        ),
        "side": args.side,
        "base_wrist_method": args.wrist_method,
        "finger_method_unchanged": args.finger_method,
        "frame_range_inclusive": [start, end],
        "static_training_frames": train_frames,
        "thresholds": {
            "contact_m": float(args.contact_threshold_m),
            "penetration_p95_m": float(args.penetration_tolerance_m),
            "penetration_ratio": float(args.penetration_ratio_tolerance),
        },
        "optimization_targets": {
            "contact_m": float(args.optimization_contact_target_m),
            "penetration_p95_m": float(args.optimization_penetration_target_m),
            "penetration_ratio": float(
                args.optimization_penetration_ratio_target
            ),
            "robust_contact_m": float(args.robust_contact_m),
            "robust_penetration_p95_m": float(args.robust_penetration_m),
            "robust_penetration_ratio": float(args.robust_penetration_ratio),
        },
        "summaries": summaries,
    }
    (output / "contact_aware_summary.json").write_text(
        json.dumps(summary_document, indent=2) + "\n"
    )
    plot_refinement_timeline(
        output / "contact_aware_timeline.png",
        frames,
        base_records,
        static_records,
        causal_records,
        causal_residuals,
        args,
    )

    keyframe = int(args.keyframe)
    if keyframe not in frames:
        keyframe = frames[len(frames) // 2]
    keyframe_index = frames.index(keyframe)
    local = local_samples[keyframe]
    entries = []
    for variant, residual in (
        ("base", zero_residual),
        ("static", static_residual),
        ("causal", causal_residuals[keyframe_index]),
    ):
        position, rotation = apply_residual(
            base_position[keyframe], base_rotation[keyframe], residual
        )
        metric, samples = evaluate_configuration(
            local,
            position,
            rotation,
            cups[keyframe],
            float(args.contact_threshold_m),
            float(args.penetration_tolerance_m),
            float(args.penetration_ratio_tolerance),
        )
        annotate_finger_contact_metrics(metric, args)
        entries.append((variant, args.finger_method, metric, samples))
    # The object point cloud is not required by optimization. Use fitted-body
    # points for a clean comparison visualization.
    cup = cups[keyframe]
    angles = np.linspace(0.0, 2.0 * np.pi, 160)
    heights = np.linspace(cup.z_min, cup.z_max, 20)
    cup_points = np.concatenate(
        [
            np.column_stack(
                [
                    cup.center_xy[0] + float(cup.radius_at(z)) * np.cos(angles),
                    cup.center_xy[1] + float(cup.radius_at(z)) * np.sin(angles),
                    np.full_like(angles, z),
                ]
            )
            for z in heights
        ],
        axis=0,
    )
    plot_keyframe(
        output / f"contact_aware_keyframe_{keyframe:06d}.png",
        keyframe,
        cup,
        cup_points,
        entries,
        point_cloud_only=True,
        dpi=300,
        point_size=1.2,
        point_alpha=0.58,
    )

    manifest = {
        "schema_version": 2,
        "experiment_only": True,
        "production_pipeline_modified": False,
        "require_all_finger_contact": bool(args.require_all_finger_contact),
        "contact_constraint": (
            "thumb_index_middle"
            if args.require_all_finger_contact
            else "thumb_and_best_opposition"
        ),
        "source_experiment_dir": str(experiment),
        "source_interaction_dir": str(interaction),
        "output_dir": str(output),
        "static_contract": "one local wrist_yaw_link residual shared by all frames",
        "causal_contract": (
            "current cup observation + current hand candidate + previous residual only"
        ),
        "finger_joint_trajectory_modified": False,
        "outputs": [
            "contact_aware_wrist_candidates_native.npz",
            "contact_aware_frame_metrics.jsonl",
            "contact_aware_optimizer.jsonl",
            "contact_aware_summary.json",
            "contact_aware_timeline.png",
            f"contact_aware_keyframe_{keyframe:06d}.png",
        ],
    }
    (output / "contact_aware_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
