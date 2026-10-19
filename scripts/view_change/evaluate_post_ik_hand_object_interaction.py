#!/usr/bin/env python3
"""Evaluate the realized arms-only IK wrist against the observed cup.

This experiment-only sidecar resamples the actual ``wrist_yaw_link`` pose
from an IK kinematics log onto the native RGB-D timeline, applies the selected
Dex3 finger trajectory, and reuses the fitted cup models for contact and
penetration scoring. It never modifies production retargeting inputs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from evaluate_hand_object_interaction import (
    CupModel,
    CupObservationLoader,
    Dex3MeshSampler,
    evaluate_configuration,
    finite_stats,
    nearest_index,
    plot_keyframe,
    summarize_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ik-kinematics", type=Path, required=True)
    parser.add_argument("--hand-kinematics", type=Path, required=True)
    parser.add_argument("--contact-candidates", type=Path, required=True)
    parser.add_argument("--object-models", type=Path, required=True)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument(
        "--hand-mode",
        choices=("dex3", "g1_brainco"),
        default="dex3",
    )
    parser.add_argument("--keyframe", type=int, default=331)
    parser.add_argument("--contact-threshold-m", type=float, default=0.008)
    parser.add_argument("--penetration-tolerance-m", type=float, default=0.005)
    parser.add_argument("--penetration-ratio-tolerance", type=float, default=0.02)
    parser.add_argument("--mesh-samples-per-geom", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip timeline/keyframe PNGs; JSONL and summary are still written.",
    )
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    return Rotation.from_quat(q[..., [1, 2, 3, 0]]).as_matrix()


def interpolate_pose(
    source_time: np.ndarray,
    position: np.ndarray,
    rotation: np.ndarray,
    query_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_time, dtype=np.float64).reshape(-1)
    query = np.asarray(query_time, dtype=np.float64).reshape(-1)
    if len(source) != len(position) or len(source) != len(rotation):
        raise ValueError("pose and timestamp lengths differ")
    if not len(source) or np.any(np.diff(source) <= 0.0):
        raise ValueError("pose timestamps must be non-empty and increasing")
    if np.any(query < source[0] - 1e-9) or np.any(query > source[-1] + 1e-9):
        raise ValueError("query time outside IK timeline")
    query = np.clip(query, source[0], source[-1])
    interpolated_position = np.column_stack(
        [np.interp(query, source, position[:, axis]) for axis in range(3)]
    )
    interpolated_rotation = Slerp(source, Rotation.from_matrix(rotation))(query).as_matrix()
    return interpolated_position, interpolated_rotation


def interpolate_columns(
    source_time: np.ndarray, values: np.ndarray, query_time: np.ndarray
) -> np.ndarray:
    source = np.asarray(source_time, dtype=np.float64).reshape(-1)
    array = np.asarray(values, dtype=np.float64)
    query = np.asarray(query_time, dtype=np.float64).reshape(-1)
    if len(source) != len(array):
        raise ValueError("value and timestamp lengths differ")
    return np.column_stack(
        [np.interp(query, source, array[:, column]) for column in range(array.shape[1])]
    )


def load_cup_models(path: Path) -> dict[int, CupModel]:
    names = {field.name for field in fields(CupModel)}
    models: dict[int, CupModel] = {}
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            model = CupModel(**{key: record[key] for key in names})
            models[int(model.frame)] = model
    if not models:
        raise ValueError(f"no cup models in {path}")
    return models


def rotation_error_deg(reference: np.ndarray, realized: np.ndarray) -> np.ndarray:
    delta = Rotation.from_matrix(reference).inv() * Rotation.from_matrix(realized)
    return np.degrees(delta.magnitude())


def plot_timeline(output: Path, records: list[dict[str, Any]]) -> None:
    frames = np.asarray([record["frame"] for record in records])
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    axes[0, 0].plot(frames, [record["thumb_surface_distance_m"] for record in records], label="thumb")
    axes[0, 0].plot(frames, [record["opposition_surface_distance_m"] for record in records], label="opposition")
    axes[0, 0].axhline(0.008, color="black", linestyle="--", linewidth=0.8)
    axes[0, 0].set_ylabel("surface distance (m)")
    axes[0, 0].legend()
    axes[0, 1].plot(frames, [record["hand_penetration_ratio"] for record in records])
    axes[0, 1].axhline(0.02, color="black", linestyle="--", linewidth=0.8)
    axes[0, 1].set_ylabel("penetrating mesh ratio")
    axes[1, 0].plot(frames, [1000.0 * record["wrist_tracking_position_m"] for record in records])
    axes[1, 0].set_ylabel("wrist position error (mm)")
    axes[1, 1].plot(frames, [record["wrist_tracking_orientation_deg"] for record in records])
    axes[1, 1].set_ylabel("wrist orientation error (deg)")
    for axis in axes.reshape(-1):
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("native source frame")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    paths = {
        "ik_kinematics": args.ik_kinematics.expanduser().resolve(),
        "hand_kinematics": args.hand_kinematics.expanduser().resolve(),
        "contact_candidates": args.contact_candidates.expanduser().resolve(),
        "object_models": args.object_models.expanduser().resolve(),
        "session_dir": args.session_dir.expanduser().resolve(),
        "mask_dir": args.mask_dir.expanduser().resolve(),
    }
    output = args.out_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("ik_kinematics", "hand_kinematics", "contact_candidates", "object_models"):
        if not paths[name].is_file():
            raise FileNotFoundError(paths[name])
    if not paths["session_dir"].is_dir() or not paths["mask_dir"].is_dir():
        raise FileNotFoundError("session or cup mask directory is missing")

    ik = load_npz(paths["ik_kinematics"])
    hand = load_npz(paths["hand_kinematics"])
    contact = load_npz(paths["contact_candidates"])
    cups = load_cup_models(paths["object_models"])

    body_names = [str(value) for value in ik["body_names"].tolist()]
    wrist_name = f"{args.side}_wrist_yaw_link"
    if wrist_name not in body_names:
        raise KeyError(wrist_name)
    wrist_index = body_names.index(wrist_name)
    fps_key = "sparsetrack_fps" if "sparsetrack_fps" in ik else "fps"
    ik_fps = float(np.asarray(ik[fps_key]).reshape(()))
    ik_time = np.arange(len(ik["body_pos_w"]), dtype=np.float64) / ik_fps
    ik_position = np.asarray(ik["body_pos_w"][:, wrist_index], dtype=np.float64)
    ik_rotation = wxyz_to_matrix(ik["body_quat_w"][:, wrist_index])

    native_time = np.asarray(contact["timestamps"], dtype=np.float64)
    native_time = native_time - native_time[0]
    frame_indices = np.asarray(contact["frame_indices"], dtype=np.int64)
    frames_to_evaluate = [int(frame) for frame in frame_indices if int(frame) in cups]
    if not frames_to_evaluate:
        raise ValueError("contact interval and cup models do not overlap")
    query_time = native_time[frames_to_evaluate]
    realized_position, realized_rotation = interpolate_pose(
        ik_time, ik_position, ik_rotation, query_time
    )

    hand_time = np.asarray(hand["timestamps"], dtype=np.float64)
    hand_time = hand_time - hand_time[0]
    hand_qpos = interpolate_columns(hand_time, hand["qpos"], query_time)
    finger_method = str(np.asarray(hand.get("source_method", "unknown")).reshape(()))
    ideal_position = np.asarray(
        contact.get("reference_causal_position", contact["causal_position"]),
        dtype=np.float64,
    )[frames_to_evaluate]
    ideal_rotation = np.asarray(
        contact.get("reference_causal_rotation", contact["causal_rotation"]),
        dtype=np.float64,
    )[frames_to_evaluate]
    position_error = np.linalg.norm(realized_position - ideal_position, axis=1)
    orientation_error = rotation_error_deg(ideal_rotation, realized_rotation)

    sampler = Dex3MeshSampler(
        args.side,
        int(args.mesh_samples_per_geom),
        int(args.seed),
        hand_mode=str(args.hand_mode),
    )
    records: list[dict[str, Any]] = []
    keyframe_samples: dict[str, np.ndarray] | None = None
    keyframe_metric: dict[str, Any] | None = None
    for offset, frame in enumerate(frames_to_evaluate):
        local = sampler.local_samples(hand_qpos[offset])
        metric, samples = evaluate_configuration(
            local,
            realized_position[offset],
            realized_rotation[offset],
            cups[frame],
            float(args.contact_threshold_m),
            float(args.penetration_tolerance_m),
            float(args.penetration_ratio_tolerance),
        )
        metric.update(
            {
                "frame": frame,
                "native_time_s": float(query_time[offset]),
                "nearest_ik_frame": int(nearest_index(ik_time, query_time[offset])),
                "side": args.side,
                "finger_method": finger_method,
                "wrist_source": "realized_arms_only_mink_ik",
                "wrist_tracking_position_m": float(position_error[offset]),
                "wrist_tracking_orientation_deg": float(orientation_error[offset]),
            }
        )
        records.append(metric)
        if frame == int(args.keyframe):
            keyframe_metric = metric
            keyframe_samples = samples

    with (output / "post_ik_interaction_metrics.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    summary = {
        "schema_version": 1,
        "experiment_only": True,
        "production_pipeline_modified": False,
        "side": args.side,
        "finger_method": finger_method,
        "native_frame_range_inclusive": [frames_to_evaluate[0], frames_to_evaluate[-1]],
        "thresholds": {
            "contact_m": float(args.contact_threshold_m),
            "penetration_p95_m": float(args.penetration_tolerance_m),
            "penetration_ratio": float(args.penetration_ratio_tolerance),
        },
        "interaction": summarize_records(records),
        "wrist_tracking": {
            "position_m": finite_stats(position_error),
            "orientation_deg": finite_stats(orientation_error),
        },
    }
    offline_meta = paths["ik_kinematics"].with_name("offline_ik.json")
    if offline_meta.is_file():
        summary["ik_configuration"] = json.loads(offline_meta.read_text())
    (output / "post_ik_interaction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    if not args.skip_plots:
        plot_timeline(output / "post_ik_interaction_timeline.png", records)

    keyframe = int(args.keyframe)
    if keyframe not in frames_to_evaluate:
        keyframe = frames_to_evaluate[len(frames_to_evaluate) // 2]
        index = frames_to_evaluate.index(keyframe)
        local = sampler.local_samples(hand_qpos[index])
        keyframe_metric, keyframe_samples = evaluate_configuration(
            local,
            realized_position[index],
            realized_rotation[index],
            cups[keyframe],
            float(args.contact_threshold_m),
            float(args.penetration_tolerance_m),
            float(args.penetration_ratio_tolerance),
        )
    assert keyframe_metric is not None and keyframe_samples is not None
    if not args.skip_plots:
        with CupObservationLoader(
            paths["session_dir"],
            paths["mask_dir"],
            dbscan_eps_m=0.012,
            dbscan_min_samples=8,
        ) as loader:
            cup_points, _ = loader.points_for_frame(keyframe)
        plot_keyframe(
            output / f"post_ik_keyframe_{keyframe:06d}.png",
            keyframe,
            cups[keyframe],
            cup_points,
            [("realized Mink IK", finger_method, keyframe_metric, keyframe_samples)],
        )

    manifest = {
        "schema_version": 1,
        "experiment_only": True,
        "production_pipeline_modified": False,
        **{key: str(value) for key, value in paths.items()},
        "output_dir": str(output),
        "coordinate_contract": (
            "actual G1 wrist_yaw_link pose from arms-only Mink IK in Z-up world; "
            "complete G1+Dex3 visual mesh evaluated against RGB-D fitted cup"
        ),
        "outputs": [
            "post_ik_interaction_metrics.jsonl",
            "post_ik_interaction_summary.json",
            "post_ik_interaction_timeline.png",
            f"post_ik_keyframe_{keyframe:06d}.png",
        ],
    }
    (output / "post_ik_interaction_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
