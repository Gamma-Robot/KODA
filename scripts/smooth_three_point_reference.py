#!/usr/bin/env python3
"""Repair wrist excursions, then smooth the three-point tracking reference.

The input reference is already sampled on the tracking timeline.  Translation
is filtered in the ScaleTrack Z-up world frame, while orientation is filtered
in a local SO(3) tangent space.  Wrist targets are updated from the exact same
smoothed poses so Kimodo preflight, ScaleTrack, and Mink share one reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from smooth_wrist_reference import (
    _repair_transient_pose_outliers,
    _validate_filter,
    savgol_rotations,
    savgol_time_series,
    timestamp_segments,
)


TRACKING_BODIES = (
    "torso_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
WRIST_TARGET_FIELDS = {
    "left_wrist_yaw_link": ("left_wrist", "left_wrist_quat_wxyz"),
    "right_wrist_yaw_link": ("right_wrist", "right_wrist_quat_wxyz"),
}
POSE_FIELDS = (
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def string_list(values: np.ndarray) -> list[str]:
    return [str(value) for value in np.asarray(values).reshape(-1).tolist()]


def rotation_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternion, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(~np.isfinite(values)) or np.any(norms <= 1.0e-12):
        raise ValueError("quaternion input is non-finite or has zero norm")
    return Rotation.from_quat(values / norms, scalar_first=True).as_matrix()


def continuous_wxyz(rotation: np.ndarray) -> np.ndarray:
    quaternion = Rotation.from_matrix(rotation).as_quat(scalar_first=True)
    if quaternion.ndim == 1:
        quaternion = quaternion[None]
    for index in range(1, len(quaternion)):
        if float(np.dot(quaternion[index - 1], quaternion[index])) < 0.0:
            quaternion[index] *= -1.0
    return quaternion


def finite_difference(values: np.ndarray, timestamps: np.ndarray, gap_factor: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    result = np.zeros_like(values)
    segments, _ = timestamp_segments(timestamps, gap_factor)
    for start, end in segments:
        length = end - start
        if length == 2:
            velocity = (values[start + 1] - values[start]) / (timestamps[start + 1] - timestamps[start])
            result[start:end] = velocity
        elif length >= 3:
            result[start:end] = np.gradient(
                values[start:end], timestamps[start:end], axis=0, edge_order=2
            )
    return result


def angular_velocity_world(
    rotation: np.ndarray, timestamps: np.ndarray, gap_factor: float
) -> np.ndarray:
    matrices = np.asarray(rotation, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    result = np.zeros((len(matrices), 3), dtype=np.float64)
    segments, _ = timestamp_segments(timestamps, gap_factor)
    for start, end in segments:
        if end - start < 2:
            continue
        for index in range(start, end):
            if index == start:
                previous, following = index, index + 1
            elif index == end - 1:
                previous, following = index - 1, index
            else:
                previous, following = index - 1, index + 1
            delta_t = timestamps[following] - timestamps[previous]
            result[index] = Rotation.from_matrix(
                matrices[following] @ matrices[previous].T
            ).as_rotvec() / delta_t
    return result


def stats(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95.0)),
        "max": float(np.max(data)),
    }


def repair_transient_wrist_outliers(
    timestamps: np.ndarray,
    positions: np.ndarray,
    rotations: np.ndarray,
    *,
    gap_factor: float,
    max_event_duration_s: float,
    position_floor_m: float,
    orientation_floor_deg: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Repair bounded excursions without pairing anchors across time gaps."""
    repaired_position = np.asarray(positions, dtype=np.float64).copy()
    repaired_rotation = np.asarray(rotations, dtype=np.float64).copy()
    segments, timeline = timestamp_segments(timestamps, gap_factor)
    repaired_indices: list[int] = []
    candidate_edges: list[int] = []
    events: list[dict[str, Any]] = []
    segment_reports: list[dict[str, Any]] = []
    for start, end in segments:
        segment_position, segment_rotation, report = _repair_transient_pose_outliers(
            timestamps[start:end],
            repaired_position[start:end],
            repaired_rotation[start:end],
            max_event_duration_s=max_event_duration_s,
            position_floor_m=position_floor_m,
            orientation_floor_deg=orientation_floor_deg,
        )
        repaired_position[start:end] = segment_position
        repaired_rotation[start:end] = segment_rotation
        segment_repaired = [
            start + int(index)
            for index in report["repaired_source_frame_indices"]
        ]
        segment_edges = [
            start + int(index) for index in report["candidate_anomalous_edges"]
        ]
        repaired_indices.extend(segment_repaired)
        candidate_edges.extend(segment_edges)
        for event in report["events"]:
            global_event = dict(event)
            for field in (
                "left_anchor_index",
                "right_anchor_index",
                "first_repaired_index",
                "last_repaired_index",
            ):
                global_event[field] = start + int(global_event[field])
            events.append(global_event)
        segment_reports.append(
            {
                "start_index": int(start),
                "end_index": int(end - 1),
                "position_jump_threshold_m": float(
                    report["position_jump_threshold_m"]
                ),
                "orientation_jump_threshold_deg": float(
                    report["orientation_jump_threshold_deg"]
                ),
                "candidate_anomalous_edges": segment_edges,
                "repaired_frame_indices": segment_repaired,
            }
        )
    return repaired_position, repaired_rotation, {
        "method": (
            "bounded jump-out-and-return excursion removal; linear position "
            "interpolation and quaternion Slerp between clean anchors"
        ),
        "coordinate_frame": "ScaleTrack Z-up world",
        "max_event_duration_s": float(max_event_duration_s),
        "position_jump_floor_m": float(position_floor_m),
        "orientation_jump_floor_deg": float(orientation_floor_deg),
        "threshold_policy": "max(floor, median + 8 * 1.4826 * MAD)",
        "endpoint_extrapolation": False,
        "cross_timestamp_gap_repair": False,
        "timeline": timeline,
        "candidate_anomalous_edges": candidate_edges,
        "repaired_frame_indices": repaired_indices,
        "repaired_frame_count": len(repaired_indices),
        "events": events,
        "segments": segment_reports,
    }


def update_d435_pose(reference: dict[str, np.ndarray], body_names: list[str]) -> bool:
    required = {
        "d435_pos_w",
        "d435_quat_wxyz",
        "torso_to_d435_position_m",
        "torso_to_d435_quat_wxyz",
    }
    if not required.issubset(reference):
        return False
    torso_index = body_names.index("torso_link")
    torso_position = np.asarray(reference["body_pos_w"][:, torso_index], dtype=np.float64)
    torso_rotation = rotation_from_wxyz(reference["body_quat_w"][:, torso_index])
    mount_position = np.asarray(reference["torso_to_d435_position_m"], dtype=np.float64).reshape(3)
    mount_rotation = rotation_from_wxyz(
        np.asarray(reference["torso_to_d435_quat_wxyz"], dtype=np.float64).reshape(1, 4)
    )[0]
    d435_position = torso_position + np.einsum(
        "tij,j->ti", torso_rotation, mount_position
    )
    d435_rotation = np.einsum("tij,jk->tik", torso_rotation, mount_rotation)
    reference["d435_pos_w"] = d435_position.astype(reference["d435_pos_w"].dtype)
    reference["d435_quat_wxyz"] = continuous_wxyz(d435_rotation).astype(
        reference["d435_quat_wxyz"].dtype
    )
    return True


def smooth_reference(
    *,
    input_reference: Path,
    input_targets: Path,
    output_reference: Path,
    output_targets: Path,
    meta_json: Path,
    window_length: int,
    polyorder: int,
    gap_factor: float,
    wrist_outlier_max_duration_s: float = 1.1,
    wrist_outlier_position_floor_m: float = 0.05,
    wrist_outlier_orientation_floor_deg: float = 45.0,
) -> dict[str, Any]:
    _validate_filter(window_length, polyorder)
    if gap_factor <= 1.0:
        raise ValueError("gap_factor must be greater than 1")
    if wrist_outlier_max_duration_s <= 0.0:
        raise ValueError("wrist_outlier_max_duration_s must be positive")
    if wrist_outlier_position_floor_m <= 0.0:
        raise ValueError("wrist_outlier_position_floor_m must be positive")
    if wrist_outlier_orientation_floor_deg <= 0.0:
        raise ValueError("wrist_outlier_orientation_floor_deg must be positive")
    for path in (input_reference, input_targets):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_reference.resolve() == input_reference.resolve():
        raise ValueError("output_reference must not overwrite input_reference")
    if output_targets.resolve() == input_targets.resolve():
        raise ValueError("output_targets must not overwrite input_targets")

    reference = load_npz(input_reference)
    targets = load_npz(input_targets)
    required_reference = {"body_names", "timestamps", *POSE_FIELDS}
    missing = sorted(required_reference - reference.keys())
    if missing:
        raise KeyError(f"reference is missing fields: {missing}")
    if "timestamps" not in targets:
        raise KeyError("targets is missing timestamps")
    body_names = string_list(reference["body_names"])
    for body in TRACKING_BODIES:
        if body not in body_names:
            raise KeyError(f"reference is missing tracking body {body}")

    timestamps = np.asarray(reference["timestamps"], dtype=np.float64)
    target_time = np.asarray(targets["timestamps"], dtype=np.float64)
    frames = len(timestamps)
    if target_time.shape != timestamps.shape or not np.allclose(
        target_time, timestamps, rtol=0.0, atol=1.0e-9
    ):
        raise ValueError("reference and targets do not share one timeline")
    if np.any(~np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("timestamps must be finite and strictly increasing")

    source_reference = {name: value.copy() for name, value in reference.items()}
    source_targets = {name: value.copy() for name, value in targets.items()}
    body_metrics: dict[str, Any] = {}
    outlier_repair: dict[str, Any] = {}
    for body in TRACKING_BODIES:
        body_index = body_names.index(body)
        raw_position = np.asarray(reference["body_pos_w"][:, body_index], dtype=np.float64)
        raw_rotation = rotation_from_wxyz(reference["body_quat_w"][:, body_index])
        filter_position = raw_position
        filter_rotation = raw_rotation
        if body in WRIST_TARGET_FIELDS:
            filter_position, filter_rotation, repair_report = (
                repair_transient_wrist_outliers(
                    timestamps,
                    raw_position,
                    raw_rotation,
                    gap_factor=gap_factor,
                    max_event_duration_s=wrist_outlier_max_duration_s,
                    position_floor_m=wrist_outlier_position_floor_m,
                    orientation_floor_deg=wrist_outlier_orientation_floor_deg,
                )
            )
            outlier_repair[body] = repair_report
        smooth_position = savgol_time_series(
            timestamps, filter_position, window_length, polyorder, gap_factor
        )
        smooth_rotation = savgol_rotations(
            timestamps, filter_rotation, window_length, polyorder, gap_factor
        )
        smooth_quaternion = continuous_wxyz(smooth_rotation)
        reference["body_pos_w"][:, body_index] = smooth_position.astype(
            reference["body_pos_w"].dtype
        )
        reference["body_quat_w"][:, body_index] = smooth_quaternion.astype(
            reference["body_quat_w"].dtype
        )
        reference["body_lin_vel_w"][:, body_index] = finite_difference(
            smooth_position, timestamps, gap_factor
        ).astype(reference["body_lin_vel_w"].dtype)
        reference["body_ang_vel_w"][:, body_index] = angular_velocity_world(
            smooth_rotation, timestamps, gap_factor
        ).astype(reference["body_ang_vel_w"].dtype)
        position_delta = np.linalg.norm(smooth_position - raw_position, axis=1)
        orientation_delta = np.degrees(
            (
                Rotation.from_matrix(raw_rotation).inv()
                * Rotation.from_matrix(smooth_rotation)
            ).magnitude()
        )
        body_metrics[body] = {
            "position_delta_m": stats(position_delta),
            "orientation_delta_deg": stats(orientation_delta),
            "savgol_position_delta_from_repaired_m": stats(
                np.linalg.norm(smooth_position - filter_position, axis=1)
            ),
            "savgol_orientation_delta_from_repaired_deg": stats(
                np.degrees(
                    (
                        Rotation.from_matrix(filter_rotation).inv()
                        * Rotation.from_matrix(smooth_rotation)
                    ).magnitude()
                )
            ),
        }
        if body in WRIST_TARGET_FIELDS:
            position_field, quaternion_field = WRIST_TARGET_FIELDS[body]
            if position_field not in targets or quaternion_field not in targets:
                raise KeyError(
                    f"targets is missing {position_field}/{quaternion_field}"
                )
            targets[position_field] = smooth_position.astype(targets[position_field].dtype)
            targets[quaternion_field] = smooth_quaternion.astype(
                targets[quaternion_field].dtype
            )

    heading_metric = None
    heading_field = "kimodo_pelvis_heading_quat_wxyz"
    if heading_field in reference:
        raw_heading = rotation_from_wxyz(reference[heading_field])
        smooth_heading = savgol_rotations(
            timestamps, raw_heading, window_length, polyorder, gap_factor
        )
        smooth_heading_quaternion = continuous_wxyz(smooth_heading)
        reference[heading_field] = smooth_heading_quaternion.astype(
            reference[heading_field].dtype
        )
        if heading_field in targets:
            targets[heading_field] = smooth_heading_quaternion.astype(
                targets[heading_field].dtype
            )
        heading_delta = np.degrees(
            (
                Rotation.from_matrix(raw_heading).inv()
                * Rotation.from_matrix(smooth_heading)
            ).magnitude()
        )
        heading_metric = {"orientation_delta_deg": stats(heading_delta)}

    d435_updated = update_d435_pose(reference, body_names)
    config = {
        "method": "bounded wrist excursion repair then timestamp-aware local Savitzky-Golay",
        "window_length": int(window_length),
        "polyorder": int(polyorder),
        "gap_factor": float(gap_factor),
        "position_space": "ScaleTrack Z-up world",
        "orientation_space": "local SO(3) tangent",
        "edge_policy": "shifted full window evaluated at actual timestamp",
        "wrist_outlier_repair": {
            "enabled": True,
            "max_event_duration_s": float(wrist_outlier_max_duration_s),
            "position_jump_floor_m": float(wrist_outlier_position_floor_m),
            "orientation_jump_floor_deg": float(
                wrist_outlier_orientation_floor_deg
            ),
        },
    }
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    for arrays in (reference, targets):
        arrays["reference_smoothing_applied"] = np.asarray(True)
        arrays["reference_smoothing_config_json"] = np.asarray(config_json)
        arrays["reference_smoothing_input_reference_sha256"] = np.asarray(
            sha256(input_reference)
        )
        arrays["reference_smoothing_input_targets_sha256"] = np.asarray(
            sha256(input_targets)
        )

    tracking_indices = [body_names.index(body) for body in TRACKING_BODIES]
    non_tracking = np.ones(len(body_names), dtype=bool)
    non_tracking[tracking_indices] = False
    for field in POSE_FIELDS:
        if not np.array_equal(
            source_reference[field][:, non_tracking], reference[field][:, non_tracking]
        ):
            raise RuntimeError(f"non-tracking bodies changed in {field}")
    for field in ("left_wrist", "right_wrist", "left_wrist_quat_wxyz", "right_wrist_quat_wxyz"):
        body = "left_wrist_yaw_link" if field.startswith("left") else "right_wrist_yaw_link"
        body_index = body_names.index(body)
        expected = (
            reference["body_pos_w"][:, body_index]
            if field.endswith("wrist")
            else reference["body_quat_w"][:, body_index]
        )
        if not np.array_equal(targets[field], expected.astype(targets[field].dtype)):
            raise RuntimeError(f"smoothed target mismatch for {field}")
    preserved_target_fields = set(source_targets) - {
        "left_wrist",
        "right_wrist",
        "left_wrist_quat_wxyz",
        "right_wrist_quat_wxyz",
        heading_field,
    }
    for field in preserved_target_fields:
        if field.startswith("reference_smoothing_"):
            continue
        if not np.array_equal(source_targets[field], targets[field]):
            raise RuntimeError(f"unrelated target field changed: {field}")

    atomic_npz(output_reference, reference)
    atomic_npz(output_targets, targets)
    _, timeline = timestamp_segments(timestamps, gap_factor)
    report = {
        "schema_version": 2,
        "stage": "post_contact_pre_tracking",
        "filter": config,
        "tracking_bodies": list(TRACKING_BODIES),
        "body_metrics": body_metrics,
        "pre_savgol_wrist_outlier_repair": outlier_repair,
        "kimodo_pelvis_heading_metric": heading_metric,
        "timeline": timeline,
        "d435_pose_recomputed_from_smoothed_torso": d435_updated,
        "inputs": {
            "reference": str(input_reference.resolve()),
            "reference_sha256": sha256(input_reference),
            "targets": str(input_targets.resolve()),
            "targets_sha256": sha256(input_targets),
        },
        "outputs": {
            "reference": str(output_reference.resolve()),
            "reference_sha256": sha256(output_reference),
            "targets": str(output_targets.resolve()),
            "targets_sha256": sha256(output_targets),
        },
        "contract": {
            "kimodo_reference": "smoothed reference and smoothed wrist targets",
            "mpc_reference": "smoothed reference",
            "mink_reference": "smoothed reference",
            "dex3_hand_qpos": "unchanged joint-contact/position-retarget trajectory",
        },
    }
    atomic_json(meta_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-reference", type=Path, required=True)
    parser.add_argument("--input-targets", type=Path, required=True)
    parser.add_argument("--output-reference", type=Path, required=True)
    parser.add_argument("--output-targets", type=Path, required=True)
    parser.add_argument("--meta-json", type=Path, required=True)
    parser.add_argument("--window-length", type=int, default=5)
    parser.add_argument("--polyorder", type=int, default=2)
    parser.add_argument("--gap-factor", type=float, default=3.0)
    parser.add_argument("--wrist-outlier-max-duration-s", type=float, default=1.1)
    parser.add_argument("--wrist-outlier-position-floor-m", type=float, default=0.05)
    parser.add_argument(
        "--wrist-outlier-orientation-floor-deg", type=float, default=45.0
    )
    args = parser.parse_args()
    report = smooth_reference(
        input_reference=args.input_reference.expanduser().resolve(),
        input_targets=args.input_targets.expanduser().resolve(),
        output_reference=args.output_reference.expanduser().resolve(),
        output_targets=args.output_targets.expanduser().resolve(),
        meta_json=args.meta_json.expanduser().resolve(),
        window_length=args.window_length,
        polyorder=args.polyorder,
        gap_factor=args.gap_factor,
        wrist_outlier_max_duration_s=args.wrist_outlier_max_duration_s,
        wrist_outlier_position_floor_m=args.wrist_outlier_position_floor_m,
        wrist_outlier_orientation_floor_deg=(
            args.wrist_outlier_orientation_floor_deg
        ),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
