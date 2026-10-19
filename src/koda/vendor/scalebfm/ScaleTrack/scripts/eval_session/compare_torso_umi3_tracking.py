#!/usr/bin/env python3
"""Compare wrist-only and torso-plus-wrists tracking kinematics offline.

Both robot runs are compared against the same three-point reference.  The
source Stera camera pose stored by the reference is expressed as a Unitree
D435-link pose.  If a run has no explicit ``d435_link`` body, its D435 pose is
computed rigidly from ``torso_link`` with the same fixed mount transform.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from build_torso_umi3_reference import (
    EXPECTED_FPS,
    EXPECTED_FRAMES,
    TORSO_BODY,
    WRIST_BODIES,
    d435_from_torso,
)


D435_BODY = "d435_link"
WRIST_POSITION_ACCEPTANCE_LIMIT_M = 0.05
PRIMARY_METRICS = (
    "left_wrist_position_m",
    "left_wrist_orientation_deg",
    "right_wrist_position_m",
    "right_wrist_orientation_deg",
    "wrist_position_m",
    "wrist_orientation_deg",
    "torso_position_m",
    "torso_orientation_deg",
    "d435_position_m",
    "d435_orientation_deg",
    "d435_horizontal_xy_m",
    "d435_height_abs_m",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _strings(value: Any) -> list[str]:
    return [str(item) for item in np.asarray(value).reshape(-1).tolist()]


def _body_names(arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> list[str]:
    embedded = _strings(arrays["body_names"]) if "body_names" in arrays else None
    from_meta = metadata.get("body_names")
    if from_meta is None and isinstance(metadata.get("dimensions"), dict):
        from_meta = metadata.get("body_index_to_name")
    if isinstance(from_meta, dict):
        from_meta = [
            from_meta[str(index)] for index in range(len(from_meta))
        ]
    names = embedded if embedded is not None else [str(item) for item in from_meta or []]
    if not names:
        raise ValueError("kinematics body names are absent from both NPZ and metadata")
    if len(set(names)) != len(names):
        raise ValueError("kinematics body names contain duplicates")
    if embedded is not None and from_meta is not None:
        meta_names = [str(item) for item in from_meta]
        if embedded != meta_names:
            raise ValueError("NPZ and metadata body names disagree")
    return names


def _normalized_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    valid = np.isfinite(quaternion).all(axis=-1, keepdims=True) & (norm > 1.0e-12)
    return np.divide(
        quaternion,
        norm,
        out=np.full_like(quaternion, np.nan),
        where=valid,
    )


def _orientation_error_deg(
    actual_wxyz: np.ndarray, reference_wxyz: np.ndarray
) -> np.ndarray:
    actual = _normalized_quaternion(actual_wxyz)
    reference = _normalized_quaternion(reference_wxyz)
    dot = np.abs(np.sum(actual * reference, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def _statistics(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {
            "count": 0,
            "total_count": int(len(values)),
            "finite_fraction": 0.0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(len(finite)),
        "total_count": int(len(values)),
        "finite_fraction": float(len(finite) / len(values)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95.0)),
        "max": float(np.max(finite)),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_reference(
    reference: dict[str, np.ndarray], expected_frames: int, expected_fps: float
) -> tuple[list[str], int, float, np.ndarray]:
    required = {
        "body_pos_w",
        "body_quat_w",
        "body_names",
        "fps",
        "timestamps",
        "d435_pos_w",
        "d435_quat_wxyz",
    }
    missing = sorted(required - reference.keys())
    if missing:
        raise KeyError(f"three-point reference is missing fields: {missing}")
    names = _strings(reference["body_names"])
    for body in (TORSO_BODY, *WRIST_BODIES):
        if body not in names:
            raise ValueError(f"reference is missing body {body!r}")
    frames = int(reference["body_pos_w"].shape[0])
    fps = float(np.asarray(reference["fps"]).reshape(()))
    timestamps = np.asarray(reference["timestamps"], dtype=np.float64)
    if frames != expected_frames:
        raise ValueError(f"reference has {frames} frames, expected {expected_frames}")
    if not np.isclose(fps, expected_fps, rtol=0.0, atol=1.0e-8):
        raise ValueError(f"reference has fps={fps}, expected {expected_fps}")
    expected_timestamps = np.arange(frames, dtype=np.float64) / fps
    if not np.array_equal(timestamps, expected_timestamps):
        max_error = float(np.max(np.abs(timestamps - expected_timestamps)))
        raise ValueError(
            "reference timestamps are not exact frame/fps alignment "
            f"(max error {max_error})"
        )
    if reference["d435_pos_w"].shape != (frames, 3):
        raise ValueError(f"invalid d435_pos_w shape {reference['d435_pos_w'].shape}")
    if reference["d435_quat_wxyz"].shape != (frames, 4):
        raise ValueError(
            f"invalid d435_quat_wxyz shape {reference['d435_quat_wxyz'].shape}"
        )
    return names, frames, fps, timestamps


def _validate_run(
    label: str,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    expected_frames: int,
    expected_fps: float,
) -> list[str]:
    for field in ("body_pos_w", "body_quat_w", "fps"):
        if field not in arrays:
            raise KeyError(f"{label} kinematics is missing {field!r}")
    names = _body_names(arrays, metadata)
    frames = int(arrays["body_pos_w"].shape[0])
    fps = float(np.asarray(arrays["fps"]).reshape(()))
    if frames != expected_frames:
        raise ValueError(f"{label} has {frames} frames, expected {expected_frames}")
    if not np.isclose(fps, expected_fps, rtol=0.0, atol=1.0e-8):
        raise ValueError(f"{label} has fps={fps}, expected {expected_fps}")
    if arrays["body_pos_w"].shape != (frames, len(names), 3):
        raise ValueError(
            f"{label} body_pos_w shape {arrays['body_pos_w'].shape} does not "
            f"match {len(names)} body names"
        )
    if arrays["body_quat_w"].shape != (frames, len(names), 4):
        raise ValueError(
            f"{label} body_quat_w shape {arrays['body_quat_w'].shape} does not "
            f"match {len(names)} body names"
        )
    for body in (TORSO_BODY, *WRIST_BODIES):
        if body not in names:
            raise ValueError(f"{label} is missing body {body!r}")
    if "timestamps" in arrays:
        expected = np.arange(frames, dtype=np.float64) / fps
        actual = np.asarray(arrays["timestamps"], dtype=np.float64)
        if not np.array_equal(actual, expected):
            raise ValueError(f"{label} embedded timestamps are not exact 50 Hz indices")
    return names


def _run_errors(
    arrays: dict[str, np.ndarray],
    body_names: list[str],
    reference: dict[str, np.ndarray],
    reference_names: list[str],
) -> tuple[dict[str, np.ndarray], str]:
    actual_position = np.asarray(arrays["body_pos_w"], dtype=np.float64)
    actual_quaternion = np.asarray(arrays["body_quat_w"], dtype=np.float64)
    reference_position = np.asarray(reference["body_pos_w"], dtype=np.float64)
    reference_quaternion = np.asarray(reference["body_quat_w"], dtype=np.float64)

    errors: dict[str, np.ndarray] = {}
    for body, prefix in (
        (WRIST_BODIES[0], "left_wrist"),
        (WRIST_BODIES[1], "right_wrist"),
        (TORSO_BODY, "torso"),
    ):
        actual_index = body_names.index(body)
        reference_index = reference_names.index(body)
        errors[f"{prefix}_position_m"] = np.linalg.norm(
            actual_position[:, actual_index] - reference_position[:, reference_index],
            axis=1,
        )
        errors[f"{prefix}_orientation_deg"] = _orientation_error_deg(
            actual_quaternion[:, actual_index],
            reference_quaternion[:, reference_index],
        )

    errors["wrist_position_m"] = 0.5 * (
        errors["left_wrist_position_m"] + errors["right_wrist_position_m"]
    )
    errors["wrist_orientation_deg"] = 0.5 * (
        errors["left_wrist_orientation_deg"]
        + errors["right_wrist_orientation_deg"]
    )

    if D435_BODY in body_names:
        d435_index = body_names.index(D435_BODY)
        d435_position = actual_position[:, d435_index]
        d435_quaternion = actual_quaternion[:, d435_index]
        d435_source = "recorded body:d435_link"
    else:
        torso_index = body_names.index(TORSO_BODY)
        d435_position, d435_quaternion = d435_from_torso(
            actual_position[:, torso_index], actual_quaternion[:, torso_index]
        )
        d435_source = "derived from torso_link and fixed torso-to-D435 mount"

    d435_delta = d435_position - np.asarray(reference["d435_pos_w"], dtype=np.float64)
    errors["d435_position_m"] = np.linalg.norm(d435_delta, axis=1)
    errors["d435_orientation_deg"] = _orientation_error_deg(
        d435_quaternion,
        np.asarray(reference["d435_quat_wxyz"], dtype=np.float64),
    )
    errors["d435_horizontal_xy_m"] = np.linalg.norm(d435_delta[:, :2], axis=1)
    errors["d435_height_abs_m"] = np.abs(d435_delta[:, 2])
    errors["d435_dx_m"] = d435_delta[:, 0]
    errors["d435_dy_m"] = d435_delta[:, 1]
    errors["d435_dz_m"] = d435_delta[:, 2]
    return errors, d435_source


def _compare_stat(
    baseline: dict[str, Any], candidate: dict[str, Any], stat: str
) -> dict[str, Any]:
    old = baseline[stat]
    new = candidate[stat]
    if old is None or new is None:
        return {
            "baseline": old,
            "candidate": new,
            "delta_candidate_minus_baseline": None,
            "relative_change_percent": None,
            "improvement_percent": None,
        }
    relative = None if abs(old) <= 1.0e-15 else 100.0 * (new - old) / old
    return {
        "baseline": old,
        "candidate": new,
        "delta_candidate_minus_baseline": new - old,
        "relative_change_percent": relative,
        "improvement_percent": None if relative is None else -relative,
    }


def compare_tracking(
    *,
    baseline_kinematics: Path,
    baseline_meta: Path,
    candidate_kinematics: Path,
    candidate_meta: Path,
    reference_npz: Path,
    output_dir: Path,
    expected_frames: int = EXPECTED_FRAMES,
    expected_fps: float = EXPECTED_FPS,
) -> dict[str, Any]:
    paths = {
        "baseline_kinematics": baseline_kinematics.expanduser().resolve(),
        "baseline_meta": baseline_meta.expanduser().resolve(),
        "candidate_kinematics": candidate_kinematics.expanduser().resolve(),
        "candidate_meta": candidate_meta.expanduser().resolve(),
        "reference_npz": reference_npz.expanduser().resolve(),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")
    output_dir = output_dir.expanduser().resolve()

    baseline = _load_npz(paths["baseline_kinematics"])
    candidate = _load_npz(paths["candidate_kinematics"])
    reference = _load_npz(paths["reference_npz"])
    baseline_metadata = _load_json(paths["baseline_meta"])
    candidate_metadata = _load_json(paths["candidate_meta"])
    reference_names, frames, fps, timestamps = _validate_reference(
        reference, expected_frames, expected_fps
    )
    baseline_names = _validate_run(
        "baseline", baseline, baseline_metadata, frames, fps
    )
    candidate_names = _validate_run(
        "candidate", candidate, candidate_metadata, frames, fps
    )
    baseline_errors, baseline_d435_source = _run_errors(
        baseline, baseline_names, reference, reference_names
    )
    candidate_errors, candidate_d435_source = _run_errors(
        candidate, candidate_names, reference, reference_names
    )

    summary = {
        "baseline": {
            metric: _statistics(baseline_errors[metric]) for metric in PRIMARY_METRICS
        },
        "candidate": {
            metric: _statistics(candidate_errors[metric]) for metric in PRIMARY_METRICS
        },
    }
    comparison: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    for metric in PRIMARY_METRICS:
        comparison[metric] = {
            stat: _compare_stat(
                summary["baseline"][metric], summary["candidate"][metric], stat
            )
            for stat in ("mean", "median", "p95", "max")
        }
        row: dict[str, Any] = {"metric": metric}
        for stat in ("mean", "median", "p95", "max"):
            values = comparison[metric][stat]
            row[f"baseline_{stat}"] = values["baseline"]
            row[f"candidate_{stat}"] = values["candidate"]
            row[f"delta_{stat}"] = values["delta_candidate_minus_baseline"]
            row[f"improvement_{stat}_percent"] = values["improvement_percent"]
        comparison_rows.append(row)

    baseline_wrist_position = summary["baseline"]["wrist_position_m"]["mean"]
    candidate_wrist_position = summary["candidate"]["wrist_position_m"]["mean"]
    baseline_wrist_orientation = summary["baseline"]["wrist_orientation_deg"]["mean"]
    candidate_wrist_orientation = summary["candidate"]["wrist_orientation_deg"]["mean"]
    baseline_camera_mean = summary["baseline"]["d435_position_m"]["mean"]
    candidate_camera_mean = summary["candidate"]["d435_position_m"]["mean"]
    baseline_camera_median = summary["baseline"]["d435_position_m"]["median"]
    candidate_camera_median = summary["candidate"]["d435_position_m"]["median"]
    baseline_camera_orientation = summary["baseline"]["d435_orientation_deg"]["mean"]
    candidate_camera_orientation = summary["candidate"]["d435_orientation_deg"]["mean"]
    if any(
        value is None
        for value in (
            baseline_wrist_position,
            candidate_wrist_position,
            baseline_wrist_orientation,
            candidate_wrist_orientation,
            baseline_camera_mean,
            candidate_camera_mean,
            baseline_camera_median,
            candidate_camera_median,
            baseline_camera_orientation,
            candidate_camera_orientation,
        )
    ):
        raise RuntimeError("acceptance metrics have no finite samples")

    camera_position_limit = baseline_camera_mean * 0.90
    camera_orientation_limit = baseline_camera_orientation * 1.05
    finite_candidate = all(
        summary["candidate"][metric]["finite_fraction"] == 1.0
        for metric in PRIMARY_METRICS
    )
    gates = {
        "candidate_all_metrics_finite": {
            "passed": finite_candidate,
        },
        "wrist_position_mean": {
            "passed": candidate_wrist_position < WRIST_POSITION_ACCEPTANCE_LIMIT_M,
            "candidate_m": candidate_wrist_position,
            "baseline_m": baseline_wrist_position,
            "limit_m": WRIST_POSITION_ACCEPTANCE_LIMIT_M,
            "rule": "candidate absolute mean < 0.05 m",
        },
        "camera_position_mean": {
            "passed": candidate_camera_mean <= camera_position_limit + 1.0e-12,
            "candidate_m": candidate_camera_mean,
            "baseline_m": baseline_camera_mean,
            "limit_m": camera_position_limit,
            "rule": "at least 10 percent lower than baseline",
        },
        "camera_position_median": {
            "passed": candidate_camera_median < baseline_camera_median - 1.0e-12,
            "candidate_m": candidate_camera_median,
            "baseline_m": baseline_camera_median,
            "rule": "candidate strictly lower than baseline",
        },
        "camera_orientation_mean": {
            "passed": candidate_camera_orientation
            <= camera_safe_limit(camera_orientation_limit),
            "candidate_deg": candidate_camera_orientation,
            "baseline_deg": baseline_camera_orientation,
            "limit_deg": camera_orientation_limit,
            "rule": "candidate <= baseline * 1.05",
        },
    }
    acceptance = {
        "passed": all(bool(gate["passed"]) for gate in gates.values()),
        "criteria_version": 2,
        "gates": gates,
        "diagnostics": {
            "wrist_orientation_mean": {
                "candidate_deg": candidate_wrist_orientation,
                "baseline_deg": baseline_wrist_orientation,
                "candidate_minus_baseline_deg": (
                    candidate_wrist_orientation - baseline_wrist_orientation
                ),
                "acceptance_role": "reported_only_no_angular_threshold",
            }
        },
        "failure_rate": {
            "status": "not_available_from_kinematics",
            "note": "Use the MPC run summary for the failure-rate gate.",
        },
    }

    frame_rows: list[dict[str, Any]] = []
    all_error_names = list(PRIMARY_METRICS) + [
        "d435_dx_m",
        "d435_dy_m",
        "d435_dz_m",
    ]
    for frame in range(frames):
        row: dict[str, Any] = {
            "frame": frame,
            "timestamp_s": float(timestamps[frame]),
        }
        for metric in all_error_names:
            old = float(baseline_errors[metric][frame])
            new = float(candidate_errors[metric][frame])
            row[f"baseline_{metric}"] = old
            row[f"candidate_{metric}"] = new
            row[f"delta_{metric}"] = new - old
        frame_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_frame_csv = output_dir / "tracking_errors_per_frame.csv"
    comparison_csv = output_dir / "tracking_comparison.csv"
    comparison_json = output_dir / "tracking_comparison.json"
    _atomic_csv(per_frame_csv, list(frame_rows[0].keys()), frame_rows)
    _atomic_csv(comparison_csv, list(comparison_rows[0].keys()), comparison_rows)

    payload: dict[str, Any] = {
        "format": "torso-plus-dual-wrist baseline comparison",
        "format_version": 2,
        "inputs": {
            label: {"path": str(path), "sha256": _sha256(path)}
            for label, path in paths.items()
        },
        "alignment": {
            "frames": frames,
            "fps": fps,
            "first_timestamp_s": float(timestamps[0]),
            "last_timestamp_s": float(timestamps[-1]),
            "policy": "strict same frame count, fps, and reference index",
        },
        "pose_conventions": {
            "position": "world xyz in metres",
            "quaternion": "world wxyz; sign invariant",
            "orientation_error": "shortest geodesic quaternion angle in degrees",
            "source_camera": "Stera camera-link converted to Unitree D435-link axes",
            "baseline_d435_pose": baseline_d435_source,
            "candidate_d435_pose": candidate_d435_source,
        },
        "raw_data_csv": str(per_frame_csv),
        "comparison_csv": str(comparison_csv),
        "summary": summary,
        "comparison": comparison,
        "acceptance": acceptance,
        "comparison_json": str(comparison_json),
    }
    _atomic_json(comparison_json, payload)
    return payload


def camera_safe_limit(limit: float) -> float:
    """Allow only floating-point noise when a baseline angle is exactly zero."""

    return max(float(limit), 1.0e-9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-kinematics", type=Path, required=True)
    parser.add_argument(
        "--baseline-meta",
        type=Path,
        help="Defaults to meta.json beside --baseline-kinematics.",
    )
    parser.add_argument("--candidate-kinematics", type=Path, required=True)
    parser.add_argument(
        "--candidate-meta",
        type=Path,
        help="Defaults to meta.json beside --candidate-kinematics.",
    )
    parser.add_argument("--reference-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=EXPECTED_FRAMES)
    parser.add_argument("--expected-fps", type=float, default=EXPECTED_FPS)
    args = parser.parse_args()
    baseline_meta = args.baseline_meta or args.baseline_kinematics.parent / "meta.json"
    candidate_meta = args.candidate_meta or args.candidate_kinematics.parent / "meta.json"
    payload = compare_tracking(
        baseline_kinematics=args.baseline_kinematics,
        baseline_meta=baseline_meta,
        candidate_kinematics=args.candidate_kinematics,
        candidate_meta=candidate_meta,
        reference_npz=args.reference_npz,
        output_dir=args.output_dir,
        expected_frames=args.expected_frames,
        expected_fps=args.expected_fps,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
