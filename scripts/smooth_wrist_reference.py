#!/usr/bin/env python3
"""Smooth the dual-wrist tracking reference stored in Stera annotations.

Only ``hand-pose/{left,right}_joints`` are updated.  The 21 keypoints of each
hand receive one rigid correction per frame, so finger articulation and bone
lengths are preserved while the wrist position and wrist-link orientation are
smoothed in the tracking pipeline's Z-up world frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_ROOT = PROJECT_ROOT / "resources"
EVAL_SCRIPT_DIR = RESOURCE_ROOT / "ScaleBFM/ScaleTrack/scripts/eval_session"
if str(EVAL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_SCRIPT_DIR))

from prepare_output_clean_raw_eval import (  # noqa: E402
    YUP_TO_ZUP,
    _historical_wrist_link_rot,
    _repair_transient_pose_outliers,
    _robust_jump_threshold,
)


DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT / "data/egocentric_data/pick_up_the_cup_smooth"
)
PROTECTED_ORIGINAL_ROOT = (
    PROJECT_ROOT / "data/egocentric_data/pick_up_the_cup"
)
SIDES = ("left", "right")
TARGET_DATASETS = {
    "hand-pose/left_joints",
    "hand-pose/right_joints",
}
FORMAT_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_scalar_or_array(digest: Any, value: Any) -> None:
    array = np.asarray(value)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    if array.dtype.kind in {"O", "S", "U"}:
        digest.update(repr(array.tolist()).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(array).tobytes())


def _hash_attrs(digest: Any, attrs: h5py.AttributeManager) -> None:
    for name in sorted(attrs.keys()):
        digest.update(name.encode("utf-8"))
        _hash_scalar_or_array(digest, attrs[name])


def _hash_dataset_data(digest: Any, dataset: h5py.Dataset) -> None:
    if dataset.shape == ():
        _hash_scalar_or_array(digest, dataset[()])
        return
    if dataset.shape[0] == 0:
        return
    bytes_per_frame = max(
        1,
        int(np.prod(dataset.shape[1:], dtype=np.int64)) * dataset.dtype.itemsize,
    )
    frames_per_chunk = max(1, (16 * 1024 * 1024) // bytes_per_frame)
    for start in range(0, dataset.shape[0], frames_per_chunk):
        values = np.asarray(dataset[start : start + frames_per_chunk])
        digest.update(np.ascontiguousarray(values).tobytes())


def _non_target_hdf5_digest(path: Path) -> str:
    """Hash all HDF5 structure and non-target values."""
    digest = hashlib.sha256()
    with h5py.File(path, "r") as archive:
        digest.update(b"root")
        _hash_attrs(digest, archive.attrs)

        def visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            digest.update(name.encode("utf-8"))
            _hash_attrs(digest, obj.attrs)
            if isinstance(obj, h5py.Group):
                digest.update(b"group")
                return
            digest.update(b"dataset")
            digest.update(str(obj.shape).encode("ascii"))
            digest.update(str(obj.dtype).encode("ascii"))
            digest.update(str(obj.chunks).encode("ascii"))
            digest.update(str(obj.compression).encode("ascii"))
            digest.update(str(obj.compression_opts).encode("ascii"))
            if name not in TARGET_DATASETS:
                _hash_dataset_data(digest, obj)

        archive.visititems(visit)
    return digest.hexdigest()


def _validate_filter(window_length: int, polyorder: int) -> None:
    if window_length < 3 or window_length % 2 != 1:
        raise ValueError("window_length must be an odd integer >= 3")
    if polyorder < 0 or polyorder >= window_length:
        raise ValueError("polyorder must satisfy 0 <= polyorder < window_length")


def _window_bounds(length: int, index: int, window_length: int) -> tuple[int, int]:
    half = window_length // 2
    start = min(max(index - half, 0), length - window_length)
    return start, start + window_length


def timestamp_segments(
    timestamps: np.ndarray, gap_factor: float = 3.0
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Split a trajectory at dropped-frame gaps without altering timestamps."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if len(timestamps) < 2:
        return [(0, len(timestamps))], {
            "median_period_s": 0.0,
            "gap_threshold_s": 0.0,
            "gap_after_frame_indices": [],
        }
    delta = np.diff(timestamps)
    if not np.all(np.isfinite(delta)) or np.any(delta <= 0.0):
        raise ValueError("timestamps must be finite and strictly increasing")
    median_period = float(np.median(delta))
    threshold = float(gap_factor * median_period)
    gap_after = np.flatnonzero(delta > threshold).astype(int)
    boundaries = [0, *[int(index + 1) for index in gap_after], len(timestamps)]
    segments = list(zip(boundaries[:-1], boundaries[1:]))
    return segments, {
        "median_period_s": median_period,
        "gap_factor": float(gap_factor),
        "gap_threshold_s": threshold,
        "max_period_s": float(np.max(delta)),
        "gap_after_frame_indices": gap_after.tolist(),
        "segments": [[int(start), int(end - 1)] for start, end in segments],
    }


def _polynomial_value_at_zero(
    relative_time: np.ndarray,
    values: np.ndarray,
    polyorder: int,
) -> np.ndarray:
    scale = max(float(np.max(np.abs(relative_time))), 1.0e-12)
    normalized_time = relative_time / scale
    design = np.vander(normalized_time, N=polyorder + 1, increasing=True)
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    return coefficients[0]


def savgol_time_series(
    timestamps: np.ndarray,
    values: np.ndarray,
    window_length: int,
    polyorder: int,
    gap_factor: float = 3.0,
) -> np.ndarray:
    """Evaluate a local Savitzky-Golay polynomial at every actual timestamp."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] != len(timestamps):
        raise ValueError("values and timestamps have different lengths")
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("timestamps must be finite and strictly increasing")
    if not np.all(np.isfinite(values)):
        raise ValueError("values contain non-finite entries")

    output = values.copy()
    segments, _ = timestamp_segments(timestamps, gap_factor)
    for segment_start, segment_end in segments:
        length = segment_end - segment_start
        if length < window_length:
            continue
        for index in range(segment_start, segment_end):
            local_index = index - segment_start
            local_start, local_end = _window_bounds(
                length, local_index, window_length
            )
            start = segment_start + local_start
            end = segment_start + local_end
            output[index] = _polynomial_value_at_zero(
                timestamps[start:end] - timestamps[index],
                values[start:end],
                polyorder,
            )
    return output


def savgol_rotations(
    timestamps: np.ndarray,
    rotation_matrices: np.ndarray,
    window_length: int,
    polyorder: int,
    gap_factor: float = 3.0,
) -> np.ndarray:
    """Savitzky-Golay filtering in a local SO(3) tangent space."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    matrices = np.asarray(rotation_matrices, dtype=np.float64)
    if matrices.shape != (len(timestamps), 3, 3):
        raise ValueError(f"invalid rotation shape {matrices.shape}")
    rotations = Rotation.from_matrix(matrices)
    output = matrices.copy()
    segments, _ = timestamp_segments(timestamps, gap_factor)
    for segment_start, segment_end in segments:
        length = segment_end - segment_start
        if length < window_length:
            continue
        for index in range(segment_start, segment_end):
            local_index = index - segment_start
            local_start, local_end = _window_bounds(
                length, local_index, window_length
            )
            start = segment_start + local_start
            end = segment_start + local_end
            local_rotvec = (
                rotations[index].inv() * rotations[start:end]
            ).as_rotvec()
            correction = _polynomial_value_at_zero(
                timestamps[start:end] - timestamps[index],
                local_rotvec,
                polyorder,
            )
            output[index] = (
                rotations[index] * Rotation.from_rotvec(correction)
            ).as_matrix()
    return output


def _wrist_pose(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (21, 3):
        raise ValueError(f"expected (frames,21,3) points, got {points.shape}")
    position = points[:, 0].copy()
    rotation = np.asarray(
        [_historical_wrist_link_rot(frame) for frame in points],
        dtype=np.float64,
    )
    return position, rotation


def _apply_pose_correction(
    points: np.ndarray,
    source_position: np.ndarray,
    source_rotation: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
) -> np.ndarray:
    delta_rotation = np.einsum(
        "tij,tkj->tik", target_rotation, source_rotation
    )
    relative = points - source_position[:, None, :]
    return (
        np.einsum("tij,tkj->tki", delta_rotation, relative)
        + target_position[:, None, :]
    )


def _optical_to_link(points: np.ndarray, optical_to_link: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ optical_to_link.T


def _link_to_world_zup(
    points: np.ndarray,
    camera_rotation: np.ndarray,
    camera_translation: np.ndarray,
) -> np.ndarray:
    world_yup = (
        np.einsum("tij,tkj->tki", camera_rotation, points)
        + camera_translation[:, None, :]
    )
    return np.einsum("ij,tkj->tki", YUP_TO_ZUP, world_yup)


def _world_zup_to_optical(
    points: np.ndarray,
    camera_rotation: np.ndarray,
    camera_translation: np.ndarray,
    optical_to_link: np.ndarray,
) -> np.ndarray:
    world_yup = np.einsum("ij,tkj->tki", YUP_TO_ZUP.T, points)
    camera_link = np.einsum(
        "tji,tkj->tki",
        camera_rotation,
        world_yup - camera_translation[:, None, :],
    )
    return camera_link @ optical_to_link


def _rotation_angle(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return (
        Rotation.from_matrix(first).inv() * Rotation.from_matrix(second)
    ).magnitude()


def _position_jerk_energy(values: np.ndarray, timestamps: np.ndarray) -> tuple[float, int]:
    energy = 0.0
    count = 0
    segments, _ = timestamp_segments(timestamps)
    for start, end in segments:
        if end - start < 4:
            continue
        dt = float(np.median(np.diff(timestamps[start:end])))
        jerk = np.diff(values[start:end], n=3, axis=0) / (dt**3)
        squared = np.sum(jerk * jerk, axis=1)
        energy += float(np.sum(squared))
        count += int(len(squared))
    return energy, count


def _angular_acceleration_energy(
    rotations: np.ndarray, timestamps: np.ndarray
) -> tuple[float, int]:
    energy = 0.0
    count = 0
    segments, _ = timestamp_segments(timestamps)
    for start, end in segments:
        if end - start < 3:
            continue
        objects = Rotation.from_matrix(rotations[start:end])
        dt = np.diff(timestamps[start:end])
        velocity = (objects[:-1].inv() * objects[1:]).as_rotvec() / dt[:, None]
        acceleration_dt = 0.5 * (dt[:-1] + dt[1:])
        acceleration = np.diff(velocity, axis=0) / acceleration_dt[:, None]
        squared = np.sum(acceleration * acceleration, axis=1)
        energy += float(np.sum(squared))
        count += int(len(squared))
    return energy, count


def _position_steps(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    parts = []
    segments, _ = timestamp_segments(timestamps)
    for start, end in segments:
        if end - start >= 2:
            parts.append(np.linalg.norm(np.diff(values[start:end], axis=0), axis=1))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)


def _orientation_steps(rotations: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    parts = []
    segments, _ = timestamp_segments(timestamps)
    for start, end in segments:
        if end - start >= 2:
            parts.append(_rotation_angle(rotations[start : end - 1], rotations[start + 1 : end]))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)


def _new_jump_crossings(
    before: np.ndarray,
    after: np.ndarray,
    threshold: float,
    neighborhood: int,
) -> int:
    """Count threshold crossings unrelated to a nearby pre-existing jump."""
    count = 0
    for index in np.flatnonzero((after > threshold) & (before <= threshold)):
        start = max(0, int(index) - neighborhood)
        end = min(len(before), int(index) + neighborhood + 1)
        if not np.any(before[start:end] > threshold):
            count += 1
    return count


def _rms_ratio(before_energy: float, after_energy: float) -> float:
    if before_energy <= 1.0e-20:
        return 0.0 if after_energy <= 1.0e-20 else float("inf")
    return float(np.sqrt(after_energy / before_energy))


def _max_pairwise_distance_error(
    before: np.ndarray, after: np.ndarray
) -> float:
    before_relative = before - before[:, :1]
    after_relative = after - after[:, :1]
    before_distance = np.linalg.norm(before_relative, axis=-1)
    after_distance = np.linalg.norm(after_relative, axis=-1)
    return float(np.max(np.abs(before_distance - after_distance)))


def _rotation_quality(rotations: np.ndarray) -> dict[str, float]:
    identity = np.eye(3, dtype=np.float64)
    product = np.einsum("tji,tjk->tik", rotations, rotations)
    orthogonality = np.linalg.norm(product - identity, axis=(1, 2))
    determinant = np.abs(np.linalg.det(rotations) - 1.0)
    return {
        "max_orthogonality_error": float(np.max(orthogonality)),
        "max_abs_determinant_error": float(np.max(determinant)),
    }


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "max": float(np.max(values)),
    }


def _process_side(
    *,
    joints_optical: np.ndarray,
    timestamps: np.ndarray,
    camera_rotation: np.ndarray,
    camera_translation: np.ndarray,
    optical_to_link: np.ndarray,
    window_length: int,
    polyorder: int,
    repair_transient_outliers: bool,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray | float | int]]:
    raw_link = _optical_to_link(joints_optical, optical_to_link)
    raw_link_position, raw_link_rotation = _wrist_pose(raw_link)

    if repair_transient_outliers:
        repaired_link_position, repaired_link_rotation, repair = (
            _repair_transient_pose_outliers(
                timestamps, raw_link_position, raw_link_rotation
            )
        )
    else:
        repaired_link_position = raw_link_position.copy()
        repaired_link_rotation = raw_link_rotation.copy()
        repair = {
            "method": "disabled",
            "candidate_anomalous_edges": [],
            "repaired_source_frame_indices": [],
            "events": [],
        }

    repaired_link = _apply_pose_correction(
        raw_link,
        raw_link_position,
        raw_link_rotation,
        repaired_link_position,
        repaired_link_rotation,
    )
    raw_world = _link_to_world_zup(
        raw_link, camera_rotation, camera_translation
    )
    repaired_world = _link_to_world_zup(
        repaired_link, camera_rotation, camera_translation
    )
    raw_world_position, raw_world_rotation = _wrist_pose(raw_world)
    repaired_world_position, repaired_world_rotation = _wrist_pose(repaired_world)

    smooth_world_position = savgol_time_series(
        timestamps,
        repaired_world_position,
        window_length,
        polyorder,
    )
    smooth_world_rotation = savgol_rotations(
        timestamps,
        repaired_world_rotation,
        window_length,
        polyorder,
    )
    smooth_world = _apply_pose_correction(
        repaired_world,
        repaired_world_position,
        repaired_world_rotation,
        smooth_world_position,
        smooth_world_rotation,
    )
    smooth_optical = _world_zup_to_optical(
        smooth_world,
        camera_rotation,
        camera_translation,
        optical_to_link,
    )

    saved_world = _link_to_world_zup(
        _optical_to_link(smooth_optical.astype(np.float32), optical_to_link),
        camera_rotation,
        camera_translation,
    )
    saved_position, saved_rotation = _wrist_pose(saved_world)
    reconstruction_position_error = np.linalg.norm(
        saved_position - smooth_world_position, axis=1
    )
    reconstruction_orientation_error = _rotation_angle(
        smooth_world_rotation, saved_rotation
    )

    correction_position = np.linalg.norm(
        smooth_world_position - raw_world_position, axis=1
    )
    correction_orientation = _rotation_angle(
        raw_world_rotation, smooth_world_rotation
    )
    sg_position = np.linalg.norm(
        smooth_world_position - repaired_world_position, axis=1
    )
    sg_orientation = _rotation_angle(
        repaired_world_rotation, smooth_world_rotation
    )

    before_position_energy, before_position_count = _position_jerk_energy(
        repaired_world_position, timestamps
    )
    after_position_energy, after_position_count = _position_jerk_energy(
        smooth_world_position, timestamps
    )
    before_angular_energy, before_angular_count = _angular_acceleration_energy(
        repaired_world_rotation, timestamps
    )
    after_angular_energy, after_angular_count = _angular_acceleration_energy(
        smooth_world_rotation, timestamps
    )

    before_position_step = _position_steps(repaired_world_position, timestamps)
    after_position_step = _position_steps(smooth_world_position, timestamps)
    before_orientation_step = _orientation_steps(repaired_world_rotation, timestamps)
    after_orientation_step = _orientation_steps(smooth_world_rotation, timestamps)

    rigid_error = _max_pairwise_distance_error(raw_world, saved_world)
    quality = _rotation_quality(smooth_world_rotation)
    _, timestamp_diagnostics = timestamp_segments(timestamps)
    position_jump_threshold = _robust_jump_threshold(
        before_position_step, floor=0.05
    )
    orientation_jump_threshold = np.deg2rad(
        _robust_jump_threshold(
            np.degrees(before_orientation_step), floor=45.0
        )
    )
    new_position_crossings = _new_jump_crossings(
        before_position_step,
        after_position_step,
        position_jump_threshold,
        window_length // 2,
    )
    new_orientation_crossings = _new_jump_crossings(
        before_orientation_step,
        after_orientation_step,
        orientation_jump_threshold,
        window_length // 2,
    )
    metrics: dict[str, Any] = {
        "frames": int(len(timestamps)),
        "source_fps_median": float(1.0 / np.median(np.diff(timestamps))),
        "timestamps": timestamp_diagnostics,
        "transient_repair": repair,
        "total_position_correction_m": _summary_stats(correction_position),
        "total_orientation_correction_rad": _summary_stats(
            correction_orientation
        ),
        "sg_position_correction_m": _summary_stats(sg_position),
        "sg_orientation_correction_rad": _summary_stats(sg_orientation),
        "position_jerk_rms_ratio": _rms_ratio(
            before_position_energy, after_position_energy
        ),
        "angular_acceleration_rms_ratio": _rms_ratio(
            before_angular_energy, after_angular_energy
        ),
        "max_position_step_before_m": float(np.max(before_position_step)),
        "max_position_step_after_m": float(np.max(after_position_step)),
        "position_jump_threshold_m": float(position_jump_threshold),
        "new_position_jump_threshold_crossings": new_position_crossings,
        "max_orientation_step_before_rad": float(
            np.max(before_orientation_step)
        ),
        "max_orientation_step_after_rad": float(np.max(after_orientation_step)),
        "orientation_jump_threshold_rad": float(orientation_jump_threshold),
        "new_orientation_jump_threshold_crossings": new_orientation_crossings,
        "max_rigid_joint_distance_error_m": rigid_error,
        "max_saved_position_reconstruction_error_m": float(
            np.max(reconstruction_position_error)
        ),
        "max_saved_orientation_reconstruction_error_rad": float(
            np.max(reconstruction_orientation_error)
        ),
        "rotation_quality": quality,
    }
    aggregates: dict[str, np.ndarray | float | int] = {
        "sg_position_correction": sg_position,
        "sg_orientation_correction": sg_orientation,
        "total_position_correction": correction_position,
        "total_orientation_correction": correction_orientation,
        "position_energy_before": before_position_energy,
        "position_energy_after": after_position_energy,
        "position_energy_count": before_position_count,
        "angular_energy_before": before_angular_energy,
        "angular_energy_after": after_angular_energy,
        "angular_energy_count": before_angular_count,
        "new_position_jump": int(
            new_position_crossings > 0
            or np.max(after_position_step) > np.max(before_position_step) + 1.0e-9
        ),
        "new_orientation_jump": int(
            new_orientation_crossings > 0
            or np.max(after_orientation_step)
            > np.max(before_orientation_step) + 1.0e-9
        ),
    }

    if rigid_error > 1.0e-6:
        raise RuntimeError(f"rigid hand geometry gate failed: {rigid_error:g} m")
    if metrics["max_saved_position_reconstruction_error_m"] > 2.0e-6:
        raise RuntimeError("saved wrist position does not reconstruct the target")
    if metrics["max_saved_orientation_reconstruction_error_rad"] > 2.0e-5:
        raise RuntimeError("saved wrist orientation does not reconstruct the target")
    if quality["max_orthogonality_error"] > 1.0e-10:
        raise RuntimeError("smoothed rotations are not orthonormal")
    if quality["max_abs_determinant_error"] > 1.0e-10:
        raise RuntimeError("smoothed rotations do not have determinant one")

    return smooth_optical.astype(np.float32), metrics, aggregates


def _load_session_inputs(session_dir: Path) -> dict[str, Any]:
    annotation = session_dir / "annotation.hdf5"
    calibration = session_dir / "calibrations/R_optical_to_link.npy"
    if not annotation.is_file():
        raise FileNotFoundError(annotation)
    if not calibration.is_file():
        raise FileNotFoundError(calibration)
    optical_to_link = np.asarray(np.load(calibration), dtype=np.float64)
    if optical_to_link.shape != (3, 3):
        raise ValueError(f"invalid optical-to-link shape {optical_to_link.shape}")
    if not np.all(np.isfinite(optical_to_link)):
        raise ValueError("optical-to-link calibration contains non-finite values")
    calibration_quality = _rotation_quality(optical_to_link[None])
    if (
        calibration_quality["max_orthogonality_error"] > 1.0e-4
        or calibration_quality["max_abs_determinant_error"] > 1.0e-4
    ):
        raise ValueError("optical-to-link calibration is not a valid rotation")

    with h5py.File(annotation, "r") as archive:
        required = {
            "hand-pose/timestamps",
            "cam-pose/timestamps",
            "cam-pose/rotations",
            "cam-pose/translations",
            "hand-pose/left_joints",
            "hand-pose/right_joints",
            "hand-pose/left_valid",
            "hand-pose/right_valid",
        }
        missing = sorted(name for name in required if name not in archive)
        if missing:
            raise KeyError(f"missing HDF5 datasets: {missing}")
        timestamps = np.asarray(
            archive["hand-pose/timestamps"], dtype=np.float64
        )
        camera_timestamps = np.asarray(
            archive["cam-pose/timestamps"], dtype=np.float64
        )
        camera_rotation = np.asarray(
            archive["cam-pose/rotations"], dtype=np.float64
        )
        camera_translation = np.asarray(
            archive["cam-pose/translations"], dtype=np.float64
        )
        frames = len(timestamps)
        if not np.array_equal(camera_timestamps, timestamps):
            raise ValueError(
                "camera and hand timestamps must be exactly aligned by frame"
            )
        if camera_rotation.shape != (frames, 3, 3):
            raise ValueError("camera rotation count does not match hand timestamps")
        if camera_translation.shape != (frames, 3):
            raise ValueError("camera translation count does not match hand timestamps")
        if not np.all(np.isfinite(camera_rotation)) or not np.all(
            np.isfinite(camera_translation)
        ):
            raise ValueError("camera poses contain non-finite values")
        camera_quality = _rotation_quality(camera_rotation)
        if (
            camera_quality["max_orthogonality_error"] > 1.0e-4
            or camera_quality["max_abs_determinant_error"] > 1.0e-4
        ):
            raise ValueError("camera rotations fail the SO(3) quality gate")
        joints = {}
        for side in SIDES:
            values = np.asarray(
                archive[f"hand-pose/{side}_joints"], dtype=np.float64
            )
            valid = np.asarray(
                archive[f"hand-pose/{side}_valid"], dtype=bool
            )
            if values.shape != (frames, 21, 3):
                raise ValueError(f"invalid {side} joints shape {values.shape}")
            if valid.shape != (frames,) or not np.all(valid):
                raise ValueError(
                    f"{side} wrist is not valid for every frame; refusing extrapolation"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{side} joints contain non-finite values")
            joints[side] = values

    return {
        "annotation": annotation,
        "timestamps": timestamps,
        "camera_rotation": camera_rotation,
        "camera_translation": camera_translation,
        "optical_to_link": optical_to_link,
        "calibration_rotation_quality": calibration_quality,
        "camera_rotation_quality": camera_quality,
        "joints": joints,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _serialize_aggregate_parts(
    parts: list[dict[str, np.ndarray | float | int]],
) -> list[dict[str, Any]]:
    return [
        {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in part.items()
        }
        for part in parts
    ]


def _deserialize_aggregate_parts(
    parts: list[dict[str, Any]],
) -> list[dict[str, np.ndarray | float | int]]:
    array_keys = {
        "sg_position_correction",
        "sg_orientation_correction",
        "total_position_correction",
        "total_orientation_correction",
    }
    return [
        {
            key: np.asarray(value, dtype=np.float64)
            if key in array_keys
            else value
            for key, value in part.items()
        }
        for part in parts
    ]


def process_session(
    session_dir: Path,
    *,
    window_length: int,
    polyorder: int,
    repair_transient_outliers: bool,
    dry_run: bool,
    force: bool,
) -> dict[str, Any]:
    session_dir = session_dir.expanduser().resolve()
    marker_path = session_dir / "wrist_smoothing.json"
    inputs = _load_session_inputs(session_dir)
    annotation: Path = inputs["annotation"]
    input_sha256 = _sha256_file(annotation)

    if marker_path.is_file() and not dry_run and not force:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            marker.get("format_version") == FORMAT_VERSION
            and marker.get("output_annotation_sha256") == input_sha256
            and marker.get("filter", {}).get("window_length") == window_length
            and marker.get("filter", {}).get("polyorder") == polyorder
        ):
            marker = dict(marker)
            aggregate_contribution = marker.pop("aggregate_contribution", [])
            marker["_aggregate_parts"] = _deserialize_aggregate_parts(
                aggregate_contribution
            )
            marker["status"] = "already_smoothed"
            return marker
        raise RuntimeError(
            f"existing smoothing metadata does not match {annotation}; "
            "use --force only with an unsmoothed source"
        )

    non_target_digest = _non_target_hdf5_digest(annotation)
    smoothed: dict[str, np.ndarray] = {}
    side_metrics: dict[str, Any] = {}
    aggregate_parts: list[dict[str, np.ndarray | float | int]] = []
    for side in SIDES:
        values, metrics, aggregates = _process_side(
            joints_optical=inputs["joints"][side],
            timestamps=inputs["timestamps"],
            camera_rotation=inputs["camera_rotation"],
            camera_translation=inputs["camera_translation"],
            optical_to_link=inputs["optical_to_link"],
            window_length=window_length,
            polyorder=polyorder,
            repair_transient_outliers=repair_transient_outliers,
        )
        smoothed[side] = values
        side_metrics[side] = metrics
        aggregate_parts.append(aggregates)

    payload: dict[str, Any] = {
        "format": "Stera dual-wrist reference smoothing",
        "format_version": FORMAT_VERSION,
        "created_at": _utc_now(),
        "status": "dry_run" if dry_run else "completed",
        "session_id": session_dir.name,
        "annotation_hdf5": str(annotation),
        "input_annotation_sha256": input_sha256,
        "filter": {
            "method": "timestamp-aware local Savitzky-Golay",
            "window_length": int(window_length),
            "polyorder": int(polyorder),
            "position_space": "ScaleTrack Z-up world",
            "orientation_space": "local SO(3) tangent",
            "edge_policy": "shifted full window and polynomial evaluation at the actual timestamp",
        },
        "preprocessing": {
            "transient_outlier_repair": bool(repair_transient_outliers),
            "coordinate_frame": "camera-link",
            "implementation": str(
                EVAL_SCRIPT_DIR / "prepare_output_clean_raw_eval.py"
            ),
        },
        "input_rotation_quality": {
            "optical_to_link": inputs["calibration_rotation_quality"],
            "camera": inputs["camera_rotation_quality"],
        },
        "modified_datasets": sorted(TARGET_DATASETS),
        "unchanged_policy": (
            "all other HDF5 datasets and attributes are value-identical; "
            "non-HDF5 session files are not modified"
        ),
        "detector_field_policy": (
            "MANO parameters, MANO vertices, detector camera parameters, and "
            "2D keypoints remain the original detector outputs; tracking derives "
            "the smoothed wrist pose exclusively from the modified 21 joints"
        ),
        "sides": side_metrics,
        "non_target_hdf5_digest": non_target_digest,
    }
    payload["_aggregate_parts"] = aggregate_parts

    if dry_run:
        return payload

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".annotation.hdf5.savgol.", suffix=".tmp", dir=session_dir
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(annotation, temporary)
        with h5py.File(temporary, "r+") as archive:
            for side in SIDES:
                archive[f"hand-pose/{side}_joints"][:] = smoothed[side]
            archive.flush()

        output_non_target_digest = _non_target_hdf5_digest(temporary)
        if output_non_target_digest != non_target_digest:
            raise RuntimeError("a non-target HDF5 value or attribute changed")
        with h5py.File(temporary, "r") as archive:
            for side in SIDES:
                saved = np.asarray(archive[f"hand-pose/{side}_joints"])
                if not np.array_equal(saved, smoothed[side]):
                    raise RuntimeError(f"saved {side} joints differ from computed values")

        output_sha256 = _sha256_file(temporary)
        os.replace(temporary, annotation)
        payload["output_annotation_sha256"] = output_sha256
        payload["non_target_hdf5_digest_after"] = output_non_target_digest
        metadata_payload = {
            key: value for key, value in payload.items() if key != "_aggregate_parts"
        }
        metadata_payload["aggregate_contribution"] = _serialize_aggregate_parts(
            aggregate_parts
        )
        _atomic_write_json(marker_path, metadata_payload)
        return payload
    finally:
        if temporary.exists():
            temporary.unlink()


def _aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    sg_position_correction = []
    sg_orientation_correction = []
    total_position_correction = []
    total_orientation_correction = []
    position_before = 0.0
    position_after = 0.0
    angular_before = 0.0
    angular_after = 0.0
    position_count = 0
    angular_count = 0
    new_position_jumps = 0
    new_orientation_jumps = 0
    tracks = 0
    for result in results:
        for part in result.get("_aggregate_parts", []):
            tracks += 1
            sg_position_correction.append(
                np.asarray(part["sg_position_correction"])
            )
            sg_orientation_correction.append(
                np.asarray(part["sg_orientation_correction"])
            )
            total_position_correction.append(
                np.asarray(part["total_position_correction"])
            )
            total_orientation_correction.append(
                np.asarray(part["total_orientation_correction"])
            )
            position_before += float(part["position_energy_before"])
            position_after += float(part["position_energy_after"])
            angular_before += float(part["angular_energy_before"])
            angular_after += float(part["angular_energy_after"])
            position_count += int(part["position_energy_count"])
            angular_count += int(part["angular_energy_count"])
            new_position_jumps += int(part["new_position_jump"])
            new_orientation_jumps += int(part["new_orientation_jump"])
    if not sg_position_correction:
        return {"tracks": 0, "available": False}
    sg_position_values = np.concatenate(sg_position_correction)
    sg_orientation_values = np.concatenate(sg_orientation_correction)
    total_position_values = np.concatenate(total_position_correction)
    total_orientation_values = np.concatenate(total_orientation_correction)
    return {
        "available": True,
        "tracks": int(tracks),
        "frames_across_tracks": int(len(sg_position_values)),
        "sg_position_correction_m": _summary_stats(sg_position_values),
        "sg_orientation_correction_rad": _summary_stats(sg_orientation_values),
        "total_position_correction_m": _summary_stats(total_position_values),
        "total_orientation_correction_rad": _summary_stats(
            total_orientation_values
        ),
        "position_jerk_rms_ratio": _rms_ratio(position_before, position_after),
        "angular_acceleration_rms_ratio": _rms_ratio(
            angular_before, angular_after
        ),
        "position_energy_sample_count": int(position_count),
        "angular_energy_sample_count": int(angular_count),
        "tracks_with_new_max_position_step": int(new_position_jumps),
        "tracks_with_new_max_orientation_step": int(new_orientation_jumps),
    }


def _acceptance(aggregate: dict[str, Any]) -> dict[str, Any]:
    if not aggregate.get("available"):
        return {"passed": False, "reason": "aggregate samples unavailable"}
    checks = {
        "position_jerk_rms_ratio_le_0_60": (
            aggregate["position_jerk_rms_ratio"] <= 0.60
        ),
        "angular_acceleration_rms_ratio_le_0_60": (
            aggregate["angular_acceleration_rms_ratio"] <= 0.60
        ),
        "mean_position_correction_le_5mm": (
            aggregate["sg_position_correction_m"]["mean"] <= 0.005
        ),
        "p95_position_correction_le_10mm": (
            aggregate["sg_position_correction_m"]["p95"] <= 0.010
        ),
        "no_new_max_position_step": (
            aggregate["tracks_with_new_max_position_step"] == 0
        ),
        "no_new_max_orientation_step": (
            aggregate["tracks_with_new_max_orientation_step"] == 0
        ),
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _discover_sessions(input_root: Path, requested: list[str]) -> list[Path]:
    if requested:
        sessions = [(input_root / session_id).resolve() for session_id in requested]
    else:
        sessions = sorted(
            path.resolve()
            for path in input_root.glob("session_*")
            if path.is_dir() and (path / "annotation.hdf5").is_file()
        )
    escaped = [str(path) for path in sessions if not _is_relative_to(path, input_root)]
    if escaped:
        raise RuntimeError(f"session paths escape the input root: {escaped}")
    protected = PROTECTED_ORIGINAL_ROOT.resolve()
    unsafe = [str(path) for path in sessions if _is_relative_to(path, protected)]
    if unsafe:
        raise RuntimeError(f"refusing protected original sessions: {unsafe}")
    missing = [str(path) for path in sessions if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"session directories not found: {missing}")
    if not sessions:
        raise RuntimeError(f"no sessions found in {input_root}")
    return sessions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--sessions", nargs="*", default=[])
    parser.add_argument("--window-length", type=int, default=5)
    parser.add_argument("--polyorder", type=int, default=2)
    parser.add_argument(
        "--skip-transient-repair",
        action="store_true",
        help="Skip the existing bounded wrist excursion repair before SG.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Required to replace annotation.hdf5 files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess a session with existing smoothing metadata.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Defaults to wrist_smoothing_summary.json under the input root.",
    )
    args = parser.parse_args()
    _validate_filter(args.window_length, args.polyorder)

    input_root = args.input_root.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    if input_root == PROTECTED_ORIGINAL_ROOT.resolve():
        raise RuntimeError(
            f"refusing to modify or dry-run the protected original root: {input_root}"
        )
    if not args.dry_run and not args.in_place:
        raise RuntimeError("--in-place is required unless --dry-run is used")
    if args.dry_run and args.in_place:
        raise ValueError("--dry-run and --in-place are mutually exclusive")

    sessions = _discover_sessions(input_root, args.sessions)
    results = []
    failures: dict[str, str] = {}
    for index, session_dir in enumerate(sessions, start=1):
        print(
            f"[{index}/{len(sessions)}] {session_dir.name}",
            flush=True,
        )
        try:
            result = process_session(
                session_dir,
                window_length=args.window_length,
                polyorder=args.polyorder,
                repair_transient_outliers=not args.skip_transient_repair,
                dry_run=args.dry_run,
                force=args.force,
            )
            results.append(result)
            print(f"  status={result['status']}", flush=True)
        except Exception as error:
            failures[session_dir.name] = f"{type(error).__name__}: {error}"
            print(f"  failed={failures[session_dir.name]}", flush=True)

    all_already_smoothed = bool(results) and all(
        result.get("status") == "already_smoothed" for result in results
    )
    if all_already_smoothed and not args.dry_run:
        summary_path = (
            args.summary_json.expanduser().resolve()
            if args.summary_json is not None
            else input_root / "wrist_smoothing_summary.json"
        )
        if summary_path.is_file():
            print(summary_path.read_text(encoding="utf-8"), end="")
            print(f"summary={summary_path}")
        else:
            print(
                json.dumps(
                    {
                        "status": "already_smoothed",
                        "successful_sessions": len(results),
                        "failed_sessions": failures,
                    },
                    indent=2,
                )
            )
        if failures:
            raise SystemExit(1)
        return

    aggregate = _aggregate_results(results)
    acceptance = _acceptance(aggregate)
    summary = {
        "format": "Stera dual-wrist reference smoothing batch summary",
        "format_version": FORMAT_VERSION,
        "created_at": _utc_now(),
        "status": "dry_run" if args.dry_run else "completed",
        "input_root": str(input_root),
        "requested_sessions": len(sessions),
        "successful_sessions": len(results),
        "failed_sessions": failures,
        "filter": {
            "window_length": int(args.window_length),
            "polyorder": int(args.polyorder),
            "transient_outlier_repair": not args.skip_transient_repair,
        },
        "aggregate": aggregate,
        "acceptance": acceptance,
        "sessions": {
            result["session_id"]: {
                key: value
                for key, value in result.items()
                if key != "_aggregate_parts"
            }
            for result in results
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if not args.dry_run:
        summary_path = (
            args.summary_json.expanduser().resolve()
            if args.summary_json is not None
            else input_root / "wrist_smoothing_summary.json"
        )
        _atomic_write_json(summary_path, summary)
        print(f"summary={summary_path}")

    if failures:
        raise SystemExit(1)
    if not acceptance.get("passed", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
