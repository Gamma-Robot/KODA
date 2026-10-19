#!/usr/bin/env python3
"""Replace a prepared dual-wrist reference with template-palm Kabsch poses.

The input carrier, timeline, camera-derived heading, and all non-wrist bodies
are preserved. Only the two G1 ``wrist_yaw_link`` pose/velocity streams and the
matching Kimodo wrist targets are replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


YUP_TO_ZUP = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
WRIST_LINK_LOCAL_RX_MINUS_90 = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)
WRIST = 0
MCP_INDICES = (5, 9, 13, 17)
INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
LITTLE_MCP = 17
ANCHOR_INDICES = (WRIST, *MCP_INDICES)
ROBOT_PALM_SITE = np.asarray([0.08, 0.0, 0.0], dtype=np.float64)
WRIST_BODIES = {
    "left": "left_wrist_yaw_link",
    "right": "right_wrist_yaw_link",
}


def normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < 1.0e-10:
        raise ValueError("degenerate direction")
    return value / norm


def palm_polygon_centroid(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    weighted = np.zeros(3, dtype=np.float64)
    total_area = 0.0
    for indices in (
        (WRIST, INDEX_MCP, MIDDLE_MCP),
        (WRIST, MIDDLE_MCP, RING_MCP),
        (WRIST, RING_MCP, LITTLE_MCP),
    ):
        triangle = p[list(indices)]
        area = 0.5 * float(
            np.linalg.norm(
                np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            )
        )
        if np.isfinite(area) and area > 1.0e-12:
            weighted += area * np.mean(triangle, axis=0)
            total_area += area
    if total_area <= 1.0e-12:
        raise ValueError("degenerate wrist-to-MCP palm polygon")
    return weighted / total_area


def landmark_frame(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    distal = np.mean(p[list(MCP_INDICES)], axis=0)
    x_axis = normalize(distal - p[WRIST])
    across = p[INDEX_MCP] - p[LITTLE_MCP]
    across = normalize(across - x_axis * float(np.dot(across, x_axis)))
    z_axis = -normalize(np.cross(across, x_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))
    return (
        np.stack([x_axis, y_axis, z_axis], axis=1)
        @ WRIST_LINK_LOCAL_RX_MINUS_90
    )


def rigid_fit(
    source: np.ndarray,
    target: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(source, dtype=np.float64)
    p = np.asarray(target, dtype=np.float64)
    if q.shape != p.shape or q.ndim != 2 or q.shape[1] != 3:
        raise ValueError(f"invalid rigid fit shapes source={q.shape} target={p.shape}")
    w = np.ones(len(q), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (len(q),) or np.any(~np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("invalid rigid-fit weights")
    if float(np.sum(w)) <= 1.0e-12:
        raise ValueError("rigid-fit weights sum to zero")
    w = w / np.sum(w)
    q_mean = np.sum(w[:, None] * q, axis=0)
    p_mean = np.sum(w[:, None] * p, axis=0)
    covariance = (w[:, None] * (q - q_mean)).T @ (p - p_mean)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    return rotation, p_mean - rotation @ q_mean


def robust_rigid_fit(
    source: np.ndarray,
    target: np.ndarray,
    *,
    huber_delta_m: float = 0.008,
    iterations: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.ones(len(source), dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    for _ in range(max(1, int(iterations))):
        rotation, translation = rigid_fit(source, target, weights=weights)
        residual = np.linalg.norm(
            (rotation @ np.asarray(source).T).T + translation - target,
            axis=1,
        )
        weights = np.minimum(
            1.0, float(huber_delta_m) / np.maximum(residual, 1.0e-12)
        )
    return rotation, translation


def build_palm_template(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    local_anchors = []
    for hand in p:
        center = palm_polygon_centroid(hand)
        rotation = landmark_frame(hand)
        local_anchors.append((hand[list(ANCHOR_INDICES)] - center) @ rotation)
    template = np.median(np.asarray(local_anchors), axis=0)
    if template.shape != (5, 3) or not np.all(np.isfinite(template)):
        raise ValueError("invalid palm template")
    return template


def template_palm_kabsch(
    points: np.ndarray,
    *,
    huber_delta_m: float = 0.008,
    iterations: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return desired G1 wrist-link positions and rotations for one hand."""
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (21, 3):
        raise ValueError(f"expected (T,21,3) hand points, got {p.shape}")
    template = build_palm_template(p)
    positions = np.empty((len(p), 3), dtype=np.float64)
    rotations = np.empty((len(p), 3, 3), dtype=np.float64)
    previous_rotation = landmark_frame(p[0])
    for frame, hand in enumerate(p):
        try:
            rotation, palm_position = robust_rigid_fit(
                template,
                hand[list(ANCHOR_INDICES)],
                huber_delta_m=huber_delta_m,
                iterations=iterations,
            )
            position = palm_position - rotation @ ROBOT_PALM_SITE
        except (ValueError, np.linalg.LinAlgError):
            rotation = previous_rotation
            position = hand[WRIST] - rotation @ ROBOT_PALM_SITE
        positions[frame] = position
        rotations[frame] = rotation
        previous_rotation = rotation
    return positions, rotations, template


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _continuous_wxyz(rotations: np.ndarray) -> np.ndarray:
    quaternions = Rotation.from_matrix(rotations).as_quat(scalar_first=True)
    for index in range(1, len(quaternions)):
        if float(np.dot(quaternions[index - 1], quaternions[index])) < 0.0:
            quaternions[index] *= -1.0
    return quaternions


def _finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.zeros_like(values)
    if len(values) > 1:
        result[0] = (values[1] - values[0]) * fps
        result[-1] = (values[-1] - values[-2]) * fps
        result[1:-1] = (values[2:] - values[:-2]) * (0.5 * fps)
    return result


def _angular_velocity_world(rotations: np.ndarray, fps: float) -> np.ndarray:
    matrices = np.asarray(rotations, dtype=np.float64)
    result = np.zeros((len(matrices), 3), dtype=np.float64)
    if len(matrices) > 1:
        result[0] = Rotation.from_matrix(matrices[1] @ matrices[0].T).as_rotvec() * fps
        result[-1] = Rotation.from_matrix(matrices[-1] @ matrices[-2].T).as_rotvec() * fps
        for index in range(1, len(matrices) - 1):
            result[index] = (
                Rotation.from_matrix(matrices[index + 1] @ matrices[index - 1].T)
                .as_rotvec()
                * (0.5 * fps)
            )
    return result


def _interpolate_invalid(
    values: np.ndarray, valid: np.ndarray, timestamps: np.ndarray, side: str
) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    good = np.asarray(valid, dtype=bool) & np.isfinite(data).all(axis=(1, 2))
    if np.count_nonzero(good) < 2:
        raise ValueError(f"{side} has fewer than two valid hand frames")
    flattened = data.reshape(len(data), -1)
    output = np.empty_like(flattened)
    for column in range(flattened.shape[1]):
        output[:, column] = np.interp(
            timestamps, timestamps[good], flattened[good, column]
        )
    return output.reshape(data.shape)


def repair_landmark_events(
    values: np.ndarray,
    timestamps: np.ndarray,
    events: list[dict[str, Any]],
    side: str,
) -> tuple[np.ndarray, list[int]]:
    """Interpolate prepared bounded outlier intervals in camera-local space."""
    data = np.asarray(values, dtype=np.float64).copy()
    time = np.asarray(timestamps, dtype=np.float64)
    repaired = np.zeros(len(data), dtype=bool)
    for event in events:
        left = int(event["left_anchor_index"])
        right = int(event["right_anchor_index"])
        if left < 0 or right >= len(data) or right <= left + 1:
            raise ValueError(
                f"invalid {side} landmark repair anchors [{left}, {right}]"
            )
        if np.any(repaired[left + 1 : right]):
            raise ValueError(f"overlapping {side} landmark repair events")
        duration = float(time[right] - time[left])
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError(f"invalid {side} landmark repair duration")
        alpha = (time[left + 1 : right] - time[left]) / duration
        data[left + 1 : right] = (
            (1.0 - alpha[:, None, None]) * data[left]
            + alpha[:, None, None] * data[right]
        )
        repaired[left + 1 : right] = True
    return data, np.flatnonzero(repaired).astype(int).tolist()


def _load_world_points_zup(
    annotation_hdf5: Path,
    calibration: Path,
    repair_policy: dict[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    with h5py.File(annotation_hdf5, "r") as archive:
        required = (
            "hand-pose/left_joints",
            "hand-pose/right_joints",
            "hand-pose/timestamps",
            "cam-pose/rotations",
            "cam-pose/translations",
            "cam-pose/timestamps",
        )
        missing = [key for key in required if key not in archive]
        if missing:
            raise KeyError(f"annotation is missing datasets: {missing}")
        hand_abs = np.asarray(archive["hand-pose/timestamps"], dtype=np.float64)
        camera_abs = np.asarray(archive["cam-pose/timestamps"], dtype=np.float64)
        camera_rotation = np.asarray(
            archive["cam-pose/rotations"], dtype=np.float64
        )
        camera_translation = np.asarray(
            archive["cam-pose/translations"], dtype=np.float64
        )
        joints = {
            side: np.asarray(archive[f"hand-pose/{side}_joints"], dtype=np.float64)
            for side in ("left", "right")
        }
        valid = {
            side: (
                np.asarray(archive[f"hand-pose/{side}_valid"], dtype=bool)
                if f"hand-pose/{side}_valid" in archive
                else np.ones(len(hand_abs), dtype=bool)
            )
            for side in ("left", "right")
        }
    if len(hand_abs) < 2 or np.any(np.diff(hand_abs) <= 0.0):
        raise ValueError("hand timestamps must be strictly increasing")
    if len(camera_abs) < 2 or np.any(np.diff(camera_abs) <= 0.0):
        raise ValueError("camera timestamps must be strictly increasing")
    landmark_repair: dict[str, Any] = {}
    repair_enabled = bool(repair_policy.get("enabled", False))
    repair_per_hand = repair_policy.get("per_hand", {})
    for side in ("left", "right"):
        if joints[side].shape != (len(hand_abs), 21, 3):
            raise ValueError(f"invalid {side} joint shape {joints[side].shape}")
        joints[side] = _interpolate_invalid(joints[side], valid[side], hand_abs, side)
        events = (
            repair_per_hand.get(side, {}).get("events", [])
            if repair_enabled
            else []
        )
        joints[side], repaired_indices = repair_landmark_events(
            joints[side], hand_abs, events, side
        )
        expected_indices = (
            repair_per_hand.get(side, {}).get("repaired_source_frame_indices", [])
            if repair_enabled
            else []
        )
        if repaired_indices != [int(index) for index in expected_indices]:
            raise ValueError(
                f"{side} landmark repair indices do not match prepare metadata"
            )
        landmark_repair[side] = {
            "event_count": len(events),
            "repaired_source_frame_indices": repaired_indices,
            "repaired_frame_count": len(repaired_indices),
        }
    optical_to_link = np.asarray(np.load(calibration), dtype=np.float64)
    if optical_to_link.shape != (3, 3) or not np.allclose(
        optical_to_link.T @ optical_to_link, np.eye(3), atol=1.0e-5
    ):
        raise ValueError(f"invalid optical-to-link rotation: {calibration}")

    hand_time = hand_abs - hand_abs[0]
    camera_time = camera_abs - hand_abs[0]
    camera_rotation_at_hand = Slerp(
        camera_time, Rotation.from_matrix(camera_rotation)
    )(np.clip(hand_time, camera_time[0], camera_time[-1])).as_matrix()
    camera_translation_at_hand = np.column_stack(
        [
            np.interp(hand_time, camera_time, camera_translation[:, axis])
            for axis in range(3)
        ]
    )
    world_from_optical = camera_rotation_at_hand @ optical_to_link
    points_zup = {}
    for side in ("left", "right"):
        points_yup = (
            np.einsum("tij,tkj->tki", world_from_optical, joints[side])
            + camera_translation_at_hand[:, None]
        )
        points_zup[side] = np.einsum("ij,tkj->tki", YUP_TO_ZUP, points_yup)
    return points_zup, hand_time, {
        "native_frames": int(len(hand_abs)),
        "left_valid_frames": int(np.count_nonzero(valid["left"])),
        "right_valid_frames": int(np.count_nonzero(valid["right"])),
        "source_coordinate_frame": "camera optical OpenCV",
        "output_coordinate_frame": "ScaleTrack Z-up world",
        "world_transform": (
            "p_zup = YUP_TO_ZUP @ (R_link_to_world @ "
            "R_optical_to_link @ p_optical + t_world)"
        ),
        "landmark_outlier_repair": {
            "enabled": repair_enabled,
            "coordinate_frame": "camera optical OpenCV",
            "method": (
                "linear interpolation of all 21 landmarks between prepared "
                "clean anchors"
            ),
            "per_hand": landmark_repair,
        },
    }


def _resample_pose(
    positions: np.ndarray,
    rotations: np.ndarray,
    source_time: np.ndarray,
    target_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    position = np.column_stack(
        [np.interp(target_time, source_time, positions[:, axis]) for axis in range(3)]
    )
    rotation = Slerp(source_time, Rotation.from_matrix(rotations))(
        np.clip(target_time, source_time[0], source_time[-1])
    ).as_matrix()
    return position, rotation


def _stats(values: np.ndarray) -> dict[str, float]:
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(value)),
        "median": float(np.median(value)),
        "p95": float(np.percentile(value, 95.0)),
        "max": float(np.max(value)),
    }


def apply_local_orientation_offset(
    positions: np.ndarray,
    rotations: np.ndarray,
    offset: np.ndarray,
    pivot_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a fixed local rotation while preserving one local pivot point."""
    position = np.asarray(positions, dtype=np.float64)
    rotation = np.asarray(rotations, dtype=np.float64)
    local_rotation = np.asarray(offset, dtype=np.float64)
    pivot = np.asarray(pivot_local, dtype=np.float64)
    if local_rotation.shape != (3, 3) or pivot.shape != (3,):
        raise ValueError("invalid wrist orientation offset or pivot shape")
    if not np.allclose(local_rotation.T @ local_rotation, np.eye(3), atol=1.0e-6):
        raise ValueError("wrist orientation offset is not orthonormal")
    if not np.isclose(np.linalg.det(local_rotation), 1.0, atol=1.0e-6):
        raise ValueError("wrist orientation offset is not a proper rotation")
    corrected_rotation = np.einsum("tij,jk->tik", rotation, local_rotation)
    pivot_world = position + np.einsum("tij,j->ti", rotation, pivot)
    corrected_position = pivot_world - np.einsum(
        "tij,j->ti", corrected_rotation, pivot
    )
    return corrected_position, corrected_rotation


def load_orientation_offset(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("composition") != "R_corrected = R_template @ R_offset_local":
        raise ValueError("unsupported wrist orientation offset composition")
    applications = payload.get("applications", {})
    for side in ("left", "right"):
        matrix = np.asarray(applications.get(side, {}).get("matrix"), dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"orientation offset is missing {side} matrix")
    pivot = np.asarray(payload.get("pivot_local_m"), dtype=np.float64)
    if pivot.shape != (3,) or not np.all(np.isfinite(pivot)):
        raise ValueError("orientation offset has invalid pivot_local_m")
    payload["_resolved_path"] = str(path.resolve())
    payload["_sha256"] = _sha256(path)
    return payload


def retarget_reference(
    *,
    annotation_hdf5: Path,
    base_reference: Path,
    base_targets: Path,
    prepare_meta: Path,
    output_reference: Path,
    output_targets: Path,
    meta_json: Path,
    orientation_offset_json: Path | None = None,
    world_z_offset_m: float = 0.0,
    huber_delta_m: float = 0.008,
    iterations: int = 4,
) -> dict[str, Any]:
    paths = [annotation_hdf5, base_reference, base_targets, prepare_meta]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    calibration = annotation_hdf5.parent / "calibrations/R_optical_to_link.npy"
    if not calibration.is_file():
        raise FileNotFoundError(calibration)
    reference = _load_npz(base_reference)
    targets = _load_npz(base_targets)
    preparation = json.loads(prepare_meta.read_text(encoding="utf-8"))
    orientation_offset = load_orientation_offset(orientation_offset_json)
    body_names = [str(value) for value in reference["body_names"].reshape(-1).tolist()]
    fps = float(np.asarray(reference["fps"]).reshape(()))
    target_time = np.asarray(targets["timestamps"], dtype=np.float64)
    frames = len(target_time)
    if reference["body_pos_w"].shape[0] != frames:
        raise ValueError("reference and target frame counts differ")
    if not np.isclose(float(targets["fps_session"]), fps, atol=1.0e-8):
        raise ValueError("reference and target FPS differ")
    selection = preparation.get("selection", {})
    start = int(selection.get("selected_source_start_index", 0))
    end = int(
        selection.get(
            "selected_source_end_index",
            start + int(preparation["source_frames"]) - 1,
        )
    )

    repair_policy = selection.get("transient_wrist_outlier_repair", {})
    world_points, native_time_all, source_meta = _load_world_points_zup(
        annotation_hdf5, calibration, repair_policy
    )
    if start < 0 or end >= len(native_time_all) or end <= start:
        raise ValueError(f"invalid selected source range [{start}, {end}]")
    native_time = native_time_all[start : end + 1] - native_time_all[start]
    if (
        target_time[0] < native_time[0] - 1.0e-9
        or target_time[-1] > native_time[-1] + 1.0e-6
    ):
        raise ValueError("prepared target timeline is outside selected hand timeline")

    output_reference_arrays = {name: value.copy() for name, value in reference.items()}
    output_target_arrays = {name: value.copy() for name, value in targets.items()}
    side_reports: dict[str, Any] = {}
    templates: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        points = world_points[side][start : end + 1]
        native_position, native_rotation, template = template_palm_kabsch(
            points, huber_delta_m=huber_delta_m, iterations=iterations
        )
        offset_report = None
        if orientation_offset is not None:
            offset_matrix = np.asarray(
                orientation_offset["applications"][side]["matrix"],
                dtype=np.float64,
            )
            pivot_local = np.asarray(
                orientation_offset["pivot_local_m"], dtype=np.float64
            )
            uncorrected_position = native_position.copy()
            native_position, native_rotation = apply_local_orientation_offset(
                native_position, native_rotation, offset_matrix, pivot_local
            )
            offset_report = {
                "angle_deg": float(
                    orientation_offset["applications"][side]["angle_deg"]
                ),
                "axis": orientation_offset["applications"][side]["axis"],
                "position_delta_m": _stats(
                    np.linalg.norm(native_position - uncorrected_position, axis=1)
                ),
            }
        position, rotation = _resample_pose(
            native_position, native_rotation, native_time, target_time
        )
        position[:, 2] += float(world_z_offset_m)
        quaternion = _continuous_wxyz(rotation)
        body_index = body_names.index(WRIST_BODIES[side])
        old_position = np.asarray(
            reference["body_pos_w"][:, body_index], dtype=np.float64
        )
        old_quaternion = np.asarray(
            reference["body_quat_w"][:, body_index], dtype=np.float64
        )
        old_rotation = Rotation.from_quat(old_quaternion, scalar_first=True)
        orientation_delta = np.degrees(
            (old_rotation.inv() * Rotation.from_matrix(rotation)).magnitude()
        )
        output_reference_arrays["body_pos_w"][:, body_index] = position.astype(
            output_reference_arrays["body_pos_w"].dtype
        )
        output_reference_arrays["body_quat_w"][:, body_index] = quaternion.astype(
            output_reference_arrays["body_quat_w"].dtype
        )
        output_reference_arrays["body_lin_vel_w"][:, body_index] = _finite_difference(
            position, fps
        ).astype(output_reference_arrays["body_lin_vel_w"].dtype)
        output_reference_arrays["body_ang_vel_w"][:, body_index] = (
            _angular_velocity_world(rotation, fps).astype(
                output_reference_arrays["body_ang_vel_w"].dtype
            )
        )
        output_target_arrays[f"{side}_wrist"] = position.astype(np.float32)
        output_target_arrays[f"{side}_wrist_quat_wxyz"] = quaternion.astype(np.float32)
        templates[side] = template
        side_reports[side] = {
            "position_delta_m": _stats(np.linalg.norm(position - old_position, axis=1)),
            "orientation_delta_deg": _stats(orientation_delta),
            "first_position_zup": position[0].tolist(),
            "first_quaternion_wxyz": quaternion[0].tolist(),
            "session_orientation_offset": offset_report,
            "world_z_offset_m": float(world_z_offset_m),
        }

    method = (
        "template_palm_kabsch_session_fixed_so3"
        if orientation_offset is not None
        else "template_palm_kabsch"
    )
    output_reference_arrays["wrist_reference_method"] = np.asarray(method)
    output_reference_arrays["wrist_reference_huber_delta_m"] = np.asarray(
        huber_delta_m, dtype=np.float64
    )
    output_reference_arrays["wrist_reference_irls_iterations"] = np.asarray(
        iterations, dtype=np.int32
    )
    output_target_arrays["wrist_reference_method"] = np.asarray(method)
    if orientation_offset is not None:
        output_reference_arrays["wrist_orientation_offset_source"] = np.asarray(
            orientation_offset["_resolved_path"]
        )
        output_reference_arrays["wrist_orientation_offset_sha256"] = np.asarray(
            orientation_offset["_sha256"]
        )
        output_reference_arrays["wrist_orientation_offset_pivot_local_m"] = (
            np.asarray(orientation_offset["pivot_local_m"], dtype=np.float64)
        )
        for side in ("left", "right"):
            matrix = np.asarray(
                orientation_offset["applications"][side]["matrix"],
                dtype=np.float64,
            )
            output_reference_arrays[f"{side}_wrist_orientation_offset_matrix"] = matrix
            output_target_arrays[f"{side}_wrist_orientation_offset_matrix"] = matrix
    output_target_arrays["left_palm_template"] = templates["left"].astype(np.float64)
    output_target_arrays["right_palm_template"] = templates["right"].astype(np.float64)

    # The adapter is intentionally unable to alter the carrier, joints, clock,
    # or Kimodo torso heading.
    wrist_indices = [body_names.index(name) for name in WRIST_BODIES.values()]
    non_wrist = np.ones(len(body_names), dtype=bool)
    non_wrist[wrist_indices] = False
    for field in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        if not np.array_equal(
            reference[field][:, non_wrist],
            output_reference_arrays[field][:, non_wrist],
        ):
            raise RuntimeError(f"non-wrist values changed in {field}")
    for field in ("joint_pos", "joint_vel", "body_names", "joint_names", "fps"):
        if not np.array_equal(reference[field], output_reference_arrays[field]):
            raise RuntimeError(f"base reference field changed: {field}")
    for field in ("timestamps", "kimodo_pelvis_heading_quat_wxyz", "fps_session"):
        if not np.array_equal(targets[field], output_target_arrays[field]):
            raise RuntimeError(f"base target field changed: {field}")

    _atomic_npz(output_reference, output_reference_arrays)
    _atomic_npz(output_targets, output_target_arrays)
    report: dict[str, Any] = {
        "schema_version": 1,
        "method": method,
        "selection_policy": preparation.get("selection_policy"),
        "selection": selection,
        "source_frames": preparation.get("source_frames"),
        "output_frames": frames,
        "output_fps": fps,
        "inputs": {
            "annotation_hdf5": str(annotation_hdf5.resolve()),
            "annotation_sha256": _sha256(annotation_hdf5),
            "base_reference": str(base_reference.resolve()),
            "base_reference_sha256": _sha256(base_reference),
            "base_targets": str(base_targets.resolve()),
            "base_targets_sha256": _sha256(base_targets),
            "prepare_meta": str(prepare_meta.resolve()),
        },
        "outputs": {
            "reference": str(output_reference.resolve()),
            "targets": str(output_targets.resolve()),
        },
        "contract": {
            "anchors": "wrist + index/middle/ring/little MCP",
            "template": "sequence median anchors in wrist-link axes",
            "fit": "per-frame Huber IRLS Kabsch",
            "palm_site_in_wrist_link_m": ROBOT_PALM_SITE.tolist(),
            "huber_delta_m": float(huber_delta_m),
            "irls_iterations": int(iterations),
            "transient_landmark_repair": (
                "apply prepare_meta bounded-event anchors to all 21 camera-local landmarks"
            ),
            "finger_retarget_method": "position (separate downstream stage)",
            "session_orientation_offset": (
                {
                    "source": orientation_offset["_resolved_path"],
                    "sha256": orientation_offset["_sha256"],
                    "method": orientation_offset.get("method"),
                    "composition": orientation_offset.get("composition"),
                    "opposite_hand_policy": orientation_offset.get(
                        "opposite_hand_policy"
                    ),
                    "pivot": orientation_offset.get("pivot"),
                    "pivot_local_m": orientation_offset.get("pivot_local_m"),
                    "applications": orientation_offset.get("applications"),
                }
                if orientation_offset is not None
                else None
            ),
            "world_z_offset_m": float(world_z_offset_m),
        },
        "source": source_meta,
        "delta_from_historical_reference": side_reports,
        "gates": {
            "non_wrist_reference_unchanged": True,
            "joint_and_timeline_fields_unchanged": True,
            "finite_output": bool(
                np.all(np.isfinite(output_reference_arrays["body_pos_w"]))
                and np.all(np.isfinite(output_reference_arrays["body_quat_w"]))
            ),
        },
    }
    if not report["gates"]["finite_output"]:
        raise RuntimeError("template-palm reference contains non-finite values")
    _atomic_json(meta_json, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-hdf5", type=Path, required=True)
    parser.add_argument("--base-reference", type=Path, required=True)
    parser.add_argument("--base-targets", type=Path, required=True)
    parser.add_argument("--prepare-meta", type=Path, required=True)
    parser.add_argument("--output-reference", type=Path, required=True)
    parser.add_argument("--output-targets", type=Path, required=True)
    parser.add_argument("--meta-json", type=Path, required=True)
    parser.add_argument("--orientation-offset-json", type=Path)
    parser.add_argument("--world-z-offset-m", type=float, default=0.0)
    parser.add_argument("--huber-delta-m", type=float, default=0.008)
    parser.add_argument("--irls-iterations", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = retarget_reference(
        annotation_hdf5=args.annotation_hdf5.expanduser().resolve(),
        base_reference=args.base_reference.expanduser().resolve(),
        base_targets=args.base_targets.expanduser().resolve(),
        prepare_meta=args.prepare_meta.expanduser().resolve(),
        output_reference=args.output_reference.expanduser().resolve(),
        output_targets=args.output_targets.expanduser().resolve(),
        meta_json=args.meta_json.expanduser().resolve(),
        orientation_offset_json=(
            args.orientation_offset_json.expanduser().resolve()
            if args.orientation_offset_json is not None
            else None
        ),
        world_z_offset_m=float(args.world_z_offset_m),
        huber_delta_m=float(args.huber_delta_m),
        iterations=int(args.irls_iterations),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
