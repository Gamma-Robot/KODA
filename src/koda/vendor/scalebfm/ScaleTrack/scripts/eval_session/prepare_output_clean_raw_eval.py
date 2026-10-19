#!/usr/bin/env python3
"""Prepare a raw-wrist-only ScaleTrack evaluation motion.

The raw hand tracks come from ``annotation.hdf5``.  Unobserved G1 bodies use a
fixed nominal MuJoCo carrier; only wrist-link positions and orientations are
dynamic tracking targets.  Wrist-link orientation is reconstructed directly
from raw hand keypoints with the historical raw-to-G1 convention.  This file
does not alter the source data or training code.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
SCALEBFM_ROOT = Path(os.environ.get("KODA_SCALEBFM_ROOT", str(ROOT))).expanduser().resolve()
MJCF = Path(os.environ.get("KODA_SCALETRACK_MJCF", str(SCALEBFM_ROOT / "ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof/g1_29dof.xml"))).expanduser().resolve()
SCHEMA_JSON = SCRIPT_DIR / "isaac_g1_29dof_order.json"
TEMPLATE = SCALEBFM_ROOT / "motion_schema_template.npz"

# Y-up world -> ScaleTrack Z-up: (x, y, z) -> (x, -z, y).
YUP_TO_ZUP = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


# Historical Stera camera-forward -> G1-forward alignment.  This is used only
# for Kimodo's separate heading condition; it is never applied to the static
# non-wrist carrier in the ScaleTrack tracking reference.
STERA_TO_G1_HEADING = np.asarray(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
)

# The old raw-keypoint conversion defines a robot wrist-yaw frame, not a hand
# frame.  The final rotation is a *local* right multiplication.
WRIST_LINK_LOCAL_RX_MINUS_90 = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64
)

# Internal detector gaps (both anchors present) are filled in the camera-local
# joint frame: wrist translation uses cubic Hermite with endpoint velocities,
# wrist orientation uses quaternion slerp, and the remaining joints keep a
# time-lerped wrist-local shape. Endpoint gaps are never extrapolated.
DEFAULT_MAX_GAP_FRAMES = 10
_PALM_AXIS_MIN_NORM = 1.0e-6


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _historical_wrist_link_rot(points_zup: np.ndarray) -> np.ndarray:
    """Reconstruct the historical raw-keypoint G1 wrist-yaw link frame.

    ``points_zup`` must contain all 21 hand keypoints in the ScaleTrack Z-up
    world frame.  This exact palm-frame construction avoids importing any
    legacy motion trajectory while retaining its raw-wrist orientation
    convention.
    """
    p = np.asarray(points_zup, dtype=np.float64).reshape(21, 3)
    x = _normalize(p[12] - p[0])  # wrist -> middle fingertip
    across = _normalize(p[8] - p[20])  # index tip -> pinky tip
    z = -_normalize(np.cross(across, x))
    y = _normalize(np.cross(z, x))
    x = _normalize(np.cross(y, z))
    return np.stack([x, y, z], axis=1) @ WRIST_LINK_LOCAL_RX_MINUS_90


def _yaw_only(rot: np.ndarray) -> np.ndarray:
    """Return the Z-up yaw component of a rotation matrix."""
    r = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    yaw = float(np.arctan2(r[1, 0], r[0, 0]))
    c, s = np.cos(yaw), np.sin(yaw)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _robust_jump_threshold(values: np.ndarray, *, floor: float) -> float:
    """Return an absolute step threshold above the stable-motion distribution."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(floor)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return float(max(floor, median + 8.0 * 1.4826 * mad))


def _wrist_rotation_from_camera_joints(points: np.ndarray) -> np.ndarray:
    """Palm-frame wrist rotation in the camera-local joint frame.

    Falls back to identity when the middle/index/pinky span is degenerate so
    a missing idle hand can still be filled from wrist translation alone.
    """
    p = np.asarray(points, dtype=np.float64).reshape(21, 3)
    x = p[12] - p[0]
    across = p[8] - p[20]
    if (
        float(np.linalg.norm(x)) < _PALM_AXIS_MIN_NORM
        or float(np.linalg.norm(across)) < _PALM_AXIS_MIN_NORM
        or float(np.linalg.norm(np.cross(across, x))) < _PALM_AXIS_MIN_NORM
    ):
        return np.eye(3, dtype=np.float64)
    return _historical_wrist_link_rot(p)


def _wrist_local_joints(points: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Return 21 joints expressed in the wrist frame ``rotation`` at joint 0."""
    points = np.asarray(points, dtype=np.float64).reshape(21, 3)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return (points - points[0]) @ rotation


def _cubic_hermite(
    u: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
) -> np.ndarray:
    """Cubic Hermite on the unit interval. ``m0``/``m1`` are ``dp/du``."""
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    p0 = np.asarray(p0, dtype=np.float64).reshape(-1)
    p1 = np.asarray(p1, dtype=np.float64).reshape(-1)
    m0 = np.asarray(m0, dtype=np.float64).reshape(-1)
    m1 = np.asarray(m1, dtype=np.float64).reshape(-1)
    u2 = u * u
    u3 = u2 * u
    h00 = (2.0 * u3 - 3.0 * u2 + 1.0)[:, None]
    h10 = (u3 - 2.0 * u2 + u)[:, None]
    h01 = (-2.0 * u3 + 3.0 * u2)[:, None]
    h11 = (u3 - u2)[:, None]
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


def _anchor_translation_velocity(
    timestamps: np.ndarray,
    positions: np.ndarray,
    usable: np.ndarray,
    index: int,
    *,
    side: str,
) -> np.ndarray:
    """One-sided finite-difference wrist velocity. Missing neighbor -> 0."""
    positions = np.asarray(positions, dtype=np.float64)
    if side == "left":
        neighbor = int(index) - 1
        if neighbor >= 0 and bool(usable[neighbor]):
            dt = float(timestamps[index] - timestamps[neighbor])
            if dt > 0.0:
                return (positions[index] - positions[neighbor]) / dt
    elif side == "right":
        neighbor = int(index) + 1
        if neighbor < len(timestamps) and bool(usable[neighbor]):
            dt = float(timestamps[neighbor] - timestamps[index])
            if dt > 0.0:
                return (positions[neighbor] - positions[index]) / dt
    else:
        raise ValueError(f"invalid velocity side {side!r}")
    return np.zeros(positions.shape[-1], dtype=np.float64)


def _slerp_rotations(rotation_left: np.ndarray, rotation_right: np.ndarray, u: np.ndarray) -> np.ndarray:
    u = np.clip(np.asarray(u, dtype=np.float64).reshape(-1), 0.0, 1.0)
    slerp = Slerp(
        [0.0, 1.0],
        Rotation.from_matrix(
            np.stack(
                [
                    np.asarray(rotation_left, dtype=np.float64).reshape(3, 3),
                    np.asarray(rotation_right, dtype=np.float64).reshape(3, 3),
                ],
                axis=0,
            )
        ),
    )
    matrices = np.asarray(slerp(u).as_matrix(), dtype=np.float64)
    if matrices.ndim == 2:
        matrices = matrices.reshape(1, 3, 3)
    return matrices


def _interpolate_camera_hand_se3(
    timestamps: np.ndarray,
    joints: np.ndarray,
    usable: np.ndarray,
    left: int,
    right: int,
    query_slice: slice,
) -> np.ndarray:
    """Fill a camera-local hand gap with Hermite translation and quaternion slerp."""
    query_t = timestamps[query_slice]
    span = float(timestamps[right] - timestamps[left])
    u = np.clip((query_t - timestamps[left]) / span, 0.0, 1.0)
    p_left = np.asarray(joints[left, 0], dtype=np.float64)
    p_right = np.asarray(joints[right, 0], dtype=np.float64)
    r_left = _wrist_rotation_from_camera_joints(joints[left])
    r_right = _wrist_rotation_from_camera_joints(joints[right])
    local_left = _wrist_local_joints(joints[left], r_left)
    local_right = _wrist_local_joints(joints[right], r_right)
    wrist_positions = joints[:, 0]
    m_left = _anchor_translation_velocity(
        timestamps, wrist_positions, usable, left, side="left"
    ) * span
    m_right = _anchor_translation_velocity(
        timestamps, wrist_positions, usable, right, side="right"
    ) * span
    position = _cubic_hermite(u, p_left, p_right, m_left, m_right)
    rotation = _slerp_rotations(r_left, r_right, u)
    local = (1.0 - u)[:, None, None] * local_left + u[:, None, None] * local_right
    return np.einsum("njp,nqp->njq", local, rotation) + position[:, None, :]


def _repair_bounded_joint_gaps(
    timestamps: np.ndarray,
    joints: np.ndarray,
    valid: np.ndarray,
    *,
    max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fill short internal detector gaps without endpoint extrapolation.

    Wrist translation uses cubic Hermite with finite-difference endpoint
    velocities (zero if a neighbor is missing). Wrist orientation uses
    quaternion slerp. Finger joints keep a timestamp-linear blend of the
    two wrist-local hand shapes. Both anchors must be valid.
    """
    timestamps = np.asarray(timestamps, dtype=np.float64)
    repaired = np.asarray(joints, dtype=np.float64).copy()
    valid = np.asarray(valid, dtype=bool).copy()
    finite = np.all(np.isfinite(repaired), axis=(1, 2))
    usable = valid & finite
    bad_indices = np.flatnonzero(~usable)
    repaired_indices: list[int] = []
    runs: list[dict] = []
    cursor = 0
    while cursor < len(bad_indices):
        start = int(bad_indices[cursor])
        end = start
        while cursor + 1 < len(bad_indices) and int(bad_indices[cursor + 1]) == end + 1:
            cursor += 1
            end = int(bad_indices[cursor])
        left, right = start - 1, end + 1
        length = end - start + 1
        repairable = bool(
            length <= int(max_gap_frames)
            and left >= 0
            and right < len(timestamps)
            and usable[left]
            and usable[right]
            and timestamps[right] > timestamps[left]
        )
        if repairable:
            repaired[start : end + 1] = _interpolate_camera_hand_se3(
                timestamps,
                repaired,
                usable,
                left,
                right,
                slice(start, end + 1),
            )
            valid[start : end + 1] = True
            usable[start : end + 1] = True
            repaired_indices.extend(range(start, end + 1))
        runs.append(
            {
                "start_index": start,
                "end_index": end,
                "length": length,
                "left_anchor_index": left if left >= 0 else None,
                "right_anchor_index": right if right < len(timestamps) else None,
                "repaired": repairable,
            }
        )
        cursor += 1
    return repaired, valid, {
        "method": (
            "camera-local wrist SE3: cubic Hermite translation with endpoint "
            "velocities, quaternion slerp, wrist-local hand-shape lerp"
        ),
        "max_gap_frames": int(max_gap_frames),
        "endpoint_extrapolation": False,
        "original_invalid_indices": bad_indices.astype(int).tolist(),
        "repaired_indices": repaired_indices,
        "runs": runs,
    }


def _repair_transient_pose_outliers(
    timestamps: np.ndarray,
    positions: np.ndarray,
    rotations: np.ndarray,
    *,
    max_event_duration_s: float = 1.1,
    position_floor_m: float = 0.05,
    orientation_floor_deg: float = 45.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Interpolate bounded pose excursions without changing the time axis.

    A repair requires two anomalous edges around a segment whose two anchors
    themselves differ by less than one anomalous jump. Dividing that
    difference by the frame span would let a 140° spike look like a smooth
    15°/frame slope and keep the spike as a "clean" return anchor. A
    one-sided jump that persists is intentionally left unchanged.
    """
    timestamps = np.asarray(timestamps, dtype=np.float64)
    repaired_position = np.asarray(positions, dtype=np.float64).copy()
    repaired_rotation = np.asarray(rotations, dtype=np.float64).copy()
    if repaired_position.shape != (len(timestamps), 3):
        raise ValueError(f"invalid pose positions shape {repaired_position.shape}")
    if repaired_rotation.shape != (len(timestamps), 3, 3):
        raise ValueError(f"invalid pose rotations shape {repaired_rotation.shape}")
    if len(timestamps) < 3:
        return repaired_position, repaired_rotation, {
            "repaired_source_frame_indices": [],
            "events": [],
        }

    position_steps = np.linalg.norm(np.diff(repaired_position, axis=0), axis=1)
    orientation_steps = np.degrees(
        (
            Rotation.from_matrix(repaired_rotation[:-1]).inv()
            * Rotation.from_matrix(repaired_rotation[1:])
        ).magnitude()
    )
    position_threshold = _robust_jump_threshold(
        position_steps, floor=position_floor_m
    )
    orientation_threshold = _robust_jump_threshold(
        orientation_steps, floor=orientation_floor_deg
    )
    anomalous_edges = np.flatnonzero(
        (position_steps > position_threshold)
        | (orientation_steps > orientation_threshold)
    ).astype(int)

    events = []
    repaired_mask = np.zeros(len(timestamps), dtype=bool)
    anomalous_set = {int(edge) for edge in anomalous_edges}
    candidates: list[dict] = []
    for left_cursor, left_edge in enumerate(anomalous_edges):
        left_anchor = int(left_edge)
        # A spike vertex (incoming and outgoing edges both anomalous) is the
        # outlier itself, not a clean start anchor.
        if (left_anchor - 1) in anomalous_set:
            continue
        for right_cursor in range(left_cursor + 1, len(anomalous_edges)):
            right_anchor = int(anomalous_edges[right_cursor]) + 1
            duration_s = float(timestamps[right_anchor] - timestamps[left_anchor])
            if duration_s > max_event_duration_s:
                break
            frame_span = right_anchor - left_anchor
            if frame_span <= 1 or duration_s <= 0.0:
                continue
            if right_anchor >= len(timestamps):
                continue

            anchor_position_delta = float(
                np.linalg.norm(
                    repaired_position[right_anchor] - repaired_position[left_anchor]
                )
            )
            anchor_orientation_delta = float(
                np.degrees(
                    (
                        Rotation.from_matrix(repaired_rotation[left_anchor]).inv()
                        * Rotation.from_matrix(repaired_rotation[right_anchor])
                    ).magnitude()
                )
            )
            if (
                anchor_position_delta > position_threshold
                or anchor_orientation_delta > orientation_threshold
            ):
                continue

            query = timestamps[left_anchor + 1 : right_anchor]
            alpha = (query - timestamps[left_anchor]) / duration_s
            interpolated_position = (
                (1.0 - alpha[:, None]) * repaired_position[left_anchor]
                + alpha[:, None] * repaired_position[right_anchor]
            )
            interpolated_rotation_object = Slerp(
                [timestamps[left_anchor], timestamps[right_anchor]],
                Rotation.from_matrix(
                    [
                        repaired_rotation[left_anchor],
                        repaired_rotation[right_anchor],
                    ]
                ),
            )(query)
            interpolated_rotation = interpolated_rotation_object.as_matrix()
            position_residual = np.linalg.norm(
                repaired_position[left_anchor + 1 : right_anchor]
                - interpolated_position,
                axis=1,
            )
            orientation_residual = np.degrees(
                (
                    interpolated_rotation_object.inv()
                    * Rotation.from_matrix(
                        repaired_rotation[left_anchor + 1 : right_anchor]
                    )
                ).magnitude()
            )
            max_position_residual = float(np.max(position_residual))
            max_orientation_residual = float(np.max(orientation_residual))
            if (
                max_position_residual <= position_threshold
                and max_orientation_residual <= orientation_threshold
            ):
                continue
            candidates.append(
                {
                    "frame_span": int(frame_span),
                    "duration_s": duration_s,
                    "left_anchor": left_anchor,
                    "right_anchor": right_anchor,
                    "interpolated_position": interpolated_position,
                    "interpolated_rotation": interpolated_rotation,
                    "max_position_residual": max_position_residual,
                    "max_orientation_residual": max_orientation_residual,
                }
            )

    # Prefer the tightest valid window so a 1-frame spike inside a longer
    # mild excursion is repaired on its own, not smeared across the longer span.
    candidates.sort(
        key=lambda item: (
            int(item["frame_span"]),
            float(item["duration_s"]),
            int(item["left_anchor"]),
        )
    )
    for candidate in candidates:
        left_anchor = int(candidate["left_anchor"])
        right_anchor = int(candidate["right_anchor"])
        first_repaired = left_anchor + 1
        if np.any(repaired_mask[first_repaired:right_anchor]):
            continue
        last_repaired = right_anchor - 1
        repaired_position[first_repaired:right_anchor] = candidate[
            "interpolated_position"
        ]
        repaired_rotation[first_repaired:right_anchor] = candidate[
            "interpolated_rotation"
        ]
        repaired_mask[first_repaired:right_anchor] = True
        events.append(
            {
                "left_anchor_index": left_anchor,
                "right_anchor_index": right_anchor,
                "first_repaired_index": first_repaired,
                "last_repaired_index": last_repaired,
                "repaired_frame_count": right_anchor - first_repaired,
                "duration_s": float(candidate["duration_s"]),
                "max_position_residual_m": float(candidate["max_position_residual"]),
                "max_orientation_residual_deg": float(
                    candidate["max_orientation_residual"]
                ),
            }
        )

    return repaired_position, repaired_rotation, {
        "method": (
            "camera-local bounded excursion removal; linear position "
            "interpolation and quaternion Slerp between adjacent clean anchors"
        ),
        "position_jump_threshold_m": position_threshold,
        "orientation_jump_threshold_deg": orientation_threshold,
        "max_event_duration_s": float(max_event_duration_s),
        "candidate_anomalous_edges": anomalous_edges.tolist(),
        "repaired_source_frame_indices": np.flatnonzero(repaired_mask)
        .astype(int)
        .tolist(),
        "events": events,
    }


def _clean_dual_wrist_interval(
    timestamps: np.ndarray,
    tracks: dict[str, tuple[np.ndarray, np.ndarray]],
    valid: dict[str, np.ndarray],
) -> tuple[int, int, dict]:
    """Find the longest dual-valid interval without position/orientation jumps."""
    n = len(timestamps)
    per_hand = {}
    edge_ok = np.ones(max(0, n - 1), dtype=bool)
    common = np.ones(n, dtype=bool)
    for side in ("left", "right"):
        pos, rot = tracks[side]
        finite = np.isfinite(pos).all(axis=1) & np.isfinite(rot).all(axis=(1, 2))
        side_valid = np.asarray(valid[side], dtype=bool) & finite
        common &= side_valid
        edge_idx = np.flatnonzero(side_valid[:-1] & side_valid[1:])
        step = np.full(max(0, n - 1), np.inf, dtype=np.float64)
        angle = np.full(max(0, n - 1), np.inf, dtype=np.float64)
        if edge_idx.size:
            step[edge_idx] = np.linalg.norm(pos[edge_idx + 1] - pos[edge_idx], axis=1)
            angle[edge_idx] = np.degrees(
                (Rotation.from_matrix(rot[edge_idx]).inv()
                 * Rotation.from_matrix(rot[edge_idx + 1])).magnitude()
            )
        dt = np.diff(timestamps)
        dt_ref = float(np.median(dt[edge_idx])) if edge_idx.size else 0.0
        pos_threshold = _robust_jump_threshold(step[edge_idx], floor=0.05)
        angle_threshold = _robust_jump_threshold(angle[edge_idx], floor=45.0)
        dt_ok = np.ones(max(0, n - 1), dtype=bool)
        if dt_ref > 0:
            dt_ok = dt <= 1.5 * dt_ref
        jumps = (
            side_valid[:-1]
            & side_valid[1:]
            & (~dt_ok | (step > pos_threshold) | (angle > angle_threshold))
        )
        edge_ok &= side_valid[:-1] & side_valid[1:] & ~jumps & dt_ok
        per_hand[side] = {
            "position_jump_threshold_m": pos_threshold,
            "orientation_jump_threshold_deg": angle_threshold,
            "median_dt_s": dt_ref,
            "position_jump_edges": np.flatnonzero(jumps).astype(int).tolist(),
            "orientation_jump_edges": np.flatnonzero(
                side_valid[:-1] & side_valid[1:] & (angle > angle_threshold)
            ).astype(int).tolist(),
            "invalid_frames": np.flatnonzero(~side_valid).astype(int).tolist(),
        }

    # An edge belongs to a usable interval only when both hands are valid and
    # neither hand has a jump. The interval endpoints are never interpolated
    # across a rejected edge.
    runs = []
    start = None
    for i, good in enumerate(edge_ok):
        if good and start is None:
            start = i
        elif not good and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, n - 1))
    if not runs:
        raise RuntimeError("No dual-wrist interval remains after validity/jump filtering")
    best_start, best_end = max(runs, key=lambda x: (x[1] - x[0] + 1, -x[0]))
    summary = {
        "selected_source_start_index": int(best_start),
        "selected_source_end_index": int(best_end),
        "selected_source_frame_count": int(best_end - best_start + 1),
        "selected_source_duration_s": float(timestamps[best_end] - timestamps[best_start]),
        "dual_valid_frame_count": int(common.sum()),
        "dual_valid_runs": [
            [int(a), int(b), int(b - a + 1)]
            for a, b in zip(
                np.flatnonzero(common & np.r_[True, ~common[:-1]]),
                np.flatnonzero(common & np.r_[~common[1:], True]),
            )
        ],
        "clean_edge_runs": [[int(a), int(b), int(b - a + 1)] for a, b in runs],
        "per_hand": per_hand,
    }
    return best_start, best_end, summary


def _raw_wrist_tracks(
    h5_path: Path,
    max_duration_s: float,
    *,
    source_start_index: int | None = None,
    source_end_index: int | None = None,
    use_full_source: bool = False,
    repair_transient_outliers: bool = False,
    max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES,
):
    r_o2l = np.load(h5_path.parent / "calibrations/R_optical_to_link.npy").astype(np.float64)
    with h5py.File(h5_path, "r") as f:
        timestamps = np.asarray(f["hand-pose/timestamps"], dtype=np.float64)
        cam_r = np.asarray(f["cam-pose/rotations"], dtype=np.float64)
        cam_t = np.asarray(f["cam-pose/translations"], dtype=np.float64)
        tracks = {}
        repaired_tracks = {}
        repair_diagnostics = {}
        missing_frame_diagnostics = {}
        valid = {}
        for side in ("left", "right"):
            joints = np.asarray(f[f"hand-pose/{side}_joints"], dtype=np.float64)
            valid[side] = np.asarray(f[f"hand-pose/{side}_valid"], dtype=bool)
            joints, valid[side], missing_frame_diagnostics[side] = (
                _repair_bounded_joint_gaps(
                    timestamps,
                    joints,
                    valid[side],
                    max_gap_frames=int(max_gap_frames),
                )
            )
            if not np.all(valid[side]) or not np.all(np.isfinite(joints)):
                raise ValueError(
                    f"{side} hand contains an unrepairable missing-keypoint gap; "
                    f"diagnostics={missing_frame_diagnostics[side]}"
                )
            points_camera = joints @ r_o2l.T
            position_camera = points_camera[:, 0].copy()
            rotation_camera = np.asarray(
                [_historical_wrist_link_rot(points) for points in points_camera]
            )
            if repair_transient_outliers:
                (
                    repaired_position_camera,
                    repaired_rotation_camera,
                    repair_diagnostics[side],
                ) = _repair_transient_pose_outliers(
                    timestamps, position_camera, rotation_camera
                )
            else:
                repaired_position_camera = position_camera
                repaired_rotation_camera = rotation_camera
                repair_diagnostics[side] = {
                    "repaired_source_frame_indices": [],
                    "events": [],
                }
            pos_y, rot_y = [], []
            repaired_pos_y, repaired_rot_y = [], []
            for i in range(len(timestamps)):
                points_y = (cam_r[i] @ points_camera[i].T).T + cam_t[i]
                points_z = points_y @ YUP_TO_ZUP.T
                pos_y.append(points_z[0])
                rot_y.append(_historical_wrist_link_rot(points_z))
                repaired_pos_y.append(
                    YUP_TO_ZUP
                    @ (
                        cam_r[i] @ repaired_position_camera[i]
                        + cam_t[i]
                    )
                )
                repaired_rot_y.append(
                    YUP_TO_ZUP @ cam_r[i] @ repaired_rotation_camera[i]
                )
            pos_z = np.asarray(pos_y)
            rot_z = np.asarray(rot_y)
            tracks[side] = (pos_z, rot_z)
            repaired_tracks[side] = (
                np.asarray(repaired_pos_y),
                np.asarray(repaired_rot_y),
            )

        # Derive the camera heading from the current raw source.  It is saved
        # separately for Kimodo and deliberately does not animate the static
        # carrier in the ScaleTrack UMI-2 reference.
        heading_z = np.asarray(
            [
                _yaw_only(YUP_TO_ZUP @ r @ YUP_TO_ZUP.T @ STERA_TO_G1_HEADING)
                for r in cam_r
            ]
        )

    clean_start, clean_end, selection = _clean_dual_wrist_interval(timestamps, tracks, valid)
    if repair_transient_outliers:
        tracks = repaired_tracks
    selection["transient_wrist_outlier_repair"] = {
        "enabled": bool(repair_transient_outliers),
        "coordinate_frame": "camera-local link frame",
        "camera_pose_modified": False,
        "per_hand": repair_diagnostics,
        "total_repaired_hand_frames": int(
            sum(
                len(values["repaired_source_frame_indices"])
                for values in repair_diagnostics.values()
            )
        ),
    }
    selection["bounded_missing_hand_frame_repair"] = {
        "enabled": True,
        "source_hdf5_modified": False,
        "per_hand": missing_frame_diagnostics,
        "total_repaired_hand_frames": int(
            sum(len(values["repaired_indices"]) for values in missing_frame_diagnostics.values())
        ),
    }
    if (source_start_index is None) != (source_end_index is None):
        raise ValueError(
            "--source_start_index and --source_end_index must be provided together"
        )
    if use_full_source and source_start_index is not None:
        raise ValueError("--use-full-source cannot be combined with an explicit source range")
    if use_full_source:
        start, end = 0, len(timestamps) - 1
        selection["selection_mode"] = "full_source_bypass_clean_interval_filter"
    elif source_start_index is None:
        start, end = clean_start, clean_end
        selection["selection_mode"] = "longest_clean_interval"
    else:
        start, end = int(source_start_index), int(source_end_index)
        if start < 0 or end >= len(timestamps) or start >= end:
            raise ValueError(
                f"Invalid explicit source range [{start}, {end}] for {len(timestamps)} frames"
            )
        _, _, recomputed = _clean_dual_wrist_interval(timestamps, tracks, valid)
        clean_edges = {
            (a, b)
            for a, b, _ in recomputed["clean_edge_runs"]
        }
        if not any(a <= start and end <= b for a, b in clean_edges):
            raise ValueError(
                f"Explicit source range [{start}, {end}] crosses an invalid or jump edge; "
                f"clean runs: {recomputed['clean_edge_runs']}"
            )
        selection["selection_mode"] = "explicit_clean_interval"
        selection["requested_source_start_index"] = start
        selection["requested_source_end_index"] = end
    source_t = timestamps[start : end + 1] - timestamps[start]
    if max_duration_s > 0:
        keep = source_t <= float(max_duration_s) + 1e-6
        if keep.sum() < 2:
            raise RuntimeError("Duration cap leaves fewer than two source frames")
        end = start + int(np.flatnonzero(keep)[-1])
        source_t = timestamps[start : end + 1] - timestamps[start]
        selection["duration_cap_s"] = float(max_duration_s)
        selection["selected_source_end_index_after_cap"] = int(end)
    else:
        selection["duration_cap_s"] = None
    selection["selected_source_start_index"] = int(start)
    selection["selected_source_end_index"] = int(end)
    selection["selected_source_frame_count"] = int(end - start + 1)
    selection["selected_source_duration_s"] = float(source_t[-1])
    return (
        source_t,
        (tracks["left"][0][start : end + 1], tracks["left"][1][start : end + 1]),
        (tracks["right"][0][start : end + 1], tracks["right"][1][start : end + 1]),
        heading_z[start : end + 1],
        float(source_t[-1]),
        selection,
    )


def _interp_rot(times: np.ndarray, mats: np.ndarray, query: np.ndarray) -> np.ndarray:
    rots = Rotation.from_matrix(mats)
    return Slerp(times, rots)(query).as_matrix()


def _continuous_quat(quat: np.ndarray) -> np.ndarray:
    """Choose quaternion signs continuously for stable motion-file interpolation."""
    out = np.asarray(quat, dtype=np.float32).copy()
    for i in range(1, len(out)):
        if float(np.dot(out[i - 1], out[i])) < 0.0:
            out[i] *= -1.0
    return out


def _load_schema():
    if SCHEMA_JSON.is_file():
        data = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))
        return [str(x) for x in data["body_names"]], [str(x) for x in data["joint_names"]]
    if TEMPLATE.is_file():
        d = np.load(TEMPLATE, allow_pickle=True)
        return [str(x) for x in d["body_names"].tolist()], [str(x) for x in d["joint_names"].tolist()]
    raise FileNotFoundError(
        f"Motion schema missing: {SCHEMA_JSON} (and fallback {TEMPLATE})"
    )


def _nominal_carrier(body_names: list[str], joint_names: list[str], root_xy: np.ndarray):
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    data.qpos[0:3] = [float(root_xy[0]), float(root_xy[1]), 0.79]
    data.qpos[3] = 1.0
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Joint not found in MuJoCo model: {name}")
        data.qpos[model.jnt_qposadr[jid]] = 0.0
    mujoco.mj_forward(model, data)
    body_pos = np.asarray([data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)] for n in body_names])
    body_quat = np.asarray([data.xquat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)] for n in body_names])
    joint_pos = np.zeros(len(joint_names), dtype=np.float32)
    return model, body_pos.astype(np.float32), body_quat.astype(np.float32), joint_pos


def _finite_diff(x: np.ndarray, fps: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    if len(x) > 1:
        out[1:-1] = (x[2:] - x[:-2]) * (0.5 * fps)
        out[0] = (x[1] - x[0]) * fps
        out[-1] = (x[-1] - x[-2]) * fps
    return out


def _angular_velocity_from_rotations(mats: np.ndarray, fps: float) -> np.ndarray:
    """Finite-difference world angular velocity for a rotation sequence."""
    rotations = Rotation.from_matrix(np.asarray(mats, dtype=np.float64))
    out = np.zeros((len(rotations), 3), dtype=np.float32)
    if len(rotations) < 2:
        return out
    out[0] = ((rotations[1] * rotations[0].inv()).as_rotvec() * fps).astype(np.float32)
    out[-1] = ((rotations[-1] * rotations[-2].inv()).as_rotvec() * fps).astype(np.float32)
    for i in range(1, len(rotations) - 1):
        out[i] = (
            (rotations[i + 1] * rotations[i - 1].inv()).as_rotvec() * (0.5 * fps)
        ).astype(np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=Path, default=ROOT / "ego_anywhere/output_clean")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Optional maximum duration after automatic clean-interval selection; 0 uses the full interval.",
    )
    ap.add_argument("--out-fps", type=float, default=50.0)
    ap.add_argument(
        "--source-start-index",
        type=int,
        default=None,
        help="Optional inclusive source-frame start for a verified clean interval.",
    )
    ap.add_argument(
        "--source-end-index",
        type=int,
        default=None,
        help="Optional inclusive source-frame end for a verified clean interval.",
    )
    ap.add_argument(
        "--use-full-source",
        action="store_true",
        help="Use every source frame in order; retain jump diagnostics but do not split/filter the track.",
    )
    ap.add_argument(
        "--repair-transient-wrist-outliers",
        action="store_true",
        help=(
            "Replace bounded camera-local wrist pose excursions with "
            "position interpolation and quaternion Slerp; preserve timestamps "
            "and camera poses."
        ),
    )
    ap.add_argument(
        "--max-gap-frames",
        type=int,
        default=DEFAULT_MAX_GAP_FRAMES,
        help=(
            "Maximum internal missing-hand gap, in source frames, that may be "
            "filled in the camera-local joint frame. Wrist translation uses "
            "cubic Hermite with endpoint velocities, wrist orientation uses "
            "quaternion slerp, and finger joints keep a wrist-local shape "
            "lerp. Endpoint gaps are never extrapolated. Default matches the "
            f"historical limit of {DEFAULT_MAX_GAP_FRAMES}."
        ),
    )
    args = ap.parse_args()
    if int(args.max_gap_frames) < 1:
        raise ValueError("--max-gap-frames must be >= 1")

    h5_path = args.episode / "annotation.hdf5"
    (
        source_t,
        (left_p, left_r),
        (right_p, right_r),
        heading_r,
        used_duration,
        selection,
    ) = _raw_wrist_tracks(
        h5_path,
        float(args.duration),
        source_start_index=args.source_start_index,
        source_end_index=args.source_end_index,
        use_full_source=bool(args.use_full_source),
        repair_transient_outliers=bool(args.repair_transient_wrist_outliers),
        max_gap_frames=int(args.max_gap_frames),
    )
    body_names, joint_names = _load_schema()
    query_t = np.arange(int(round(used_duration * args.out_fps)) + 1, dtype=np.float64) / args.out_fps
    query_t = query_t[query_t <= used_duration + 1e-7]
    if len(query_t) < 2:
        raise RuntimeError("Prepared reference has fewer than two frames")

    left_pos = np.stack([np.interp(query_t, source_t, left_p[:, k]) for k in range(3)], axis=1)
    right_pos = np.stack([np.interp(query_t, source_t, right_p[:, k]) for k in range(3)], axis=1)
    raw_left_rot = _interp_rot(source_t, left_r, query_t)
    raw_right_rot = _interp_rot(source_t, right_r, query_t)
    heading_rot = _interp_rot(source_t, heading_r, query_t)

    # The carrier is intentionally static.  HDF5 has no full-body reference,
    # and rotating/translating this filler to match a legacy motion creates an
    # internally inconsistent target (the failure mode this adapter is meant
    # to avoid).
    root_xy = np.zeros(2, dtype=np.float64)
    _, nominal_pos, nominal_quat, joint_pos0 = _nominal_carrier(body_names, joint_names, root_xy)
    n = len(query_t)
    body_pos = np.repeat(nominal_pos[None], n, axis=0).astype(np.float32)
    body_quat = np.repeat(nominal_quat[None], n, axis=0).astype(np.float32)
    li = body_names.index("left_wrist_yaw_link")
    ri = body_names.index("right_wrist_yaw_link")
    body_pos[:, li] = left_pos.astype(np.float32)
    body_pos[:, ri] = right_pos.astype(np.float32)
    body_quat[:, li] = _continuous_quat(
        Rotation.from_matrix(raw_left_rot).as_quat(scalar_first=True)
    )
    body_quat[:, ri] = _continuous_quat(
        Rotation.from_matrix(raw_right_rot).as_quat(scalar_first=True)
    )

    body_lin = _finite_diff(body_pos, args.out_fps)
    body_ang = np.zeros_like(body_pos)
    body_ang[:, li] = _angular_velocity_from_rotations(raw_left_rot, args.out_fps)
    body_ang[:, ri] = _angular_velocity_from_rotations(raw_right_rot, args.out_fps)
    joint_pos = np.repeat(joint_pos0[None], n, axis=0)
    joint_vel = np.zeros_like(joint_pos)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    motion_path = args.out_dir / "raw_dual_wrist_reference.npz"
    target_path = args.out_dir / "raw_dual_wrist_targets.npz"
    np.savez_compressed(
        motion_path,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin,
        body_ang_vel_w=body_ang,
        body_names=np.asarray(body_names),
        joint_names=np.asarray(joint_names),
        fps=np.asarray(args.out_fps, dtype=np.float32),
        source=np.asarray(str(h5_path.resolve())),
        source_coordinate_system=np.asarray("output_clean Y-up -> ScaleTrack Z-up"),
        raw_left_wrist_rot_zup=Rotation.from_matrix(raw_left_rot)
        .as_quat(scalar_first=True)
        .astype(np.float32),
        raw_right_wrist_rot_zup=Rotation.from_matrix(raw_right_rot)
        .as_quat(scalar_first=True)
        .astype(np.float32),
        kimodo_pelvis_heading_quat_wxyz=_continuous_quat(
            Rotation.from_matrix(heading_rot).as_quat(scalar_first=True)
        ),
        rotation_source=np.asarray(
            "raw_keypoints_historical_palm_frame_local_Rx_minus_90"
        ),
    )
    np.savez_compressed(
        target_path,
        left_wrist=left_pos.astype(np.float32),
        right_wrist=right_pos.astype(np.float32),
        left_wrist_quat_wxyz=_continuous_quat(
            Rotation.from_matrix(raw_left_rot).as_quat(scalar_first=True)
        ),
        right_wrist_quat_wxyz=_continuous_quat(
            Rotation.from_matrix(raw_right_rot).as_quat(scalar_first=True)
        ),
        kimodo_pelvis_heading_quat_wxyz=_continuous_quat(
            Rotation.from_matrix(heading_rot).as_quat(scalar_first=True)
        ),
        fps_session=np.asarray(args.out_fps, dtype=np.float32),
        timestamps=query_t,
        source=np.asarray(str(h5_path.resolve())),
    )
    selection_mode = str(selection.get("selection_mode", ""))
    selection_policy = (
        (
            "full source sequence; bounded camera-local wrist outliers "
            "interpolated; timestamps and camera poses preserved"
        )
        if (
            selection_mode == "full_source_bypass_clean_interval_filter"
            and args.repair_transient_wrist_outliers
        )
        else "full source sequence; jump diagnostics retained and no interval filtering"
        if selection_mode == "full_source_bypass_clean_interval_filter"
        else "longest dual-valid interval after robust position/orientation jump filtering"
    )
    meta = {
        "source_hdf5": str(h5_path.resolve()),
        "source_frames": int(len(source_t)),
        "source_duration_s": float(used_duration),
        "output_frames": int(n),
        "output_fps": float(args.out_fps),
        "output_duration_s": float(query_t[-1]),
        "selection_policy": selection_policy,
        "selection": selection,
        "unobserved_body_policy": "fixed nominal MuJoCo G1 carrier; not a tracking target",
        "rotation": {
            "source": "raw_hdf5_keypoints",
            "convention": "historical_palm_frame_then_local_Rx_minus_90_deg",
            "legacy_motion_imported": False,
        },
        "kimodo_heading": {
            "source": "raw_hdf5_cam_pose",
            "convention": "Y-up camera rotation -> Z-up -> Stera-to-G1 yaw",
        },
        "first_pelvis_quat_wxyz": body_quat[0, body_names.index("pelvis")].tolist(),
        "first_pelvis_yaw_deg": float(
            Rotation.from_matrix(heading_rot[0]).as_euler("xyz", degrees=True)[2]
        ),
        "first_left_wrist_quat_wxyz": body_quat[0, li].tolist(),
        "first_right_wrist_quat_wxyz": body_quat[0, ri].tolist(),
        "root_position_zup": [float(root_xy[0]), float(root_xy[1]), 0.79],
        "first_left_wrist_zup": left_pos[0].tolist(),
        "first_right_wrist_zup": right_pos[0].tolist(),
        "first_wrist_distance_m": float(np.linalg.norm(left_pos[0] - right_pos[0])),
        "motion_npz": str(motion_path.resolve()),
        "targets_npz": str(target_path.resolve()),
    }
    (args.out_dir / "prepare_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
