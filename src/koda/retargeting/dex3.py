#!/usr/bin/env python3
"""Retarget EgoDex hand keypoints to Unitree Dex3 joint positions and velocities.

The legacy input is an HDF5 file containing 25 EgoDex/AVP keypoint transforms
under ``transforms/*``.  A captured EgoAnywhere session can also be supplied;
its 21-joint hand tracks are transformed from camera optical coordinates into
world coordinates before retargeting.  Retargeting always runs on the native
hand timestamps.  If a body kinematics NPZ is supplied, the resulting joint
positions are then linearly resampled onto that file's timeline.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import site
import sys
import types
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from koda.paths import repo_root


WORKSPACE_ROOT = repo_root()
XR_ROOT = Path(
    os.environ.get("XR_TELEOPERATE_ROOT", WORKSPACE_ROOT / "resources/xr_teleoperate")
).expanduser().resolve()
ASSETS = XR_ROOT / "assets"
_SCRIPT_CONFIGS = Path(__file__).resolve().parent / "configs"
DEX3_YML = {
    "dexpilot": ASSETS / "unitree_hand" / "unitree_dex3.yml",
    "vector": ASSETS / "unitree_hand" / "unitree_dex3_vector.yml",
    # The contact pipeline explicitly selects position retargeting.
    "position": _SCRIPT_CONFIGS / "unitree_dex3_position.yml",
}

_FINGER_CHAINS = {
    "Thumb": [
        "ThumbKnuckle",
        "ThumbIntermediateBase",
        "ThumbIntermediateTip",
        "ThumbTip",
    ],
    "Index": [
        "IndexFingerMetacarpal",
        "IndexFingerKnuckle",
        "IndexFingerIntermediateBase",
        "IndexFingerIntermediateTip",
        "IndexFingerTip",
    ],
    "Middle": [
        "MiddleFingerMetacarpal",
        "MiddleFingerKnuckle",
        "MiddleFingerIntermediateBase",
        "MiddleFingerIntermediateTip",
        "MiddleFingerTip",
    ],
    "Ring": [
        "RingFingerMetacarpal",
        "RingFingerKnuckle",
        "RingFingerIntermediateBase",
        "RingFingerIntermediateTip",
        "RingFingerTip",
    ],
    "Little": [
        "LittleFingerMetacarpal",
        "LittleFingerKnuckle",
        "LittleFingerIntermediateBase",
        "LittleFingerIntermediateTip",
        "LittleFingerTip",
    ],
}

LEFT_JOINT_NAMES = [
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
]
RIGHT_JOINT_NAMES = [
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
]
HAND_JOINT_NAMES = LEFT_JOINT_NAMES + RIGHT_JOINT_NAMES

# Joint limits from g1_with_visual_dex3_hands.xml.  These values and the API
# ordering intentionally match the historical bytecode implementation.
DEX3_JNT_RANGE = {
    "left_hand_thumb_0_joint": (-1.04719755, 1.04719755),
    "left_hand_thumb_1_joint": (-0.72431163, 1.04719755),
    "left_hand_thumb_2_joint": (0.0, 1.74532925),
    "left_hand_middle_0_joint": (-1.57079632, 0.0),
    "left_hand_middle_1_joint": (-1.74532925, 0.0),
    "left_hand_index_0_joint": (-1.57079632, 0.0),
    "left_hand_index_1_joint": (-1.74532925, 0.0),
    "right_hand_thumb_0_joint": (-1.04719755, 1.04719755),
    "right_hand_thumb_1_joint": (-1.04719755, 0.72431163),
    "right_hand_thumb_2_joint": (-1.74532925, 0.0),
    "right_hand_middle_0_joint": (0.0, 1.57079632),
    "right_hand_middle_1_joint": (0.0, 1.74532925),
    "right_hand_index_0_joint": (0.0, 1.57079632),
    "right_hand_index_1_joint": (0.0, 1.74532925),
}


def _side_key(side: str, suffix: str) -> str:
    return f"{side}{suffix}"


def build_25_keypoints(f: h5py.File, side: str) -> np.ndarray:
    """Return legacy EgoDex/AVP keypoint world positions with shape (T, 25, 3)."""
    hand_key = f"transforms/{_side_key(side, 'Hand')}"
    if hand_key not in f:
        raise KeyError(f"Missing {hand_key} in hdf5")
    hand = np.asarray(f[hand_key][..., :3, 3], dtype=np.float64)
    pts = np.zeros((hand.shape[0], 25, 3), dtype=np.float64)
    pts[:, 0] = hand
    idx = 1
    for chain in ("Thumb", "Index", "Middle", "Ring", "Little"):
        for name in _FINGER_CHAINS[chain]:
            key = f"transforms/{_side_key(side, name)}"
            if key not in f:
                raise KeyError(f"Missing {key} in hdf5")
            value = np.asarray(f[key][..., :3, 3], dtype=np.float64)
            if value.shape != hand.shape:
                raise ValueError(
                    f"{key} has shape {value.shape}; expected {hand.shape}"
                )
            pts[:, idx] = value
            idx += 1
    if idx != 25:
        raise RuntimeError(f"Expected 25 keypoints, built {idx}")
    return pts


def _mano21_to_avp25(joints: np.ndarray) -> np.ndarray:
    """Map wrist + five 4-joint MANO chains to the AVP 25-point layout.

    Dex3's shipped DexPilot configuration consumes indices 0, 4, 9, and 14.
    Those are exact wrist/thumb-tip/index-tip/middle-tip copies.  Each
    non-thumb chain needs one extra AVP slot, so its MCP is duplicated in the
    otherwise-unused metacarpal slot.
    """
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1:] != (21, 3):
        raise ValueError(f"Expected (T, 21, 3) hand joints, got {joints.shape}")
    points = np.empty((joints.shape[0], 25, 3), dtype=np.float64)
    points[:, 0] = joints[:, 0]
    points[:, 1:5] = joints[:, 1:5]
    for dst, src in zip((5, 10, 15, 20), (5, 9, 13, 17)):
        points[:, dst] = joints[:, src]
        points[:, dst + 1 : dst + 5] = joints[:, src : src + 4]
    return points


def _ensure_logging_mp_stub() -> None:
    """Provide the small logging_mp surface imported by xr_teleoperate."""
    try:
        import logging_mp  # noqa: F401
    except ModuleNotFoundError:
        stub = types.ModuleType("logging_mp")
        stub.getLogger = logging.getLogger  # type: ignore[attr-defined]
        sys.modules["logging_mp"] = stub


def _prefer_environment_pinocchio() -> None:
    """Exclude the unrelated user-site ``pinocchio==0.1`` package."""

    import pinocchio

    if hasattr(pinocchio, "buildModelFromUrdf"):
        return

    user_site = Path(site.getusersitepackages()).resolve()
    filtered: list[str] = []
    for entry in sys.path:
        try:
            resolved = Path(entry).resolve()
        except (OSError, RuntimeError):
            filtered.append(entry)
            continue
        if resolved == user_site or user_site in resolved.parents:
            continue
        filtered.append(entry)
    sys.path[:] = filtered
    for module_name in [
        name
        for name in sys.modules
        if name == "pinocchio" or name.startswith("pinocchio.")
    ]:
        del sys.modules[module_name]
    importlib.invalidate_caches()
    import pinocchio as pinocchio_reload

    if not hasattr(pinocchio_reload, "buildModelFromUrdf"):
        raise ImportError(
            "robotics Pinocchio is unavailable; run this script in the "
            "dex_retarget environment (the unrelated user package pinocchio==0.1 "
            "is intentionally ignored)"
        )


def _init_retargeting(
    config_yml: Path,
    *,
    scaling_factor: float | None = None,
    project_dist: float | None = None,
    escape_dist: float | None = None,
    low_pass_alpha: float | None = None,
) -> dict[str, Any]:
    """Build stateful left/right retargeters from a Dex3 YAML file."""
    _ensure_logging_mp_stub()
    _prefer_environment_pinocchio()
    robot_control = XR_ROOT / "teleop" / "robot_control"
    dex_src = robot_control / "dex-retargeting" / "src"
    for import_root in (robot_control, dex_src):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))

    from dex_retargeting import RetargetingConfig

    RetargetingConfig.set_default_urdf_dir(str(ASSETS))
    with config_yml.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict) or "left" not in cfg or "right" not in cfg:
        raise ValueError(f"Invalid Dex3 retarget config: {config_yml}")
    for side in ("left", "right"):
        if scaling_factor is not None:
            cfg[side]["scaling_factor"] = float(scaling_factor)
        if project_dist is not None:
            cfg[side]["project_dist"] = float(project_dist)
        if escape_dist is not None:
            cfg[side]["escape_dist"] = float(escape_dist)
        if low_pass_alpha is not None:
            cfg[side]["low_pass_alpha"] = float(low_pass_alpha)

    left = RetargetingConfig.from_dict(cfg["left"]).build()
    right = RetargetingConfig.from_dict(cfg["right"]).build()
    left_map = [left.joint_names.index(name) for name in LEFT_JOINT_NAMES]
    right_map = [right.joint_names.index(name) for name in RIGHT_JOINT_NAMES]
    return {
        "left": left,
        "right": right,
        "left_indices": left.optimizer.target_link_human_indices,
        "right_indices": right.optimizer.target_link_human_indices,
        "left_map": left_map,
        "right_map": right_map,
        "cfg_left": cfg["left"],
        "cfg_right": cfg["right"],
    }


def _position_targets(
    points25: np.ndarray, indices: np.ndarray, scaling_factor: float
) -> np.ndarray:
    """Absolute link targets for PositionOptimizer.

    Scale tip offsets about the wrist (index 0 in the AVP-25 layout) so a
    robot/human size ratio can be applied without moving the wrist origin.
    """
    points = np.asarray(points25, dtype=np.float64).reshape(25, 3)
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    selected = points[idx].copy()
    scale = float(scaling_factor)
    if abs(scale - 1.0) > 1e-12:
        wrist = points[0]
        selected = wrist[None, :] + (selected - wrist[None, :]) * scale
    return selected.astype(np.float64)


def soft_open_and_clip(
    qpos: np.ndarray,
    joint_names: list[str],
    *,
    flexion_scale: float = 1.0,
    limit_margin: float = 0.01,
) -> np.ndarray:
    """Scale joint curl about zero and clip it inside the Dex3 limits."""
    q = np.asarray(qpos, dtype=np.float64).copy()
    if q.ndim != 2 or q.shape[1] != len(joint_names):
        raise ValueError(f"qpos shape {q.shape} vs {len(joint_names)} joints")
    scale = float(flexion_scale)
    if not 0.0 < scale <= 2.5:
        raise ValueError(f"flexion_scale must be in (0, 2.5], got {scale}")
    if abs(scale - 1.0) > 1e-12:
        q *= scale
    margin = max(0.0, float(limit_margin))
    for joint_index, name in enumerate(joint_names):
        lo, hi = DEX3_JNT_RANGE[name]
        applied_margin = min(margin, max(0.0, 0.5 * (hi - lo) - 1e-4))
        q[:, joint_index] = np.clip(
            q[:, joint_index], lo + applied_margin, hi - applied_margin
        )
    return q.astype(np.float32)


def _human_closedness_01(points25: np.ndarray) -> np.ndarray:
    """Estimate per-frame closedness from thumb/index/middle tip distances."""
    tips = points25[:, [4, 9, 14], :]
    d01 = np.linalg.norm(tips[:, 0] - tips[:, 1], axis=1)
    d02 = np.linalg.norm(tips[:, 0] - tips[:, 2], axis=1)
    d12 = np.linalg.norm(tips[:, 1] - tips[:, 2], axis=1)
    gap = (d01 + d02 + d12) / 3.0
    lo, hi = float(np.percentile(gap, 5)), float(np.percentile(gap, 95))
    if hi <= lo + 1e-8:
        return np.zeros(points25.shape[0], dtype=np.float64)
    open_fraction = np.clip((gap - lo) / (hi - lo), 0.0, 1.0)
    return 1.0 - open_fraction


def apply_human_close_gate(
    qpos: np.ndarray,
    left_points: np.ndarray,
    right_points: np.ndarray,
    *,
    gamma: float = 1.0,
) -> np.ndarray:
    """Modulate curl by the observed human hand closedness."""
    q = np.asarray(qpos, dtype=np.float64).copy()
    exponent = max(1e-6, float(gamma))
    q[:, :7] *= _human_closedness_01(left_points)[:, None] ** exponent
    q[:, 7:] *= _human_closedness_01(right_points)[:, None] ** exponent
    return q.astype(np.float32)


def _interpolate_invalid_frames(
    values: np.ndarray, valid: np.ndarray, side: str
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    finite = np.all(np.isfinite(values), axis=tuple(range(1, values.ndim)))
    valid = valid & finite
    if not np.any(valid):
        raise ValueError(f"No valid {side} hand frames")
    if np.all(valid):
        return values
    source_index = np.flatnonzero(valid)
    frame_index = np.arange(values.shape[0], dtype=np.float64)
    flat = values.reshape(values.shape[0], -1)
    filled = np.empty_like(flat)
    for column in range(flat.shape[1]):
        filled[:, column] = np.interp(
            frame_index, source_index, flat[source_index, column]
        )
    return filled.reshape(values.shape)


def _normalize_timestamps(
    timestamps: np.ndarray, expected_frames: int, label: str
) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if timestamps.shape != (expected_frames,):
        raise ValueError(
            f"{label} timestamps have shape {timestamps.shape}; expected ({expected_frames},)"
        )
    if not np.all(np.isfinite(timestamps)):
        raise ValueError(f"{label} timestamps contain non-finite values")
    timestamps = timestamps - timestamps[0]
    if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"{label} timestamps must be strictly increasing")
    return timestamps


def _interpolate_rotation_samples(
    rotations: np.ndarray, source_time: np.ndarray, target_time: np.ndarray
) -> np.ndarray:
    """Interpolate rotation matrices and project each result back onto SO(3)."""
    rotations = np.asarray(rotations, dtype=np.float64)
    if np.array_equal(source_time, target_time):
        return rotations
    flat = rotations.reshape(rotations.shape[0], 9)
    interpolated = np.column_stack(
        [np.interp(target_time, source_time, flat[:, index]) for index in range(9)]
    ).reshape(-1, 3, 3)
    for index, matrix in enumerate(interpolated):
        u, _, vt = np.linalg.svd(matrix)
        result = u @ vt
        if np.linalg.det(result) < 0.0:
            u[:, -1] *= -1.0
            result = u @ vt
        interpolated[index] = result
    return interpolated


def _load_session_keypoints(
    annotation_path: Path, session_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    with h5py.File(annotation_path, "r") as h5:
        required = (
            "hand-pose/left_joints",
            "hand-pose/right_joints",
            "hand-pose/timestamps",
            "cam-pose/rotations",
            "cam-pose/translations",
            "cam-pose/timestamps",
        )
        missing = [key for key in required if key not in h5]
        if missing:
            raise KeyError(f"Missing session datasets in {annotation_path}: {missing}")
        left_joints = np.asarray(h5["hand-pose/left_joints"], dtype=np.float64)
        right_joints = np.asarray(h5["hand-pose/right_joints"], dtype=np.float64)
        hand_time_absolute = np.asarray(h5["hand-pose/timestamps"], dtype=np.float64)
        camera_time_absolute = np.asarray(h5["cam-pose/timestamps"], dtype=np.float64)
        camera_rotations = np.asarray(h5["cam-pose/rotations"], dtype=np.float64)
        camera_translations = np.asarray(h5["cam-pose/translations"], dtype=np.float64)
        left_valid = (
            np.asarray(h5["hand-pose/left_valid"], dtype=bool)
            if "hand-pose/left_valid" in h5
            else np.ones(left_joints.shape[0], dtype=bool)
        )
        right_valid = (
            np.asarray(h5["hand-pose/right_valid"], dtype=bool)
            if "hand-pose/right_valid" in h5
            else np.ones(right_joints.shape[0], dtype=bool)
        )

    if left_joints.shape != right_joints.shape or left_joints.shape[1:] != (21, 3):
        raise ValueError(
            f"Session hand shape mismatch: left={left_joints.shape}, right={right_joints.shape}"
        )
    frame_count = left_joints.shape[0]
    if left_valid.shape != (frame_count,) or right_valid.shape != (frame_count,):
        raise ValueError("Session hand validity arrays do not match hand frame count")
    if camera_rotations.shape != (camera_time_absolute.size, 3, 3):
        raise ValueError(f"Unexpected camera rotation shape {camera_rotations.shape}")
    if camera_translations.shape != (camera_time_absolute.size, 3):
        raise ValueError(
            f"Unexpected camera translation shape {camera_translations.shape}"
        )

    hand_time = _normalize_timestamps(hand_time_absolute, frame_count, "hand")
    camera_time_absolute = np.asarray(camera_time_absolute, dtype=np.float64).reshape(
        -1
    )
    if not np.all(np.isfinite(camera_time_absolute)):
        raise ValueError("camera timestamps contain non-finite values")
    if camera_time_absolute.size > 1 and np.any(np.diff(camera_time_absolute) <= 0.0):
        raise ValueError("camera timestamps must be strictly increasing")
    # Use the hand clock origin for both streams so a non-zero capture offset
    # is preserved instead of accidentally aligning both first samples.
    camera_time = camera_time_absolute - hand_time_absolute[0]
    left_joints = _interpolate_invalid_frames(left_joints, left_valid, "left")
    right_joints = _interpolate_invalid_frames(right_joints, right_valid, "right")

    calibration_path = session_dir / "calibrations" / "R_optical_to_link.npy"
    if not calibration_path.is_file():
        raise FileNotFoundError(
            f"Session optical-to-link calibration is required: {calibration_path}"
        )
    optical_to_link = np.asarray(np.load(calibration_path), dtype=np.float64)
    if optical_to_link.shape != (3, 3):
        raise ValueError(
            f"Expected a 3x3 R_optical_to_link, got {optical_to_link.shape}"
        )
    if not np.allclose(optical_to_link.T @ optical_to_link, np.eye(3), atol=1e-5):
        raise ValueError(f"R_optical_to_link is not orthonormal: {calibration_path}")

    camera_rotations = _interpolate_rotation_samples(
        camera_rotations, camera_time, hand_time
    )
    camera_translations = np.column_stack(
        [
            np.interp(hand_time, camera_time, camera_translations[:, axis])
            for axis in range(3)
        ]
    )
    world_from_optical = camera_rotations @ optical_to_link

    def to_world(joints: np.ndarray) -> np.ndarray:
        return (
            np.einsum("tij,tkj->tki", world_from_optical, joints)
            + camera_translations[:, None]
        )

    left_points = _mano21_to_avp25(to_world(left_joints))
    right_points = _mano21_to_avp25(to_world(right_joints))
    metadata = {
        "source_schema": "ego_anywhere_hand_pose_21_v1",
        "source_session_dir": str(session_dir),
        "source_hdf5": str(annotation_path),
        "source_coordinate_frame": "camera_optical_opencv",
        "retarget_coordinate_frame": "world",
        "world_transform": "p_world = R_link_to_world @ R_optical_to_link @ p_optical + t_world",
        "r_optical_to_link": optical_to_link.tolist(),
        "left_valid_frames": int(np.count_nonzero(left_valid)),
        "right_valid_frames": int(np.count_nonzero(right_valid)),
        "native_frames": int(frame_count),
    }
    return left_points, right_points, hand_time, hand_time_absolute, metadata


def _load_legacy_keypoints(
    hdf5_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    with h5py.File(hdf5_path, "r") as h5:
        left_points = build_25_keypoints(h5, "left")
        right_points = build_25_keypoints(h5, "right")
        if left_points.shape[0] != right_points.shape[0]:
            raise ValueError("Legacy left/right hand tracks have different lengths")
        frame_count = left_points.shape[0]
        timestamp_key = next(
            (
                key
                for key in ("hand-pose/timestamps", "timestamps", "time")
                if key in h5
            ),
            None,
        )
        if timestamp_key is None:
            timestamps_absolute = np.arange(frame_count, dtype=np.float64) / 30.0
            timestamp_source = "inferred_30_hz"
        else:
            timestamps_absolute = np.asarray(h5[timestamp_key], dtype=np.float64)
            timestamp_source = f"hdf5:{timestamp_key}"
    timestamps = _normalize_timestamps(timestamps_absolute, frame_count, "legacy hand")
    metadata = {
        "source_schema": "legacy_egodex_avp_transforms_25_v1",
        "source_session_dir": "",
        "source_hdf5": str(hdf5_path),
        "source_coordinate_frame": "world",
        "retarget_coordinate_frame": "world",
        "world_transform": "already_world",
        "left_valid_frames": int(frame_count),
        "right_valid_frames": int(frame_count),
        "native_frames": int(frame_count),
        "native_timestamp_source": timestamp_source,
    }
    return left_points, right_points, timestamps, timestamps_absolute, metadata


def _retarget_points(
    left_points: np.ndarray,
    right_points: np.ndarray,
    *,
    retarget_type: str,
    config_yml: Path | None,
    scaling_factor: float | None,
    project_dist: float | None,
    escape_dist: float | None,
    low_pass_alpha: float | None,
    flexion_scale: float,
    limit_margin: float,
    human_close_gamma: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if retarget_type not in DEX3_YML:
        raise ValueError(f"Unsupported retarget type: {retarget_type}")
    yml = Path(config_yml) if config_yml is not None else DEX3_YML[retarget_type]
    if not yml.is_file():
        raise FileNotFoundError(yml)
    retarget = _init_retargeting(
        yml,
        scaling_factor=scaling_factor,
        project_dist=project_dist,
        escape_dist=escape_dist,
        low_pass_alpha=low_pass_alpha,
    )
    if left_points.shape != right_points.shape or left_points.shape[1:] != (25, 3):
        raise ValueError(
            f"Retarget point shape mismatch: left={left_points.shape}, right={right_points.shape}"
        )

    frame_count = left_points.shape[0]
    left_qpos = np.zeros((frame_count, 7), dtype=np.float32)
    right_qpos = np.zeros((frame_count, 7), dtype=np.float32)
    # PositionOptimizer ignores YAML scaling_factor; scale wrist-centered offsets.
    position_scale = float(
        scaling_factor
        if scaling_factor is not None
        else retarget["cfg_left"].get("scaling_factor", 1.0)
    )
    for frame_index in range(frame_count):
        left_indices = retarget["left_indices"]
        right_indices = retarget["right_indices"]
        if retarget_type == "position":
            left_reference = _position_targets(
                left_points[frame_index], left_indices, position_scale
            )
            right_reference = _position_targets(
                right_points[frame_index], right_indices, position_scale
            )
        else:
            left_reference = (
                left_points[frame_index, left_indices[1, :]]
                - left_points[frame_index, left_indices[0, :]]
            )
            right_reference = (
                right_points[frame_index, right_indices[1, :]]
                - right_points[frame_index, right_indices[0, :]]
            )
        left_qpos[frame_index] = np.asarray(
            retarget["left"].retarget(left_reference)[retarget["left_map"]],
            dtype=np.float32,
        )
        right_qpos[frame_index] = np.asarray(
            retarget["right"].retarget(right_reference)[retarget["right_map"]],
            dtype=np.float32,
        )

    qpos = np.concatenate((left_qpos, right_qpos), axis=1)
    if human_close_gamma is not None:
        qpos = apply_human_close_gate(
            qpos, left_points, right_points, gamma=float(human_close_gamma)
        )
    qpos = soft_open_and_clip(
        qpos,
        HAND_JOINT_NAMES,
        flexion_scale=flexion_scale,
        limit_margin=limit_margin,
    )
    parameters = {
        "method": f"dex_retargeting_{retarget_type}",
        "retarget_type": retarget_type,
        "config_yml": str(yml.resolve()),
        "flexion_scale": float(flexion_scale),
        "limit_margin": float(limit_margin),
        "scaling_factor": float(
            scaling_factor
            if scaling_factor is not None
            else retarget["cfg_left"].get("scaling_factor", 1.0)
        ),
        "low_pass_alpha": float(
            low_pass_alpha
            if low_pass_alpha is not None
            else retarget["cfg_left"].get("low_pass_alpha", 0.2)
        ),
        "project_dist": (
            float(project_dist)
            if project_dist is not None
            else retarget["cfg_left"].get("project_dist")
        ),
        "escape_dist": (
            float(escape_dist)
            if escape_dist is not None
            else retarget["cfg_left"].get("escape_dist")
        ),
        "human_close_gamma": (
            float(human_close_gamma) if human_close_gamma is not None else None
        ),
    }
    return qpos, parameters


def _infer_timeline_from_npz(path: Path) -> tuple[np.ndarray, float, str, int]:
    with np.load(path, allow_pickle=False) as data:
        if "timestamps" in data.files:
            raw_timestamps = np.asarray(data["timestamps"], dtype=np.float64)
            frame_count = int(raw_timestamps.size)
            timestamps = _normalize_timestamps(raw_timestamps, frame_count, "target")
            for frame_key in ("joint_pos", "body_pos_w", "root_pos", "qpos"):
                if frame_key in data.files and np.asarray(data[frame_key]).ndim >= 1:
                    field_frames = int(np.asarray(data[frame_key]).shape[0])
                    if field_frames != frame_count:
                        raise ValueError(
                            f"Target {frame_key} has {field_frames} frames but timestamps "
                            f"has {frame_count}: {path}"
                        )
            if frame_count > 1:
                fps = float(1.0 / np.median(np.diff(timestamps)))
            else:
                fps = (
                    float(np.asarray(data["fps"]).item())
                    if "fps" in data.files
                    else 0.0
                )
            source = "target_npz:timestamps"
        else:
            frame_key = next(
                (
                    key
                    for key in ("joint_pos", "body_pos_w", "root_pos", "qpos")
                    if key in data.files and np.asarray(data[key]).ndim >= 1
                ),
                None,
            )
            if frame_key is None:
                raise KeyError(
                    f"Cannot infer frame count from {path}; expected timestamps, "
                    "joint_pos, body_pos_w, root_pos, or qpos"
                )
            frame_count = int(np.asarray(data[frame_key]).shape[0])
            if "fps" not in data.files:
                raise KeyError(f"{path} has no timestamps or fps")
            fps = float(np.asarray(data["fps"]).item())
            if not np.isfinite(fps) or fps <= 0.0:
                raise ValueError(f"Invalid target fps {fps} in {path}")
            timestamps = np.arange(frame_count, dtype=np.float64) / fps
            source = "target_npz:fps"
    return timestamps, fps, source, frame_count


def _resample_qpos(
    qpos: np.ndarray, source_timestamps: np.ndarray, target_timestamps: np.ndarray
) -> np.ndarray:
    if qpos.shape[0] != source_timestamps.size:
        raise ValueError("qpos and source timeline lengths differ")
    if np.array_equal(source_timestamps, target_timestamps):
        return qpos.copy()
    return np.column_stack(
        [
            np.interp(target_timestamps, source_timestamps, qpos[:, joint_index])
            for joint_index in range(qpos.shape[1])
        ]
    ).astype(np.float32)


def _differentiate(qpos: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    if qpos.shape[0] != timestamps.size:
        raise ValueError("qpos and velocity timeline lengths differ")
    if timestamps.size == 1:
        return np.zeros_like(qpos, dtype=np.float32)
    # np.gradient uses centered differences internally and one-sided endpoint
    # differences with edge_order=1, including for non-uniform timestamps.
    return np.gradient(
        np.asarray(qpos, dtype=np.float64), timestamps, axis=0, edge_order=1
    ).astype(np.float32)


def retarget_episode(
    hdf5_path: Path,
    *,
    session_dir: Path | None = None,
    target_kinematics_npz: Path | None = None,
    retarget_type: str = "dexpilot",
    config_yml: Path | None = None,
    scaling_factor: float | None = None,
    project_dist: float | None = None,
    escape_dist: float | None = None,
    low_pass_alpha: float | None = None,
    flexion_scale: float = 0.7,
    limit_margin: float = 0.01,
    human_close_gamma: float | None = None,
) -> dict[str, np.ndarray]:
    """Retarget one legacy HDF5 or EgoAnywhere session to Dex3 kinematics."""
    hdf5_path = Path(hdf5_path).expanduser().resolve()
    if not hdf5_path.is_file():
        raise FileNotFoundError(hdf5_path)
    with h5py.File(hdf5_path, "r") as h5:
        is_session_schema = "hand-pose/left_joints" in h5

    if is_session_schema:
        resolved_session = (
            Path(session_dir).expanduser().resolve()
            if session_dir is not None
            else hdf5_path.parent
        )
        (
            left_points,
            right_points,
            native_time,
            native_time_absolute,
            source_metadata,
        ) = _load_session_keypoints(hdf5_path, resolved_session)
    else:
        if session_dir is not None:
            raise ValueError(
                f"--session-dir points to a legacy HDF5 without hand-pose datasets: {hdf5_path}"
            )
        resolved_session = None
        (
            left_points,
            right_points,
            native_time,
            native_time_absolute,
            source_metadata,
        ) = _load_legacy_keypoints(hdf5_path)

    native_qpos, retarget_metadata = _retarget_points(
        left_points,
        right_points,
        retarget_type=retarget_type,
        config_yml=config_yml,
        scaling_factor=scaling_factor,
        project_dist=project_dist,
        escape_dist=escape_dist,
        low_pass_alpha=low_pass_alpha,
        flexion_scale=flexion_scale,
        limit_margin=limit_margin,
        human_close_gamma=human_close_gamma,
    )
    native_qvel = _differentiate(native_qpos, native_time)

    if target_kinematics_npz is None:
        timestamps = native_time.copy()
        qpos = native_qpos.copy()
        target_fps = (
            float(1.0 / np.median(np.diff(timestamps))) if timestamps.size > 1 else 0.0
        )
        timeline_source = "native_hand_timestamps"
        target_path = ""
    else:
        target = Path(target_kinematics_npz).expanduser().resolve()
        if not target.is_file():
            raise FileNotFoundError(target)
        timestamps, target_fps, timeline_source, _ = _infer_timeline_from_npz(target)
        qpos = _resample_qpos(native_qpos, native_time, timestamps)
        target_path = str(target)
    qvel = _differentiate(qpos, timestamps)

    native_fps = (
        float(1.0 / np.median(np.diff(native_time))) if native_time.size > 1 else 0.0
    )
    metadata: dict[str, Any] = {
        "schema_version": "dex3_hand_kinematics_v2",
        **source_metadata,
        **retarget_metadata,
        "target_kinematics_npz": target_path,
        "timeline_source": timeline_source,
        "native_fps": native_fps,
        "fps": target_fps,
        "frames": int(qpos.shape[0]),
        "timestamp_origin": "seconds_from_first_frame",
        "qpos_units": "radian",
        "qvel_units": "radian_per_second",
        "hand_joint_order": HAND_JOINT_NAMES,
        "native_time_range_seconds": [float(native_time[0]), float(native_time[-1])],
        "target_time_range_seconds": [float(timestamps[0]), float(timestamps[-1])],
        "target_samples_before_native_range": int(
            np.count_nonzero(timestamps < native_time[0])
        ),
        "target_samples_after_native_range": int(
            np.count_nonzero(timestamps > native_time[-1])
        ),
    }
    return {
        # Historical fields.
        "left_qpos": qpos[:, :7],
        "right_qpos": qpos[:, 7:],
        "qpos": qpos,
        "left_joint_names": np.asarray(LEFT_JOINT_NAMES),
        "right_joint_names": np.asarray(RIGHT_JOINT_NAMES),
        "hand_joint_names": np.asarray(HAND_JOINT_NAMES),
        "source_hdf5": np.asarray(str(hdf5_path)),
        "method": np.asarray(retarget_metadata["method"]),
        "retarget_type": np.asarray(retarget_type),
        "config_yml": np.asarray(retarget_metadata["config_yml"]),
        "flexion_scale": np.asarray(float(flexion_scale)),
        "limit_margin": np.asarray(float(limit_margin)),
        "scaling_factor": np.asarray(float(retarget_metadata["scaling_factor"])),
        "low_pass_alpha": np.asarray(float(retarget_metadata["low_pass_alpha"])),
        "human_close_gamma": np.asarray(
            float("nan") if human_close_gamma is None else float(human_close_gamma)
        ),
        # Time-aligned hand kinematics.
        "left_qvel": qvel[:, :7],
        "right_qvel": qvel[:, 7:],
        "qvel": qvel,
        "timestamps": timestamps,
        "fps": np.asarray(target_fps, dtype=np.float64),
        # Native-rate diagnostics, before body-timeline interpolation.
        "native_left_qpos": native_qpos[:, :7],
        "native_right_qpos": native_qpos[:, 7:],
        "native_qpos": native_qpos,
        "native_left_qvel": native_qvel[:, :7],
        "native_right_qvel": native_qvel[:, 7:],
        "native_qvel": native_qvel,
        "native_timestamps": native_time,
        "native_timestamps_absolute": native_time_absolute,
        "native_fps": np.asarray(native_fps, dtype=np.float64),
        "target_kinematics_npz": np.asarray(target_path),
        "source_schema": np.asarray(source_metadata["source_schema"]),
        "source_session_dir": np.asarray(
            str(resolved_session) if resolved_session is not None else ""
        ),
        "schema_version": np.asarray(metadata["schema_version"]),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--hdf5",
        type=Path,
        help="Legacy transforms HDF5, or a session annotation.hdf5.",
    )
    source.add_argument(
        "--session-dir",
        type=Path,
        help="EgoAnywhere session directory containing annotation.hdf5 and calibrations/.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target-kinematics-npz",
        type=Path,
        default=None,
        help="Resample qpos and compute qvel on this body kinematics timeline.",
    )
    parser.add_argument(
        "--retarget-type",
        choices=("dexpilot", "vector", "position"),
        default="dexpilot",
        help=(
            "dexpilot (default pipeline), vector, or position "
            "(offline absolute tip/wrist positions; see configs/unitree_dex3_position.yml)."
        ),
    )
    parser.add_argument(
        "--config-yml",
        type=Path,
        default=None,
        help="Override the shipped Unitree Dex3 retarget config.",
    )
    parser.add_argument("--scaling-factor", type=float, default=None)
    parser.add_argument("--project-dist", type=float, default=None)
    parser.add_argument("--escape-dist", type=float, default=None)
    parser.add_argument("--low-pass-alpha", type=float, default=None)
    parser.add_argument(
        "--flexion-scale",
        type=float,
        default=0.7,
        help="Scale retargeted curl about zero before clipping (default: 0.70).",
    )
    parser.add_argument("--human-close-gamma", type=float, default=None)
    parser.add_argument(
        "--limit-margin",
        type=float,
        default=0.01,
        help="Clip each Dex3 joint this far inside its XML limits.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Write only the first N aligned frames; native diagnostics remain complete.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.session_dir is not None:
        session_dir = args.session_dir.expanduser().resolve()
        hdf5_path = session_dir / "annotation.hdf5"
    else:
        session_dir = None
        hdf5_path = args.hdf5.expanduser().resolve()
    output = args.output.expanduser().resolve()

    data = retarget_episode(
        hdf5_path,
        session_dir=session_dir,
        target_kinematics_npz=args.target_kinematics_npz,
        retarget_type=str(args.retarget_type),
        config_yml=(
            args.config_yml.expanduser().resolve() if args.config_yml else None
        ),
        scaling_factor=args.scaling_factor,
        project_dist=args.project_dist,
        escape_dist=args.escape_dist,
        low_pass_alpha=args.low_pass_alpha,
        flexion_scale=float(args.flexion_scale),
        limit_margin=float(args.limit_margin),
        human_close_gamma=args.human_close_gamma,
    )
    if args.max_frames is not None:
        frame_count = int(args.max_frames)
        if frame_count <= 0:
            raise ValueError(f"--max-frames must be positive, got {frame_count}")
        aligned_fields = (
            "left_qpos",
            "right_qpos",
            "qpos",
            "left_qvel",
            "right_qvel",
            "qvel",
            "timestamps",
        )
        for key in aligned_fields:
            data[key] = data[key][:frame_count]

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **data)
    metadata = json.loads(str(data["metadata_json"]))
    metadata["written_frames"] = int(data["qpos"].shape[0])
    metadata_path = output.with_suffix(".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        f"[hand_retarget] wrote {output} qpos={data['qpos'].shape} "
        f"native={data['native_qpos'].shape} method={str(data['method'])}"
    )
    print(
        f"[hand_retarget] fps={float(data['fps']):.6g} "
        f"native_fps={float(data['native_fps']):.6g} qvel={data['qvel'].shape}"
    )
    print(
        f"[hand_retarget] range left=[{data['left_qpos'].min():.3f},"
        f"{data['left_qpos'].max():.3f}] right=[{data['right_qpos'].min():.3f},"
        f"{data['right_qpos'].max():.3f}] metadata={metadata_path}"
    )


if __name__ == "__main__":
    main()
