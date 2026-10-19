#!/usr/bin/env python3
"""Build an isolated torso-plus-wrists reference from a Stera session.

The input ScaleTrack motion is copied verbatim except for the four torso body
pose/velocity fields.  The source camera-link pose is converted to the Unitree
D435-link convention and the fixed torso-to-D435 mount is inverted to obtain
the torso reference.  No policy, simulator, or MPC code is imported here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


EXPECTED_FRAMES = 539
EXPECTED_FPS = 50.0
TORSO_BODY = "torso_link"
WRIST_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
POSE_FIELDS = (
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)

# Change of basis from the Stera Y-up world/local basis to ScaleTrack Z-up:
# (x, y, z) -> (x, -z, y).
YUP_TO_ZUP = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)

# After the change of basis, Stera's link looks along +Y.  Unitree's D435
# link looks along +X, with right mapped to -Y and up preserved as +Z.
STERA_ZUP_TO_UNITREE_D435 = np.asarray(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)

TORSO_TO_D435_POSITION_M = np.asarray(
    [0.0576235, 0.01753, 0.41987], dtype=np.float64
)
TORSO_TO_D435_RPY_RAD = np.asarray(
    [0.0, 0.8307767239493009, 0.0], dtype=np.float64
)
TORSO_TO_D435_ROTATION = Rotation.from_euler(
    "xyz", TORSO_TO_D435_RPY_RAD
).as_matrix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _string_list(values: np.ndarray) -> list[str]:
    return [str(item) for item in np.asarray(values).reshape(-1).tolist()]


def _continuous_wxyz(rotation: Rotation) -> np.ndarray:
    quat = np.asarray(rotation.as_quat(scalar_first=True), dtype=np.float64)
    if quat.ndim == 1:
        quat = quat[None, :]
    for index in range(1, len(quat)):
        if float(np.dot(quat[index - 1], quat[index])) < 0.0:
            quat[index] *= -1.0
    return quat


def _rotation_from_wxyz(quaternion: np.ndarray) -> Rotation:
    quat = np.asarray(quaternion, dtype=np.float64)
    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
    if not np.all(np.isfinite(quat)) or np.any(norms <= 1.0e-12):
        raise ValueError("quaternion input is non-finite or has zero norm")
    return Rotation.from_quat(quat / norms, scalar_first=True)


def _finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.zeros_like(values)
    if len(values) > 1:
        result[0] = (values[1] - values[0]) * fps
        result[-1] = (values[-1] - values[-2]) * fps
        result[1:-1] = (values[2:] - values[:-2]) * (0.5 * fps)
    return result


def _angular_velocity_world(rotation: Rotation, fps: float) -> np.ndarray:
    matrices = rotation.as_matrix()
    result = np.zeros((len(matrices), 3), dtype=np.float64)
    if len(matrices) > 1:
        result[0] = (
            Rotation.from_matrix(matrices[1] @ matrices[0].T).as_rotvec() * fps
        )
        result[-1] = (
            Rotation.from_matrix(matrices[-1] @ matrices[-2].T).as_rotvec() * fps
        )
        for index in range(1, len(matrices) - 1):
            result[index] = (
                Rotation.from_matrix(matrices[index + 1] @ matrices[index - 1].T)
                .as_rotvec()
                * (0.5 * fps)
            )
    return result


def _interpolate_d435_pose(
    annotation_hdf5: Path, query_timestamps: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with h5py.File(annotation_hdf5, "r") as annotation:
        for key in (
            "cam-pose/timestamps",
            "cam-pose/translations",
            "cam-pose/rotations",
        ):
            if key not in annotation:
                raise KeyError(f"annotation is missing {key!r}")
        source_abs_t = np.asarray(
            annotation["cam-pose/timestamps"], dtype=np.float64
        )
        source_position_yup = np.asarray(
            annotation["cam-pose/translations"], dtype=np.float64
        )
        source_rotation_yup = np.asarray(
            annotation["cam-pose/rotations"], dtype=np.float64
        )

    source_frames = len(source_abs_t)
    if source_frames < 2:
        raise ValueError("camera pose requires at least two source frames")
    if source_position_yup.shape != (source_frames, 3):
        raise ValueError(
            f"invalid camera translation shape {source_position_yup.shape}"
        )
    if source_rotation_yup.shape != (source_frames, 3, 3):
        raise ValueError(f"invalid camera rotation shape {source_rotation_yup.shape}")
    if not np.all(np.isfinite(source_abs_t)) or np.any(np.diff(source_abs_t) <= 0.0):
        raise ValueError("camera timestamps must be finite and strictly increasing")
    if not np.all(np.isfinite(source_position_yup)) or not np.all(
        np.isfinite(source_rotation_yup)
    ):
        raise ValueError("camera pose contains non-finite values")

    source_t = source_abs_t - source_abs_t[0]
    query_timestamps = np.asarray(query_timestamps, dtype=np.float64)
    tolerance = 1.0e-9
    if (
        query_timestamps[0] < source_t[0] - tolerance
        or query_timestamps[-1] > source_t[-1] + tolerance
    ):
        raise ValueError(
            "requested reference timestamps are outside the camera interval: "
            f"query=[{query_timestamps[0]}, {query_timestamps[-1]}], "
            f"camera=[{source_t[0]}, {source_t[-1]}]"
        )

    d435_position_source = source_position_yup @ YUP_TO_ZUP.T
    stera_rotation_zup = np.einsum(
        "ij,tjk,kl->til",
        YUP_TO_ZUP,
        source_rotation_yup,
        YUP_TO_ZUP.T,
    )
    d435_rotation_source = (
        stera_rotation_zup @ STERA_ZUP_TO_UNITREE_D435
    )
    rotation_validity_error = float(
        np.max(
            np.abs(
                d435_rotation_source
                @ np.swapaxes(d435_rotation_source, -1, -2)
                - np.eye(3)
            )
        )
    )
    if rotation_validity_error > 1.0e-4:
        raise ValueError(
            f"camera rotations are not orthonormal (max error {rotation_validity_error})"
        )

    d435_position = np.stack(
        [
            np.interp(query_timestamps, source_t, d435_position_source[:, axis])
            for axis in range(3)
        ],
        axis=1,
    )
    d435_rotation = Slerp(
        source_t, Rotation.from_matrix(d435_rotation_source)
    )(query_timestamps)
    d435_quaternion = _continuous_wxyz(d435_rotation)
    source_info = {
        "source_frames": source_frames,
        "source_start_timestamp_s": float(source_abs_t[0]),
        "source_end_timestamp_s": float(source_abs_t[-1]),
        "source_duration_s": float(source_t[-1]),
        "source_median_period_s": float(np.median(np.diff(source_t))),
        "source_nominal_fps": float(1.0 / np.median(np.diff(source_t))),
        "rotation_orthonormality_max_abs": rotation_validity_error,
        "interpolation": "linear translation and quaternion Slerp",
    }
    return d435_position, d435_quaternion, source_info


def torso_from_d435(
    d435_position_w: np.ndarray, d435_quaternion_wxyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Invert the fixed torso-to-D435 transform for a world D435 pose."""

    d435_rotation = _rotation_from_wxyz(d435_quaternion_wxyz).as_matrix()
    torso_rotation = d435_rotation @ TORSO_TO_D435_ROTATION.T
    torso_position = np.asarray(d435_position_w, dtype=np.float64) - np.einsum(
        "tij,j->ti", torso_rotation, TORSO_TO_D435_POSITION_M
    )
    torso_quaternion = _continuous_wxyz(Rotation.from_matrix(torso_rotation))
    return torso_position, torso_quaternion


def d435_from_torso(
    torso_position_w: np.ndarray, torso_quaternion_wxyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the fixed torso-to-D435 transform to a world torso pose."""

    torso_rotation = _rotation_from_wxyz(torso_quaternion_wxyz).as_matrix()
    d435_rotation = torso_rotation @ TORSO_TO_D435_ROTATION
    d435_position = np.asarray(torso_position_w, dtype=np.float64) + np.einsum(
        "tij,j->ti", torso_rotation, TORSO_TO_D435_POSITION_M
    )
    d435_quaternion = _continuous_wxyz(Rotation.from_matrix(d435_rotation))
    return d435_position, d435_quaternion


def _rotation_distance_rad(first_wxyz: np.ndarray, second_wxyz: np.ndarray) -> np.ndarray:
    first = _rotation_from_wxyz(first_wxyz)
    second = _rotation_from_wxyz(second_wxyz)
    return (first * second.inv()).magnitude()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_base_reference(
    arrays: dict[str, np.ndarray], expected_frames: int, expected_fps: float
) -> tuple[list[str], int, float]:
    required = {
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "body_names",
        "joint_names",
        "fps",
    }
    missing = sorted(required - arrays.keys())
    if missing:
        raise KeyError(f"base reference is missing fields: {missing}")
    body_names = _string_list(arrays["body_names"])
    if len(set(body_names)) != len(body_names):
        raise ValueError("base reference body_names contains duplicates")
    for name in (TORSO_BODY, *WRIST_BODIES):
        if name not in body_names:
            raise ValueError(f"base reference is missing body {name!r}")
    frame_count = int(np.asarray(arrays["body_pos_w"]).shape[0])
    fps = float(np.asarray(arrays["fps"]).reshape(()))
    if frame_count != expected_frames:
        raise ValueError(
            f"base reference has {frame_count} frames, expected {expected_frames}"
        )
    if not np.isclose(fps, expected_fps, rtol=0.0, atol=1.0e-8):
        raise ValueError(f"base reference has fps={fps}, expected {expected_fps}")
    body_count = len(body_names)
    expected_shapes = {
        "body_pos_w": (frame_count, body_count, 3),
        "body_quat_w": (frame_count, body_count, 4),
        "body_lin_vel_w": (frame_count, body_count, 3),
        "body_ang_vel_w": (frame_count, body_count, 3),
    }
    for field, shape in expected_shapes.items():
        if arrays[field].shape != shape:
            raise ValueError(f"{field} shape {arrays[field].shape}, expected {shape}")
        if not np.all(np.isfinite(arrays[field])):
            raise ValueError(f"{field} contains non-finite values")
    return body_names, frame_count, fps


def _unchanged_gates(
    source: dict[str, np.ndarray], result: dict[str, np.ndarray], body_names: list[str]
) -> dict[str, Any]:
    torso_index = body_names.index(TORSO_BODY)
    non_torso = np.ones(len(body_names), dtype=bool)
    non_torso[torso_index] = False
    for field in POSE_FIELDS:
        if not np.array_equal(source[field][:, non_torso], result[field][:, non_torso]):
            raise RuntimeError(f"non-torso values changed in {field}")
    for field in source:
        if field not in POSE_FIELDS and not np.array_equal(source[field], result[field]):
            raise RuntimeError(f"base field changed unexpectedly: {field}")

    wrist_checks: dict[str, dict[str, bool]] = {}
    for wrist in WRIST_BODIES:
        wrist_index = body_names.index(wrist)
        wrist_checks[wrist] = {}
        for field in POSE_FIELDS:
            equal = bool(
                np.array_equal(source[field][:, wrist_index], result[field][:, wrist_index])
            )
            wrist_checks[wrist][field] = equal
            if not equal:
                raise RuntimeError(f"strict wrist equality failed for {wrist}/{field}")
    return {
        "all_original_non_torso_values_bitwise_equal": True,
        "all_original_non_pose_fields_bitwise_equal": True,
        "wrist_fields_bitwise_equal": wrist_checks,
    }


def build_reference(
    *,
    annotation_hdf5: Path,
    base_reference: Path,
    output_npz: Path,
    meta_json: Path | None = None,
    expected_frames: int = EXPECTED_FRAMES,
    expected_fps: float = EXPECTED_FPS,
    roundtrip_tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    annotation_hdf5 = annotation_hdf5.expanduser().resolve()
    base_reference = base_reference.expanduser().resolve()
    output_npz = output_npz.expanduser().resolve()
    meta_json = (
        output_npz.parent / "reference_meta.json"
        if meta_json is None
        else meta_json.expanduser().resolve()
    )
    if not annotation_hdf5.is_file():
        raise FileNotFoundError(annotation_hdf5)
    if not base_reference.is_file():
        raise FileNotFoundError(base_reference)
    if output_npz == base_reference:
        raise ValueError("output NPZ must not overwrite the base wrist reference")

    source = _load_npz(base_reference)
    body_names, frame_count, fps = _validate_base_reference(
        source, expected_frames, expected_fps
    )
    timestamps = np.arange(frame_count, dtype=np.float64) / fps
    d435_position, d435_quaternion, camera_info = _interpolate_d435_pose(
        annotation_hdf5, timestamps
    )
    torso_position, torso_quaternion = torso_from_d435(
        d435_position, d435_quaternion
    )
    torso_rotation = _rotation_from_wxyz(torso_quaternion)
    torso_linear_velocity = _finite_difference(torso_position, fps)
    torso_angular_velocity = _angular_velocity_world(torso_rotation, fps)

    result = {name: values.copy() for name, values in source.items()}
    torso_index = body_names.index(TORSO_BODY)
    result["body_pos_w"][:, torso_index] = torso_position.astype(
        result["body_pos_w"].dtype
    )
    result["body_quat_w"][:, torso_index] = torso_quaternion.astype(
        result["body_quat_w"].dtype
    )
    result["body_lin_vel_w"][:, torso_index] = torso_linear_velocity.astype(
        result["body_lin_vel_w"].dtype
    )
    result["body_ang_vel_w"][:, torso_index] = torso_angular_velocity.astype(
        result["body_ang_vel_w"].dtype
    )
    result.update(
        {
            "timestamps": timestamps,
            "d435_pos_w": d435_position.astype(np.float32),
            "d435_quat_wxyz": d435_quaternion.astype(np.float32),
            "tracking_body_names": np.asarray(
                [TORSO_BODY, *WRIST_BODIES], dtype=np.str_
            ),
            "yup_to_zup_matrix": YUP_TO_ZUP.astype(np.float64),
            "stera_zup_to_unitree_d435_matrix": (
                STERA_ZUP_TO_UNITREE_D435.astype(np.float64)
            ),
            "torso_to_d435_position_m": TORSO_TO_D435_POSITION_M.copy(),
            "torso_to_d435_quat_wxyz": _continuous_wxyz(
                Rotation.from_matrix(TORSO_TO_D435_ROTATION)
            )[0],
        }
    )

    unchanged = _unchanged_gates(source, result, body_names)
    reconstructed_position, reconstructed_quaternion = d435_from_torso(
        result["body_pos_w"][:, torso_index], result["body_quat_w"][:, torso_index]
    )
    roundtrip_position_max = float(
        np.max(np.linalg.norm(reconstructed_position - result["d435_pos_w"], axis=1))
    )
    roundtrip_orientation_max = float(
        np.max(
            _rotation_distance_rad(
                reconstructed_quaternion, result["d435_quat_wxyz"]
            )
        )
    )
    if roundtrip_position_max > roundtrip_tolerance:
        raise RuntimeError(
            "camera-to-torso position round-trip failed: "
            f"{roundtrip_position_max} > {roundtrip_tolerance}"
        )
    if roundtrip_orientation_max > roundtrip_tolerance:
        raise RuntimeError(
            "camera-to-torso orientation round-trip failed: "
            f"{roundtrip_orientation_max} > {roundtrip_tolerance}"
        )

    _atomic_save_npz(output_npz, result)
    saved = _load_npz(output_npz)
    saved_unchanged = _unchanged_gates(source, saved, body_names)
    for field in ("timestamps", "d435_pos_w", "d435_quat_wxyz"):
        if not np.array_equal(saved[field], result[field]):
            raise RuntimeError(f"saved reference changed appended field {field}")

    metadata: dict[str, Any] = {
        "format": "ScaleTrack torso-plus-dual-wrist reference",
        "format_version": 1,
        "output_npz": str(output_npz),
        "inputs": {
            "base_wrist_reference": str(base_reference),
            "base_wrist_reference_sha256": _sha256(base_reference),
            "annotation_hdf5": str(annotation_hdf5),
            "annotation_hdf5_sha256": _sha256(annotation_hdf5),
            "camera_datasets": [
                "cam-pose/timestamps",
                "cam-pose/translations",
                "cam-pose/rotations",
            ],
        },
        "sampling": {
            "frames": frame_count,
            "fps": fps,
            "period_s": 1.0 / fps,
            "timestamps_field": "timestamps",
            "first_timestamp_s": float(timestamps[0]),
            "last_timestamp_s": float(timestamps[-1]),
            "alignment": "exactly base-reference frame indices at 50 Hz",
            "source_camera": camera_info,
        },
        "tracking_bodies": [TORSO_BODY, *WRIST_BODIES],
        "modified_base_fields": {
            field: {"body_name": TORSO_BODY, "body_index": torso_index}
            for field in POSE_FIELDS
        },
        "appended_fields": {
            "timestamps": {"shape": [frame_count], "unit": "s"},
            "d435_pos_w": {"shape": [frame_count, 3], "unit": "m"},
            "d435_quat_wxyz": {
                "shape": [frame_count, 4],
                "order": "wxyz",
            },
            "tracking_body_names": [TORSO_BODY, *WRIST_BODIES],
        },
        "transforms": {
            "camera_pose_input": "Stera camera-link to Y-up world",
            "yup_to_zup": YUP_TO_ZUP.tolist(),
            "stera_zup_to_unitree_d435": STERA_ZUP_TO_UNITREE_D435.tolist(),
            "d435_rotation_formula": "R_w_d435 = M R_w_stera M^T A",
            "d435_position_formula": "p_w_d435 = M p_w_stera",
            "torso_from_d435_formula": "T_w_torso = T_w_d435 inv(T_torso_d435)",
            "torso_to_d435": {
                "position_m": TORSO_TO_D435_POSITION_M.tolist(),
                "rpy_rad": TORSO_TO_D435_RPY_RAD.tolist(),
                "quaternion_wxyz": _continuous_wxyz(
                    Rotation.from_matrix(TORSO_TO_D435_ROTATION)
                )[0].tolist(),
            },
        },
        "gates": {
            **unchanged,
            "saved_file_recheck": saved_unchanged,
            "roundtrip_tolerance": roundtrip_tolerance,
            "camera_torso_camera_position_max_m": roundtrip_position_max,
            "camera_torso_camera_orientation_max_rad": roundtrip_orientation_max,
            "passed": True,
        },
        "meta_json": str(meta_json),
    }
    _atomic_save_json(meta_json, metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-hdf5", type=Path, required=True)
    parser.add_argument("--base-reference", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument(
        "--meta-json",
        type=Path,
        help="Defaults to reference_meta.json beside --output-npz.",
    )
    parser.add_argument("--expected-frames", type=int, default=EXPECTED_FRAMES)
    parser.add_argument("--expected-fps", type=float, default=EXPECTED_FPS)
    parser.add_argument("--roundtrip-tolerance", type=float, default=1.0e-6)
    args = parser.parse_args()
    metadata = build_reference(
        annotation_hdf5=args.annotation_hdf5,
        base_reference=args.base_reference,
        output_npz=args.output_npz,
        meta_json=args.meta_json,
        expected_frames=args.expected_frames,
        expected_fps=args.expected_fps,
        roundtrip_tolerance=args.roundtrip_tolerance,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
