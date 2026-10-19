#!/usr/bin/env python3
"""Inject a contact-aware wrist goal before Kimodo, ScaleTrack, and Mink.

The template-palm reference remains the carrier.  The correction is a full
SE(3) residual on ``*_wrist_yaw_link``; it is never a Dex3 yaw-joint offset.
The ``edge_hold`` policy applies the first contact residual to every frame
before contact, each optimized residual during contact, and the last contact
residual after contact.  The unified pipeline enables this policy explicitly;
the legacy ``transition`` mode remains available for regression tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


WRIST_BODIES = {
    "left": "left_wrist_yaw_link",
    "right": "right_wrist_yaw_link",
}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]).copy() for key in archive.files}


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_schedule(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Persist the expanded 6D residual schedule next to the references."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_time(values: np.ndarray) -> np.ndarray:
    time = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(time) < 2 or np.any(~np.isfinite(time)) or np.any(np.diff(time) <= 0.0):
        raise ValueError("timestamps must be finite and strictly increasing")
    return time - time[0]


def continuous_wxyz(rotations: np.ndarray) -> np.ndarray:
    quaternions = Rotation.from_matrix(rotations).as_quat(scalar_first=True)
    for index in range(1, len(quaternions)):
        if float(np.dot(quaternions[index - 1], quaternions[index])) < 0.0:
            quaternions[index] *= -1.0
    return quaternions


def wxyz_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(
        np.asarray(quaternions, dtype=np.float64), scalar_first=True
    ).as_matrix()


def finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    velocity = np.zeros_like(data)
    if len(data) > 1:
        velocity[0] = (data[1] - data[0]) * fps
        velocity[-1] = (data[-1] - data[-2]) * fps
        velocity[1:-1] = (data[2:] - data[:-2]) * (0.5 * fps)
    return velocity


def angular_velocity_world(rotations: np.ndarray, fps: float) -> np.ndarray:
    matrices = np.asarray(rotations, dtype=np.float64)
    velocity = np.zeros((len(matrices), 3), dtype=np.float64)
    if len(matrices) > 1:
        velocity[0] = (
            Rotation.from_matrix(matrices[1] @ matrices[0].T).as_rotvec() * fps
        )
        velocity[-1] = (
            Rotation.from_matrix(matrices[-1] @ matrices[-2].T).as_rotvec() * fps
        )
        for index in range(1, len(matrices) - 1):
            velocity[index] = (
                Rotation.from_matrix(matrices[index + 1] @ matrices[index - 1].T)
                .as_rotvec()
                * (0.5 * fps)
            )
    return velocity


def resample_pose(
    source_time: np.ndarray,
    positions: np.ndarray,
    rotations: np.ndarray,
    target_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_time = np.asarray(source_time, dtype=np.float64).reshape(-1)
    query = np.asarray(target_time, dtype=np.float64)
    if not len(source_time):
        raise ValueError("contact pose timeline is empty")
    if len(source_time) == 1:
        return (
            np.repeat(np.asarray(positions, dtype=np.float64)[:1], len(query), axis=0),
            np.repeat(np.asarray(rotations, dtype=np.float64)[:1], len(query), axis=0),
        )
    if np.any(query < source_time[0] - 1.0e-9) or np.any(
        query > source_time[-1] + 1.0e-9
    ):
        raise ValueError("contact query is outside the native candidate timeline")
    query = np.clip(query, source_time[0], source_time[-1])
    position = np.column_stack(
        [np.interp(query, source_time, positions[:, axis]) for axis in range(3)]
    )
    rotation = Slerp(source_time, Rotation.from_matrix(rotations))(query).as_matrix()
    return position, rotation


def smoothstep(values: np.ndarray) -> np.ndarray:
    alpha = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def apply_scaled_local_correction(
    base_position: np.ndarray,
    base_rotation: np.ndarray,
    translation_world: np.ndarray,
    rotation_local_vector: np.ndarray,
    alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(alpha, dtype=np.float64).reshape(-1)
    position = base_position + weights[:, None] * translation_world
    correction = Rotation.from_rotvec(
        weights[:, None] * rotation_local_vector
    ).as_matrix()
    rotation = np.einsum("tij,tjk->tik", base_rotation, correction)
    return position, rotation


def stats(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95.0)),
        "max": float(np.max(data)),
    }


def inject_contact_reference(
    *,
    base_reference: Path,
    base_targets: Path,
    base_meta: Path,
    contact_candidates: Path,
    output_reference: Path,
    output_targets: Path,
    output_meta: Path,
    side: str = "right",
    transition_frames: int = 15,
    edge_hold_policy: str = "transition",
    schedule_output: Path | None = None,
) -> dict[str, Any]:
    for path in (base_reference, base_targets, base_meta, contact_candidates):
        if not path.is_file():
            raise FileNotFoundError(path)
    if side not in WRIST_BODIES:
        raise ValueError(f"unsupported side {side}")
    if transition_frames < 0:
        raise ValueError("transition_frames must be non-negative")
    if edge_hold_policy not in {"edge_hold", "transition"}:
        raise ValueError("edge_hold_policy must be edge_hold or transition")

    reference = load_npz(base_reference)
    targets = load_npz(base_targets)
    contact = load_npz(contact_candidates)
    base_report = json.loads(base_meta.read_text(encoding="utf-8"))
    required_contact = (
        "timestamps",
        "frame_indices",
        "causal_position",
        "causal_rotation",
    )
    missing = [key for key in required_contact if key not in contact]
    if missing:
        raise KeyError(f"contact candidates missing {missing}")

    target_time = normalized_time(targets["timestamps"])
    native_time = normalized_time(contact["timestamps"])
    frame_indices = np.asarray(contact["frame_indices"], dtype=np.int64).reshape(-1)
    if not len(frame_indices) or np.any(np.diff(frame_indices) != 1):
        raise ValueError("contact frame_indices must be non-empty and contiguous")
    if frame_indices[0] < 0 or frame_indices[-1] >= len(native_time):
        raise ValueError("contact frame_indices are outside the native timeline")

    contact_position = np.asarray(contact["causal_position"], dtype=np.float64)
    contact_rotation = np.asarray(contact["causal_rotation"], dtype=np.float64)
    if contact_position.shape != (len(native_time), 3):
        raise ValueError(f"invalid causal_position shape {contact_position.shape}")
    if contact_rotation.shape != (len(native_time), 3, 3):
        raise ValueError(f"invalid causal_rotation shape {contact_rotation.shape}")
    if np.any(~np.isfinite(contact_position)) or np.any(~np.isfinite(contact_rotation)):
        raise ValueError("contact-aware pose contains non-finite values")

    interval_start = float(native_time[frame_indices[0]])
    interval_end = float(native_time[frame_indices[-1]])
    selected = np.flatnonzero(
        (target_time >= interval_start) & (target_time <= interval_end)
    )
    if not len(selected):
        if len(frame_indices) != 1:
            raise ValueError("contact interval does not overlap the reference timeline")
        selected = np.asarray(
            [int(np.argmin(np.abs(target_time - interval_start)))], dtype=np.int64
        )

    body_names = [str(value) for value in reference["body_names"].reshape(-1)]
    wrist_name = WRIST_BODIES[side]
    if wrist_name not in body_names:
        raise KeyError(wrist_name)
    wrist_index = body_names.index(wrist_name)
    fps = float(np.asarray(reference["fps"]).reshape(()))
    if not np.isclose(float(np.asarray(targets["fps_session"]).reshape(())), fps):
        raise ValueError("reference and target FPS differ")
    if len(target_time) != reference["body_pos_w"].shape[0]:
        raise ValueError("reference and target frame counts differ")

    output_reference_arrays = {key: value.copy() for key, value in reference.items()}
    output_target_arrays = {key: value.copy() for key, value in targets.items()}
    base_position = np.asarray(reference["body_pos_w"][:, wrist_index], dtype=np.float64)
    base_rotation = wxyz_to_matrix(reference["body_quat_w"][:, wrist_index])
    output_position = base_position.copy()
    output_rotation = base_rotation.copy()

    # Build the requested causal schedule.  Translation is a world-space
    # residual, while rotation is a local residual composed after the base
    # wrist orientation.  Contact frames are sampled from the optimized
    # candidate; outside the interval the nearest edge residual is held.
    contact_position = contact_position[frame_indices]
    contact_rotation = contact_rotation[frame_indices]
    first_target = int(np.argmin(np.abs(target_time - interval_start)))
    last_target = int(np.argmin(np.abs(target_time - interval_end)))
    edge_base_position, edge_base_rotation = resample_pose(
        target_time,
        base_position,
        base_rotation,
        np.asarray([interval_start, interval_end], dtype=np.float64),
    )
    first_translation_world = contact_position[0] - edge_base_position[0]
    first_rotation_local = Rotation.from_matrix(
        edge_base_rotation[0].T @ contact_rotation[0]
    ).as_rotvec()
    last_translation_world = contact_position[-1] - edge_base_position[-1]
    last_rotation_local = Rotation.from_matrix(
        edge_base_rotation[-1].T @ contact_rotation[-1]
    ).as_rotvec()

    if edge_hold_policy == "edge_hold":
        inside = selected
        if len(inside):
            inside_position, inside_rotation = resample_pose(
                native_time[frame_indices],
                contact_position,
                contact_rotation,
                target_time[inside],
            )
            output_position[inside] = inside_position
            output_rotation[inside] = inside_rotation
        before = np.arange(0, int(inside[0]), dtype=np.int64)
        after = np.arange(int(inside[-1]) + 1, len(target_time), dtype=np.int64)
        if len(before):
            output_position[before] = base_position[before] + first_translation_world
            output_rotation[before] = np.einsum(
                "tij,jk->tik", base_rotation[before],
                Rotation.from_rotvec(first_rotation_local).as_matrix()
            )
        if len(after):
            output_position[after] = base_position[after] + last_translation_world
            output_rotation[after] = np.einsum(
                "tij,jk->tik", base_rotation[after],
                Rotation.from_rotvec(last_rotation_local).as_matrix()
            )
        selected = inside
    else:
        selected_position, selected_rotation = resample_pose(
            native_time[frame_indices],
            contact_position,
            contact_rotation,
            target_time[selected],
        )
        output_position[selected] = selected_position
        output_rotation[selected] = selected_rotation
        before = np.arange(
            max(0, int(selected[0]) - transition_frames), int(selected[0]), dtype=np.int64
        )
        after = np.arange(
            int(selected[-1]) + 1,
            min(len(target_time), int(selected[-1]) + 1 + transition_frames),
            dtype=np.int64,
        )
        start_translation = selected_position[0] - base_position[selected[0]]
        start_rotation = Rotation.from_matrix(
            base_rotation[selected[0]].T @ selected_rotation[0]
        ).as_rotvec()
        if len(before):
            alpha = smoothstep(
                np.arange(1, len(before) + 1, dtype=np.float64) / (len(before) + 1)
            )
            output_position[before], output_rotation[before] = apply_scaled_local_correction(
                base_position[before], base_rotation[before], start_translation,
                start_rotation, alpha,
            )
        end_translation = selected_position[-1] - base_position[selected[-1]]
        end_rotation = Rotation.from_matrix(
            base_rotation[selected[-1]].T @ selected_rotation[-1]
        ).as_rotvec()
        if len(after):
            alpha = smoothstep(
                1.0 - np.arange(1, len(after) + 1, dtype=np.float64)
                / (len(after) + 1)
            )
            output_position[after], output_rotation[after] = apply_scaled_local_correction(
                base_position[after], base_rotation[after], end_translation,
                end_rotation, alpha,
            )

    quaternion = continuous_wxyz(output_rotation)
    output_reference_arrays["body_pos_w"][:, wrist_index] = output_position.astype(
        reference["body_pos_w"].dtype
    )
    output_reference_arrays["body_quat_w"][:, wrist_index] = quaternion.astype(
        reference["body_quat_w"].dtype
    )
    output_reference_arrays["body_lin_vel_w"][:, wrist_index] = finite_difference(
        output_position, fps
    ).astype(reference["body_lin_vel_w"].dtype)
    output_reference_arrays["body_ang_vel_w"][:, wrist_index] = angular_velocity_world(
        output_rotation, fps
    ).astype(reference["body_ang_vel_w"].dtype)
    output_target_arrays[f"{side}_wrist"] = output_position.astype(np.float32)
    output_target_arrays[f"{side}_wrist_quat_wxyz"] = quaternion.astype(np.float32)

    method = "template_palm_kabsch_contact_aware_feedback1"
    source_hash = sha256(contact_candidates)
    for arrays in (output_reference_arrays, output_target_arrays):
        arrays["wrist_reference_method"] = np.asarray(method)
        arrays["contact_aware_wrist_side"] = np.asarray(side)
        arrays["contact_aware_reference_source"] = np.asarray(
            str(contact_candidates.resolve())
        )
        arrays["contact_aware_reference_sha256"] = np.asarray(source_hash)
        arrays["contact_aware_source_frame_indices"] = frame_indices
        arrays["contact_aware_target_indices"] = selected
        arrays["contact_aware_transition_before_indices"] = before
        arrays["contact_aware_transition_after_indices"] = after
        arrays["contact_aware_edge_hold_policy"] = np.asarray(edge_hold_policy)
        arrays["optimization_space"] = np.asarray(f"{wrist_name}_SE3")
        arrays["not_joint_yaw_offset"] = np.asarray(True)

    wrist_mask = np.zeros(len(body_names), dtype=bool)
    wrist_mask[wrist_index] = True
    for field in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        if not np.array_equal(
            reference[field][:, ~wrist_mask], output_reference_arrays[field][:, ~wrist_mask]
        ):
            raise RuntimeError(f"non-selected-wrist values changed in {field}")
    for field in ("joint_pos", "joint_vel", "body_names", "joint_names", "fps"):
        if not np.array_equal(reference[field], output_reference_arrays[field]):
            raise RuntimeError(f"base reference field changed: {field}")
    other_side = "left" if side == "right" else "right"
    for field in (
        "timestamps",
        "fps_session",
        "kimodo_pelvis_heading_quat_wxyz",
        f"{other_side}_wrist",
        f"{other_side}_wrist_quat_wxyz",
    ):
        if not np.array_equal(targets[field], output_target_arrays[field]):
            raise RuntimeError(f"base target field changed: {field}")

    modified = np.concatenate((before, selected, after))
    position_delta = np.linalg.norm(
        output_position[selected] - base_position[selected], axis=1
    )
    orientation_delta = np.degrees(
        (
            Rotation.from_matrix(base_rotation[selected]).inv()
            * Rotation.from_matrix(output_rotation[selected])
        ).magnitude()
    )
    if np.any(~np.isfinite(output_position)) or np.any(~np.isfinite(quaternion)):
        raise RuntimeError("contact-aware reference contains non-finite values")

    if schedule_output is not None:
        target_translation_world = output_position - base_position
        target_rotation_local = np.asarray([
            Rotation.from_matrix(base_rotation[i].T @ output_rotation[i]).as_rotvec()
            for i in range(len(output_rotation))
        ])
        atomic_schedule(
            schedule_output,
            {
                "target_indices": np.arange(len(target_time), dtype=np.int64),
                "target_timestamps": target_time,
                "contact_frame_indices": frame_indices,
                "first_contact": np.asarray(frame_indices[0], dtype=np.int64),
                "last_contact": np.asarray(frame_indices[-1], dtype=np.int64),
                "translation_world_m": target_translation_world,
                "rotation_local_rotvec_rad": target_rotation_local,
                "scheduled_position": output_position,
                "scheduled_rotation": output_rotation,
                "policy": np.asarray(edge_hold_policy),
                "optimization_space": np.asarray("right_wrist_yaw_link_SE3"),
                "not_joint_yaw_offset": np.asarray(True),
            },
        )

    atomic_npz(output_reference, output_reference_arrays)
    atomic_npz(output_targets, output_target_arrays)
    report = dict(base_report)
    report.update(
        {
            "schema_version": 2,
            "method": method,
            "output_frames": int(len(target_time)),
            "output_fps": fps,
            "contact_aware_injection": {
                "side": side,
                "source": str(contact_candidates.resolve()),
                "source_sha256": source_hash,
                "tracking_fields": "causal_position/causal_rotation",
                "evaluation_only_fields": (
                    "reference_causal_position/reference_causal_rotation"
                ),
                "native_frame_range_inclusive": [
                    int(frame_indices[0]),
                    int(frame_indices[-1]),
                ],
                "native_time_range_s": [interval_start, interval_end],
                "target_frame_range_inclusive": [
                    int(selected[0]),
                    int(selected[-1]),
                ],
                "target_time_range_s": [
                    float(target_time[selected[0]]),
                    float(target_time[selected[-1]]),
                ],
                "transition_frames_requested": int(transition_frames),
                "edge_hold_policy": edge_hold_policy,
                "transition_before_indices": before.tolist(),
                "transition_after_indices": after.tolist(),
                "modified_frame_count": int(len(modified)),
                "position_delta_m": stats(position_delta),
                "orientation_delta_deg": stats(orientation_delta),
            },
            "outputs": {
                "reference": str(output_reference.resolve()),
                "targets": str(output_targets.resolve()),
            },
        }
    )
    report.setdefault("contract", {})
    report["contract"]["contact_aware_reference_stage"] = (
        "template-palm carrier -> feedback1 causal wrist goal -> Kimodo -> "
        "ScaleTrack/MPC -> Mink"
    )
    report["contract"]["contact_transition"] = (
        "first/last contact SE(3) residual held outside contact by default"
        if edge_hold_policy == "edge_hold"
        else "cubic smoothstep SE(3) correction"
    )
    report["contract"]["cached_kinematics_policy"] = (
        "no post-IK body pose or cached contact-aware kinematics are consumed"
    )
    report.setdefault("gates", {})
    report["gates"].update(
        {
            "only_selected_wrist_reference_changed": True,
            "joint_and_timeline_fields_unchanged": True,
            "matching_kimodo_wrist_target_updated": True,
            "finite_output": True,
        }
    )
    atomic_json(output_meta, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-reference", type=Path, required=True)
    parser.add_argument("--base-targets", type=Path, required=True)
    parser.add_argument("--base-meta", type=Path, required=True)
    parser.add_argument("--contact-candidates", type=Path, required=True)
    parser.add_argument("--output-reference", type=Path, required=True)
    parser.add_argument("--output-targets", type=Path, required=True)
    parser.add_argument("--output-meta", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--transition-frames", type=int, default=15)
    parser.add_argument(
        "--edge-hold-policy", choices=("edge_hold", "transition"), default="transition"
    )
    parser.add_argument(
        "--schedule-output", type=Path,
        help="Optional NPZ path for the expanded per-frame 6D wrist offset schedule.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = inject_contact_reference(
        base_reference=args.base_reference.expanduser().resolve(),
        base_targets=args.base_targets.expanduser().resolve(),
        base_meta=args.base_meta.expanduser().resolve(),
        contact_candidates=args.contact_candidates.expanduser().resolve(),
        output_reference=args.output_reference.expanduser().resolve(),
        output_targets=args.output_targets.expanduser().resolve(),
        output_meta=args.output_meta.expanduser().resolve(),
        side=args.side,
        transition_frames=int(args.transition_frames),
        edge_hold_policy=args.edge_hold_policy,
        schedule_output=(
            args.schedule_output.expanduser().resolve()
            if args.schedule_output is not None else None
        ),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
