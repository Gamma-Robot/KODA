#!/usr/bin/env python3
"""Package contact-aware wrist goals and selected Dex3 fingers at 50 Hz.

The output is an experiment input for arms-only Mink IK. The source body
trajectory is copied unchanged except for the selected wrist goal inside the
evaluated contact interval. The production motion and view-change pipeline
are never modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


HAND_JOINT_NAMES = np.asarray(
    [
        "left_hand_thumb_0_joint",
        "left_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
        "left_hand_index_0_joint",
        "left_hand_index_1_joint",
        "right_hand_thumb_0_joint",
        "right_hand_thumb_1_joint",
        "right_hand_thumb_2_joint",
        "right_hand_middle_0_joint",
        "right_hand_middle_1_joint",
        "right_hand_index_0_joint",
        "right_hand_index_1_joint",
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-kinematics", type=Path, required=True)
    parser.add_argument("--contact-candidates", type=Path, required=True)
    parser.add_argument(
        "--wrist-schedule",
        type=Path,
        default=None,
        help="Optional expanded causal full-timeline wrist schedule.",
    )
    parser.add_argument("--finger-candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--finger-method", default="position")
    parser.add_argument("--transition-frames", type=int, default=15)
    return parser.parse_args()


def normalized_time(arrays: dict[str, np.ndarray], count: int, fps: float) -> np.ndarray:
    if "timestamps" in arrays:
        time = np.asarray(arrays["timestamps"], dtype=np.float64)
        if time.shape != (count,):
            raise ValueError(f"timestamp shape {time.shape} != {(count,)}")
        return time - time[0]
    return np.arange(count, dtype=np.float64) / fps


def resample_rotation(
    source_time: np.ndarray, rotations: np.ndarray, target_time: np.ndarray
) -> np.ndarray:
    clipped = np.clip(target_time, source_time[0], source_time[-1])
    if len(source_time) == 1:
        return np.repeat(rotations[:1], len(target_time), axis=0)
    return Slerp(source_time, Rotation.from_matrix(rotations))(clipped).as_matrix()


def matrix_to_wxyz(rotations: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(rotations).as_quat()
    return xyzw[:, [3, 0, 1, 2]]


def wxyz_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternions, dtype=np.float64)
    return Rotation.from_quat(value[:, [1, 2, 3, 0]]).as_matrix()


def main() -> None:
    args = parse_args()
    body_path = args.body_kinematics.expanduser().resolve()
    contact_path = args.contact_candidates.expanduser().resolve()
    finger_path = args.finger_candidates.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for path in (body_path, contact_path, finger_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(body_path, allow_pickle=False) as archive:
        body = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(contact_path, allow_pickle=False) as archive:
        contact = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(finger_path, allow_pickle=False) as archive:
        finger = {key: np.asarray(archive[key]) for key in archive.files}

    required_body = (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_names",
        "joint_names",
        "fps",
    )
    missing = [key for key in required_body if key not in body]
    if missing:
        raise KeyError(f"body kinematics missing {missing}")
    body_count = int(body["joint_pos"].shape[0])
    body_fps = float(np.asarray(body["fps"]).reshape(()))
    if abs(body_fps - 50.0) > 1e-6:
        raise ValueError(f"expected 50 Hz body kinematics, got {body_fps}")
    body_time = normalized_time(body, body_count, body_fps)
    native_time = np.asarray(contact["timestamps"], dtype=np.float64)
    native_time = native_time - native_time[0]
    finger_time = np.asarray(finger["timestamps"], dtype=np.float64)
    finger_time = finger_time - finger_time[0]
    frame_indices = np.asarray(contact["frame_indices"], dtype=np.int64)
    if not len(frame_indices) or np.any(np.diff(frame_indices) != 1):
        raise ValueError("contact frame_indices must be non-empty and contiguous")
    start_native = int(frame_indices[0])
    end_native = int(frame_indices[-1])
    interval_start = float(native_time[start_native])
    interval_end = float(native_time[end_native])
    target_mask = (body_time >= interval_start) & (body_time <= interval_end)
    target_indices = np.flatnonzero(target_mask)
    if not len(target_indices):
        raise ValueError("contact and body timelines do not overlap")
    target_time = body_time[target_indices]

    body_names = [str(value) for value in body["body_names"].tolist()]
    wrist_name = f"{args.side}_wrist_yaw_link"
    if wrist_name not in body_names:
        raise KeyError(wrist_name)
    wrist_index = body_names.index(wrist_name)
    causal_position = np.asarray(contact["causal_position"], dtype=np.float64)
    causal_rotation = np.asarray(contact["causal_rotation"], dtype=np.float64)
    if causal_position.shape != (len(native_time), 3):
        raise ValueError(causal_position.shape)
    if causal_rotation.shape != (len(native_time), 3, 3):
        raise ValueError(causal_rotation.shape)

    schedule_data = None
    if args.wrist_schedule is not None:
        with np.load(args.wrist_schedule.expanduser().resolve(), allow_pickle=False) as archive:
            schedule_data = {key: np.asarray(archive[key]) for key in archive.files}
        required_schedule = ("scheduled_position", "scheduled_rotation", "contact_frame_indices")
        missing_schedule = [key for key in required_schedule if key not in schedule_data]
        if missing_schedule:
            raise KeyError(f"wrist schedule missing {missing_schedule}")
        if schedule_data["scheduled_position"].shape != (len(native_time), 3):
            raise ValueError("scheduled_position does not cover the native timeline")
        if schedule_data["scheduled_rotation"].shape != (len(native_time), 3, 3):
            raise ValueError("scheduled_rotation does not cover the native timeline")
        interval_time = native_time
        interval_position = np.asarray(schedule_data["scheduled_position"], dtype=np.float64)
        interval_rotation = np.asarray(schedule_data["scheduled_rotation"], dtype=np.float64)
        target_mask = (body_time >= interval_time[0]) & (body_time <= interval_time[-1])
        target_indices = np.flatnonzero(target_mask)
    else:
        interval_time = native_time[start_native : end_native + 1]
        interval_position = causal_position[start_native : end_native + 1]
        interval_rotation = causal_rotation[start_native : end_native + 1]
    target_time = body_time[target_indices]
    resampled_position = np.column_stack(
        [
            np.interp(target_time, interval_time, interval_position[:, axis])
            for axis in range(3)
        ]
    )
    resampled_rotation = resample_rotation(
        interval_time, interval_rotation, target_time
    )

    motion = {key: np.asarray(value).copy() for key, value in body.items()}
    motion["body_pos_w"][target_indices, wrist_index] = resampled_position.astype(
        motion["body_pos_w"].dtype
    )
    motion["body_quat_w"][target_indices, wrist_index] = matrix_to_wxyz(
        resampled_rotation
    ).astype(motion["body_quat_w"].dtype)
    transition_count = max(0, int(args.transition_frames))
    original_position_all = np.asarray(body["body_pos_w"], dtype=np.float64)[
        :, wrist_index
    ]
    original_rotation_all = wxyz_to_matrix(
        np.asarray(body["body_quat_w"], dtype=np.float64)[:, wrist_index]
    )
    before_indices = np.arange(
        max(0, int(target_indices[0]) - transition_count),
        int(target_indices[0]),
        dtype=np.int64,
    )
    after_indices = np.arange(
        int(target_indices[-1]) + 1,
        min(body_count, int(target_indices[-1]) + 1 + transition_count),
        dtype=np.int64,
    )

    def apply_transition(
        indices: np.ndarray,
        delta_position: np.ndarray,
        delta_rotation_vector: np.ndarray,
        alphas: np.ndarray,
    ) -> None:
        if not len(indices):
            return
        position = original_position_all[indices] + alphas[:, None] * delta_position
        correction = Rotation.from_rotvec(
            alphas[:, None] * delta_rotation_vector
        ).as_matrix()
        rotation = np.einsum(
            "tij,tjk->tik", original_rotation_all[indices], correction
        )
        motion["body_pos_w"][indices, wrist_index] = position.astype(
            motion["body_pos_w"].dtype
        )
        motion["body_quat_w"][indices, wrist_index] = matrix_to_wxyz(
            rotation
        ).astype(motion["body_quat_w"].dtype)

    start_delta_position = (
        resampled_position[0] - original_position_all[target_indices[0]]
    )
    start_delta_rotation = Rotation.from_matrix(
        original_rotation_all[target_indices[0]].T @ resampled_rotation[0]
    ).as_rotvec()
    apply_transition(
        before_indices,
        start_delta_position,
        start_delta_rotation,
        np.arange(1, len(before_indices) + 1, dtype=np.float64)
        / (len(before_indices) + 1),
    )
    end_delta_position = (
        resampled_position[-1] - original_position_all[target_indices[-1]]
    )
    end_delta_rotation = Rotation.from_matrix(
        original_rotation_all[target_indices[-1]].T @ resampled_rotation[-1]
    ).as_rotvec()
    apply_transition(
        after_indices,
        end_delta_position,
        end_delta_rotation,
        1.0
        - np.arange(1, len(after_indices) + 1, dtype=np.float64)
        / (len(after_indices) + 1),
    )
    motion["contact_aware_wrist"] = np.asarray(wrist_name)
    motion["contact_aware_target_indices"] = target_indices
    motion["contact_aware_source_frames"] = frame_indices
    motion["mjlab_g1_body_names"] = np.asarray(body["body_names"])
    motion["mjlab_g1_joint_names"] = np.asarray(body["joint_names"])
    motion_path = output / "motion_targets_50hz.npz"
    np.savez_compressed(motion_path, **motion)

    root_position = np.asarray(
        body.get("root_pos_w", body["body_pos_w"][:, body_names.index("pelvis")])
    )
    root_quaternion = np.asarray(
        body.get(
            "root_quat_wxyz",
            body["body_quat_w"][:, body_names.index("pelvis")],
        )
    )
    zeros = np.zeros(body_count, dtype=np.float32)
    seed_path = output / "kinematics_seed_st.npz"
    np.savez_compressed(
        seed_path,
        sparsetrack_frame=np.arange(body_count, dtype=np.int32),
        egodex_frame=np.arange(body_count, dtype=np.int32),
        joint_pos=np.asarray(body["joint_pos"], dtype=np.float32),
        joint_vel=np.asarray(body["joint_vel"], dtype=np.float32),
        root_pos=np.asarray(root_position, dtype=np.float32),
        root_quat_wxyz=np.asarray(root_quaternion, dtype=np.float32),
        body_pos_w=np.asarray(body["body_pos_w"], dtype=np.float32),
        body_quat_w=np.asarray(body["body_quat_w"], dtype=np.float32),
        joint_names=np.asarray(body["joint_names"]),
        body_names=np.asarray(body["body_names"]),
        sparsetrack_fps=np.asarray(body_fps, dtype=np.float32),
        error_left_wrist_m=zeros,
        error_right_wrist_m=zeros.copy(),
        error_wrist_mean_m=zeros.copy(),
        error_body_pos_g=zeros.copy(),
        error_body_pos=zeros.copy(),
    )

    qpos_key = f"{args.finger_method}__qpos"
    if qpos_key not in finger:
        raise KeyError(qpos_key)
    native_qpos = np.asarray(finger[qpos_key], dtype=np.float64)
    qpos = np.column_stack(
        [
            np.interp(body_time, finger_time, native_qpos[:, joint])
            for joint in range(native_qpos.shape[1])
        ]
    )
    qvel = np.gradient(qpos, body_time, axis=0, edge_order=1)
    hand_path = output / "hand_kinematics_50hz.npz"
    np.savez_compressed(
        hand_path,
        qpos=qpos.astype(np.float32),
        qvel=qvel.astype(np.float32),
        joint_names=HAND_JOINT_NAMES,
        timestamps=body_time,
        fps=np.asarray(body_fps),
        source_method=np.asarray(args.finger_method),
    )

    original_position = np.asarray(body["body_pos_w"], dtype=np.float64)[
        target_indices, wrist_index
    ]
    original_quaternion = np.asarray(body["body_quat_w"], dtype=np.float64)[
        target_indices, wrist_index
    ]
    original_xyzw = original_quaternion[:, [1, 2, 3, 0]]
    correction_angle = (
        Rotation.from_quat(original_xyzw).inv()
        * Rotation.from_matrix(resampled_rotation)
    ).magnitude()
    manifest = {
        "schema_version": 1,
        "experiment_only": True,
        "production_pipeline_modified": False,
        "body_kinematics": str(body_path),
        "contact_candidates": str(contact_path),
        "finger_candidates": str(finger_path),
        "motion_targets_50hz": str(motion_path),
        "kinematics_seed_st": str(seed_path),
        "hand_kinematics_50hz": str(hand_path),
        "wrist": wrist_name,
        "finger_method": args.finger_method,
        "wrist_schedule": (
            str(args.wrist_schedule.expanduser().resolve())
            if args.wrist_schedule is not None else None
        ),
        "native_frame_range_inclusive": [start_native, end_native],
        "target_50hz_frame_range_inclusive": [
            int(target_indices[0]),
            int(target_indices[-1]),
        ],
        "target_frames": int(len(target_indices)),
        "transition_frames_requested": transition_count,
        "transition_50hz_frames_before": before_indices.tolist(),
        "transition_50hz_frames_after": after_indices.tolist(),
        "target_time_range_s": [float(target_time[0]), float(target_time[-1])],
        "goal_correction_from_original": {
            "position_median_m": float(
                np.median(np.linalg.norm(resampled_position - original_position, axis=1))
            ),
            "position_p95_m": float(
                np.percentile(
                    np.linalg.norm(resampled_position - original_position, axis=1), 95
                )
            ),
            "orientation_median_deg": float(np.degrees(np.median(correction_angle))),
            "orientation_p95_deg": float(
                np.degrees(np.percentile(correction_angle, 95))
            ),
        },
    }
    (output / "target_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
