"""Export SIL clips plus source kinematics into an Isaac-GR00T SONIC LeRobot dataset."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

BODY_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
LEFT_HAND_STATE_NAMES = (
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
)
RIGHT_HAND_STATE_NAMES = (
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
)
STATE_NAMES = (
    *BODY_JOINTS[:22],
    *LEFT_HAND_STATE_NAMES,
    *BODY_JOINTS[22:],
    *RIGHT_HAND_STATE_NAMES,
)
HAND_STATE_NAMES = LEFT_HAND_STATE_NAMES + RIGHT_HAND_STATE_NAMES
TELEOP_RIGHT_HAND_NAMES = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
)
REFERENCE_RIGHT_HAND_ACTION = np.asarray(
    [0.0, -0.7, -0.7, 1.0, 1.5, 1.0, 1.5], dtype=np.float32
)
HAND_PROFILE_DEX3 = "dex3"
HAND_PROFILE_BRAINCO = "brainco"
BRAINCO_HAND_JOINT_NAMES = (
    "left_thumb_metacarpal_joint",
    "left_thumb_proximal_joint",
    "left_index_proximal_joint",
    "left_middle_proximal_joint",
    "left_ring_proximal_joint",
    "left_pinky_proximal_joint",
    "right_thumb_metacarpal_joint",
    "right_thumb_proximal_joint",
    "right_index_proximal_joint",
    "right_middle_proximal_joint",
    "right_ring_proximal_joint",
    "right_pinky_proximal_joint",
)
BRAINCO_STATE_HAND_NAMES = (
    "left_Thumb",
    "left_Thumb_aux",
    "left_Index",
    "left_Middle",
    "left_Ring",
    "left_Pinky",
    "right_Thumb",
    "right_Thumb_aux",
    "right_Index",
    "right_Middle",
    "right_Ring",
    "right_Pinky",
)
BRAINCO_STATE_NAMES = BODY_JOINTS + BRAINCO_STATE_HAND_NAMES
BRAINCO_HAND_JOINT_RANGE = np.asarray(
    [1.5184, 1.0472, 1.4661, 1.4661, 1.4661, 1.4661] * 2,
    dtype=np.float64,
)
BRAINCO_OPEN_HAND = np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
BRAINCO_CLOSE_HAND = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
OPTIMIZED_KINEMATICS_FILE = "optimized_kinematics.npz"
SOURCE_KINEMATICS_RELATIVE = Path("final_50hz") / "kinematics.npz"
SOURCE_VIDEO_RELATIVE = Path("final_50hz") / "rgb_final_50hz.mp4"
VIDEO_KEY = "observation.images.ego_view"
FPS = 50
CHUNKS_SIZE = 1000
GRAVITY_WORLD = np.array([0.0, 0.0, -1.0], dtype=np.float64)


@dataclass(frozen=True)
class ClipExportPlan:
    session_id: str
    episode_index: int
    source_kinematics: str
    source_video: str
    sil_kinematics: str
    output_parquet: str
    output_video: str
    action: str


def compute_projected_gravity(base_quat: np.ndarray) -> np.ndarray:
    """Project world gravity ``[0, 0, -1]`` into the body frame.

    Matches GR00T-WholeBodyControl ``compute_projected_gravity``:
    ``Rotation.from_quat(wxyz, scalar_first=True).inv().apply([0, 0, -1])``.
    Accepts one quaternion ``(4,)`` or a batch ``(N, 4)``.
    """

    quat = np.asarray(base_quat, dtype=np.float64)
    single = quat.ndim == 1
    if single:
        quat = quat.reshape(1, 4)
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError(f"base_quat must have shape (4,) or (N, 4), got {quat.shape}")
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    if np.any(norms < 1.0e-12):
        raise ValueError("base quaternion must be non-zero")
    quat = quat / norms
    rotation = _wxyz_to_rotation_matrix(quat)
    projected = np.einsum("nji,j->ni", rotation, GRAVITY_WORLD)
    if single:
        return projected[0].astype(np.float32)
    return projected.astype(np.float32)


def _wxyz_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(np.asarray(quat, dtype=np.float64), -1, 0)
    result = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    result[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    result[..., 0, 1] = 2.0 * (x * y - w * z)
    result[..., 0, 2] = 2.0 * (x * z + w * y)
    result[..., 1, 0] = 2.0 * (x * y + w * z)
    result[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    result[..., 1, 2] = 2.0 * (y * z - w * x)
    result[..., 2, 0] = 2.0 * (x * z - w * y)
    result[..., 2, 1] = 2.0 * (y * z + w * x)
    result[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return result


def _string_names(values: object, field: str) -> tuple[str, ...]:
    names = tuple(str(value) for value in np.asarray(values).reshape(-1).tolist())
    if not names or any(not name for name in names):
        raise ValueError(f"{field} must contain non-empty names")
    if len(set(names)) != len(names):
        raise ValueError(f"{field} must be unique")
    return names


def remap_named(
    values: np.ndarray, source_names: Sequence[str], target_names: Sequence[str], *, field: str
) -> np.ndarray:
    source = tuple(source_names)
    target = tuple(target_names)
    if set(source) != set(target) or len(source) != len(target):
        missing = sorted(set(target).difference(source))
        extra = sorted(set(source).difference(target))
        raise ValueError(f"{field} names do not match; missing={missing} extra={extra}")
    order = [source.index(name) for name in target]
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != len(source):
        raise ValueError(f"{field} must have shape (T, {len(source)}), got {array.shape}")
    return np.ascontiguousarray(array[:, order])


def motion_token_on_fsq_grid(token: np.ndarray) -> bool:
    values = np.asarray(token, dtype=np.float64)
    return bool(np.allclose(values * 16.0, np.round(values * 16.0), rtol=0.0, atol=1.0e-5))


def resolve_hand_profile(hand_names: Sequence[str]) -> str:
    names = tuple(hand_names)
    if set(names) == set(HAND_STATE_NAMES) and len(names) == len(HAND_STATE_NAMES):
        return HAND_PROFILE_DEX3
    if set(names) == set(BRAINCO_HAND_JOINT_NAMES) and len(names) == len(
        BRAINCO_HAND_JOINT_NAMES
    ):
        return HAND_PROFILE_BRAINCO
    raise ValueError(
        "hand_joint_names must be Dex3 14-DoF or BrainCo 12-DoF names; "
        f"got {list(names)}"
    )


def state_names_for_profile(hand_profile: str) -> tuple[str, ...]:
    if hand_profile == HAND_PROFILE_BRAINCO:
        return BRAINCO_STATE_NAMES
    if hand_profile == HAND_PROFILE_DEX3:
        return STATE_NAMES
    raise ValueError(f"unknown hand profile: {hand_profile}")


def _binary_contact(values: object, rows: int, *, field: str) -> np.ndarray:
    contact = np.asarray(values, dtype=np.int8).reshape(-1)
    if contact.shape[0] < rows or not np.all(np.isin(contact[:rows], (0, 1))):
        raise ValueError(f"{field}[:T] must be a binary vector of length T")
    return contact[:rows].copy()


def _time_vector(source: Mapping[str, np.ndarray], key: str, length: int) -> np.ndarray:
    if key in source:
        stamps = np.asarray(source[key], dtype=np.float64).reshape(-1)
        if stamps.shape != (length,):
            raise ValueError(f"{key} must have shape ({length},), got {stamps.shape}")
        return stamps
    return np.arange(length, dtype=np.float64) / float(FPS)


def _upsample_contact(
    signal: np.ndarray, source_time: np.ndarray, target_time: np.ndarray
) -> np.ndarray:
    if signal.shape != source_time.shape:
        raise ValueError(
            f"contact signal length {signal.shape} does not match timestamps "
            f"{source_time.shape}"
        )
    right = np.searchsorted(source_time, target_time, side="left")
    right = np.clip(right, 0, len(source_time) - 1)
    left = np.clip(right - 1, 0, len(source_time) - 1)
    use_right = np.abs(source_time[right] - target_time) < np.abs(
        target_time - source_time[left]
    )
    nearest = np.where(use_right, right, left)
    return np.asarray(signal, dtype=np.int8)[nearest]


def _per_hand_contact(
    source: Mapping[str, np.ndarray], side: str, rows: int
) -> np.ndarray:
    key_50 = f"{side}_contact"
    key_30 = f"{side}_contact_30hz"
    source_frames = int(np.asarray(source["joint_pos"]).shape[0])
    if key_50 in source:
        raw_50 = np.asarray(source[key_50]).reshape(-1)
        if raw_50.shape[0] != source_frames:
            raise ValueError(
                f"{key_50} must have length {source_frames}, got {raw_50.shape[0]}"
            )
        contact_50 = _binary_contact(raw_50, rows, field=key_50)
        if np.any(contact_50 == 1) or key_30 not in source:
            return contact_50
    if key_30 not in source:
        raise ValueError(f"source kinematics is missing {key_50} and {key_30}")
    contact_30 = np.asarray(source[key_30], dtype=np.int8).reshape(-1)
    if contact_30.size < 1 or not np.all(np.isin(contact_30, (0, 1))):
        raise ValueError(f"{key_30} must be a non-empty binary vector")
    source_time = _time_vector(
        source, "contact_30hz_timestamps", int(contact_30.shape[0])
    )
    target_time = _time_vector(source, "timestamps", source_frames)
    upsampled = _upsample_contact(contact_30, source_time, target_time)
    return _binary_contact(upsampled, rows, field=key_30)


def _contact_targets(
    contact: np.ndarray, open_pose: np.ndarray, close_pose: np.ndarray
) -> np.ndarray:
    mask = np.asarray(contact, dtype=np.float32).reshape(-1, 1)
    return np.where(mask == 1.0, close_pose, open_pose).astype(np.float32)


def pack_observation_state(
    *,
    body: np.ndarray,
    body_names: Sequence[str],
    hand: np.ndarray,
    hand_names: Sequence[str],
) -> np.ndarray:
    profile = resolve_hand_profile(hand_names)
    body_ordered = remap_named(body, body_names, BODY_JOINTS, field="body joints")
    frames = body_ordered.shape[0]
    if profile == HAND_PROFILE_BRAINCO:
        hand_ordered = remap_named(
            hand, hand_names, BRAINCO_HAND_JOINT_NAMES, field="hand joints"
        )
        hand_values = np.clip(
            np.asarray(hand_ordered, dtype=np.float64) / BRAINCO_HAND_JOINT_RANGE,
            0.0,
            1.0,
        )
        target_names = BRAINCO_STATE_NAMES
        hand_columns = BRAINCO_STATE_HAND_NAMES
    else:
        hand_values = remap_named(hand, hand_names, HAND_STATE_NAMES, field="hand joints")
        target_names = STATE_NAMES
        hand_columns = HAND_STATE_NAMES
    state = np.zeros((frames, len(target_names)), dtype=np.float64)
    index = {name: i for i, name in enumerate(target_names)}
    for column, name in enumerate(BODY_JOINTS):
        state[:, index[name]] = body_ordered[:, column]
    for column, name in enumerate(hand_columns):
        state[:, index[name]] = hand_values[:, column]
    return state


def assemble_episode_arrays(
    source: Mapping[str, np.ndarray],
    kinematics: Mapping[str, np.ndarray],
    *,
    hand_profile: str | None = None,
) -> dict[str, np.ndarray]:
    """Build Isaac-GR00T SONIC columns for one clip."""

    source_frames = int(np.asarray(source["joint_pos"]).shape[0])
    token = np.asarray(kinematics["motion_token"], dtype=np.float64)
    rows = int(token.shape[0])
    if token.ndim != 2 or token.shape[1] != 64:
        raise ValueError(f"motion_token must have shape (T, 64), got {token.shape}")
    if source_frames != rows + 1:
        raise ValueError(
            f"source must have T+1 frames and SIL kinematics T frames, "
            f"got source={source_frames} sil={rows}"
        )
    if not motion_token_on_fsq_grid(token):
        raise ValueError("motion_token is not on the official FSQ 1/16 grid")

    body_names = _string_names(kinematics["joint_names"], "joint_names")
    source_hand_names = _string_names(source["hand_joint_names"], "hand_joint_names")
    detected = resolve_hand_profile(source_hand_names)
    if hand_profile is not None and detected != hand_profile:
        raise ValueError(
            f"clip hand profile {detected} does not match dataset {hand_profile}"
        )
    profile = detected
    source_hand = np.asarray(source["hand_qpos"][:rows], dtype=np.float64)

    state = pack_observation_state(
        body=np.asarray(kinematics["joint_pos"], dtype=np.float64),
        body_names=body_names,
        hand=source_hand,
        hand_names=source_hand_names,
    )
    gravity = compute_projected_gravity(
        np.asarray(kinematics["root_quat_wxyz"], dtype=np.float64)
    ).astype(np.float64)
    if profile == HAND_PROFILE_BRAINCO:
        left_contact = _per_hand_contact(source, "left", rows)
        right_contact = _per_hand_contact(source, "right", rows)
        left_hand = _contact_targets(left_contact, BRAINCO_OPEN_HAND, BRAINCO_CLOSE_HAND)
        right_hand = _contact_targets(
            right_contact, BRAINCO_OPEN_HAND, BRAINCO_CLOSE_HAND
        )
    else:
        if "contact_50hz" not in source:
            raise ValueError("source kinematics is missing contact_50hz")
        contact = _binary_contact(source["contact_50hz"], rows, field="contact_50hz")
        right_hand = (
            np.asarray(contact, dtype=np.float32)[:, None] * REFERENCE_RIGHT_HAND_ACTION
        )
        left_hand = np.zeros((rows, 7), dtype=np.float32)
    return {
        "observation.state": state,
        "observation.projected_gravity": gravity,
        "action.motion_token": token,
        "teleop.left_hand_joints": left_hand,
        "teleop.right_hand_joints": right_hand.astype(np.float32),
        "timestamp": np.arange(rows, dtype=np.float64) * (1.0 / FPS),
        "frame_index": np.arange(rows, dtype=np.int64),
        "task_index": np.zeros(rows, dtype=np.int64),
    }


def modality_config(hand_profile: str = HAND_PROFILE_DEX3) -> dict[str, Any]:
    if hand_profile == HAND_PROFILE_BRAINCO:
        state = {
            "left_leg": {"start": 0, "end": 6},
            "right_leg": {"start": 6, "end": 12},
            "waist": {"start": 12, "end": 15},
            "left_arm": {"start": 15, "end": 22},
            "right_arm": {"start": 22, "end": 29},
            "left_hand": {"start": 29, "end": 35},
            "right_hand": {"start": 35, "end": 41},
            "projected_gravity": {
                "start": 0,
                "end": 3,
                "original_key": "observation.projected_gravity",
            },
        }
        hand_dim = 6
    elif hand_profile == HAND_PROFILE_DEX3:
        state = {
            "left_leg": {"start": 0, "end": 6},
            "right_leg": {"start": 6, "end": 12},
            "waist": {"start": 12, "end": 15},
            "left_arm": {"start": 15, "end": 22},
            "left_hand": {"start": 22, "end": 29},
            "right_arm": {"start": 29, "end": 36},
            "right_hand": {"start": 36, "end": 43},
            "projected_gravity": {
                "start": 0,
                "end": 3,
                "original_key": "observation.projected_gravity",
            },
        }
        hand_dim = 7
    else:
        raise ValueError(f"unknown hand profile: {hand_profile}")
    return {
        "state": state,
        "action": {
            "motion_token": {
                "start": 0,
                "end": 64,
                "original_key": "action.motion_token",
            },
            "left_hand_joints": {
                "start": 0,
                "end": hand_dim,
                "original_key": "teleop.left_hand_joints",
            },
            "right_hand_joints": {
                "start": 0,
                "end": hand_dim,
                "original_key": "teleop.right_hand_joints",
            },
        },
        "video": {"ego_view": {"original_key": VIDEO_KEY}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def dataset_features(hand_profile: str = HAND_PROFILE_DEX3) -> dict[str, Any]:
    names = state_names_for_profile(hand_profile)
    hand_dim = 6 if hand_profile == HAND_PROFILE_BRAINCO else 7
    return {
        VIDEO_KEY: {
            "dtype": "video",
            "shape": [376, 672, 3],
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float64",
            "shape": [len(names)],
            "names": list(names),
        },
        "observation.projected_gravity": {
            "dtype": "float64",
            "shape": [3],
            "names": ["gravity_x", "gravity_y", "gravity_z"],
        },
        "action.motion_token": {
            "dtype": "float64",
            "shape": [64],
            "names": "motion_token",
        },
        "teleop.left_hand_joints": {
            "dtype": "float32",
            "shape": [hand_dim],
            "names": "left_hand_joints",
        },
        "teleop.right_hand_joints": {
            "dtype": "float32",
            "shape": [hand_dim],
            "names": "right_hand_joints",
        },
        "timestamp": {"dtype": "float64", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }


def list_sil_sessions(clips_root: str | Path) -> tuple[str, ...]:
    root = Path(clips_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"clips root does not exist: {root}")
    sessions: list[str] = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / OPTIMIZED_KINEMATICS_FILE).is_file():
            sessions.append(path.name)
    return tuple(sessions)


def episode_paths(output_root: Path, episode_index: int) -> tuple[Path, Path]:
    chunk = episode_index // CHUNKS_SIZE
    parquet = (
        output_root
        / "data"
        / f"chunk-{chunk:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )
    video = (
        output_root
        / "videos"
        / f"chunk-{chunk:03d}"
        / VIDEO_KEY
        / f"episode_{episode_index:06d}.mp4"
    )
    return parquet, video


def episode_is_complete(output_root: Path, episode_index: int) -> bool:
    parquet, video = episode_paths(output_root, episode_index)
    return parquet.is_file() and parquet.stat().st_size > 0 and video.is_file() and video.stat().st_size > 0


def plan_groot_export(
    *,
    source_root: str | Path,
    clips_root: str | Path,
    output_root: str | Path,
    skip_existing: bool = True,
) -> tuple[ClipExportPlan, ...]:
    source = Path(source_root).expanduser().resolve()
    clips = Path(clips_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    plans: list[ClipExportPlan] = []
    for episode_index, session_id in enumerate(list_sil_sessions(clips)):
        parquet, video = episode_paths(output, episode_index)
        action = "skip" if skip_existing and episode_is_complete(output, episode_index) else "export"
        plans.append(
            ClipExportPlan(
                session_id=session_id,
                episode_index=episode_index,
                source_kinematics=str(source / session_id / SOURCE_KINEMATICS_RELATIVE),
                source_video=str(source / session_id / SOURCE_VIDEO_RELATIVE),
                sil_kinematics=str(clips / session_id / OPTIMIZED_KINEMATICS_FILE),
                output_parquet=str(parquet),
                output_video=str(video),
                action=action,
            )
        )
    return tuple(plans)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )


def trim_video(source_video: Path, destination: Path, frames: int) -> tuple[int, int]:
    """Copy the first ``frames`` of ``source_video`` into ``destination``."""

    import cv2

    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open source video: {source_video}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width < 1 or height < 1:
            raise ValueError(f"invalid source video size {width}x{height}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(destination),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(FPS),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot open video writer: {destination}")
        written = 0
        try:
            while written < frames:
                ok, frame = capture.read()
                if not ok:
                    break
                writer.write(frame)
                written += 1
        finally:
            writer.release()
    finally:
        capture.release()
    if written != frames:
        if destination.exists():
            destination.unlink()
        raise ValueError(
            f"source video {source_video} has {written} frames, need {frames}"
        )
    return height, width


def _write_parquet(path: Path, columns: Mapping[str, np.ndarray]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    fields = []
    arrays = []
    for name, values in columns.items():
        array = np.asarray(values)
        if array.ndim == 2:
            fields.append(pa.field(name, pa.list_(pa.from_numpy_dtype(array.dtype))))
            arrays.append(pa.array(array.tolist(), type=fields[-1].type))
        elif np.issubdtype(array.dtype, np.floating):
            fields.append(pa.field(name, pa.float64()))
            arrays.append(pa.array(array.astype(np.float64), type=pa.float64()))
        else:
            fields.append(pa.field(name, pa.int64()))
            arrays.append(pa.array(array.astype(np.int64), type=pa.int64()))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_arrays(arrays, schema=pa.schema(fields)), path)


def export_one_clip(
    plan: ClipExportPlan,
    *,
    global_index: int,
    task_index: int = 0,
    hand_profile: str | None = None,
) -> int:
    source = _load_npz(Path(plan.source_kinematics))
    kinematics = _load_npz(Path(plan.sil_kinematics))
    arrays = assemble_episode_arrays(source, kinematics, hand_profile=hand_profile)
    rows = int(arrays["observation.state"].shape[0])
    arrays["episode_index"] = np.full(rows, plan.episode_index, dtype=np.int64)
    arrays["index"] = np.arange(global_index, global_index + rows, dtype=np.int64)
    arrays["task_index"] = np.full(rows, task_index, dtype=np.int64)
    _write_parquet(Path(plan.output_parquet), arrays)
    trim_video(Path(plan.source_video), Path(plan.output_video), rows)
    return rows


def _write_dataset_metadata(
    output: Path,
    *,
    task: str,
    completed: Sequence[tuple[ClipExportPlan, int]],
    hand_profile: str = HAND_PROFILE_DEX3,
) -> None:
    total_frames = sum(rows for _plan, rows in completed)
    total_episodes = len(completed)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "g1",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_episodes,
        "total_chunks": 1 if total_episodes == 0 else (max(plan.episode_index for plan, _ in completed) // CHUNKS_SIZE) + 1,
        "chunks_size": CHUNKS_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": dataset_features(hand_profile),
    }
    _write_json(output / "meta" / "info.json", info)
    _write_json(output / "meta" / "modality.json", modality_config(hand_profile))
    _write_jsonl(
        output / "meta" / "tasks.jsonl",
        [{"task_index": 0, "task": task}],
    )
    _write_jsonl(
        output / "meta" / "episodes.jsonl",
        [
            {
                "episode_index": plan.episode_index,
                "tasks": [task],
                "length": rows,
                "session_id": plan.session_id,
            }
            for plan, rows in completed
        ],
    )


def _infer_hand_profile(plans: Sequence[ClipExportPlan]) -> str:
    for plan in plans:
        path = Path(plan.source_kinematics)
        if not path.is_file():
            continue
        payload = _load_npz(path)
        return resolve_hand_profile(
            _string_names(payload["hand_joint_names"], "hand_joint_names")
        )
    return HAND_PROFILE_DEX3


def export_groot_dataset(
    *,
    source_root: str | Path,
    clips_root: str | Path,
    output_root: str | Path,
    task: str,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> dict[str, Any]:
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")
    output = Path(output_root).expanduser().resolve()
    plans = plan_groot_export(
        source_root=source_root,
        clips_root=clips_root,
        output_root=output,
        skip_existing=skip_existing,
    )
    hand_profile = _infer_hand_profile(plans)
    summary: dict[str, Any] = {
        "source_root": str(Path(source_root).expanduser().resolve()),
        "clips_root": str(Path(clips_root).expanduser().resolve()),
        "output_root": str(output),
        "task": task,
        "hand_profile": hand_profile,
        "embodiment_tag": (
            "UNITREE_G1_SONIC_BRAINCO"
            if hand_profile == HAND_PROFILE_BRAINCO
            else "UNITREE_G1_SONIC"
        ),
        "dry_run": bool(dry_run),
        "assignments": [asdict(item) for item in plans],
        "num_export": sum(1 for item in plans if item.action == "export"),
        "num_skip": sum(1 for item in plans if item.action == "skip"),
        "failures": [],
        "completed": [],
    }
    if dry_run:
        _write_json(output / "export_summary.json", summary)
        return summary

    output.mkdir(parents=True, exist_ok=True)
    completed: list[tuple[ClipExportPlan, int]] = []
    global_index = 0
    for plan in plans:
        if plan.action == "skip":
            try:
                import pyarrow.parquet as pq

                table = pq.read_table(plan.output_parquet, columns=["index"])
                rows = table.num_rows
                if rows:
                    last = int(table.column("index")[rows - 1].as_py())
                    global_index = max(global_index, last + 1)
            except Exception:
                rows = 0
            completed.append((plan, rows))
            continue
        try:
            rows = export_one_clip(
                plan, global_index=global_index, hand_profile=hand_profile
            )
        except Exception as exc:
            summary["failures"].append(
                {
                    "session_id": plan.session_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        global_index += rows
        completed.append((plan, rows))
        summary["completed"].append(
            {
                "session_id": plan.session_id,
                "episode_index": plan.episode_index,
                "rows": rows,
            }
        )

    _write_dataset_metadata(
        output, task=task, completed=completed, hand_profile=hand_profile
    )
    summary["success"] = not summary["failures"]
    summary["num_completed"] = len(summary["completed"]) + summary["num_skip"]
    _write_json(output / "export_summary.json", summary)
    return summary
