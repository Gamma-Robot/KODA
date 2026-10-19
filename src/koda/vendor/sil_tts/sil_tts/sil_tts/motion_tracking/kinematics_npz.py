"""Convert ego2humanoid ``kinematics.npz`` into Any4HDMI plus a hand sidecar."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from sil_tts.motion_tracking.any4hdmi import _ROOT_QPOS_NAMES, resolve_any4hdmi_motion

_HAND_SIDECAR_SUFFIX = ".hands.npz"
_HAND_FIELDS = ("hand_qpos", "hand_qvel", "hand_joint_names")
_SESSION_PREFIX = "session_"
_KINEMATICS_RELATIVE = Path("final_50hz") / "kinematics.npz"
CLIP_DURATIONS_FILE = "clip_durations.json"
INGEST_FAILURES_FILE = "ingest_failures.json"
INGEST_SUMMARY_FILE = "ingest_summary.json"
DEFAULT_KINEMATICS_FPS = 50.0
DEFAULT_ROBOT_RELATIVE = Path("configs") / "robots" / "g1_mjwarp_43dof_dex3.yaml"


def _string_names(values: object, field: str) -> tuple[str, ...]:
    names = tuple(str(value) for value in np.asarray(values).reshape(-1).tolist())
    if not names or any(not name for name in names):
        raise ValueError(f"{field} must contain non-empty names")
    if len(set(names)) != len(names):
        raise ValueError(f"{field} must be unique")
    return names


def convert_kinematics_npz(path: str | Path) -> dict[str, Any]:
    """Turn one 50 Hz kinematics archive into Any4HDMI qpos plus hand arrays."""

    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as payload:
        required = {
            "joint_pos",
            "joint_vel",
            "joint_names",
            "root_pos_w",
            "root_quat_wxyz",
            "hand_qpos",
            "hand_qvel",
            "hand_joint_names",
            "fps",
        }
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"kinematics archive is missing fields: {missing}")
        fps = np.asarray(payload["fps"])
        if fps.shape != () or int(fps) != 50:
            raise ValueError(f"kinematics fps must be scalar 50, got {fps}")
        joint_names = _string_names(payload["joint_names"], "joint_names")
        hand_names = _string_names(payload["hand_joint_names"], "hand_joint_names")
        joint_pos = np.asarray(payload["joint_pos"], dtype=np.float64)
        root_pos = np.asarray(payload["root_pos_w"], dtype=np.float64)
        root_quat = np.asarray(payload["root_quat_wxyz"], dtype=np.float64)
        hand_qpos = np.asarray(payload["hand_qpos"], dtype=np.float64)
        hand_qvel = np.asarray(payload["hand_qvel"], dtype=np.float64)

    frames = joint_pos.shape[0]
    if frames < 2:
        raise ValueError("kinematics archive must contain at least two frames")
    if joint_pos.shape != (frames, len(joint_names)):
        raise ValueError("joint_pos shape does not match joint_names")
    if root_pos.shape != (frames, 3) or root_quat.shape != (frames, 4):
        raise ValueError("root trajectory shapes do not match joint_pos")
    if hand_qpos.shape != (frames, len(hand_names)) or hand_qvel.shape != (
        frames,
        len(hand_names),
    ):
        raise ValueError("hand trajectory shapes do not match hand_joint_names")
    if not np.isfinite(joint_pos).all() or not np.isfinite(root_pos).all():
        raise ValueError("body kinematics must be finite")
    if not np.isfinite(root_quat).all() or not np.isfinite(hand_qpos).all():
        raise ValueError("orientation and hand kinematics must be finite")
    if not np.isfinite(hand_qvel).all():
        raise ValueError("hand_qvel must be finite")
    norms = np.linalg.norm(root_quat, axis=1)
    if np.any(norms < 1.0e-8):
        raise ValueError("root quaternions must be non-zero")
    root_quat = root_quat / norms[:, None]

    qpos = np.concatenate((root_pos, root_quat, joint_pos), axis=1)
    qpos_names = list(_ROOT_QPOS_NAMES + joint_names)
    return {
        "timestep": 0.02,
        "qpos": qpos,
        "qpos_names": qpos_names,
        "hand_qpos": hand_qpos,
        "hand_qvel": hand_qvel,
        "hand_joint_names": hand_names,
    }


def clip_duration_seconds(num_frames: int, *, fps: float = DEFAULT_KINEMATICS_FPS) -> float:
    """Return ``(T-1)/fps`` so duration lands on the source frame grid."""

    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames < 2:
        raise ValueError("clip duration requires at least two frames")
    rate = float(fps)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("fps must be finite and positive")
    return (num_frames - 1) / rate


def remap_converted_kinematics(
    converted: Mapping[str, Any],
    *,
    joint_names: Sequence[str],
    hand_joint_names: Sequence[str],
) -> dict[str, Any]:
    """Reorder body qpos and hand sidecars onto a shared canonical name list."""

    qpos = np.asarray(converted["qpos"], dtype=np.float64)
    qpos_names = _string_names(converted["qpos_names"], "qpos_names")
    if qpos.ndim != 2 or qpos.shape[1] != len(qpos_names):
        raise ValueError("converted qpos does not match qpos_names")
    if qpos_names[:7] != _ROOT_QPOS_NAMES:
        raise ValueError("converted qpos_names must begin with canonical free-root names")
    target_joints = _string_names(joint_names, "joint_names")
    source_joints = qpos_names[7:]
    joint_order = _name_permutation(source_joints, target_joints, "body joints")
    remapped_qpos = np.concatenate((qpos[:, :7], qpos[:, 7:][:, joint_order]), axis=1)

    source_hands = _string_names(converted["hand_joint_names"], "hand_joint_names")
    target_hands = _string_names(hand_joint_names, "hand_joint_names")
    hand_order = _name_permutation(source_hands, target_hands, "hand joints")
    return {
        "timestep": float(converted["timestep"]),
        "qpos": np.ascontiguousarray(remapped_qpos),
        "qpos_names": list(_ROOT_QPOS_NAMES + target_joints),
        "hand_qpos": np.ascontiguousarray(
            np.asarray(converted["hand_qpos"], dtype=np.float64)[:, hand_order]
        ),
        "hand_qvel": np.ascontiguousarray(
            np.asarray(converted["hand_qvel"], dtype=np.float64)[:, hand_order]
        ),
        "hand_joint_names": target_hands,
    }


def iter_session_kinematics(source_root: str | Path) -> tuple[tuple[str, Path], ...]:
    """Enumerate ``session_*/final_50hz/kinematics.npz`` under any source tree."""

    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"kinematics source root does not exist: {root}")
    sessions: list[tuple[str, Path]] = []
    for session_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if not session_dir.name.startswith(_SESSION_PREFIX):
            continue
        archive = session_dir / _KINEMATICS_RELATIVE
        if archive.is_file():
            sessions.append((session_dir.name, archive))
    return tuple(sessions)


def load_robot_joint_layout(robot_path: str | Path) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """Read command joints, Dex3 names, and MJCF entrypoint from a robot YAML."""

    path = Path(robot_path).expanduser().resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"robot config must be a mapping: {path}")
    joints = _string_names(payload.get("joint_names", ()), "joint_names")
    hands = _string_names(payload.get("kinematic_joint_names", ()), "kinematic_joint_names")
    mjcf = payload.get("entrypoint")
    if not isinstance(mjcf, str) or not mjcf:
        raise ValueError("robot.entrypoint must be a non-empty string")
    return joints, hands, mjcf


def write_any4hdmi_session(
    converted: dict[str, Any],
    dataset_root: str | Path,
    *,
    motion_id: str,
    dataset_name: str,
    mjcf: str,
) -> tuple[Path, Path, Path]:
    """Write Any4HDMI v2 files and a sibling hand sidecar, appending if needed."""

    root = Path(dataset_root)
    motions = root / "motions"
    motions.mkdir(parents=True, exist_ok=True)
    qpos = np.asarray(converted["qpos"], dtype=np.float64)
    qpos_names = list(converted["qpos_names"])
    timestep = float(converted["timestep"])
    manifest = _compatible_manifest(
        root,
        dataset_name=dataset_name,
        mjcf=mjcf,
        timestep=timestep,
        qpos_names=qpos_names,
        qpos_dim=int(qpos.shape[1]),
    )
    motion_path = motions / f"{motion_id}.npz"
    np.savez(motion_path, qpos=qpos)
    hands_path = motions / f"{motion_id}{_HAND_SIDECAR_SUFFIX}"
    np.savez(
        hands_path,
        hand_qpos=np.asarray(converted["hand_qpos"], dtype=np.float64),
        hand_qvel=np.asarray(converted["hand_qvel"], dtype=np.float64),
        hand_joint_names=np.asarray(converted["hand_joint_names"]),
    )
    manifest["num_motions"] = len(_motion_ids_on_disk(motions))
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path, motion_path, hands_path


def ingest_session_kinematics(
    source_root: str | Path,
    output_root: str | Path,
    *,
    joint_names: Sequence[str],
    hand_joint_names: Sequence[str],
    dataset_name: str,
    mjcf: str,
) -> dict[str, Any]:
    """Convert every complete session clip under ``source_root`` into one dataset."""

    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    sessions = iter_session_kinematics(source)
    durations: dict[str, float] = {}
    ingested: list[str] = []
    failures: list[dict[str, str]] = []
    for session_id, archive in sessions:
        try:
            converted = remap_converted_kinematics(
                convert_kinematics_npz(archive),
                joint_names=joint_names,
                hand_joint_names=hand_joint_names,
            )
            frames = int(np.asarray(converted["qpos"]).shape[0])
            duration = clip_duration_seconds(frames)
            write_any4hdmi_session(
                converted,
                output,
                motion_id=session_id,
                dataset_name=dataset_name,
                mjcf=mjcf,
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            failures.append(
                {
                    "session_id": session_id,
                    "path": str(archive),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        durations[session_id] = duration
        ingested.append(session_id)

    _write_json(output / CLIP_DURATIONS_FILE, durations)
    _write_json(output / INGEST_FAILURES_FILE, failures)
    summary = {
        "source_root": str(source),
        "output_root": str(output),
        "dataset_name": dataset_name,
        "mjcf": mjcf,
        "num_sessions_found": len(sessions),
        "num_ingested": len(ingested),
        "num_failures": len(failures),
        "motion_ids": ingested,
        "qpos_names": list(_ROOT_QPOS_NAMES + tuple(joint_names)),
        "hand_joint_names": list(hand_joint_names),
    }
    _write_json(output / INGEST_SUMMARY_FILE, summary)
    return summary


def load_clip_durations(dataset_root: str | Path) -> dict[str, float]:
    """Load per-motion clip durations written by session ingest."""

    path = Path(dataset_root) / CLIP_DURATIONS_FILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{CLIP_DURATIONS_FILE} must be a JSON object")
    durations: dict[str, float] = {}
    for motion_id, value in payload.items():
        if not isinstance(motion_id, str) or not motion_id:
            raise ValueError("clip duration keys must be non-empty motion ids")
        try:
            duration = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid clip duration for {motion_id!r}") from exc
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError(f"clip duration for {motion_id!r} must be finite and positive")
        durations[motion_id] = duration
    return durations


def _name_permutation(
    source_names: Sequence[str],
    target_names: Sequence[str],
    label: str,
) -> np.ndarray:
    source = tuple(source_names)
    target = tuple(target_names)
    if set(source) != set(target) or len(source) != len(target):
        missing = sorted(set(target).difference(source))
        extra = sorted(set(source).difference(target))
        raise ValueError(f"{label} do not match canonical names; missing={missing} extra={extra}")
    return np.asarray([source.index(name) for name in target], dtype=np.int64)


def _motion_ids_on_disk(motions_dir: Path) -> tuple[str, ...]:
    ids: list[str] = []
    for path in sorted(motions_dir.glob("*.npz")):
        if path.name.endswith(_HAND_SIDECAR_SUFFIX):
            continue
        ids.append(path.stem)
    return tuple(ids)


def _compatible_manifest(
    root: Path,
    *,
    dataset_name: str,
    mjcf: str,
    timestep: float,
    qpos_names: Sequence[str],
    qpos_dim: int,
) -> dict[str, Any]:
    names = list(qpos_names)
    expected = {
        "format_version": 2,
        "dataset_name": dataset_name,
        "mjcf": mjcf,
        "motions_subdir": "motions",
        "timestep": timestep,
        "qpos_dim": qpos_dim,
        "qpos_names": names,
    }
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return {**expected, "num_motions": 0}
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(existing, dict):
        raise ValueError("existing Any4HDMI manifest must be a JSON object")
    for field, value in expected.items():
        if existing.get(field) != value:
            raise ValueError(
                f"cannot append to dataset {root}: manifest {field} mismatch "
                f"(existing={existing.get(field)!r}, new={value!r})"
            )
    return {**existing, **expected}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_hand_sidecar(
    dataset_root: str | Path,
    motion_id: str,
    *,
    joint_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Load the hand sidecar next to an Any4HDMI motion, remapped by name."""

    motion_path = resolve_any4hdmi_motion(dataset_root, motion_id)
    sidecar = motion_path.with_name(motion_path.stem + _HAND_SIDECAR_SUFFIX)
    if not sidecar.is_file():
        raise FileNotFoundError(f"hand sidecar does not exist: {sidecar}")
    requested = _string_names(joint_names, "kinematic_joint_names")
    with np.load(sidecar, allow_pickle=False) as payload:
        missing = [name for name in _HAND_FIELDS if name not in payload.files]
        extra = sorted(set(payload.files).difference(_HAND_FIELDS))
        if missing or extra:
            raise ValueError(
                "hand sidecar must contain exactly "
                f"{list(_HAND_FIELDS)}; missing={missing} extra={extra}"
            )
        source_names = _string_names(payload["hand_joint_names"], "hand_joint_names")
        hand_qpos = np.asarray(payload["hand_qpos"], dtype=np.float64)
        hand_qvel = np.asarray(payload["hand_qvel"], dtype=np.float64)
    if set(source_names) != set(requested) or len(source_names) != len(requested):
        raise ValueError("hand sidecar joints do not match kinematic_joint_names")
    if hand_qpos.ndim != 2 or hand_qpos.shape[1] != len(source_names):
        raise ValueError("hand_qpos must have shape (T, n_hand)")
    if hand_qvel.shape != hand_qpos.shape:
        raise ValueError("hand_qvel must match hand_qpos")
    if not np.isfinite(hand_qpos).all() or not np.isfinite(hand_qvel).all():
        raise ValueError("hand sidecar must contain only finite values")
    order = np.asarray([source_names.index(name) for name in requested], dtype=np.int64)
    return np.ascontiguousarray(hand_qpos[:, order]), np.ascontiguousarray(
        hand_qvel[:, order]
    )


def executed_kinematics_arrays(
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    timeline: np.ndarray,
    initial_qpos: np.ndarray,
    initial_qvel: np.ndarray,
    joint_names: tuple[str, ...],
    joint_qpos_addresses: np.ndarray,
    joint_dof_addresses: np.ndarray,
    hand_joint_names: tuple[str, ...],
    hand_qpos_addresses: np.ndarray,
    hand_dof_addresses: np.ndarray,
    root_qpos_address: int,
    root_dof_address: int,
    motion_token: np.ndarray,
    fps: int = 50,
) -> dict[str, np.ndarray]:
    """Split executed MuJoCo state into the source kinematics layout plus tokens.

    ``qpos``/``qvel``/``timeline`` are post-step samples of length ``T``.
    ``motion_token[t]`` is inferred at ``kinematics[t]``. The last post-step
    pose is dropped so both sequences have length ``T``: ``kinematics[0]`` is
    the reset, ``kinematics[1:]`` are post-step poses ``[0, T-2]``.
    """

    post_pose = np.asarray(qpos, dtype=np.float64)
    post_vel = np.asarray(qvel, dtype=np.float64)
    post_times = np.asarray(timeline, dtype=np.float64)
    start_pose = np.asarray(initial_qpos, dtype=np.float64)
    start_vel = np.asarray(initial_qvel, dtype=np.float64)
    token = np.asarray(motion_token)
    steps = post_pose.shape[0]
    if (
        post_pose.ndim != 2
        or post_vel.ndim != 2
        or post_pose.shape[0] != post_vel.shape[0]
    ):
        raise ValueError("qpos and qvel must have shape (T, n)")
    if steps < 1:
        raise ValueError("executed kinematics require at least one Control Step")
    if start_pose.shape != (post_pose.shape[1],):
        raise ValueError("initial_qpos must match post-step qpos width")
    if start_vel.shape != (post_vel.shape[1],):
        raise ValueError("initial_qvel must match post-step qvel width")
    if post_times.shape != (steps,):
        raise ValueError("timeline must have shape (T,)")
    if token.ndim != 2 or token.shape[0] != steps:
        raise ValueError("motion_token must have shape (T, 64)")
    if token.shape[1] != 64:
        raise ValueError("motion_token must have width 64")
    if int(fps) != fps or fps < 1:
        raise ValueError("fps must be a positive integer")
    if steps == 1:
        period = float(post_times[0])
    else:
        period = float(post_times[1] - post_times[0])
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("timeline spacing must be a positive control period")
    times = float(post_times[0] - period) + period * np.arange(
        steps, dtype=np.float64
    )
    pose = np.concatenate((start_pose[None, :], post_pose[:-1]), axis=0)
    vel = np.concatenate((start_vel[None, :], post_vel[:-1]), axis=0)
    joint_qpos_addresses = np.asarray(joint_qpos_addresses, dtype=np.int64)
    joint_dof_addresses = np.asarray(joint_dof_addresses, dtype=np.int64)
    hand_qpos_addresses = np.asarray(hand_qpos_addresses, dtype=np.int64)
    hand_dof_addresses = np.asarray(hand_dof_addresses, dtype=np.int64)
    if joint_qpos_addresses.shape != (len(joint_names),):
        raise ValueError("joint qpos addresses must match joint_names")
    if joint_dof_addresses.shape != (len(joint_names),):
        raise ValueError("joint dof addresses must match joint_names")
    if hand_qpos_addresses.shape != (len(hand_joint_names),):
        raise ValueError("hand qpos addresses must match hand_joint_names")
    if hand_dof_addresses.shape != (len(hand_joint_names),):
        raise ValueError("hand dof addresses must match hand_joint_names")
    root = int(root_qpos_address)
    root_dof = int(root_dof_address)
    if root < 0 or root + 7 > pose.shape[1]:
        raise ValueError("root qpos address does not fit qpos")
    if root_dof < 0 or root_dof + 6 > vel.shape[1]:
        raise ValueError("root dof address does not fit qvel")
    arrays: dict[str, np.ndarray] = {
        "fps": np.int64(fps),
        "timestamps": np.ascontiguousarray(times),
        "root_pos_w": np.ascontiguousarray(pose[:, root : root + 3]),
        "root_quat_wxyz": np.ascontiguousarray(pose[:, root + 3 : root + 7]),
        "root_lin_vel_w": np.ascontiguousarray(vel[:, root_dof : root_dof + 3]),
        "root_ang_vel_w": np.ascontiguousarray(vel[:, root_dof + 3 : root_dof + 6]),
        "joint_pos": np.ascontiguousarray(pose[:, joint_qpos_addresses]),
        "joint_vel": np.ascontiguousarray(vel[:, joint_dof_addresses]),
        "joint_names": np.asarray(joint_names),
        "motion_token": np.ascontiguousarray(token),
    }
    if hand_joint_names:
        arrays["hand_qpos"] = np.ascontiguousarray(pose[:, hand_qpos_addresses])
        arrays["hand_qvel"] = np.ascontiguousarray(vel[:, hand_dof_addresses])
        arrays["hand_joint_names"] = np.asarray(hand_joint_names)
    return arrays


def write_optimized_kinematics(path: str | Path, arrays: Mapping[str, np.ndarray]) -> Path:
    """Write executed optimized kinematics bound to motion tokens."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    required = {
        "fps",
        "timestamps",
        "root_pos_w",
        "root_quat_wxyz",
        "joint_pos",
        "joint_vel",
        "joint_names",
        "motion_token",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"optimized kinematics missing fields: {missing}")
    token = np.asarray(arrays["motion_token"])
    frames = int(np.asarray(arrays["timestamps"]).shape[0])
    if frames < 1:
        raise ValueError("optimized kinematics must contain at least one frame")
    if token.shape != (frames, 64):
        raise ValueError(
            "motion_token[t] is inferred at kinematics[t], so both must have "
            "length T after dropping the final post-step pose"
        )
    if np.asarray(arrays["joint_pos"]).shape[0] != frames:
        raise ValueError("joint_pos must align with timestamps")
    np.savez_compressed(destination, **arrays)
    return destination
