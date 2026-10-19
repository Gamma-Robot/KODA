"""Load ScaleTrack reference NPZ files into SIL ``MotionGuidance``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sil_trackers.contracts import MotionGuidance
from sil_tts.motion_tracking.scaletrack_kinematics import (
    SCALETRACK_BODY_NAMES,
    SCALETRACK_JOINT_NAMES,
)


def scaletrack_clip_duration_seconds(path: str | Path) -> float:
    """Return ``(T-1) / fps`` for a ScaleTrack reference, requiring 50 Hz."""

    with np.load(Path(path), allow_pickle=False) as data:
        frames = int(np.asarray(data["body_pos_w"]).shape[0])
        fps = float(np.asarray(data["fps"]).reshape(()))
    if frames < 2:
        raise ValueError(f"reference has only {frames} frame(s)")
    if not np.isclose(fps, 50.0, rtol=0.0, atol=1.0e-8):
        raise ValueError(f"reference fps must be 50, got {fps}")
    return (frames - 1) / fps


def load_scaletrack_guidance(
    path: str | Path,
    *,
    joint_names: tuple[str, ...],
    body_names: tuple[str, ...],
    control_period: float,
) -> MotionGuidance:
    """Build a batch-1 dense guidance clip from ScaleTrack ``body_*`` / ``joint_*``."""

    period = float(control_period)
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("control_period must be finite and positive")
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as data:
        required = ("body_pos_w", "body_quat_w", "joint_pos", "fps")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"reference missing keys {missing}: {source}")
        body_pos = np.asarray(data["body_pos_w"], dtype=np.float64)
        body_quat = np.asarray(data["body_quat_w"], dtype=np.float64)
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
        fps = float(np.asarray(data["fps"]).reshape(()))
        joint_vel = (
            np.asarray(data["joint_vel"], dtype=np.float64)
            if "joint_vel" in data.files
            else None
        )
        body_lin = (
            np.asarray(data["body_lin_vel_w"], dtype=np.float64)
            if "body_lin_vel_w" in data.files
            else None
        )
        body_ang = (
            np.asarray(data["body_ang_vel_w"], dtype=np.float64)
            if "body_ang_vel_w" in data.files
            else None
        )

    if body_pos.ndim != 3 or body_pos.shape[1:] != (len(SCALETRACK_BODY_NAMES), 3):
        raise ValueError(
            "body_pos_w must have shape (T, 30, 3) in ScaleTrack body order, "
            f"got {body_pos.shape}"
        )
    if body_quat.shape != (*body_pos.shape[:2], 4):
        raise ValueError(f"body_quat_w shape {body_quat.shape} does not match body_pos_w")
    if joint_pos.ndim != 2 or joint_pos.shape[1] != len(SCALETRACK_JOINT_NAMES):
        raise ValueError(
            "joint_pos must have shape (T, 29) in ScaleTrack joint order, "
            f"got {joint_pos.shape}"
        )
    frames = int(body_pos.shape[0])
    if frames < 2:
        raise ValueError("reference must contain at least two frames")
    if not np.isclose(fps, 1.0 / period, rtol=0.0, atol=1.0e-6):
        raise ValueError(
            f"reference fps {fps} does not match control_period {period}"
        )
    if joint_vel is None:
        joint_vel = np.gradient(joint_pos, period, axis=0)
    if body_lin is None:
        body_lin = np.gradient(body_pos, period, axis=0)
    if body_ang is None:
        body_ang = np.gradient(body_quat[..., 1:], period, axis=0)
    if joint_vel.shape != joint_pos.shape:
        raise ValueError("joint_vel shape does not match joint_pos")
    if body_lin.shape != body_pos.shape or body_ang.shape != body_pos.shape:
        raise ValueError("body velocity shapes do not match body_pos_w")

    source_joint_index = {name: index for index, name in enumerate(SCALETRACK_JOINT_NAMES)}
    try:
        joint_order = [source_joint_index[name] for name in joint_names]
    except KeyError as exc:
        raise ValueError(f"command joint missing from ScaleTrack joint list: {exc}") from exc
    source_body_index = {name: index for index, name in enumerate(SCALETRACK_BODY_NAMES)}
    try:
        body_order = [source_body_index[name] for name in body_names]
    except KeyError as exc:
        raise ValueError(f"guidance body missing from ScaleTrack body list: {exc}") from exc

    quat = body_quat[:, body_order]
    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norms < 1.0e-8):
        raise ValueError("reference contains a zero-norm body quaternion")
    quat = quat / norms
    negative = quat[..., 0] < 0.0
    quat[negative] = -quat[negative]

    return MotionGuidance(
        query_offsets=tuple(range(frames)),
        joint_names=tuple(joint_names),
        body_names=tuple(body_names),
        joint_position=np.ascontiguousarray(joint_pos[None, :, joint_order]),
        joint_velocity=np.ascontiguousarray(joint_vel[None, :, joint_order]),
        body_position_world=np.ascontiguousarray(body_pos[None, :, body_order]),
        body_orientation_world=np.ascontiguousarray(quat[None]),
        body_linear_velocity_world=np.ascontiguousarray(body_lin[None, :, body_order]),
        body_angular_velocity_world=np.ascontiguousarray(body_ang[None, :, body_order]),
    )


def apply_init_state_npz(
    path: str | Path,
    *,
    qpos_template: np.ndarray,
    qvel_template: np.ndarray,
    joint_names: tuple[str, ...],
    joint_qpos_addresses: np.ndarray,
    joint_dof_addresses: np.ndarray,
    root_qpos_address: int,
    root_dof_address: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Overlay a Kimodo ``init_state.npz`` onto one MuJoCo qpos/qvel by joint name."""

    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=True) as data:
        required = ("root_pos_w", "root_quat_wxyz", "joint_pos", "joint_names")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"init state missing keys {missing}: {source}")
        root_pos = np.asarray(data["root_pos_w"], dtype=np.float64).reshape(3)
        root_quat = _normalise_quat(data["root_quat_wxyz"])
        root_lin = np.asarray(
            data["root_lin_vel_w"] if "root_lin_vel_w" in data.files else np.zeros(3),
            dtype=np.float64,
        ).reshape(3)
        root_ang = np.asarray(
            data["root_ang_vel_w"] if "root_ang_vel_w" in data.files else np.zeros(3),
            dtype=np.float64,
        ).reshape(3)
        names = _strings(data["joint_names"])
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float64).reshape(-1)
        joint_vel = np.asarray(
            data["joint_vel"] if "joint_vel" in data.files else np.zeros_like(joint_pos),
            dtype=np.float64,
        ).reshape(-1)

    if len(names) != joint_pos.size or joint_vel.size != joint_pos.size:
        raise ValueError("init state joint_names/pos/vel lengths disagree")
    if len(set(names)) != len(names):
        raise ValueError("init state contains duplicate joint names")
    expected = set(joint_names)
    supplied = set(names)
    missing_sim = sorted(expected - supplied)
    extra = sorted(supplied - expected)
    if missing_sim or extra:
        raise ValueError(
            "initial state joints do not match command joints; "
            f"missing={missing_sim}, extra={extra}"
        )

    qpos = np.array(qpos_template, dtype=np.float64, copy=True)
    qvel = np.array(qvel_template, dtype=np.float64, copy=True)
    qpos[root_qpos_address : root_qpos_address + 3] = root_pos
    qpos[root_qpos_address + 3 : root_qpos_address + 7] = root_quat
    qvel[root_dof_address : root_dof_address + 3] = root_lin
    qvel[root_dof_address + 3 : root_dof_address + 6] = root_ang
    values = dict(zip(names, joint_pos))
    velocities = dict(zip(names, joint_vel))
    addresses = np.asarray(joint_qpos_addresses, dtype=np.int64)
    dofs = np.asarray(joint_dof_addresses, dtype=np.int64)
    for index, name in enumerate(joint_names):
        qpos[int(addresses[index])] = values[name]
        qvel[int(dofs[index])] = velocities[name]
    if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)):
        raise ValueError("initial qpos/qvel contains non-finite values")
    return qpos, qvel


def _strings(value: np.ndarray) -> list[str]:
    return [str(item) for item in np.asarray(value).reshape(-1).tolist()]


def _normalise_quat(q: np.ndarray) -> np.ndarray:
    quat = np.asarray(q, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(quat)):
        raise ValueError("initial root quaternion contains non-finite values")
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-8:
        raise ValueError("initial root quaternion has near-zero norm")
    quat = quat / norm
    if quat[0] < 0.0:
        quat = -quat
    return quat
