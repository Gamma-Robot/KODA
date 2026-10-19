"""FK executed SIL qpos into the ScaleTrack (T, 30) kinematics NPZ Mink expects."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np

SCALETRACK_KINEMATICS_FILE = "scaletrack_kinematics.npz"
SCALETRACK_META_FILE = "meta.json"
SCALETRACK_BODY_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
SCALETRACK_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)


def default_g1_mjx_xml() -> Path:
    """Resolve the 29-DoF MJX G1 model used by ScaleTrack / Mink."""

    root = Path(os.environ.get("KODA_HOME", Path(__file__).resolve().parents[6]))
    candidates = (
        root / "resources/ScaleBFM/scalebridge/data/robot/g1_29dof/g1_29dof_mjx.xml",
        root / "src/koda/vendor/scalebfm/scalebridge/data/robot/g1_29dof/g1_29dof_mjx.xml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "g1_29dof_mjx.xml not found; expected a ScaleBFM ScaleBridge checkout"
    )


def write_executed_scaletrack_kinematics(
    path: str | Path,
    *,
    meta_path: str | Path | None = None,
    initial_qpos: np.ndarray,
    initial_qvel: np.ndarray,
    post_qpos: np.ndarray,
    post_qvel: np.ndarray,
    source_joint_names: tuple[str, ...],
    source_joint_qpos_addresses: np.ndarray,
    source_joint_dof_addresses: np.ndarray,
    source_root_qpos_address: int,
    source_root_dof_address: int,
    xml_path: str | Path | None = None,
    fps: int = 50,
    extra_meta: Mapping[str, Any] | None = None,
) -> Path:
    """FK SIL executed qpos through ``g1_29dof_mjx.xml`` and write ScaleTrack NPZ."""

    if fps != 50:
        raise ValueError(f"ScaleTrack motion data must be 50 Hz, got {fps}")
    start_qpos = np.asarray(initial_qpos, dtype=np.float64)
    start_qvel = np.asarray(initial_qvel, dtype=np.float64)
    post_pose = np.asarray(post_qpos, dtype=np.float64)
    post_vel = np.asarray(post_qvel, dtype=np.float64)
    if post_pose.ndim != 2 or post_vel.ndim != 2 or post_pose.shape[0] != post_vel.shape[0]:
        raise ValueError("post_qpos/post_qvel must have shape (T, n)")
    if start_qpos.shape != (post_pose.shape[1],) or start_qvel.shape != (post_vel.shape[1],):
        raise ValueError("initial qpos/qvel width must match executed trajectories")
    frames = [(_extract_root_and_joints(
        start_qpos,
        start_qvel,
        source_joint_names=source_joint_names,
        source_joint_qpos_addresses=source_joint_qpos_addresses,
        source_joint_dof_addresses=source_joint_dof_addresses,
        source_root_qpos_address=source_root_qpos_address,
        source_root_dof_address=source_root_dof_address,
    ))]
    for index in range(post_pose.shape[0]):
        frames.append(
            _extract_root_and_joints(
                post_pose[index],
                post_vel[index],
                source_joint_names=source_joint_names,
                source_joint_qpos_addresses=source_joint_qpos_addresses,
                source_joint_dof_addresses=source_joint_dof_addresses,
                source_root_qpos_address=source_root_qpos_address,
                source_root_dof_address=source_root_dof_address,
            )
        )
    model = mujoco.MjModel.from_xml_path(str(xml_path or default_g1_mjx_xml()))
    return write_scaletrack_kinematics(
        path,
        model=model,
        state_frames=[
            _assemble_mjx_state(model, root_pos, root_quat, root_lin, root_ang, joints, joint_vel)
            for (
                root_pos,
                root_quat,
                root_lin,
                root_ang,
                joints,
                joint_vel,
            ) in frames
        ],
        body_names=SCALETRACK_BODY_NAMES,
        joint_names=SCALETRACK_JOINT_NAMES,
        fps=fps,
        meta_path=meta_path,
        extra_meta=extra_meta,
    )


def write_scaletrack_kinematics(
    path: str | Path,
    *,
    model: mujoco.MjModel,
    state_frames: list[tuple[np.ndarray, np.ndarray]],
    body_names: tuple[str, ...] = SCALETRACK_BODY_NAMES,
    joint_names: tuple[str, ...] = SCALETRACK_JOINT_NAMES,
    fps: int = 50,
    meta_path: str | Path | None = None,
    extra_meta: Mapping[str, Any] | None = None,
) -> Path:
    """Save executed MuJoCo states in the ScaleTrack training NPZ schema."""

    if not state_frames:
        raise ValueError("cannot save an empty kinematics trajectory")
    if fps != 50:
        raise ValueError(f"ScaleTrack motion data must be 50 Hz, got {fps}")
    if not body_names or body_names[0] != "pelvis":
        raise ValueError("ScaleTrack body order must start with the pelvis/root")
    if len(set(body_names)) != len(body_names):
        raise ValueError("ScaleTrack body names contain duplicates")
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("ScaleTrack joint names contain duplicates")

    body_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names],
        dtype=np.int32,
    )
    missing_bodies = [name for name, body_id in zip(body_names, body_ids) if body_id < 0]
    if missing_bodies:
        raise ValueError(f"MuJoCo bodies not found: {missing_bodies}")
    joint_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names],
        dtype=np.int32,
    )
    missing_joints = [name for name, joint_id in zip(joint_names, joint_ids) if joint_id < 0]
    if missing_joints:
        raise ValueError(f"MuJoCo joints not found: {missing_joints}")

    qpos_addresses = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int32)
    qvel_addresses = np.asarray(model.jnt_dofadr[joint_ids], dtype=np.int32)
    frame_count = len(state_frames)
    body_count = len(body_names)
    joint_count = len(joint_names)
    joint_pos = np.empty((frame_count, joint_count), dtype=np.float32)
    joint_vel = np.empty((frame_count, joint_count), dtype=np.float32)
    body_pos_w = np.empty((frame_count, body_count, 3), dtype=np.float32)
    body_quat_w = np.empty((frame_count, body_count, 4), dtype=np.float32)
    body_lin_vel_w = np.empty((frame_count, body_count, 3), dtype=np.float32)
    body_ang_vel_w = np.empty((frame_count, body_count, 3), dtype=np.float32)

    data = mujoco.MjData(model)
    object_velocity = np.empty(6, dtype=np.float64)
    for frame_index, (qpos, qvel) in enumerate(state_frames):
        qpos = np.asarray(qpos, dtype=np.float64)
        qvel = np.asarray(qvel, dtype=np.float64)
        if qpos.shape != (model.nq,) or qvel.shape != (model.nv,):
            raise ValueError(
                f"state frame {frame_index} has qpos/qvel shapes "
                f"{qpos.shape}/{qvel.shape}, expected {(model.nq,)}/{(model.nv,)}"
            )
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)
        joint_pos[frame_index] = qpos[qpos_addresses]
        joint_vel[frame_index] = qvel[qvel_addresses]
        body_pos_w[frame_index] = data.xpos[body_ids]
        body_quat_w[frame_index] = data.xquat[body_ids]
        for body_index, body_id in enumerate(body_ids):
            mujoco.mj_objectVelocity(
                model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                int(body_id),
                object_velocity,
                0,
            )
            body_ang_vel_w[frame_index, body_index] = object_velocity[:3]
            body_lin_vel_w[frame_index, body_index] = object_velocity[3:]

    arrays = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
        "fps": np.asarray(fps, dtype=np.int64),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    np.savez_compressed(destination, **arrays)

    metadata: dict[str, Any] = {
        "format": "ScaleTrack motion NPZ",
        "format_version": 1,
        "source": "executed SIL MuJoCo/MJWarp robot states",
        "mpc_backend": "sil-tts",
        "sampling": {
            "fps": fps,
            "period_s": 1.0 / fps,
            "frame_0": "initial simulator state",
            "frame_t": "simulator state after control transition t",
        },
        "dimensions": {
            "T": frame_count,
            "J": joint_count,
            "B": body_count,
        },
        "joint_names": list(joint_names),
        "body_names": list(body_names),
    }
    if extra_meta:
        metadata.update(dict(extra_meta))
    meta_destination = Path(meta_path) if meta_path is not None else destination.with_name(
        SCALETRACK_META_FILE if destination.name == SCALETRACK_KINEMATICS_FILE else f"{destination.stem}_meta.json"
    )
    meta_destination.write_text(json.dumps(metadata, indent=2) + "\n")
    return destination


def _extract_root_and_joints(
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    source_joint_names: tuple[str, ...],
    source_joint_qpos_addresses: np.ndarray,
    source_joint_dof_addresses: np.ndarray,
    source_root_qpos_address: int,
    source_root_dof_address: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float], dict[str, float]]:
    root = int(source_root_qpos_address)
    root_dof = int(source_root_dof_address)
    joints = {
        name: float(qpos[int(address)])
        for name, address in zip(source_joint_names, source_joint_qpos_addresses)
    }
    joint_vel = {
        name: float(qvel[int(address)])
        for name, address in zip(source_joint_names, source_joint_dof_addresses)
    }
    return (
        np.asarray(qpos[root : root + 3], dtype=np.float64),
        np.asarray(qpos[root + 3 : root + 7], dtype=np.float64),
        np.asarray(qvel[root_dof : root_dof + 3], dtype=np.float64),
        np.asarray(qvel[root_dof + 3 : root_dof + 6], dtype=np.float64),
        joints,
        joint_vel,
    )


def _assemble_mjx_state(
    model: mujoco.MjModel,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    root_lin: np.ndarray,
    root_ang: np.ndarray,
    joints: Mapping[str, float],
    joint_vel: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.array(model.qpos0, dtype=np.float64, copy=True)
    qvel = np.zeros(model.nv, dtype=np.float64)
    free = np.flatnonzero(np.asarray(model.jnt_type) == int(mujoco.mjtJoint.mjJNT_FREE))
    if free.size != 1:
        raise ValueError("ScaleTrack MJX model must have exactly one free joint")
    root = int(model.jnt_qposadr[int(free[0])])
    root_dof = int(model.jnt_dofadr[int(free[0])])
    qpos[root : root + 3] = root_pos
    qpos[root + 3 : root + 7] = root_quat
    qvel[root_dof : root_dof + 3] = root_lin
    qvel[root_dof + 3 : root_dof + 6] = root_ang
    missing = [name for name in SCALETRACK_JOINT_NAMES if name not in joints]
    if missing:
        raise ValueError(f"executed qpos missing ScaleTrack joints: {missing}")
    for name in SCALETRACK_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos[int(model.jnt_qposadr[joint_id])] = joints[name]
        qvel[int(model.jnt_dofadr[joint_id])] = joint_vel[name]
    return qpos, qvel
