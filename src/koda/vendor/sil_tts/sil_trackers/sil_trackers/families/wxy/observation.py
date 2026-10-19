"""Pure-numpy WXY-WBC observation formulas over common currencies."""

from __future__ import annotations

import numpy as np

HISTORY_LENGTH = 5
ACTION_DIM = 29
LIMB_POSE_DIM = 36  # 4 limbs × (3 pos + 6 rot)
MOTION_COMMAND_DIM = 241
PROPRIOCEPTION_DIM = 645

ANCHOR_BODY = "pelvis"
TRACKING_ANCHOR_BODY = "torso_link"
LIMB_BODIES = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)

_GRAVITY_W = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    shape = v.shape
    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)
    q_w, q_vec = q[:, 0], q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0)[:, None]
    b = np.cross(q_vec, v) * q_w[:, None] * 2.0
    c = q_vec * np.sum(q_vec * v, axis=1, keepdims=True) * 2.0
    return (a - b + c).reshape(shape)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    if q1.shape != q2.shape:
        raise ValueError(f"quaternion shape mismatch: {q1.shape} != {q2.shape}")
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return np.stack([w, x, y, z], axis=-1).reshape(shape)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    shape = q.shape
    q = q.reshape(-1, 4)
    return np.concatenate((q[:, :1], -q[:, 1:]), axis=-1).reshape(shape)


def matrix_from_quat(quaternions: np.ndarray) -> np.ndarray:
    original = quaternions.shape[:-1]
    quaternions = quaternions.reshape(-1, 4)
    r, i, j, k = quaternions.T
    two_s = 2.0 / np.sum(quaternions * quaternions, axis=-1)
    o = np.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        axis=-1,
    )
    return o.reshape(original + (3, 3))


def safe_unit_quat(quat: np.ndarray) -> np.ndarray:
    values = np.asarray(quat, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    safe = np.divide(values, np.maximum(norms, np.float32(1.0e-8)), dtype=np.float32)
    identity = np.zeros_like(safe)
    identity[..., 0] = 1.0
    return np.where(norms < 1.0e-8, identity, safe).astype(np.float32, copy=False)


def limb_pose_in_anchor_frame(
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    *,
    limb_indices: np.ndarray,
    anchor_index: int,
) -> np.ndarray:
    """Return batched limb pose features in the anchor frame, shape (B, 36)."""

    positions = np.asarray(body_pos_w, dtype=np.float32)
    quaternions = np.asarray(body_quat_w, dtype=np.float32)
    limb_pos_w = positions[:, limb_indices]
    limb_quat_w = quaternions[:, limb_indices]
    anchor_pos_w = positions[:, anchor_index]
    anchor_quat_w = safe_unit_quat(quaternions[:, anchor_index])
    anchor_quat_limb = np.broadcast_to(anchor_quat_w[:, None, :], limb_quat_w.shape)
    limb_pos_b = quat_rotate_inverse(
        anchor_quat_limb, limb_pos_w - anchor_pos_w[:, None, :]
    )
    limb_quat_b = quat_mul(quat_conjugate(anchor_quat_limb), limb_quat_w)
    limb_rot6d = matrix_from_quat(limb_quat_b)[..., :, :2].reshape(
        positions.shape[0], len(limb_indices), 6
    )
    return np.concatenate((limb_pos_b, limb_rot6d), axis=-1).reshape(
        positions.shape[0], -1
    )


def projected_gravity(root_quat_w: np.ndarray) -> np.ndarray:
    quat = safe_unit_quat(root_quat_w)
    gravity = np.broadcast_to(_GRAVITY_W, (*quat.shape[:-1], 3))
    return quat_rotate_inverse(quat, gravity).astype(np.float32, copy=False)


def base_ang_vel_body(
    root_quat_w: np.ndarray, root_ang_vel_w: np.ndarray
) -> np.ndarray:
    return quat_rotate_inverse(
        safe_unit_quat(root_quat_w), np.asarray(root_ang_vel_w, dtype=np.float32)
    ).astype(np.float32, copy=False)


def append_history(
    history: np.ndarray, value: np.ndarray, initialized: np.ndarray
) -> np.ndarray:
    """Oldest-to-newest history; fill all slots on first sample per lane."""

    out = history.copy()
    out[:, :-1] = out[:, 1:]
    out[:, -1] = value
    init = initialized.astype(bool)
    if np.any(~init):
        out[~init] = value[~init, None, :]
    return out


def pack_motion_command(
    ref_joint_pos: np.ndarray,
    ref_joint_vel: np.ndarray,
    ref_limb_history: np.ndarray,
    motion_ref_ang_vel: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        (
            ref_joint_pos,
            ref_joint_vel,
            ref_limb_history.reshape(ref_limb_history.shape[0], -1),
            motion_ref_ang_vel,
        ),
        axis=-1,
        dtype=np.float32,
    )


def pack_proprioception(
    robot_limb_history: np.ndarray,
    gravity_history: np.ndarray,
    ang_vel_history: np.ndarray,
    joint_pos_history: np.ndarray,
    joint_vel_history: np.ndarray,
    action_history: np.ndarray,
) -> np.ndarray:
    batch = robot_limb_history.shape[0]
    return np.concatenate(
        (
            robot_limb_history.reshape(batch, -1),
            gravity_history.reshape(batch, -1),
            ang_vel_history.reshape(batch, -1),
            joint_pos_history.reshape(batch, -1),
            joint_vel_history.reshape(batch, -1),
            action_history.reshape(batch, -1),
        ),
        axis=-1,
        dtype=np.float32,
    )
