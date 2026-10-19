"""Pure NumPy observation semantics for Mimic-Lite 32x8192-huge."""

from __future__ import annotations

import numpy as np


ACTION_DIM = 29
HISTORY_LENGTH = 17
HISTORY_LAGS = (0, 1, 2, 3, 4, 8, 16)
PREVIOUS_ACTIONS = 3
QUERY_OFFSETS = (-8, -4, -2, 0, 1, 2, 3, 4)
POLICY_DIM = 535
COMMAND_DIM = 304
_CURRENT_QUERY = QUERY_OFFSETS.index(0)
_DOWN = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)


def _unit_quat(value: np.ndarray) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-8):
        raise ValueError("Mimic-Lite received a zero-norm quaternion")
    return (quat / norm).astype(np.float32, copy=False)


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    value = np.asarray(quat, dtype=np.float32)
    return np.concatenate((value[..., :1], -value[..., 1:]), axis=-1)


def quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if left.shape != right.shape:
        raise ValueError(f"quaternion shape mismatch: {left.shape} != {right.shape}")
    w1, x1, y1, z1 = np.moveaxis(left, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def quat_rotate(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quat = _unit_quat(quat)
    vector = np.asarray(vector, dtype=np.float32)
    scalar, xyz = quat[..., :1], quat[..., 1:]
    return (
        vector * (2.0 * scalar * scalar - 1.0)
        + 2.0 * scalar * np.cross(xyz, vector)
        + 2.0 * xyz * np.sum(xyz * vector, axis=-1, keepdims=True)
    ).astype(np.float32, copy=False)


def quat_rotate_inverse(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return quat_rotate(quat_conjugate(_unit_quat(quat)), vector)


def matrix_from_quat(quat: np.ndarray) -> np.ndarray:
    quat = _unit_quat(quat)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(*quat.shape[:-1], 3, 3).astype(np.float32, copy=False)


def projected_yaw_quat(quat: np.ndarray) -> np.ndarray:
    """Match the upstream horizontal-axis heading convention."""

    quat = _unit_quat(quat)
    x_axis = np.zeros((*quat.shape[:-1], 3), dtype=np.float32)
    z_axis = np.zeros_like(x_axis)
    x_axis[..., 0] = 1.0
    z_axis[..., 2] = 1.0
    x_world = quat_rotate(quat, x_axis)
    z_world = quat_rotate(quat, z_axis)
    heading = np.where(
        np.linalg.norm(x_world[..., :2], axis=-1, keepdims=True) > 0.1,
        x_world[..., :2],
        np.where(x_world[..., 2:3] < 0.0, z_world[..., :2], -z_world[..., :2]),
    )
    yaw = np.arctan2(heading[..., 1], heading[..., 0])
    result = np.zeros((*yaw.shape, 4), dtype=np.float32)
    result[..., 0] = np.cos(yaw / 2.0)
    result[..., 3] = np.sin(yaw / 2.0)
    return _unit_quat(result)


def append_newest(history: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Advance a current-first, zero-warmup history by one control step."""

    result = history.copy()
    result[:, 1:] = history[:, :-1]
    result[:, 0] = value
    return result


def root_angular_velocity_body(
    root_orientation: np.ndarray, angular_velocity_world: np.ndarray
) -> np.ndarray:
    return quat_rotate_inverse(root_orientation, angular_velocity_world)


def projected_gravity(root_orientation: np.ndarray) -> np.ndarray:
    down = np.broadcast_to(_DOWN, (*root_orientation.shape[:-1], 3))
    return quat_rotate_inverse(root_orientation, down)


def heading_offset(
    robot_root_orientation: np.ndarray,
    reference_root_orientation: np.ndarray,
) -> np.ndarray:
    """Return the sticky left-multiplier aligning robot yaw to reference yaw."""

    return quat_mul(
        projected_yaw_quat(reference_root_orientation),
        quat_conjugate(projected_yaw_quat(robot_root_orientation)),
    )


def pack_policy(
    root_angular_velocity: np.ndarray,
    gravity: np.ndarray,
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    previous_actions: np.ndarray,
) -> np.ndarray:
    lags = np.asarray(HISTORY_LAGS, dtype=np.int64)
    batch = root_angular_velocity.shape[0]
    result = np.concatenate(
        (
            root_angular_velocity[:, lags].reshape(batch, -1),
            gravity[:, lags].reshape(batch, -1),
            joint_position[:, lags].reshape(batch, -1),
            joint_velocity[:, lags].reshape(batch, -1),
            previous_actions.reshape(batch, -1),
        ),
        axis=-1,
    ).astype(np.float32, copy=False)
    if result.shape != (batch, POLICY_DIM):
        raise ValueError(f"Mimic-Lite policy observation has shape {result.shape}")
    return result


def pack_command(
    reference_root_position: np.ndarray,
    reference_root_orientation: np.ndarray,
    reference_anchor_position: np.ndarray,
    reference_anchor_orientation: np.ndarray,
    reference_joint_position: np.ndarray,
    robot_root_orientation: np.ndarray,
    fixed_heading_offset: np.ndarray,
) -> np.ndarray:
    batch, queries = reference_root_position.shape[:2]
    if queries != len(QUERY_OFFSETS):
        raise ValueError(f"Mimic-Lite requires {len(QUERY_OFFSETS)} guidance queries")
    anchor_position = reference_anchor_position[:, _CURRENT_QUERY].copy()
    anchor_position[:, 2] = 0.0
    delta_world = reference_root_position - anchor_position[:, None]
    anchor_yaw = projected_yaw_quat(
        reference_anchor_orientation[:, _CURRENT_QUERY]
    )
    position_local = quat_rotate_inverse(
        np.broadcast_to(anchor_yaw[:, None], reference_root_orientation.shape),
        delta_world,
    )

    aligned_robot = quat_mul(fixed_heading_offset, robot_root_orientation)
    robot = np.broadcast_to(aligned_robot[:, None], reference_root_orientation.shape)
    relative = quat_mul(
        quat_conjugate(_unit_quat(robot)),
        _unit_quat(reference_root_orientation),
    )
    orientation_6d = matrix_from_quat(relative)[..., :2, :3]
    result = np.concatenate(
        (
            position_local.reshape(batch, -1),
            orientation_6d.reshape(batch, -1),
            reference_joint_position.reshape(batch, -1),
        ),
        axis=-1,
    ).astype(np.float32, copy=False)
    if result.shape != (batch, COMMAND_DIM):
        raise ValueError(f"Mimic-Lite command observation has shape {result.shape}")
    return result


__all__ = [
    "ACTION_DIM",
    "COMMAND_DIM",
    "HISTORY_LAGS",
    "HISTORY_LENGTH",
    "POLICY_DIM",
    "PREVIOUS_ACTIONS",
    "QUERY_OFFSETS",
    "append_newest",
    "heading_offset",
    "pack_command",
    "pack_policy",
    "projected_gravity",
    "root_angular_velocity_body",
]
