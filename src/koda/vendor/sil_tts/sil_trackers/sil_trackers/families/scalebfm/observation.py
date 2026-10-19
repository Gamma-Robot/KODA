"""Pure NumPy observations for the Torso-UMI-3 ScaleBFM policy."""

from __future__ import annotations

import numpy as np

ACTION_DIM = 29
HISTORY_LENGTH = 3
QUERY_OFFSETS = (0, 1, 2, 3, 4, 5)
SELECTED_BODY_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
TRACKING_BODY_NAMES: tuple[str, ...] = (
    "torso_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
TRACKING_SELECTED_SLOTS: tuple[int, ...] = (7, 10, 13)
NUM_SELECTED = len(SELECTED_BODY_NAMES)
NUM_FUTURE = len(QUERY_OFFSETS)


def append_history(
    history: np.ndarray,
    value: np.ndarray,
    initialized: np.ndarray,
) -> np.ndarray:
    """Append current values to oldest-to-newest histories with first fill."""

    result = history.copy()
    result[:, :-1] = history[:, 1:]
    result[:, -1] = value
    missing = ~initialized.astype(bool)
    if np.any(missing):
        result[missing] = value[missing, None]
    return result


def quat_apply_inverse(q_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(q_wxyz, dtype=np.float64)
    values = np.asarray(vector, dtype=np.float64)
    inverse = np.concatenate(
        (quaternion[..., :1], -quaternion[..., 1:]), axis=-1
    )
    tangent = 2.0 * np.cross(inverse[..., 1:], values)
    return values + inverse[..., :1] * tangent + np.cross(inverse[..., 1:], tangent)


def quat_mul_inverse_left(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    left = np.asarray(q1, dtype=np.float64)
    right = np.asarray(q2, dtype=np.float64)
    w1, x1, y1, z1 = np.moveaxis(left, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(right, -1, 0)
    wi, xi, yi, zi = w1, -x1, -y1, -z1
    return np.stack(
        (
            wi * w2 - xi * x2 - yi * y2 - zi * z2,
            wi * x2 + xi * w2 + yi * z2 - zi * y2,
            wi * y2 - xi * z2 + yi * w2 + zi * x2,
            wi * z2 + xi * y2 - yi * x2 + zi * w2,
        ),
        axis=-1,
    )


def pack_three_point_targets(
    *,
    root_position: np.ndarray,
    root_orientation: np.ndarray,
    link_position_world: np.ndarray,
    link_orientation_world: np.ndarray,
    tracking_slots: tuple[int, ...] = TRACKING_SELECTED_SLOTS,
    num_selected: int = NUM_SELECTED,
) -> tuple[np.ndarray, np.ndarray]:
    """Pelvis-frame future poses of 14 selected bodies, 3-link slots filled."""

    position = np.asarray(link_position_world, dtype=np.float64)
    orientation = np.asarray(link_orientation_world, dtype=np.float64)
    batch, horizon, num_links, _ = position.shape
    if orientation.shape[:3] != (batch, horizon, num_links):
        raise ValueError("link orientation batch/horizon/link shape mismatch")
    if len(tracking_slots) != num_links:
        raise ValueError("tracking_slots length must match link count")
    root_pos = np.asarray(root_position, dtype=np.float64)
    root_quat = np.asarray(root_orientation, dtype=np.float64)
    target_pos = np.zeros((batch, horizon, num_selected, 3), dtype=np.float64)
    target_rot = np.zeros((batch, horizon, num_selected, 4), dtype=np.float64)
    target_rot[..., 0] = 1.0
    local_pos = quat_apply_inverse(
        root_quat[:, None, None, :],
        position - root_pos[:, None, None, :],
    )
    local_rot = quat_mul_inverse_left(root_quat[:, None, None, :], orientation)
    for index, slot in enumerate(tracking_slots):
        target_pos[:, :, slot] = local_pos[:, :, index]
        target_rot[:, :, slot] = local_rot[:, :, index]
    return target_pos.astype(np.float32), target_rot.astype(np.float32)
