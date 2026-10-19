"""Pure NumPy observation semantics for the released Humanoid-GPT policy."""

from __future__ import annotations

import numpy as np

ACTION_DIM = 29
PROPRIOCEPTION_DIM = 93
MOTION_COMMAND_DIM = 43
QUERY_OFFSETS = (0, 1)
CURRENT_OFFSET = 0
NEXT_OFFSET = 1

_DOWN = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
_NAVI_EPS = np.float32(1.0e-8)


def _unit_quat(value: np.ndarray) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-8):
        raise ValueError("Humanoid-GPT received a zero-norm quaternion")
    return (quat / norm).astype(np.float32, copy=False)


def wrap_pi(angle: np.ndarray) -> np.ndarray:
    angle = np.asarray(angle, dtype=np.float32)
    return np.arctan2(np.sin(angle), np.cos(angle)).astype(np.float32, copy=False)


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


def quat_from_yaw(yaw: np.ndarray) -> np.ndarray:
    yaw = np.asarray(yaw, dtype=np.float32)
    half = 0.5 * yaw
    result = np.zeros((*yaw.shape, 4), dtype=np.float32)
    result[..., 0] = np.cos(half)
    result[..., 3] = np.sin(half)
    return result


def quat_to_yaw(quaternion: np.ndarray) -> np.ndarray:
    quat = _unit_quat(quaternion)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return wrap_pi(np.arctan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z)))


def quat_rotate_inverse(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quat = _unit_quat(quaternion)
    vector = np.asarray(vector, dtype=np.float32)
    scalar = quat[..., :1]
    xyz = quat[..., 1:]
    return (
        vector * (2.0 * scalar * scalar - 1.0)
        - 2.0 * scalar * np.cross(xyz, vector)
        + 2.0 * xyz * np.sum(xyz * vector, axis=-1, keepdims=True)
    ).astype(np.float32, copy=False)


def matrix_from_quat(quaternion: np.ndarray) -> np.ndarray:
    quat = _unit_quat(quaternion)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return (
        np.stack(
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
        )
        .reshape(*quat.shape[:-1], 3, 3)
        .astype(np.float32, copy=False)
    )


def rotate_xy(xy: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32)
    cosine = np.cos(yaw).astype(np.float32)
    sine = np.sin(yaw).astype(np.float32)
    return np.stack(
        (
            cosine * xy[..., 0] - sine * xy[..., 1],
            sine * xy[..., 0] + cosine * xy[..., 1],
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def rotate_vector_z(vector: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).copy()
    vector[..., :2] = rotate_xy(vector[..., :2], yaw)
    return vector.astype(np.float32, copy=False)


def batch_base_to_navi(base_to_world: np.ndarray) -> np.ndarray:
    """Yaw-aligned gravity-view frame; columns are world-frame axes."""

    x_proj = np.asarray(base_to_world[..., :, 0], dtype=np.float32)
    z_axis = np.zeros_like(x_proj)
    z_axis[..., 2] = 1.0
    y_axis = np.cross(z_axis, x_proj)
    y_norm = np.linalg.norm(y_axis, axis=-1, keepdims=True)
    fallback = np.zeros_like(y_axis)
    fallback[..., 1] = 1.0
    y_axis = np.where(
        y_norm > _NAVI_EPS, y_axis / np.clip(y_norm, _NAVI_EPS, None), fallback
    )
    x_axis = np.cross(y_axis, z_axis)
    return np.stack((x_axis, y_axis, z_axis), axis=-1).astype(np.float32, copy=False)


def lock_root_alignment(
    initialized: np.ndarray,
    robot_init_yaw: np.ndarray,
    robot_init_xy: np.ndarray,
    ref_init_yaw: np.ndarray,
    ref_init_xy: np.ndarray,
    robot_root_orientation: np.ndarray,
    reference_root_position: np.ndarray,
    reference_root_orientation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    missing = ~np.asarray(initialized, dtype=bool)
    next_robot_yaw = np.asarray(robot_init_yaw, dtype=np.float32).copy()
    next_robot_xy = np.asarray(robot_init_xy, dtype=np.float32).copy()
    next_ref_yaw = np.asarray(ref_init_yaw, dtype=np.float32).copy()
    next_ref_xy = np.asarray(ref_init_xy, dtype=np.float32).copy()
    if np.any(missing):
        next_robot_yaw[missing] = quat_to_yaw(robot_root_orientation[missing])
        next_robot_xy[missing] = 0.0
        next_ref_yaw[missing] = quat_to_yaw(reference_root_orientation[missing])
        next_ref_xy[missing] = np.asarray(
            reference_root_position[missing, :2], dtype=np.float32
        )
    return (
        np.ones(len(initialized), dtype=bool),
        next_robot_yaw,
        next_robot_xy,
        next_ref_yaw,
        next_ref_xy,
    )


def rebias_root_pose(
    reference_root_position: np.ndarray,
    reference_root_orientation: np.ndarray,
    robot_init_yaw: np.ndarray,
    robot_init_xy: np.ndarray,
    ref_init_yaw: np.ndarray,
    ref_init_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = np.asarray(reference_root_position, dtype=np.float32).copy()
    yaw_offset = wrap_pi(robot_init_yaw - ref_init_yaw)
    ref_yaw = quat_to_yaw(reference_root_orientation)
    yaw_delta = wrap_pi(
        robot_init_yaw + wrap_pi(ref_yaw - ref_init_yaw) - ref_yaw
    )
    position[..., :2] = robot_init_xy + rotate_xy(
        position[..., :2] - ref_init_xy, yaw_offset
    )
    orientation = _unit_quat(
        quat_mul(quat_from_yaw(yaw_delta), _unit_quat(reference_root_orientation))
    )
    return position, orientation, yaw_offset


def pack_proprioception(
    root_orientation_world: np.ndarray,
    root_angular_velocity_world: np.ndarray,
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    previous_action: np.ndarray,
    default_joint_position: np.ndarray,
) -> np.ndarray:
    gravity = np.broadcast_to(_DOWN, (*root_orientation_world.shape[:-1], 3))
    result = np.concatenate(
        (
            quat_rotate_inverse(root_orientation_world, root_angular_velocity_world),
            quat_rotate_inverse(root_orientation_world, gravity),
            np.asarray(joint_position, dtype=np.float32)
            - default_joint_position[None],
            np.asarray(joint_velocity, dtype=np.float32),
            np.asarray(previous_action, dtype=np.float32),
        ),
        axis=-1,
    )
    if result.shape[-1] != PROPRIOCEPTION_DIM:
        raise ValueError(
            f"Humanoid-GPT proprioception must contain {PROPRIOCEPTION_DIM} values"
        )
    return result.astype(np.float32, copy=False)


def pack_motion_command(
    next_joint_position: np.ndarray,
    next_root_position: np.ndarray,
    next_root_orientation: np.ndarray,
    next_root_angular_velocity_world: np.ndarray,
    next_root_linear_velocity_world: np.ndarray,
    current_root_position: np.ndarray,
    current_root_orientation: np.ndarray,
    robot_root_position: np.ndarray,
    robot_root_orientation: np.ndarray,
    yaw_offset: np.ndarray,
    default_joint_position: np.ndarray,
) -> np.ndarray:
    gravity_view = batch_base_to_navi(matrix_from_quat(next_root_orientation))
    angular = rotate_vector_z(next_root_angular_velocity_world, yaw_offset)
    linear = rotate_vector_z(next_root_linear_velocity_world, yaw_offset)
    rotate = np.swapaxes(gravity_view, -1, -2)
    cvel = np.concatenate(
        (
            np.matmul(rotate, angular[..., None])[..., 0],
            np.matmul(rotate, linear[..., None])[..., 0],
        ),
        axis=-1,
    )
    down = np.broadcast_to(_DOWN, (*next_root_orientation.shape[:-1], 3))
    yaw_error = wrap_pi(
        quat_to_yaw(current_root_orientation) - quat_to_yaw(robot_root_orientation)
    )
    xy_error = rotate_xy(
        current_root_position[..., :2] - robot_root_position[..., :2],
        -quat_to_yaw(robot_root_orientation),
    )
    result = np.concatenate(
        (
            np.asarray(next_joint_position, dtype=np.float32)
            - default_joint_position[None],
            np.asarray(next_root_position, dtype=np.float32)[..., 2:3],
            quat_rotate_inverse(next_root_orientation, down),
            cvel.astype(np.float32, copy=False),
            np.stack((np.cos(yaw_error), np.sin(yaw_error)), axis=-1).astype(
                np.float32, copy=False
            ),
            xy_error,
        ),
        axis=-1,
    )
    if result.shape[-1] != MOTION_COMMAND_DIM:
        raise ValueError(
            f"Humanoid-GPT motion_command must contain {MOTION_COMMAND_DIM} values"
        )
    return result.astype(np.float32, copy=False)
