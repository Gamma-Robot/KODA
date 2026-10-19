"""Shared NumPy construction of the ScaleTrack policy target inputs."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _quat_apply_inverse_np(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(np.asarray(q_wxyz, dtype=np.float64), -1, 0)
    qinv = np.stack([w, -x, -y, -z], axis=-1)
    qw, qx, qy, qz = np.moveaxis(qinv, -1, 0)
    t = 2.0 * np.cross(qinv[..., 1:], v)
    return v + qw[..., None] * t + np.cross(qinv[..., 1:], t)


def _quat_mul_inverse_left_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    wi, xi, yi, zi = w1, -x1, -y1, -z1
    return np.stack(
        [
            wi * w2 - xi * x2 - yi * y2 - zi * z2,
            wi * x2 + xi * w2 + yi * z2 - zi * y2,
            wi * y2 - xi * z2 + yi * w2 + zi * x2,
            wi * z2 + xi * y2 - yi * x2 + zi * w2,
        ],
        axis=-1,
    )


def build_policy_targets_single(
    ref_body_pos: np.ndarray,
    ref_body_quat: np.ndarray,
    t: int,
    future_idx: Sequence[int],
    selected_idx: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    lw_sel: int,
    rw_sel: int,
    wrist_pos_w: np.ndarray | None = None,
    wrist_quat_w: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one policy target window in the robot-base frame."""
    idxs = np.clip(int(t) + np.asarray(future_idx, dtype=np.int64), 0, ref_body_pos.shape[0] - 1)
    body_pos = np.asarray(ref_body_pos[idxs][:, selected_idx, :], dtype=np.float64).copy()
    body_quat = np.asarray(ref_body_quat[idxs][:, selected_idx, :], dtype=np.float64).copy()
    if wrist_pos_w is not None:
        body_pos[:, lw_sel, :] = wrist_pos_w[:, 0, :]
        body_pos[:, rw_sel, :] = wrist_pos_w[:, 1, :]
    if wrist_quat_w is not None:
        body_quat[:, lw_sel, :] = wrist_quat_w[:, 0, :]
        body_quat[:, rw_sel, :] = wrist_quat_w[:, 1, :]
    target_pos = _quat_apply_inverse_np(
        np.asarray(root_quat)[None, None, :], body_pos - np.asarray(root_pos)[None, None, :]
    ).astype(np.float32)
    target_rot = _quat_mul_inverse_left_np(
        np.broadcast_to(np.asarray(root_quat)[None, None, :], body_quat.shape), body_quat
    ).astype(np.float32)
    return target_pos, target_rot


def build_policy_targets_batch(
    ref_body_pos: np.ndarray,
    ref_body_quat: np.ndarray,
    t: int,
    future_idx: Sequence[int],
    selected_idx: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    lw_sel: int,
    rw_sel: int,
    wrist_pos_w: np.ndarray,
    wrist_quat_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the same target window for N batched MJX environments."""
    idxs = np.clip(int(t) + np.asarray(future_idx, dtype=np.int64), 0, ref_body_pos.shape[0] - 1)
    base_pos = np.asarray(ref_body_pos[idxs][:, selected_idx, :], dtype=np.float64)
    base_quat = np.asarray(ref_body_quat[idxs][:, selected_idx, :], dtype=np.float64)

    # The non-wrist reference is shared by every candidate.  Transform it by
    # broadcast, then overwrite the two candidate-specific wrist outputs.  This
    # is elementwise equivalent to copying and overwriting full body tensors.
    target_pos = _quat_apply_inverse_np(
        root_quat[:, None, None, :], base_pos[None, ...] - root_pos[:, None, None, :]
    ).astype(np.float32)
    target_rot = _quat_mul_inverse_left_np(
        root_quat[:, None, None, :], base_quat[None, ...]
    ).astype(np.float32)

    wrist_root_pos = root_pos[:, None, :]
    wrist_root_quat = root_quat[:, None, :]
    target_pos[:, :, lw_sel, :] = _quat_apply_inverse_np(
        wrist_root_quat, wrist_pos_w[:, :, 0, :] - wrist_root_pos
    ).astype(np.float32)
    target_pos[:, :, rw_sel, :] = _quat_apply_inverse_np(
        wrist_root_quat, wrist_pos_w[:, :, 1, :] - wrist_root_pos
    ).astype(np.float32)
    target_rot[:, :, lw_sel, :] = _quat_mul_inverse_left_np(
        wrist_root_quat, wrist_quat_w[:, :, 0, :]
    ).astype(np.float32)
    target_rot[:, :, rw_sel, :] = _quat_mul_inverse_left_np(
        wrist_root_quat, wrist_quat_w[:, :, 1, :]
    ).astype(np.float32)
    return target_pos, target_rot
