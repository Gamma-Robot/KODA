"""WXY-WBC action → PDCommand mapping (checkpoint-resolved scales/gains)."""

from __future__ import annotations

import numpy as np

from sil_trackers.contracts import PDCommand
from sil_trackers.families.wxy.observation import ACTION_DIM


def action_to_pd(
    action: np.ndarray,
    *,
    policy_joint_names: tuple[str, ...],
    command_joint_names: tuple[str, ...],
    default_joint_pos: np.ndarray,
    action_scale: np.ndarray,
    joint_kp: np.ndarray,
    joint_kd: np.ndarray,
) -> PDCommand:
    """Map policy action in policy-joint order to a PDCommand."""

    values = np.asarray(action, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != ACTION_DIM:
        raise ValueError(f"action must have shape (B, {ACTION_DIM}), got {values.shape}")
    if len(policy_joint_names) != ACTION_DIM or set(policy_joint_names) != set(
        command_joint_names
    ):
        raise ValueError("policy and command names must describe the 29 WXY joints")
    vectors = (default_joint_pos, action_scale, joint_kp, joint_kd)
    if any(vector.shape != (ACTION_DIM,) for vector in vectors):
        raise ValueError(f"WXY command vectors must have shape ({ACTION_DIM},)")
    batch = values.shape[0]
    position_policy = default_joint_pos[None, :] + values * action_scale[None, :]
    order = [policy_joint_names.index(name) for name in command_joint_names]
    position = position_policy[:, order]
    zeros = np.zeros_like(position)
    return PDCommand(
        joint_names=command_joint_names,
        position_target=position.astype(np.float32, copy=False),
        velocity_target=zeros,
        feedforward_torque=zeros,
        proportional_gain=np.broadcast_to(joint_kp[order], (batch, ACTION_DIM)).astype(
            np.float32, copy=True
        ),
        derivative_gain=np.broadcast_to(joint_kd[order], (batch, ACTION_DIM)).astype(
            np.float32, copy=True
        ),
    )
