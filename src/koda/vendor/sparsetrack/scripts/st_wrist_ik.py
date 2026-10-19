"""Shared G1 arm-joint naming helpers used by the Mink wrist solver."""
from __future__ import annotations

import mujoco


ARMS_ONLY = (
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


def bare_joint_name(name: str) -> str:
    """Return the MuJoCo joint token from a namespaced archive name."""
    value = str(name)
    for separator in ("::", "/"):
        if separator in value:
            value = value.rsplit(separator, 1)[-1]
    return value


def resolve_joint(model: mujoco.MjModel, name: str) -> int:
    """Resolve an exact or namespaced hinge-joint name in ``model``."""
    target = bare_joint_name(name)
    joint_id = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, target)
    )
    if joint_id >= 0:
        return joint_id

    matches = []
    for candidate_id in range(model.njnt):
        candidate = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, candidate_id
        )
        if candidate is not None and bare_joint_name(candidate) == target:
            matches.append(candidate_id)
    if len(matches) == 1:
        return int(matches[0])
    if len(matches) > 1:
        raise KeyError(f"ambiguous joint {name!r}: ids={matches}")
    raise KeyError(f"joint {name!r} is absent from the MuJoCo model")
