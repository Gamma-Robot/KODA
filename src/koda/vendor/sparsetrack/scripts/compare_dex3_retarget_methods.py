"""Small self-contained Dex3 fingertip evaluator used by KODA's contact stage.

The original SparseTrack checkout treated this experiment helper as an
untracked local script.  Keeping the evaluator here makes the offline contact
precompute reproducible without a second repository checkout.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from dex3_kinematics_utils import load_dex3_model


_ASSETS = Path(__file__).resolve().parents[1] / "assets"
_XML = _ASSETS / "g1_with_visual_dex3_hands.xml"
_TIP_SUFFIXES = ("thumb_2_link", "index_1_link", "middle_1_link")


@dataclass
class FixedBaseFK:
    side: str
    model: mujoco.MjModel
    data: mujoco.MjData
    joint_ids: np.ndarray
    tip_ids: np.ndarray
    wrist_id: int


def _build_fixed_base_fk(side: str) -> FixedBaseFK:
    if side not in {"left", "right"}:
        raise ValueError(f"unsupported hand side: {side}")
    model = load_dex3_model(_XML, meshdir=_ASSETS, with_d435=False)
    data = mujoco.MjData(model)
    joint_names = [
        f"{side}_hand_thumb_0_joint",
        f"{side}_hand_thumb_1_joint",
        f"{side}_hand_thumb_2_joint",
        f"{side}_hand_middle_0_joint",
        f"{side}_hand_middle_1_joint",
        f"{side}_hand_index_0_joint",
        f"{side}_hand_index_1_joint",
    ]
    joint_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names],
        dtype=np.int32,
    )
    if np.any(joint_ids < 0):
        missing = [name for name, idx in zip(joint_names, joint_ids) if idx < 0]
        raise ValueError(f"Dex3 model is missing joints: {missing}")
    tip_ids = np.asarray(
        [
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_{suffix}"
            )
            for suffix in _TIP_SUFFIXES
        ],
        dtype=np.int32,
    )
    wrist_id = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
    )
    if np.any(tip_ids < 0) or wrist_id < 0:
        raise ValueError(f"Dex3 model is missing {side} wrist/fingertip bodies")
    return FixedBaseFK(side, model, data, joint_ids, tip_ids, wrist_id)


def _human_wrist_local_tips(hand_world: np.ndarray) -> np.ndarray:
    """Return thumb/index/middle tips in a wrist-relative frame."""
    hand = np.asarray(hand_world, dtype=np.float64)
    if hand.ndim != 3 or hand.shape[2:] != (3,) or hand.shape[1] not in (21, 25):
        raise ValueError(f"expected (T,21,3) or (T,25,3) hand landmarks, got {hand.shape}")
    wrist = hand[:, 0:1]
    return hand[:, (4, 9, 14)] - wrist


def _robot_tips(fk: FixedBaseFK, qpos: np.ndarray) -> np.ndarray:
    q = np.asarray(qpos, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 7:
        raise ValueError(f"expected (T,7) hand qpos, got {q.shape}")
    out = np.empty((len(q), 3, 3), dtype=np.float64)
    addresses = np.asarray([fk.model.jnt_qposadr[int(i)] for i in fk.joint_ids])
    wrist = np.asarray(fk.data.xpos[fk.wrist_id], dtype=np.float64)
    for frame, values in enumerate(q):
        fk.data.qpos[:] = 0.0
        fk.data.qpos[addresses] = values
        mujoco.mj_forward(fk.model, fk.data)
        wrist = np.asarray(fk.data.xpos[fk.wrist_id], dtype=np.float64)
        out[frame] = np.asarray(fk.data.xpos[fk.tip_ids], dtype=np.float64) - wrist
    return out


def _evaluate_side(fk: FixedBaseFK, qpos: np.ndarray, human: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    human_rel = _human_wrist_local_tips(human)
    robot_rel = _robot_tips(fk, qpos)
    rmse = np.sqrt(np.mean((robot_rel - human_rel) ** 2, axis=(1, 2)))
    human_scale = np.linalg.norm(human_rel[:, 1] - human_rel[:, 0], axis=1)
    robot_scale = np.linalg.norm(robot_rel[:, 1] - robot_rel[:, 0], axis=1)
    valid = (human_scale > 1e-8) & (robot_scale > 1e-8)
    scale = float(np.median(human_scale[valid] / robot_scale[valid])) if np.any(valid) else 1.0
    return robot_rel, rmse, scale


def _evaluate_method(
    method: str,
    qpos: np.ndarray,
    left_world: np.ndarray,
    right_world: np.ndarray,
    left_fk: FixedBaseFK,
    right_fk: FixedBaseFK,
) -> dict[str, Any]:
    q = np.asarray(qpos, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 14:
        raise ValueError(f"expected combined Dex3 qpos (T,14), got {q.shape}")
    left_robot, left_rmse, left_scale = _evaluate_side(left_fk, q[:, :7], left_world)
    right_robot, right_rmse, right_scale = _evaluate_side(right_fk, q[:, 7:], right_world)
    return {
        "method": str(method),
        "tip_procrustes_rmse_m": {
            "median_scale_left": left_scale,
            "median_scale_right": right_scale,
            "median_left": float(np.median(left_rmse)),
            "median_right": float(np.median(right_rmse)),
            "mean_left": float(np.mean(left_rmse)),
            "mean_right": float(np.mean(right_rmse)),
        },
        "arrays": {
            "left_robot_tips": left_robot,
            "right_robot_tips": right_robot,
            "left_rmse": left_rmse,
            "right_rmse": right_rmse,
        },
    }
