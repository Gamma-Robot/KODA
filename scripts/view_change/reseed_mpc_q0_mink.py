#!/usr/bin/env python3
"""Re-seed a ScaleTrack init state with zero-arm Mink wrist IK.

The root, waist, and lower-body values come from the supplied Kimodo-derived
state.  Only the 14 arm joints are reset to the robot default (zero) before a
single dual-wrist Mink solve.  This is intentionally a q0 initializer; it does
not modify the reference motion or any later per-frame IK.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import mink
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KODA_SRC = PROJECT_ROOT / "src"
if str(KODA_SRC) not in sys.path:
    sys.path.insert(0, str(KODA_SRC))

from koda.ik.mink_wrist import (  # noqa: E402
    ARMS_ONLY,
    MinkDualWristIK,
    build_model_qpos,
    extract_joint_qpos,
)

DEFAULT_MODEL = (
    PROJECT_ROOT
    / "resources/ScaleBFM/scalebridge/data/robot/g1_29dof/g1_29dof_mjx.xml"
)
WRIST_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
WAIST_JOINTS = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kimodo-init", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-iters", type=int, default=120)
    parser.add_argument("--posture-cost", type=float, default=0.12)
    parser.add_argument("--orientation-cost", type=float, default=0.08)
    parser.add_argument("--collision-margin", type=float, default=0.015)
    return parser.parse_args()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]).copy() for key in archive.files}


def strings(values: np.ndarray) -> list[str]:
    return [str(value) for value in np.asarray(values).reshape(-1).tolist()]


def norm_quat(value: np.ndarray) -> np.ndarray:
    q = np.asarray(value, dtype=np.float64)
    return q / np.linalg.norm(q)


def main() -> None:
    args = parse_args()
    init = load(args.kimodo_init)
    reference = load(args.reference)
    joint_names = strings(init["joint_names"])
    body_names = strings(init.get("body_names", np.asarray([])))
    if len(joint_names) != 29:
        raise ValueError(f"expected 29 joints, got {len(joint_names)}")
    for key in ("root_pos_w", "root_quat_wxyz", "joint_pos"):
        if key not in init:
            raise KeyError(f"Kimodo init missing {key}")
    ref_joint_names = strings(reference.get("joint_names", np.asarray([])))
    ref_body_names = strings(reference["body_names"])
    if not body_names:
        # The ScaleTrack init contract does not require body_names.  The
        # reference carries the canonical body order and is checked below.
        body_names = ref_body_names
    model = mujoco.MjModel.from_xml_path(str(args.model.expanduser().resolve()))
    if (model.nq, model.nv) != (36, 35):
        raise ValueError(f"expected body model nq=36,nv=35, got {model.nq},{model.nv}")
    body_joint_ids = {
        name: int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        for name in joint_names
    }
    missing = [name for name, jid in body_joint_ids.items() if jid < 0]
    if missing:
        raise ValueError(f"init joints absent from model: {missing}")
    for wrist in WRIST_BODIES:
        if wrist not in ref_body_names:
            raise ValueError(f"reference missing {wrist}")
    if "body_pos_w" not in reference or "body_quat_w" not in reference:
        raise KeyError("reference must contain body_pos_w and body_quat_w")
    data = mujoco.MjData(model)
    q_seed = np.asarray(init["joint_pos"], dtype=np.float64).copy()
    arm_indices = np.asarray([joint_names.index(name) for name in ARMS_ONLY], dtype=np.int32)
    q_seed[arm_indices] = 0.0
    seed_qpos = build_model_qpos(
        model,
        root_pos=np.asarray(init["root_pos_w"], dtype=np.float64),
        root_quat_wxyz=norm_quat(init["root_quat_wxyz"]),
        joint_qpos=q_seed,
        joint_names=joint_names,
    )
    left_body = ref_body_names.index(WRIST_BODIES[0])
    right_body = ref_body_names.index(WRIST_BODIES[1])
    left_pos = np.asarray(reference["body_pos_w"][0, left_body], dtype=np.float64)
    right_pos = np.asarray(reference["body_pos_w"][0, right_body], dtype=np.float64)
    left_quat = norm_quat(reference["body_quat_w"][0, left_body])
    right_quat = norm_quat(reference["body_quat_w"][0, right_body])

    # q0 is not subject to the 50 Hz inter-frame delta cap.  A large allowed
    # step lets Mink choose a reachable arm branch from the zero posture.
    q0_ik_names = list(WAIST_JOINTS) + list(ARMS_ONLY)
    solver = MinkDualWristIK(
        model,
        left_frame=WRIST_BODIES[0],
        right_frame=WRIST_BODIES[1],
        position_cost=1.0,
        orientation_cost=float(args.orientation_cost),
        posture_cost=float(args.posture_cost),
        dt=0.05,
        max_iters=int(args.max_iters),
        pos_tol_m=0.01,
        ori_tol_rad=0.2,
        # The bytecode helper's legacy collision names do not match the
        # current g1_29dof_mjx.xml.  Attach the same explicit pairs used by
        # refine_three_point_kinematics_mink.py below.
        use_collision=False,
        collision_margin_m=float(args.collision_margin),
        arm_vel_limit=20.0,
        max_arm_delta_rad=3.0,
        joint_names=q0_ik_names,
    )
    collision_pairs = [
        (
            ["left_hand_coll", "left_forearm_coll"],
            ["torso_coll_l", "torso_coll_r", "pelvis_coll", "head_coll"],
        ),
        (
            ["right_hand_coll", "right_forearm_coll"],
            ["torso_coll_l", "torso_coll_r", "pelvis_coll", "head_coll"],
        ),
        (
            ["left_hand_coll", "left_forearm_coll"],
            ["right_hand_coll", "right_forearm_coll"],
        ),
    ]
    solver.limits.append(
        mink.CollisionAvoidanceLimit(
            model,
            collision_pairs,
            gain=0.85,
            minimum_distance_from_collisions=float(args.collision_margin),
            collision_detection_distance=max(0.08, float(args.collision_margin)),
        )
    )
    result = solver.solve(
        seed_qpos=seed_qpos,
        left_pos=left_pos,
        left_quat_wxyz=left_quat,
        right_pos=right_pos,
        right_quat_wxyz=right_quat,
        joint_names=q0_ik_names,
    )
    solved = extract_joint_qpos(model, result.qpos, joint_names)
    solved[[i for i in range(29) if i not in set(arm_indices)]] = q_seed[
        [i for i in range(29) if i not in set(arm_indices)]
    ]
    if not np.all(np.isfinite(solved)):
        raise RuntimeError("Mink q0 result contains non-finite values")

    output = args.output.expanduser().resolve()
    summary_path = args.summary.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "root_pos_w": np.asarray(init["root_pos_w"], dtype=np.float32),
        "root_quat_wxyz": norm_quat(init["root_quat_wxyz"]).astype(np.float32),
        "root_lin_vel_w": np.asarray(init.get("root_lin_vel_w", np.zeros(3)), dtype=np.float32),
        "root_ang_vel_w": np.asarray(init.get("root_ang_vel_w", np.zeros(3)), dtype=np.float32),
        "joint_pos": solved.astype(np.float32),
        "joint_vel": np.zeros(29, dtype=np.float32),
        "joint_names": np.asarray(joint_names),
        "source": np.asarray("kimodo_torso_zero_arm_mink_q0"),
    }
    with output.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {
        "method": "kimodo_torso_zero_arm_mink_q0",
        "input": str(args.kimodo_init.expanduser().resolve()),
        "reference": str(args.reference.expanduser().resolve()),
        "arm_joint_count": len(ARMS_ONLY),
        "arm_seed": "all_zero",
        "torso_and_lower_body_source": "kimodo_init",
        "solver": {
            "success": bool(result.success),
            "iterations": int(result.n_iters),
            "max_arm_delta_rad": float(result.max_arm_delta_rad),
            "left_position_error_m": float(result.left_pos_err_m),
            "right_position_error_m": float(result.right_pos_err_m),
            "left_orientation_error_deg": float(np.degrees(result.left_ori_err_rad)),
            "right_orientation_error_deg": float(np.degrees(result.right_ori_err_rad)),
        },
        "q_seed_arm_norm_rad": float(np.linalg.norm(q_seed[arm_indices])),
        "q_output_arm_norm_rad": float(np.linalg.norm(solved[arm_indices])),
    }
    summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
