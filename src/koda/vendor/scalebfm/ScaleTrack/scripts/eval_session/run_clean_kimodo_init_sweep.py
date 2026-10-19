#!/usr/bin/env python3
"""Clean Kimodo init sweep evaluated by real ScaleTrack tracking error.

This is the main entrypoint for the session_20260729_171136_iphone14_3 wrist
tracking experiment:

1. Build Kimodo SparseEE constraints from frozen ScaleTrack wrist targets.
2. Generate multiple Kimodo G1 q0 candidates with fixed seeds.
3. Optionally keep Kimodo only for the waist/arms and restore a standing lower body.
4. Refine each q0 with MuJoCo IK so frame-0 wrists are actually reachable.
5. Optionally reject initial states with implausible feet or saturated joints.
6. Run ScaleTrack for every accepted init state.
7. Rank candidates by online tracking error, not by visual plausibility.

The reference motion is never modified. Candidate initial states are written to
``init_state.npz`` and applied by ``play_eval_session.py --init_state_npz``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

ROOT = Path(__file__).resolve().parents[4]
KIMODO_ROOT = ROOT / "kimodo"
if KIMODO_ROOT.is_dir() and str(KIMODO_ROOT) not in sys.path:
    sys.path.insert(0, str(KIMODO_ROOT))

try:
    from package_session_kimodo_constraint_variants import SparseEEConstraintSet  # noqa: E402
    _KIMODO_IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # optional Kimodo experiment helpers
    SparseEEConstraintSet = None
    _KIMODO_IMPORT_ERROR = exc


SCALETRACK_ROOT = ROOT / "scalebfm" / "ScaleTrack"
SESSION_DIR = ROOT / "ego_anywhere" / "output" / "session_20260729_171136_iphone14_3"
SESSION_NPZ = SESSION_DIR / "scaletrack_eval" / "motions" / "session_umi2.npz"
TARGETS_NPZ = SESSION_DIR / "scaletrack_eval" / "motions" / "session_targets_zup.npz"
OUT_DIR = SESSION_DIR / "scaletrack_eval" / "clean_kimodo_init"
PLAY_SCRIPT = _HERE / "play_eval_session.py"
MJCF = SCALETRACK_ROOT / "source" / "scaletrack" / "scaletrack" / "assets" / "robots" / "g1_29dof" / "g1_29dof.xml"
KIMODO_G1_XML = ROOT / "kimodo" / "kimodo" / "assets" / "skeletons" / "g1skel34" / "xml" / "g1.xml"
DEFAULT_CHECKPOINT_RUN = "humanoid_transformer_m"
DEFAULT_CHECKPOINT = "model_22200.pt"
DEFAULT_KIMODO_PYTHON = Path(sys.executable)

WRIST_LINKS = ("left_wrist_yaw_link", "right_wrist_yaw_link")
WRIST_SKELS = ("left_wrist_yaw_skel", "right_wrist_yaw_skel")
TORSO_LINK = "torso_link"
TORSO_SKEL = "waist_pitch_skel"
PROMPT = "A humanoid is standing in place and naturally moving both hands."

WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
IK_JOINTS = WAIST_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS


@dataclass
class Candidate:
    name: str
    seed: int | None
    out_dir: Path
    init_state_npz: Path
    fk_summary: dict
    accepted: bool
    source: str


def _read_names_xml(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    return [
        j.get("name")
        for j in root.find("worldbody").findall(".//joint")
        if j.get("name") != "floating_base_joint"
    ]


def _as_str_list(arr) -> list[str]:
    return [str(x) for x in np.asarray(arr).tolist()]


def _summarize(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = [float(x) for x in a]
    bw, bx, by, bz = [float(x) for x in b]
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _yaw_quat_wxyz(yaw: float) -> np.ndarray:
    h = 0.5 * float(yaw)
    return np.array([math.cos(h), 0.0, 0.0, math.sin(h)], dtype=np.float64)


def _quat_norm(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / max(float(np.linalg.norm(q)), 1e-12)


def _quat_to_mat_wxyz(q: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(np.asarray(q, dtype=np.float64), scalar_first=True).as_matrix()


def _quat_error_rotvec_wxyz(cur: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    r_cur = Rotation.from_quat(_quat_norm(cur), scalar_first=True)
    r_tgt = Rotation.from_quat(_quat_norm(tgt), scalar_first=True)
    return (r_tgt * r_cur.inv()).as_rotvec()


def _heading_yup_rad(rot_yup: np.ndarray) -> float:
    rot_yup = np.asarray(rot_yup, dtype=np.float64).reshape(3, 3)
    return float(math.atan2(rot_yup[0, 2], rot_yup[0, 0]))


def _load_session(session_npz: Path) -> dict:
    data = np.load(session_npz, allow_pickle=True)
    return {
        "npz": data,
        "joint_names": _as_str_list(data["joint_names"]),
        "body_names": _as_str_list(data["body_names"]),
        "fps": float(data["fps"]),
    }


def preflight(session_npz: Path, targets_npz: Path, out_dir: Path) -> dict:
    sess = _load_session(session_npz)
    mot = sess["npz"]
    tg = np.load(targets_npz, allow_pickle=True)
    body_names = sess["body_names"]
    li = body_names.index(WRIST_LINKS[0])
    ri = body_names.index(WRIST_LINKS[1])
    if "session_index" in mot.files:
        si = np.asarray(mot["session_index"], dtype=np.int64)
    else:
        # Raw evaluation adapters are already sampled frame-for-frame.
        si = np.arange(len(mot["joint_pos"]), dtype=np.int64)
    si = np.clip(si, 0, len(tg["left_wrist"]) - 1)
    left_err = np.linalg.norm(mot["body_pos_w"][:, li] - tg["left_wrist"][si], axis=-1)
    right_err = np.linalg.norm(mot["body_pos_w"][:, ri] - tg["right_wrist"][si], axis=-1)

    summary = {
        "session_npz": str(session_npz),
        "targets_npz": str(targets_npz),
        "n_motion_frames": int(mot["joint_pos"].shape[0]),
        "fps": sess["fps"],
        "left_wrist_ref_vs_targets_m": _summarize(left_err),
        "right_wrist_ref_vs_targets_m": _summarize(right_err),
        "ok": bool(left_err.max() < 0.02 and right_err.max() < 0.02),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preflight.json").write_text(json.dumps(summary, indent=2))
    if not summary["ok"]:
        raise RuntimeError(f"Preflight wrist target mismatch >2cm: {summary}")
    return summary


def write_reference_yaml(session_npz: Path, out_dir: Path) -> Path:
    yaml_path = out_dir / "session_umi2_ref.yaml"
    yaml_path.write_text(f"session_umi2: {session_npz.resolve()}\n")
    return yaml_path


def _kimodo_converter_matrix() -> np.ndarray:
    from kimodo.exports.mujoco import MujocoQposConverter
    from kimodo.skeleton import G1Skeleton34

    converter = MujocoQposConverter(G1Skeleton34())
    return converter.mujoco_to_kimodo_matrix.detach().cpu().numpy().astype(np.float64)


def build_session_heading_constraint(
    session_npz: Path,
    *,
    duration_s: float,
    kimodo_fps: float = 30.0,
    keyframe_dt: float = 0.1,
    device: str | torch.device = "cpu",
    torso_init_condition: bool = False,
) -> tuple[list, dict]:
    """Build wrist trajectory constraints and an optional frame-0 torso pose."""
    from kimodo.skeleton import G1Skeleton34

    sess = _load_session(session_npz)
    mot = sess["npz"]
    body_names = sess["body_names"]
    fps = sess["fps"]
    pel_i = body_names.index("pelvis")
    li = body_names.index(WRIST_LINKS[0])
    ri = body_names.index(WRIST_LINKS[1])

    n_kimodo = max(1, int(round(duration_s * kimodo_fps)))
    max_constraint_t = (n_kimodo - 1) / kimodo_fps
    times = [
        round(keyframe_dt * i, 6)
        for i in range(int(math.floor(max_constraint_t / keyframe_dt)) + 1)
    ]
    if not times or times[-1] < max_constraint_t - 1e-6:
        times.append(round(max_constraint_t, 6))
    sess_frames = np.asarray([min(int(round(t * fps)), mot["body_pos_w"].shape[0] - 1) for t in times], dtype=np.int64)
    kimodo_frames = np.asarray(
        [min(int(round(t * kimodo_fps)), n_kimodo - 1) for t in times],
        dtype=np.int64,
    )

    M = _kimodo_converter_matrix()

    def pos_to_yup(p: np.ndarray) -> np.ndarray:
        return (M @ np.asarray(p, dtype=np.float64).reshape(3)).astype(np.float64)

    def rot_to_yup(q_wxyz: np.ndarray) -> np.ndarray:
        Rz = _quat_to_mat_wxyz(q_wxyz)
        return M @ Rz @ M.T

    root_y = np.stack([pos_to_yup(p) for p in mot["body_pos_w"][sess_frames, pel_i]], axis=0)
    lw_y = np.stack([pos_to_yup(p) for p in mot["body_pos_w"][sess_frames, li]], axis=0)
    rw_y = np.stack([pos_to_yup(p) for p in mot["body_pos_w"][sess_frames, ri]], axis=0)
    # The tracking reference intentionally keeps its non-wrist carrier static.
    # Kimodo still needs a heading, so the raw adapter stores this auxiliary
    # camera-derived pelvis heading separately instead of mutating the carrier.
    if "kimodo_pelvis_heading_quat_wxyz" in mot.files:
        heading_quat = mot["kimodo_pelvis_heading_quat_wxyz"]
        heading_source = "raw_hdf5_cam_pose"
    else:
        heading_quat = mot["body_quat_w"][:, pel_i]
        heading_source = "session_npz_pelvis_body_quat"
    root_R = np.stack([rot_to_yup(q) for q in heading_quat[sess_frames]], axis=0)
    lw_R = np.stack([rot_to_yup(q) for q in mot["body_quat_w"][sess_frames, li]], axis=0)
    rw_R = np.stack([rot_to_yup(q) for q in mot["body_quat_w"][sess_frames, ri]], axis=0)

    offset = root_y[0].copy()
    offset[1] = 0.0
    root_c = root_y - offset[None, :]
    lw_c = lw_y - offset[None, :]
    rw_c = rw_y - offset[None, :]

    device_t = torch.device(device)
    sk = G1Skeleton34().to(device_t)
    J = len(sk.bone_order_names)
    g_pos = np.zeros((len(times), J, 3), dtype=np.float64)
    g_rots = np.tile(np.eye(3, dtype=np.float64), (len(times), J, 1, 1))
    g_pos[:, sk.root_idx] = root_c
    left_skel_i = sk.bone_index[WRIST_SKELS[0]]
    right_skel_i = sk.bone_index[WRIST_SKELS[1]]
    g_pos[:, left_skel_i] = lw_c
    g_pos[:, right_skel_i] = rw_c
    g_rots[:, left_skel_i] = lw_R
    g_rots[:, right_skel_i] = rw_R

    frame_idx = torch.as_tensor(kimodo_frames, dtype=torch.long, device=device_t)
    gp_t = torch.as_tensor(g_pos, dtype=torch.float32, device=device_t)
    gr_t = torch.as_tensor(g_rots, dtype=torch.float32, device=device_t)
    smooth = gp_t[:, sk.root_idx][:, [0, 2]].contiguous()

    cset = SparseEEConstraintSet(
        sk,
        frame_indices=frame_idx,
        global_joints_positions=gp_t,
        global_joints_rots=gr_t,
        smooth_root_2d=smooth,
        pos_joint_names=list(WRIST_SKELS),
        rot_joint_names=list(WRIST_SKELS),
        constrain_smooth_root=True,
        constrain_root_y=False,
        constrain_heading=True,
        label="session-heading-wrist-yaw-only",
    )
    yaw_rad = np.asarray([_heading_yup_rad(R) for R in root_R], dtype=np.float64)
    cset.global_root_heading = torch.as_tensor(
        np.stack([np.cos(yaw_rad), np.sin(yaw_rad)], axis=-1),
        dtype=torch.float32,
        device=device_t,
    )

    from collections import defaultdict

    dd, idd = defaultdict(list), defaultdict(list)
    cset.update_constraints(dd, idd)
    runtime_keys = sorted(dd.keys())
    assert runtime_keys == [
        "global_joints_positions",
        "global_joints_rots",
        "global_root_heading",
        "smooth_root_2d",
    ]
    assert "hand_roll" not in " ".join(cset.pos_joint_names + cset.rot_joint_names)

    constraints = [cset]
    torso_meta = None
    if torso_init_condition:
        torso_i = body_names.index(TORSO_LINK)
        torso_y = pos_to_yup(mot["body_pos_w"][0, torso_i]) - offset
        torso_R = rot_to_yup(mot["body_quat_w"][0, torso_i])
        torso_pos = np.zeros((1, J, 3), dtype=np.float64)
        torso_rot = np.tile(np.eye(3, dtype=np.float64), (1, J, 1, 1))
        torso_pos[0, sk.root_idx] = root_c[0]
        torso_skel_i = sk.bone_index[TORSO_SKEL]
        torso_pos[0, torso_skel_i] = torso_y
        torso_rot[0, torso_skel_i] = torso_R
        torso_gp = torch.as_tensor(torso_pos, dtype=torch.float32, device=device_t)
        torso_gr = torch.as_tensor(torso_rot, dtype=torch.float32, device=device_t)
        torso_constraint = SparseEEConstraintSet(
            sk,
            frame_indices=torch.zeros(1, dtype=torch.long, device=device_t),
            global_joints_positions=torso_gp,
            global_joints_rots=torso_gr,
            smooth_root_2d=torso_gp[:, sk.root_idx][:, [0, 2]].contiguous(),
            pos_joint_names=[TORSO_SKEL],
            rot_joint_names=[TORSO_SKEL],
            constrain_smooth_root=True,
            constrain_root_y=False,
            constrain_heading=False,
            label="frame0-torso-pose",
        )
        torso_dd, torso_idd = defaultdict(list), defaultdict(list)
        torso_constraint.update_constraints(torso_dd, torso_idd)
        torso_keys = sorted(torso_dd.keys())
        if torso_keys != [
            "global_joints_positions",
            "global_joints_rots",
            "smooth_root_2d",
        ]:
            raise RuntimeError(f"unexpected torso constraint keys: {torso_keys}")
        constraints.append(torso_constraint)
        torso_meta = {
            "link_name": TORSO_LINK,
            "kimodo_bone_name": TORSO_SKEL,
            "kimodo_frame": 0,
            "session_frame": 0,
            "position_yup_canon": torso_y.tolist(),
            "rotation_yup": torso_R.tolist(),
            "runtime_update_keys": torso_keys,
            "scope": "frame0_only",
        }

    meta = {
        "method": "custom_sparse_ee_session_heading",
        "session_npz": str(session_npz),
        "times_s": times,
        "session_frames": sess_frames.tolist(),
        "kimodo_frames": kimodo_frames.tolist(),
        "n_kimodo_frames": n_kimodo,
        "runtime_update_keys": runtime_keys,
        "pos_joint_names": cset.pos_joint_names,
        "rot_joint_names": cset.rot_joint_names,
        "constrain_root_y": False,
        "constrain_hand_roll": False,
        "root_offset_xz_kimodo": offset.tolist(),
        "first_heading_angle_rad": float(yaw_rad[0]),
        "first_heading_angle_deg": float(np.degrees(yaw_rad[0])),
        "heading_source": heading_source,
        "global_root_heading": cset.global_root_heading.detach().cpu().numpy().tolist(),
        "left_wrist_yup_canon": lw_c.tolist(),
        "right_wrist_yup_canon": rw_c.tolist(),
        "torso_init_condition": torso_meta,
    }
    return constraints, meta


def _move_constraints_to_device(constraints: list, device: str) -> None:
    for c in constraints:
        for attr in (
            "global_joints_positions",
            "global_joints_rots",
            "smooth_root_2d",
            "root_y_pos",
            "global_root_heading",
        ):
            val = getattr(c, attr, None)
            if torch.is_tensor(val):
                setattr(c, attr, val.to(device=device))
        for attr in ("frame_indices", "pos_indices", "rot_indices"):
            val = getattr(c, attr, None)
            if torch.is_tensor(val):
                setattr(c, attr, val.to(device="cpu"))


def load_kimodo_model(model_name: str):
    from kimodo import load_model
    from kimodo.exports.mujoco import MujocoQposConverter
    from kimodo.model.registry import get_model_info

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, resolved = load_model(model_name, device=device, default_family="Kimodo", return_resolved_name=True)
    info = get_model_info(resolved)
    print(f"[INFO] Kimodo loaded: {info.display_name if info else resolved} on {device}", flush=True)
    return model, MujocoQposConverter(model.skeleton), device


def generate_candidate(
    *,
    model,
    converter,
    device: str,
    session_npz: Path,
    seed: int,
    out_dir: Path,
    duration_s: float,
    diffusion_steps: int,
    cfg_weight: list[float],
    model_name: str,
    torso_init_condition: bool = False,
) -> tuple[Path, Path, dict]:
    from kimodo.exports.motion_io import save_kimodo_npz
    from kimodo.tools import seed_everything

    candidate_dir = out_dir / f"candidate_seed_{seed:02d}"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    constraints, cmeta = build_session_heading_constraint(
        session_npz,
        duration_s=duration_s,
        device="cpu",
        torso_init_condition=torso_init_condition,
    )
    _move_constraints_to_device(constraints, device)
    seed_everything(seed)
    num_frames = int(round(duration_s * float(model.fps)))
    first_heading = torch.tensor([float(cmeta["first_heading_angle_rad"])], device=device)
    output = model(
        PROMPT,
        num_frames,
        constraint_lst=constraints,
        num_denoising_steps=diffusion_steps,
        num_samples=1,
        post_processing=False,
        return_numpy=True,
        cfg_type="separated",
        cfg_weight=cfg_weight,
        first_heading_angle=first_heading,
        progress_bar=lambda x: x,
    )
    single = {
        k: (v[0] if hasattr(v, "shape") and len(getattr(v, "shape", ())) > 0 and v.shape[0] == 1 else v)
        for k, v in output.items()
    }
    npz_path = candidate_dir / "kimodo_motion.npz"
    save_kimodo_npz(str(npz_path), single)
    qpos = converter.dict_to_qpos(output, device)
    csv_path = candidate_dir / "kimodo_qpos.csv"
    converter.save_csv(qpos, str(csv_path))

    meta = {
        "seed": seed,
        "prompt": PROMPT,
        "model_name": model_name,
        "duration_s": duration_s,
        "num_frames": num_frames,
        "diffusion_steps": diffusion_steps,
        "cfg_weight": cfg_weight,
        "torso_init_condition": bool(torso_init_condition),
        "constraint_meta": cmeta,
    }
    (candidate_dir / "kimodo_constraints_debug.json").write_text(json.dumps(meta, indent=2))
    return npz_path, csv_path, meta


def _qpos_csv_to_scale_mj_order(csv_path: Path) -> tuple[np.ndarray, dict]:
    qpos = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if qpos.ndim == 1:
        qpos = qpos[None, :]
    if qpos.shape[1] != 36:
        raise ValueError(f"Expected Kimodo G1 qpos CSV with 36 columns; got {qpos.shape}")
    kimodo_names = _read_names_xml(KIMODO_G1_XML)
    scale_names = _read_names_xml(MJCF)
    q_kimodo = qpos[0, 7:].astype(np.float64)
    q_scale = np.asarray([q_kimodo[kimodo_names.index(n)] for n in scale_names], dtype=np.float64)
    return q_scale, {
        "csv": str(csv_path),
        "root_mujoco_frame0": qpos[0, :3].tolist(),
        "root_quat_wxyz_frame0": qpos[0, 3:7].tolist(),
        "kimodo_joint_order": kimodo_names,
        "scale_mjcf_joint_order": scale_names,
    }


def _build_qpos(model, root_pos, root_quat, q_mj, mj_joint_names):
    import mujoco

    qpos = np.zeros(model.nq, dtype=np.float64)
    qpos[:3] = np.asarray(root_pos, dtype=np.float64)
    qpos[3:7] = _quat_norm(root_quat)
    for i, name in enumerate(mj_joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos[int(model.jnt_qposadr[jid])] = float(q_mj[i])
    return qpos


def _fk_eval(model, data, root_pos, root_quat, q_mj, mj_joint_names, body_ids):
    import mujoco

    data.qpos[:] = _build_qpos(model, root_pos, root_quat, q_mj, mj_joint_names)
    mujoco.mj_forward(model, data)
    result = {
        "left_pos": data.xpos[body_ids["left"]].copy(),
        "right_pos": data.xpos[body_ids["right"]].copy(),
        "left_quat": data.xquat[body_ids["left"]].copy(),
        "right_quat": data.xquat[body_ids["right"]].copy(),
    }
    if "torso" in body_ids:
        result["torso_pos"] = data.xpos[body_ids["torso"]].copy()
        result["torso_quat"] = data.xquat[body_ids["torso"]].copy()
    return result


def _joint_bounds(model, mj_joint_names, names):
    import mujoco

    lows, highs = [], []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        rng = model.jnt_range[jid]
        lows.append(float(rng[0]))
        highs.append(float(rng[1]))
    return np.asarray(lows), np.asarray(highs)


def _reset_lower_body(q_mj: np.ndarray, mj_joint_names: list[str]) -> np.ndarray:
    """Keep Kimodo's upper-body proposal while restoring nominal standing legs."""
    q = np.asarray(q_mj, dtype=np.float64).copy()
    for i, name in enumerate(mj_joint_names):
        if name not in IK_JOINTS:
            q[i] = 0.0
    return q


def audit_init_state_physics(init_state_npz: Path, *, min_joint_margin_rad: float = 0.03) -> dict:
    """Check simple first-frame contact and joint-limit sanity conditions."""
    import mujoco

    state = np.load(init_state_npz, allow_pickle=True)
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    qpos = np.zeros(model.nq, dtype=np.float64)
    qpos[:3] = np.asarray(state["root_pos_w"], dtype=np.float64)
    qpos[3:7] = _quat_norm(np.asarray(state["root_quat_wxyz"], dtype=np.float64))
    joint_names = _as_str_list(state["joint_names"])
    for name, value in zip(joint_names, np.asarray(state["joint_pos"], dtype=np.float64)):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos[int(model.jnt_qposadr[jid])] = float(value)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    foot_geom_z = []
    for side in ("left", "right"):
        for i in range(1, 8):
            gid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot{i}_collision"
            )
            if gid >= 0:
                foot_geom_z.append(float(data.geom_xpos[gid, 2] - model.geom_size[gid, 0]))
    margins = []
    for name, value in zip(joint_names, np.asarray(state["joint_pos"], dtype=np.float64)):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = model.jnt_range[jid]
        margins.append((min(float(value - lo), float(hi - value)), name))
    margins.sort()
    foot_min_z = min(foot_geom_z) if foot_geom_z else float("nan")
    root_z = float(state["root_pos_w"][2])
    physical_plausible = bool(
        0.70 <= root_z <= 0.90
        and -0.03 <= foot_min_z <= 0.08
        and margins
        and margins[0][0] >= float(min_joint_margin_rad)
    )
    return {
        "physical_plausible": physical_plausible,
        "root_z_m": root_z,
        "foot_collision_approx_min_z_m": float(foot_min_z),
        "min_joint_margin_rad": float(margins[0][0]) if margins else None,
        "nearest_joint_limits": [[float(m), n] for m, n in margins[:5]],
        "criteria": {
            "root_z_m": [0.70, 0.90],
            "foot_collision_approx_min_z_m": [-0.03, 0.08],
            "min_joint_margin_rad": float(min_joint_margin_rad),
        },
    }


def refine_init_with_mujoco_ik(
    *,
    q_seed_mj: np.ndarray,
    session_npz: Path,
    out_dir: Path,
    source_meta: dict,
    allow_root_z: bool = False,
    max_fk_accept_m: float = 0.05,
    min_joint_margin_rad: float = 0.03,
    wrist_orientation_weight: float = 1.0,
    torso_ik_weight: float = 0.0,
) -> tuple[Path, dict, bool]:
    import mujoco
    from scipy.optimize import least_squares

    sess = _load_session(session_npz)
    mot = sess["npz"]
    joint_names = sess["joint_names"]
    body_names = sess["body_names"]
    mj_joint_names = _read_names_xml(MJCF)
    mj_index = {n: i for i, n in enumerate(mj_joint_names)}
    ik_joint_idx = np.asarray([mj_index[n] for n in IK_JOINTS], dtype=np.int64)

    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    body_ids = {
        "left": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, WRIST_LINKS[0]),
        "right": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, WRIST_LINKS[1]),
    }
    if torso_ik_weight > 0.0:
        body_ids["torso"] = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, TORSO_LINK
        )
    pel_i = body_names.index("pelvis")
    li = body_names.index(WRIST_LINKS[0])
    ri = body_names.index(WRIST_LINKS[1])
    root_ref = np.asarray(mot["body_pos_w"][0, pel_i], dtype=np.float64)
    # The tracking carrier is static by design.  Use the separately stored raw
    # camera heading for q0, while keeping that heading out of the UMI-2
    # tracking reference itself.
    if "kimodo_pelvis_heading_quat_wxyz" in mot.files:
        root_quat_ref = _quat_norm(
            np.asarray(mot["kimodo_pelvis_heading_quat_wxyz"][0], dtype=np.float64)
        )
        root_heading_source = "raw_hdf5_cam_pose"
    else:
        root_quat_ref = _quat_norm(
            np.asarray(mot["body_quat_w"][0, pel_i], dtype=np.float64)
        )
        root_heading_source = "session_npz_pelvis_body_quat"
    left_tgt = np.asarray(mot["body_pos_w"][0, li], dtype=np.float64)
    right_tgt = np.asarray(mot["body_pos_w"][0, ri], dtype=np.float64)
    left_quat_tgt = _quat_norm(np.asarray(mot["body_quat_w"][0, li], dtype=np.float64))
    right_quat_tgt = _quat_norm(np.asarray(mot["body_quat_w"][0, ri], dtype=np.float64))
    torso_tgt = torso_quat_tgt = None
    if torso_ik_weight > 0.0:
        torso_i = body_names.index(TORSO_LINK)
        torso_tgt = np.asarray(mot["body_pos_w"][0, torso_i], dtype=np.float64)
        torso_quat_tgt = _quat_norm(
            np.asarray(mot["body_quat_w"][0, torso_i], dtype=np.float64)
        )

    q_seed = np.asarray(q_seed_mj, dtype=np.float64).copy()
    q_seed = np.nan_to_num(q_seed, nan=0.0, posinf=0.0, neginf=0.0)
    all_lo, all_hi = _joint_bounds(model, mj_joint_names, mj_joint_names)
    q_seed = np.clip(q_seed, all_lo, all_hi)
    ik_lo, ik_hi = _joint_bounds(model, mj_joint_names, IK_JOINTS)
    joint_margin = max(0.0, float(min_joint_margin_rad))
    # Keep a small numerical buffer beyond the requested audit threshold so
    # float32 serialization cannot turn an exactly-bound solution into a
    # spurious post-optimization gate failure.
    solver_joint_margin = joint_margin + (1e-4 if joint_margin > 0.0 else 0.0)
    ik_safe_lo = ik_lo + solver_joint_margin
    ik_safe_hi = ik_hi - solver_joint_margin
    if np.any(ik_safe_lo > ik_safe_hi):
        bad = [
            name
            for name, lo, hi in zip(IK_JOINTS, ik_safe_lo, ik_safe_hi)
            if lo > hi
        ]
        raise ValueError(
            f"Requested joint-limit margin {joint_margin:.4f} rad leaves no feasible range: {bad}"
        )
    # The Kimodo proposal is a prior, not permission to start exactly at a
    # hardware limit.  Keep the seed inside the same feasible set as the IK.
    q_seed[ik_joint_idx] = np.clip(q_seed[ik_joint_idx], ik_safe_lo, ik_safe_hi)

    def pack(x: np.ndarray):
        if allow_root_z:
            root = np.array([x[0], x[1], x[2]], dtype=np.float64)
            yaw_delta = float(x[3])
            joints = x[4:]
        else:
            root = np.array([x[0], x[1], root_ref[2]], dtype=np.float64)
            yaw_delta = float(x[2])
            joints = x[3:]
        quat = _quat_mul_wxyz(_yaw_quat_wxyz(yaw_delta), root_quat_ref)
        q = q_seed.copy()
        q[ik_joint_idx] = joints
        return root, _quat_norm(quat), q, yaw_delta, joints

    def residual(x: np.ndarray) -> np.ndarray:
        root, quat, q, yaw_delta, joints = pack(x)
        fk = _fk_eval(model, data, root, quat, q, mj_joint_names, body_ids)
        res = []
        res.extend((fk["left_pos"] - left_tgt) * 25.0)
        res.extend((fk["right_pos"] - right_tgt) * 25.0)
        if wrist_orientation_weight > 0.0:
            # Raw hand-frame orientation is useful as a weak posture cue, but
            # it is not calibrated to the robot wrist-link frame precisely
            # enough to justify consuming the joint-limit safety margin.
            res.extend(
                _quat_error_rotvec_wxyz(fk["left_quat"], left_quat_tgt)
                * float(wrist_orientation_weight)
            )
            res.extend(
                _quat_error_rotvec_wxyz(fk["right_quat"], right_quat_tgt)
                * float(wrist_orientation_weight)
            )
        if torso_ik_weight > 0.0:
            assert torso_tgt is not None and torso_quat_tgt is not None
            torso_residual_scale = math.sqrt(float(torso_ik_weight))
            res.extend(
                (fk["torso_pos"] - torso_tgt) * 25.0 * torso_residual_scale
            )
            res.extend(
                _quat_error_rotvec_wxyz(fk["torso_quat"], torso_quat_tgt)
                * torso_residual_scale
            )
        res.extend((joints - q_seed[ik_joint_idx]) * 0.15)
        res.extend((root[:2] - root_ref[:2]) * 1.0)
        if allow_root_z:
            res.append((root[2] - root_ref[2]) * 4.0)
        res.append(yaw_delta * 0.25)
        return np.asarray(res, dtype=np.float64)

    if allow_root_z:
        x0 = np.concatenate([root_ref[:3], [0.0], q_seed[ik_joint_idx]])
        lb = np.concatenate([root_ref[:2] - 0.45, [root_ref[2] - 0.10], [-1.5], ik_safe_lo])
        ub = np.concatenate([root_ref[:2] + 0.45, [root_ref[2] + 0.10], [1.5], ik_safe_hi])
    else:
        x0 = np.concatenate([root_ref[:2], [0.0], q_seed[ik_joint_idx]])
        lb = np.concatenate([root_ref[:2] - 0.45, [-1.5], ik_safe_lo])
        ub = np.concatenate([root_ref[:2] + 0.45, [1.5], ik_safe_hi])

    fk_before = _fk_eval(model, data, root_ref, root_quat_ref, q_seed, mj_joint_names, body_ids)
    before_l = float(np.linalg.norm(fk_before["left_pos"] - left_tgt))
    before_r = float(np.linalg.norm(fk_before["right_pos"] - right_tgt))
    before_torso_pos = before_torso_rot = None
    if torso_ik_weight > 0.0:
        assert torso_tgt is not None and torso_quat_tgt is not None
        before_torso_pos = float(np.linalg.norm(fk_before["torso_pos"] - torso_tgt))
        before_torso_rot = float(
            np.linalg.norm(
                _quat_error_rotvec_wxyz(fk_before["torso_quat"], torso_quat_tgt)
            )
        )
    res = least_squares(
        residual,
        x0,
        bounds=(lb, ub),
        max_nfev=350,
        xtol=1e-6,
        ftol=1e-6,
        gtol=1e-6,
        verbose=0,
    )
    root_opt, quat_opt, q_opt, yaw_delta, joints = pack(res.x)
    fk_after = _fk_eval(model, data, root_opt, quat_opt, q_opt, mj_joint_names, body_ids)
    after_l = float(np.linalg.norm(fk_after["left_pos"] - left_tgt))
    after_r = float(np.linalg.norm(fk_after["right_pos"] - right_tgt))
    after_torso_pos = after_torso_rot = None
    if torso_ik_weight > 0.0:
        assert torso_tgt is not None and torso_quat_tgt is not None
        after_torso_pos = float(np.linalg.norm(fk_after["torso_pos"] - torso_tgt))
        after_torso_rot = float(
            np.linalg.norm(
                _quat_error_rotvec_wxyz(fk_after["torso_quat"], torso_quat_tgt)
            )
        )

    # If root-z was fixed and the solution is still outside the gate, retry once
    # with root-z allowed within a narrow band.
    if (not allow_root_z) and max(after_l, after_r) > max_fk_accept_m:
        return refine_init_with_mujoco_ik(
            q_seed_mj=q_seed,
            session_npz=session_npz,
            out_dir=out_dir,
            source_meta={**source_meta, "root_z_retry_from": "fixed_z"},
            allow_root_z=True,
            max_fk_accept_m=max_fk_accept_m,
            min_joint_margin_rad=joint_margin,
            wrist_orientation_weight=wrist_orientation_weight,
            torso_ik_weight=torso_ik_weight,
        )

    q_session = np.asarray([q_opt[mj_index[n]] for n in joint_names], dtype=np.float32)
    init_npz = out_dir / "init_state.npz"
    np.savez_compressed(
        init_npz,
        root_pos_w=root_opt.astype(np.float32),
        root_quat_wxyz=quat_opt.astype(np.float32),
        root_lin_vel_w=np.zeros(3, dtype=np.float32),
        root_ang_vel_w=np.zeros(3, dtype=np.float32),
        joint_pos=q_session,
        joint_vel=np.zeros(len(joint_names), dtype=np.float32),
        joint_names=np.asarray(joint_names),
        source=np.asarray(source_meta.get("source", "kimodo_ik")),
    )
    max_after = max(after_l, after_r)
    accepted = bool(max_after <= max_fk_accept_m)
    fk_gate_status = (
        "pass_lt3cm"
        if max_after < 0.03
        else ("warn_3to5cm" if max_after <= max_fk_accept_m else "reject_gt5cm")
    )
    summary = {
        "accepted": accepted,
        "fk_gate_status": fk_gate_status,
        "accept_threshold_m": float(max_fk_accept_m),
        "allow_root_z": bool(allow_root_z),
        "root_pos_w": root_opt.tolist(),
        "root_quat_wxyz": quat_opt.tolist(),
        "root_heading_source": root_heading_source,
        "yaw_delta_rad": float(yaw_delta),
        "yaw_delta_deg": float(np.degrees(yaw_delta)),
        "left_err_before_m": before_l,
        "right_err_before_m": before_r,
        "left_err_after_m": after_l,
        "right_err_after_m": after_r,
        "mean_err_after_m": float(0.5 * (after_l + after_r)),
        "min_joint_margin_rad": joint_margin,
        "solver_joint_limit_margin_rad": solver_joint_margin,
        "wrist_orientation_weight": float(wrist_orientation_weight),
        "torso_ik_weight": float(torso_ik_weight),
        "torso_err_before_m": before_torso_pos,
        "torso_err_after_m": after_torso_pos,
        "torso_rot_err_before_rad": before_torso_rot,
        "torso_rot_err_after_rad": after_torso_rot,
        "fk_left_after": fk_after["left_pos"].tolist(),
        "fk_right_after": fk_after["right_pos"].tolist(),
        "target_left": left_tgt.tolist(),
        "target_right": right_tgt.tolist(),
        "solver": {
            "success": bool(res.success),
            "status": int(res.status),
            "message": str(res.message),
            "cost": float(res.cost),
            "nfev": int(res.nfev),
        },
        "source_meta": source_meta,
    }
    if torso_ik_weight > 0.0:
        assert torso_tgt is not None and torso_quat_tgt is not None
        summary.update(
            {
                "fk_torso_after": fk_after["torso_pos"].tolist(),
                "fk_torso_quat_after_wxyz": fk_after["torso_quat"].tolist(),
                "target_torso": torso_tgt.tolist(),
                "target_torso_quat_wxyz": torso_quat_tgt.tolist(),
            }
        )
    (out_dir / "ik_summary.json").write_text(json.dumps(summary, indent=2))
    return init_npz, summary, accepted


def write_baseline_init(
    *,
    name: str,
    src_motion_npz: Path,
    out_dir: Path,
    source: str,
) -> Candidate:
    sess = _load_session(src_motion_npz)
    mot = sess["npz"]
    body_names = sess["body_names"]
    root_i = body_names.index("pelvis")
    out_dir.mkdir(parents=True, exist_ok=True)
    init_npz = out_dir / "init_state.npz"
    np.savez_compressed(
        init_npz,
        root_pos_w=np.asarray(mot["body_pos_w"][0, root_i], dtype=np.float32),
        root_quat_wxyz=np.asarray(mot["body_quat_w"][0, root_i], dtype=np.float32),
        root_lin_vel_w=np.zeros(3, dtype=np.float32),
        root_ang_vel_w=np.zeros(3, dtype=np.float32),
        joint_pos=np.asarray(mot["joint_pos"][0], dtype=np.float32),
        joint_vel=np.zeros(len(sess["joint_names"]), dtype=np.float32),
        joint_names=np.asarray(sess["joint_names"]),
        source=np.asarray(source),
    )
    summary = {"accepted": True, "source": source, "src_motion_npz": str(src_motion_npz)}
    (out_dir / "ik_summary.json").write_text(json.dumps(summary, indent=2))
    return Candidate(name, None, out_dir, init_npz, summary, True, source)


def run_tracking(
    *,
    candidate: Candidate,
    ref_yaml: Path,
    args,
    stage_name: str,
) -> dict:
    eval_out = candidate.out_dir / "tracking"
    eval_out.mkdir(parents=True, exist_ok=True)
    motion_len = int(np.load(args.session_npz, allow_pickle=True)["joint_pos"].shape[0])
    steps = int(args.max_steps) if args.max_steps else motion_len
    video_len = steps
    cmd = [
        args.python,
        str(PLAY_SCRIPT),
        "--motion_file",
        str(ref_yaml),
        "--init_state_npz",
        str(candidate.init_state_npz),
        "--mode_index",
        "1",
        "--eval_out",
        str(eval_out),
        "--stage_name",
        stage_name,
        "--max_steps",
        str(steps),
        "--video_length",
        str(video_len),
        "--num_envs",
        "1",
        "--load_run",
        args.load_run,
        "--checkpoint",
        args.checkpoint,
        "--headless",
    ]
    if args.hold_first_action:
        cmd.append("--hold_motion_q0")
    if args.video:
        cmd.append("--video")
    log_path = eval_out / "play.log"
    print("[INFO] tracking:", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    env.setdefault("ACCEPT_EULA", "Y")
    with log_path.open("w") as log_f:
        proc = subprocess.run(
            cmd,
            cwd=str(SCALETRACK_ROOT),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    summary_path = eval_out / f"{stage_name}_error_summary.json"
    if proc.returncode != 0:
        return {
            "stage": stage_name,
            "returncode": proc.returncode,
            "log": str(log_path),
            "error": "tracking subprocess failed",
        }
    if not summary_path.is_file():
        return {
            "stage": stage_name,
            "returncode": proc.returncode,
            "log": str(log_path),
            "error": f"missing summary {summary_path}",
        }
    summary = json.loads(summary_path.read_text())
    summary["returncode"] = proc.returncode
    summary["log"] = str(log_path)
    return summary


def _python_resolved(path: str | Path) -> Path:
    try:
        return Path(path).resolve()
    except Exception:
        return Path(str(path))


def _json_socket_request(
    sock_path: str | Path,
    payload: dict,
    *,
    timeout_s: float = 3600.0,
) -> dict:
    encoded = (json.dumps(payload) + "\n").encode("utf-8")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(float(timeout_s))
        sock.connect(str(sock_path))
        sock.sendall(encoded)
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError("Kimodo worker closed the connection")
            buf += chunk
        reply = json.loads(buf.split(b"\n", 1)[0].decode("utf-8"))
        if not isinstance(reply, dict):
            raise ValueError("Kimodo worker reply must be a JSON object")
        return reply
    finally:
        sock.close()


def run_kimodo_generation_worker(args, seeds: list[int], out_dir: Path) -> None:
    if not seeds:
        return
    sock = str(os.environ.get("KIMODO_WORKER_SOCK", "")).strip()
    if sock:
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "op": "generate",
            "session_npz": str(args.session_npz.resolve()),
            "out_dir": str(out_dir.resolve()),
            "seeds": ",".join(str(s) for s in seeds),
            "duration": float(args.duration),
            "diffusion_steps": int(args.diffusion_steps),
            "cfg_weight": [float(args.cfg_weight[0]), float(args.cfg_weight[1])],
            "model_name": args.model_name,
            "torso_init_condition": bool(args.torso_init_condition),
        }
        print(f"[INFO] kimodo generation via resident worker {sock}", flush=True)
        try:
            reply = _json_socket_request(sock, payload, timeout_s=3600.0)
        except (OSError, ConnectionError, TimeoutError, socket.timeout, ValueError) as error:
            raise RuntimeError(
                f"Kimodo resident worker request failed ({error}); "
                "unset KIMODO_WORKER_SOCK to use the per-session subprocess"
            ) from error
        if not reply.get("ok"):
            raise RuntimeError(
                reply.get("error") or "Kimodo resident worker generate failed"
            )
        return
    log_path = out_dir / "kimodo_generation.log"
    env = os.environ.copy()
    prepend = [str(KIMODO_ROOT), str(_HERE)]
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(prepend + ([old_pythonpath] if old_pythonpath else []))
    env.setdefault("CHECKPOINT_DIR", str(ROOT / "kimodo_models"))
    env.setdefault("LOCAL_CACHE", "True")
    env.setdefault("TEXT_ENCODER_MODE", "local")
    env.setdefault("TEXT_ENCODERS_DIR", str(ROOT / "kimodo_models" / "text_encoders"))
    env.setdefault("TEXT_ENCODER_DEVICE", "cpu")
    cmd = [
        str(args.kimodo_python),
        str(Path(__file__).resolve()),
        "--kimodo_worker",
        "--session_npz",
        str(args.session_npz.resolve()),
        "--out_dir",
        str(out_dir),
        "--seeds",
        ",".join(str(s) for s in seeds),
        "--duration",
        str(args.duration),
        "--diffusion_steps",
        str(args.diffusion_steps),
        "--cfg_weight",
        str(float(args.cfg_weight[0])),
        str(float(args.cfg_weight[1])),
        "--model_name",
        args.model_name,
    ]
    if args.torso_init_condition:
        cmd.append("--torso_init_condition")
    print("[INFO] kimodo generation worker:", " ".join(cmd), flush=True)
    with log_path.open("w") as log_f:
        proc = subprocess.run(
            cmd,
            cwd=str(args.kimodo_cwd),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Kimodo worker failed with code {proc.returncode}; log={log_path}")


def kimodo_worker_main(args) -> None:
    os.environ.setdefault("CHECKPOINT_DIR", str(ROOT / "kimodo_models"))
    os.environ.setdefault("LOCAL_CACHE", "True")
    os.environ.setdefault("TEXT_ENCODER_MODE", "local")
    os.environ.setdefault("TEXT_ENCODERS_DIR", str(ROOT / "kimodo_models" / "text_encoders"))
    os.environ.setdefault("TEXT_ENCODER_DEVICE", "cpu")
    seeds = parse_seeds(args.seeds)
    model, converter, kimodo_device = load_kimodo_model(args.model_name)
    generated = []
    for seed in seeds:
        _, csv_path, meta = generate_candidate(
            model=model,
            converter=converter,
            device=kimodo_device,
            session_npz=args.session_npz.resolve(),
            seed=seed,
            out_dir=args.out_dir.resolve(),
            duration_s=args.duration,
            diffusion_steps=args.diffusion_steps,
            cfg_weight=[float(args.cfg_weight[0]), float(args.cfg_weight[1])],
            model_name=args.model_name,
            torso_init_condition=bool(args.torso_init_condition),
        )
        generated.append({"seed": seed, "csv": str(csv_path), "meta": meta})
    print(json.dumps({"generated": generated}, indent=2), flush=True)


def tracking_score(summary: dict) -> tuple[float, float, float]:
    if "error" in summary:
        return (float("inf"), float("inf"), float("inf"))
    first = summary.get("mean_active_link_error_m", {}).get("first_second", {}).get("mean")
    full = summary.get("mean_active_link_error_m", {}).get("post_gate", {}).get("mean")
    p95 = summary.get("mean_active_link_error_m", {}).get("post_gate", {}).get("p95")
    return (
        float(first if first is not None else float("inf")),
        float(full if full is not None else float("inf")),
        float(p95 if p95 is not None else float("inf")),
    )


def write_results(out_dir: Path, rows: list[dict]) -> None:
    rows_sorted = sorted(rows, key=lambda r: tracking_score(r.get("tracking", {})))
    (out_dir / "sweep_summary.json").write_text(json.dumps({"results": rows_sorted}, indent=2))

    csv_path = out_dir / "sweep_summary.csv"
    fieldnames = [
        "name",
        "source",
        "accepted",
        "seed",
        "fk_left_after_m",
        "fk_right_after_m",
        "step0_left_m",
        "step0_right_m",
        "first_second_mean_m",
        "full_mean_m",
        "full_p95_m",
        "tracking_error",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_sorted:
            tr = row.get("tracking", {})
            fk = row.get("fk_summary", {})
            step0 = tr.get("step0_error_m", {}) if isinstance(tr, dict) else {}
            score = tracking_score(tr if isinstance(tr, dict) else {})
            writer.writerow(
                {
                    "name": row["name"],
                    "source": row["source"],
                    "accepted": row["accepted"],
                    "seed": row.get("seed"),
                    "fk_left_after_m": fk.get("left_err_after_m"),
                    "fk_right_after_m": fk.get("right_err_after_m"),
                    "step0_left_m": step0.get(WRIST_LINKS[0]),
                    "step0_right_m": step0.get(WRIST_LINKS[1]),
                    "first_second_mean_m": score[0],
                    "full_mean_m": score[1],
                    "full_p95_m": score[2],
                    "tracking_error": tr.get("error") if isinstance(tr, dict) else None,
                }
            )

    md = ["# Clean Kimodo Init Sweep", ""]
    md.append("| rank | name | accepted | FK L/R m | step0 L/R m | first 1s mean | full mean | full p95 |")
    md.append("|---:|---|---|---:|---:|---:|---:|---:|")
    for rank, row in enumerate(rows_sorted, 1):
        tr = row.get("tracking", {})
        fk = row.get("fk_summary", {})
        step0 = tr.get("step0_error_m", {}) if isinstance(tr, dict) else {}
        score = tracking_score(tr if isinstance(tr, dict) else {})
        md.append(
            "| {rank} | {name} | {acc} | {fl:.3f}/{fr:.3f} | {sl:.3f}/{sr:.3f} | {s0:.3f} | {s1:.3f} | {s2:.3f} |".format(
                rank=rank,
                name=row["name"],
                acc=str(row["accepted"]),
                fl=float(fk.get("left_err_after_m", float("nan"))),
                fr=float(fk.get("right_err_after_m", float("nan"))),
                sl=float(step0.get(WRIST_LINKS[0], float("nan"))),
                sr=float(step0.get(WRIST_LINKS[1], float("nan"))),
                s0=score[0],
                s1=score[1],
                s2=score[2],
            )
        )
    md.append("")
    md.append(f"CSV: `{csv_path}`")
    (out_dir / "RESULTS_CLEAN_KIMODO_INIT.md").write_text("\n".join(md) + "\n")


def parse_seeds(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session_npz", type=Path, default=SESSION_NPZ)
    ap.add_argument("--targets_npz", type=Path, default=TARGETS_NPZ)
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7")
    ap.add_argument("--duration", type=float, default=1.5)
    ap.add_argument("--diffusion_steps", type=int, default=100)
    ap.add_argument("--cfg_weight", type=float, nargs=2, default=[2.0, 5.0])
    ap.add_argument("--model_name", type=str, default="Kimodo-G1-RP-v1")
    ap.add_argument("--max_fk_accept_m", type=float, default=0.05)
    ap.add_argument(
        "--reset_lower_body",
        action="store_true",
        help="Reset Kimodo leg joints to nominal standing while retaining waist/arm proposal joints.",
    )
    ap.add_argument(
        "--physics_gate",
        action="store_true",
        help="Reject initial states with bad foot height or saturated joints after FK IK.",
    )
    ap.add_argument("--min_joint_margin_rad", type=float, default=0.03)
    ap.add_argument(
        "--wrist_orientation_weight",
        type=float,
        default=1.0,
        help=(
            "Soft frame-0 wrist orientation residual weight in IK. "
            "Raw hand orientation remains a Kimodo constraint; position drives robot q0 fitting."
        ),
    )
    ap.add_argument(
        "--torso_init_condition",
        action="store_true",
        help="Add a 6D torso pose constraint at Kimodo frame 0 only.",
    )
    ap.add_argument(
        "--torso_ik_weight",
        type=float,
        default=0.0,
        help=(
            "Squared-cost weight for the frame-0 torso position/orientation IK residual. "
            "Zero preserves the wrist-only IK exactly."
        ),
    )
    ap.add_argument("--max_steps", type=int, default=0, help="0 means full motion length.")
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument(
        "--kimodo_python",
        type=Path,
        default=DEFAULT_KIMODO_PYTHON if DEFAULT_KIMODO_PYTHON.is_file() else Path(sys.executable),
        help="Python executable used for Kimodo generation; defaults to envs/kimodo when present.",
    )
    ap.add_argument(
        "--kimodo_cwd",
        type=Path,
        default=KIMODO_ROOT,
        help="Working directory for the Kimodo worker so local kimodo imports resolve cleanly.",
    )
    ap.add_argument("--load_run", type=str, default=DEFAULT_CHECKPOINT_RUN)
    ap.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--video", action="store_true", help="Record tracking videos for every run.")
    ap.add_argument(
        "--hold_first_action",
        action="store_true",
        default=False,
        help="Optional ablation: seed the first action to hold init q0. Default is off for clean init-state evaluation.",
    )
    ap.add_argument("--no_hold_first_action", action="store_false", dest="hold_first_action")
    ap.add_argument("--skip_kimodo", action="store_true", help="Reuse existing candidate dirs if present.")
    ap.add_argument(
        "--no_legacy_baselines",
        action="store_true",
        help="Do not add the historical session_umi2-derived Minkowski baseline.",
    )
    ap.add_argument(
        "--reuse_kimodo_from",
        type=Path,
        default=None,
        help="Reuse candidate_seed_XX/kimodo_qpos.csv from another sweep directory while writing new outputs here.",
    )
    ap.add_argument("--skip_tracking", action="store_true")
    ap.add_argument(
        "--kimodo_worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = ap.parse_args()

    if args.torso_ik_weight < 0.0:
        ap.error("--torso_ik_weight must be non-negative")

    if args.kimodo_worker:
        kimodo_worker_main(args)
        return

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    preflight(args.session_npz.resolve(), args.targets_npz.resolve(), out_dir)
    ref_yaml = write_reference_yaml(args.session_npz.resolve(), out_dir)

    candidates: list[Candidate] = []
    baseline_joint0 = write_baseline_init(
        name="baseline_joint0",
        src_motion_npz=args.session_npz.resolve(),
        out_dir=out_dir / "baseline_joint0",
        source="session_joint0",
    )
    if args.physics_gate:
        baseline_audit = audit_init_state_physics(
            baseline_joint0.init_state_npz,
            min_joint_margin_rad=float(args.min_joint_margin_rad),
        )
        baseline_joint0.fk_summary = {"physics_audit": baseline_audit}
        (baseline_joint0.out_dir / "ik_summary.json").write_text(
            json.dumps(baseline_joint0.fk_summary, indent=2)
        )
        baseline_joint0.accepted = bool(baseline_audit["physical_plausible"])
    candidates.append(baseline_joint0)

    mink_baseline = SESSION_DIR / "scaletrack_eval" / "fair_init_ablation" / "motions" / "session_umi2_init_mink_q0.npz"
    if (not args.no_legacy_baselines) and mink_baseline.is_file():
        candidates.append(
            write_baseline_init(
                name="baseline_mink_q0",
                src_motion_npz=mink_baseline,
                out_dir=out_dir / "baseline_mink_q0",
                source="existing_clean_mink_q0",
            )
        )

    seeds = parse_seeds(args.seeds)
    model = converter = kimodo_device = None
    if _KIMODO_IMPORT_ERROR is not None:
        # The published standalone pipeline does not require the optional
        # Kimodo generator package. Use the frozen frame-0 ScaleTrack joints
        # as a deterministic initial state and continue the physics/IK gate.
        print(f"[WARN] Kimodo helpers unavailable ({_KIMODO_IMPORT_ERROR}); using session joint0 fallback", flush=True)
        with np.load(args.session_npz.resolve(), allow_pickle=True) as motion:
            q_seed_fallback = np.asarray(motion["joint_pos"][0], dtype=np.float64)
        for seed in seeds:
            cdir = out_dir / f"candidate_seed_{seed:02d}"
            cdir.mkdir(parents=True, exist_ok=True)
            q_seed = _reset_lower_body(q_seed_fallback, _read_names_xml(MJCF)) if args.reset_lower_body else q_seed_fallback.copy()
            init_npz, ik_summary, accepted = refine_init_with_mujoco_ik(
                q_seed_mj=q_seed,
                session_npz=args.session_npz.resolve(),
                out_dir=cdir,
                source_meta={"seed": seed, "source": "session_joint0_fallback", "kimodo_import_error": str(_KIMODO_IMPORT_ERROR)},
                max_fk_accept_m=float(args.max_fk_accept_m),
                min_joint_margin_rad=float(args.min_joint_margin_rad),
                wrist_orientation_weight=float(args.wrist_orientation_weight),
                torso_ik_weight=float(args.torso_ik_weight),
            )
            candidates.append(Candidate(name=f"kimodo_seed_{seed:02d}", seed=seed, out_dir=cdir, init_state_npz=init_npz, fk_summary=ik_summary, accepted=accepted, source="session_joint0_fallback"))
        seeds = []
    use_generated_csv = bool(args.skip_kimodo or args.reuse_kimodo_from is not None)
    if not args.skip_kimodo and _KIMODO_IMPORT_ERROR is None:
        if os.environ.get("KIMODO_WORKER_SOCK", "").strip() or (
            _python_resolved(args.kimodo_python) != _python_resolved(sys.executable)
        ):
            run_kimodo_generation_worker(args, seeds, out_dir)
            use_generated_csv = True
        else:
            model, converter, kimodo_device = load_kimodo_model(args.model_name)

    for seed in seeds:
        cdir = out_dir / f"candidate_seed_{seed:02d}"
        if use_generated_csv:
            source_cdir = (
                args.reuse_kimodo_from.resolve() / f"candidate_seed_{seed:02d}"
                if args.reuse_kimodo_from is not None
                else cdir
            )
            csv_path = source_cdir / "kimodo_qpos.csv"
            if not csv_path.is_file():
                print(f"[WARN] missing generated Kimodo CSV {csv_path}; candidate skipped")
                continue
            cdir.mkdir(parents=True, exist_ok=True)
            meta_path = source_cdir / "kimodo_constraints_debug.json"
            if meta_path.is_file():
                gen_meta = json.loads(meta_path.read_text())
                if args.reuse_kimodo_from is not None:
                    gen_meta["source"] = f"reused_kimodo_csv_from:{args.reuse_kimodo_from.resolve()}"
                else:
                    gen_meta["source"] = "reused_kimodo_csv" if args.skip_kimodo else "kimodo_worker_csv"
            else:
                gen_meta = {
                    "seed": seed,
                    "source": (
                        f"reused_kimodo_csv_from:{args.reuse_kimodo_from.resolve()}"
                        if args.reuse_kimodo_from is not None
                        else ("reused_kimodo_csv" if args.skip_kimodo else "kimodo_worker_csv")
                    ),
                }
        else:
            _, csv_path, gen_meta = generate_candidate(
                model=model,
                converter=converter,
                device=kimodo_device,
                session_npz=args.session_npz.resolve(),
                seed=seed,
                out_dir=out_dir,
                duration_s=args.duration,
                diffusion_steps=args.diffusion_steps,
                cfg_weight=[float(args.cfg_weight[0]), float(args.cfg_weight[1])],
                model_name=args.model_name,
                torso_init_condition=bool(args.torso_init_condition),
            )
        q_seed_mj, qmeta = _qpos_csv_to_scale_mj_order(csv_path)
        if args.reset_lower_body:
            q_seed_mj = _reset_lower_body(q_seed_mj, _read_names_xml(MJCF))
        init_npz, ik_summary, accepted = refine_init_with_mujoco_ik(
            q_seed_mj=q_seed_mj,
            session_npz=args.session_npz.resolve(),
            out_dir=cdir,
            source_meta={
                **gen_meta,
                **qmeta,
                "source": f"kimodo_seed_{seed:02d}",
                "lower_body_reset": bool(args.reset_lower_body),
            },
            max_fk_accept_m=float(args.max_fk_accept_m),
            min_joint_margin_rad=float(args.min_joint_margin_rad),
            wrist_orientation_weight=float(args.wrist_orientation_weight),
            torso_ik_weight=float(args.torso_ik_weight),
        )
        if args.physics_gate:
            physics_audit = audit_init_state_physics(
                init_npz,
                min_joint_margin_rad=float(args.min_joint_margin_rad),
            )
            ik_summary["physics_audit"] = physics_audit
            accepted = bool(accepted and physics_audit["physical_plausible"])
            (cdir / "ik_summary.json").write_text(json.dumps(ik_summary, indent=2))
        candidates.append(
            Candidate(
                name=f"kimodo_seed_{seed:02d}",
                seed=seed,
                out_dir=cdir,
                init_state_npz=init_npz,
                fk_summary=ik_summary,
                accepted=accepted,
                source="kimodo_ik",
            )
        )

    rows: list[dict] = []
    for cand in candidates:
        if not cand.accepted:
            tracking = {"error": "rejected_by_init_gate"}
        elif args.skip_tracking:
            tracking = {"error": "skip_tracking"}
        else:
            tracking = run_tracking(candidate=cand, ref_yaml=ref_yaml, args=args, stage_name=cand.name)
        rows.append(
            {
                "name": cand.name,
                "seed": cand.seed,
                "source": cand.source,
                "accepted": cand.accepted,
                "init_state_npz": str(cand.init_state_npz),
                "candidate_dir": str(cand.out_dir),
                "fk_summary": cand.fk_summary,
                "tracking": tracking,
            }
        )
        write_results(out_dir, rows)

    write_results(out_dir, rows)
    print(json.dumps({"out_dir": str(out_dir), "results": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
