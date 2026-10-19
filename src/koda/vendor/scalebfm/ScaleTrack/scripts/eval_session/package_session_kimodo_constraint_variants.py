#!/usr/bin/env python3
"""Fair-init Kimodo SparseEE wrist variants (NO Mink).

Fair-init rule
--------------
``body_pos_w`` / ``body_quat_w`` must stay identical to joint0/mink.
Only ``joint_pos[0]`` may differ. Packs that rewrite pelvis xyz/quat
(``wrist_noyaw``, ``session_heading_ik``, ``armik``) are INVALID.

Deliverable: ``wrist_heading_tf``
  1. Measure session pelvis yaw (Y-up) ≈ +90°.
  2. Δyaw = 0 − session_yaw (align wrists into Kimodo canonical heading ≈ 0°).
  3. Rotate wrist pos+ori about vertical through root by Δyaw; SparseEE wrists
     (+ API-forced smooth_root_2d); NO heading / root_y / Mink.
  4. Generate Kimodo; pack ONLY frame-0 joints onto frozen joint0 body_*.
  Absolute Kimodo world yaw never enters the ScaleTrack reference.

Legacy variants ``wrist_noyaw`` / ``cam_torso_wrists`` kept for continuity but
their ``pack_q0`` path that rewrites pelvis is not fair-init.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from package_kimodo_q0_ablation import (  # noqa: E402
    ISAAC_ORDER_JSON,
    _write_yaml,
)
from package_session_kimodo_wrist_q0_ablation import (  # noqa: E402
    _pos_zup_to_yup,
    _quat_wxyz_to_mat,
    _rot_zup_to_yup,
    run_kimodo_generate,
)

KODA_HOME = Path(os.environ.get("KODA_HOME", Path.cwd()))
KODA_CACHE = Path(os.environ.get("KODA_CACHE", KODA_HOME / ".cache"))
DEFAULT_SESSION = Path(os.environ.get("KODA_SESSION_MOTION", KODA_CACHE / "session_umi2.npz"))
DEFAULT_CAMERA_TARGETS = Path(os.environ.get("KODA_CAMERA_TARGETS", KODA_CACHE / "session_targets_zup.npz"))
DEFAULT_ABLATION = Path(os.environ.get("KODA_ABLATION_ROOT", KODA_CACHE / "fair_init_ablation"))
DEFAULT_OUT = DEFAULT_ABLATION / "motions"

# session torso_link ↔ Kimodo G1 waist_pitch_skel (last waist DOF; parent of shoulders;
# G1Skeleton34 has no separate torso_skel).
TORSO_SESSION = "torso_link"
TORSO_KIMODO = "waist_pitch_skel"
WRIST_SESSION = ("left_wrist_yaw_link", "right_wrist_yaw_link")
WRIST_KIMODO = ("left_wrist_yaw_skel", "right_wrist_yaw_skel")
DEFAULT_JOINT0 = DEFAULT_OUT / "session_umi2_init_joint0.npz"


def _yaw_deg_about_y(R: np.ndarray) -> float:
    """Heading degrees from a Y-up rotation matrix (atan2 of X-row xz)."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return float(np.degrees(np.arctan2(R[0, 2], R[0, 0])))


def _rot_y(deg: float) -> np.ndarray:
    a = np.radians(float(deg))
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _apply_heading_delta_yup(
    *,
    root_c: np.ndarray,
    root_R: np.ndarray,
    lw_c: np.ndarray,
    rw_c: np.ndarray,
    lw_R: np.ndarray,
    rw_R: np.ndarray,
    torso_c: np.ndarray | None = None,
    torso_R: np.ndarray | None = None,
    target_yaw_deg: float = 0.0,
) -> dict:
    """Rotate EE poses about vertical through each-frame root by Δyaw.

    Δyaw = target_yaw − session_pelvis_yaw(frame0). Positions: p' = root + Ry·(p−root).
    Rotations: R' = Ry · R. Root xz stays put (smooth_root unchanged).
    """
    yaw0 = _yaw_deg_about_y(root_R[0])
    delta = float(target_yaw_deg) - yaw0
    Ry = _rot_y(delta)

    def _tf_pos(p: np.ndarray) -> np.ndarray:
        out = np.empty_like(p, dtype=np.float64)
        for t in range(len(p)):
            out[t] = root_c[t] + Ry @ (p[t] - root_c[t])
        return out

    def _tf_R(Rm: np.ndarray) -> np.ndarray:
        return np.einsum("ij,tjk->tik", Ry, Rm)

    out = {
        "delta_yaw_deg": delta,
        "session_yaw_deg_frame0": yaw0,
        "target_yaw_deg": float(target_yaw_deg),
        "lw_c": _tf_pos(lw_c),
        "rw_c": _tf_pos(rw_c),
        "lw_R": _tf_R(lw_R),
        "rw_R": _tf_R(rw_R),
        "root_c": np.asarray(root_c, dtype=np.float64).copy(),
        "root_R": _tf_R(root_R),
    }
    if torso_c is not None:
        out["torso_c"] = _tf_pos(torso_c)
        out["torso_R"] = _tf_R(torso_R) if torso_R is not None else None
    else:
        out["torso_c"] = None
        out["torso_R"] = None
    return out


class SparseEEConstraintSet:
    """Minimal EE constraint: only listed bone pos/rot; optional root fields.

    Does **not** call skeleton.expand_joint_names (which forces LeftHand→hand_roll
    pos and always pairs with stock update_constraints that pins heading/root_y).
    """

    name = "end-effector"

    def __init__(
        self,
        skeleton,
        frame_indices: Tensor,
        global_joints_positions: Tensor,
        global_joints_rots: Tensor,
        smooth_root_2d: Optional[Tensor],
        *,
        pos_joint_names: list[str],
        rot_joint_names: list[str],
        constrain_smooth_root: bool = True,
        constrain_root_y: bool = False,
        constrain_heading: bool = False,
        label: str = "sparse-ee",
    ) -> None:
        from kimodo.constraints import compute_global_heading, create_pairs

        self.skeleton = skeleton
        self.frame_indices = frame_indices
        self.joint_names = list(dict.fromkeys(pos_joint_names + rot_joint_names))
        self.pos_joint_names = list(pos_joint_names)
        self.rot_joint_names = list(rot_joint_names)
        self.label = label
        self.constrain_smooth_root = constrain_smooth_root
        self.constrain_root_y = constrain_root_y
        self.constrain_heading = constrain_heading

        self.pos_indices = torch.tensor(
            [skeleton.bone_index[n] for n in pos_joint_names], dtype=torch.long
        )
        self.rot_indices = torch.tensor(
            [skeleton.bone_index[n] for n in rot_joint_names], dtype=torch.long
        )

        assert len(global_joints_positions) == len(frame_indices)
        assert len(global_joints_rots) == len(frame_indices)

        if smooth_root_2d is not None and smooth_root_2d.shape[-1] == 3:
            smooth_root_2d = smooth_root_2d[..., [0, 1]]
        if smooth_root_2d is None:
            smooth_root_2d = global_joints_positions[:, skeleton.root_idx, [0, 2]]
        assert len(smooth_root_2d) == len(frame_indices)

        self.global_joints_positions = global_joints_positions
        self.global_joints_rots = global_joints_rots
        self.smooth_root_2d = smooth_root_2d
        self.root_y_pos = global_joints_positions[:, skeleton.root_idx, 1]
        # Stored for inspection only; NOT appended unless constrain_heading.
        self.global_root_heading = compute_global_heading(global_joints_positions, skeleton)
        self._create_pairs = create_pairs

    def update_constraints(self, data_dict: dict, index_dict: dict) -> None:
        crop_frames_indexing = torch.arange(
            len(self.frame_indices), device=self.frame_indices.device
        )
        create_pairs = self._create_pairs

        if len(self.pos_indices):
            pos_indices_real = create_pairs(self.frame_indices, self.pos_indices)
            pos_indices_crop = create_pairs(crop_frames_indexing, self.pos_indices)
            data_dict["global_joints_positions"].append(
                self.global_joints_positions[tuple(pos_indices_crop.T)]
            )
            index_dict["global_joints_positions"].append(pos_indices_real)

        if len(self.rot_indices):
            rot_indices_real = create_pairs(self.frame_indices, self.rot_indices)
            rot_indices_crop = create_pairs(crop_frames_indexing, self.rot_indices)
            data_dict["global_joints_rots"].append(
                self.global_joints_rots[tuple(rot_indices_crop.T)]
            )
            index_dict["global_joints_rots"].append(rot_indices_real)

        # API: global positions require smooth_root xz mask (kimodo_motionrep.py).
        if self.constrain_smooth_root:
            data_dict["smooth_root_2d"].append(self.smooth_root_2d)
            index_dict["smooth_root_2d"].append(self.frame_indices)

        if self.constrain_root_y:
            data_dict["root_y_pos"].append(self.root_y_pos)
            index_dict["root_y_pos"].append(self.frame_indices)

        if self.constrain_heading:
            data_dict["global_root_heading"].append(self.global_root_heading)
            index_dict["global_root_heading"].append(self.frame_indices)

    def crop_move(self, start: int, end: int) -> "SparseEEConstraintSet":
        mask = (self.frame_indices >= start) & (self.frame_indices < end)
        return SparseEEConstraintSet(
            self.skeleton,
            self.frame_indices[mask] - start,
            self.global_joints_positions[mask],
            self.global_joints_rots[mask],
            self.smooth_root_2d[mask],
            pos_joint_names=self.pos_joint_names,
            rot_joint_names=self.rot_joint_names,
            constrain_smooth_root=self.constrain_smooth_root,
            constrain_root_y=self.constrain_root_y,
            constrain_heading=self.constrain_heading,
            label=self.label,
        )

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "SparseEEConstraintSet":
        from kimodo.constraints import _tensor_to

        self.frame_indices = _tensor_to(self.frame_indices, device, dtype)
        self.pos_indices = _tensor_to(self.pos_indices, device, dtype)
        self.rot_indices = _tensor_to(self.rot_indices, device, dtype)
        self.root_y_pos = _tensor_to(self.root_y_pos, device, dtype)
        self.global_joints_positions = _tensor_to(
            self.global_joints_positions, device, dtype
        )
        self.global_root_heading = _tensor_to(self.global_root_heading, device, dtype)
        self.global_joints_rots = _tensor_to(self.global_joints_rots, device, dtype)
        self.smooth_root_2d = _tensor_to(self.smooth_root_2d, device, dtype)
        if device is not None and hasattr(self.skeleton, "to"):
            self.skeleton = self.skeleton.to(device)
        return self


def _session_pose_tracks(
    session_npz: Path,
    times_s: list[float],
    session_fps: float,
    *,
    include_torso: bool,
    camera_targets_npz: Path | None = None,
):
    """Wrist (+ optional camera torso) tracks in Kimodo Y-up, XZ-canon at t0.

    Variant B torso MUST come from ``session_targets_zup.npz`` (camera torso),
    not from ``session_umi2.npz`` body torso_link (standing-recipe offset ~0.11 m).
    """
    sess = np.load(session_npz, allow_pickle=True)
    bn = [str(x) for x in sess["body_names"]]
    pel_i = bn.index("pelvis") if "pelvis" in bn else 0
    lw_i = bn.index(WRIST_SESSION[0])
    rw_i = bn.index(WRIST_SESSION[1])

    sess_frames = np.array([int(round(t * session_fps)) for t in times_s], dtype=np.int64)
    body_pos = np.asarray(sess["body_pos_w"], dtype=np.float64)
    body_quat = np.asarray(sess["body_quat_w"], dtype=np.float64)

    root_z = body_pos[sess_frames, pel_i]
    root_q = body_quat[sess_frames, pel_i]
    lw_z = body_pos[sess_frames, lw_i]
    rw_z = body_pos[sess_frames, rw_i]
    lw_q = body_quat[sess_frames, lw_i]
    rw_q = body_quat[sess_frames, rw_i]

    root_y = np.stack([_pos_zup_to_yup(p) for p in root_z], axis=0)
    lw_y = np.stack([_pos_zup_to_yup(p) for p in lw_z], axis=0)
    rw_y = np.stack([_pos_zup_to_yup(p) for p in rw_z], axis=0)
    root_R = np.stack([_rot_zup_to_yup(_quat_wxyz_to_mat(q)) for q in root_q], axis=0)
    lw_R = np.stack([_rot_zup_to_yup(_quat_wxyz_to_mat(q)) for q in lw_q], axis=0)
    rw_R = np.stack([_rot_zup_to_yup(_quat_wxyz_to_mat(q)) for q in rw_q], axis=0)

    torso_y = torso_R = None
    cam_frames = None
    torso_source = None
    if include_torso:
        if camera_targets_npz is None or not Path(camera_targets_npz).is_file():
            raise FileNotFoundError(
                f"Variant cam_torso_wrists requires camera targets npz; got {camera_targets_npz}"
            )
        cam = np.load(camera_targets_npz, allow_pickle=True)
        cam_fps = float(cam["fps_session"])
        cam_frames = np.array(
            [min(int(round(t * cam_fps)), len(cam["torso_pos"]) - 1) for t in times_s],
            dtype=np.int64,
        )
        tz = np.asarray(cam["torso_pos"], dtype=np.float64)[cam_frames]
        tq = np.asarray(cam["torso_quat_wxyz"], dtype=np.float64)[cam_frames]
        # Prefer camera wrist positions when available (match session at t0).
        if "left_wrist" in cam.files and "right_wrist" in cam.files:
            lw_z = np.asarray(cam["left_wrist"], dtype=np.float64)[cam_frames]
            rw_z = np.asarray(cam["right_wrist"], dtype=np.float64)[cam_frames]
            lw_y = np.stack([_pos_zup_to_yup(p) for p in lw_z], axis=0)
            rw_y = np.stack([_pos_zup_to_yup(p) for p in rw_z], axis=0)
        torso_y = np.stack([_pos_zup_to_yup(p) for p in tz], axis=0)
        torso_R = np.stack([_rot_zup_to_yup(_quat_wxyz_to_mat(q)) for q in tq], axis=0)
        torso_source = {
            "file": str(Path(camera_targets_npz).resolve()),
            "fields": ["torso_pos", "torso_quat_wxyz", "left_wrist", "right_wrist"],
            "cam_frames": cam_frames.tolist(),
            "cam_fps": cam_fps,
            "note": (
                "Camera torso from session_targets_zup; wrist ori still from "
                "session_umi2 body_quat_w (targets lack wrist quat)."
            ),
        }

    # Canon: pelvis XZ ≈ 0 at t0 (API smooth_root filler shares this frame).
    off = root_y[0].copy()
    off[1] = 0.0
    root_c = root_y - off[None, :]
    lw_c = lw_y - off[None, :]
    rw_c = rw_y - off[None, :]
    torso_c = (torso_y - off[None, :]) if torso_y is not None else None

    return {
        "sess_frames": sess_frames,
        "root_c": root_c,
        "root_R": root_R,
        "root_quat_zup_wxyz": root_q,
        "lw_c": lw_c,
        "rw_c": rw_c,
        "lw_R": lw_R,
        "rw_R": rw_R,
        "torso_c": torso_c,
        "torso_R": torso_R,
        "off": off,
        "source_npz": str(session_npz),
        "body_fields": ["body_pos_w", "body_quat_w"],
        "include_torso": include_torso,
        "torso_source": torso_source,
    }


def build_sparse_constraint(
    *,
    session_npz: Path,
    variant: str,
    duration_s: float = 1.5,
    kimodo_fps: float = 30.0,
    session_fps: float = 50.0,
    times_s: list[float] | None = None,
    device: str | torch.device = "cpu",
    camera_targets_npz: Path | None = None,
    heading_target_yaw_deg: float = 0.0,
):
    """Build SparseEE: wrists (+ optional torso) + smooth_root_2d only.

    ``wrist_heading_tf``: rotate session wrists by Δyaw into Kimodo canonical
    heading (default 0°) before constraining. NO Kimodo heading constraint —
    absolute Kimodo world yaw is not packed into the ScaleTrack reference.
    """
    from kimodo.skeleton import G1Skeleton34

    assert variant in ("wrist_noyaw", "cam_torso_wrists", "wrist_heading_tf")
    include_torso = variant == "cam_torso_wrists"
    apply_heading_tf = variant == "wrist_heading_tf"
    if camera_targets_npz is None:
        camera_targets_npz = DEFAULT_CAMERA_TARGETS

    if times_s is None:
        times_s = [round(0.1 * i, 3) for i in range(0, 11)]
    times_s = [t for t in times_s if t <= duration_s + 1e-6]
    kimodo_frames = np.array([int(round(t * kimodo_fps)) for t in times_s], dtype=np.int64)
    n_kimodo = int(round(duration_s * kimodo_fps))
    assert int(kimodo_frames.max()) < n_kimodo

    tracks = _session_pose_tracks(
        session_npz,
        times_s,
        session_fps,
        include_torso=include_torso,
        camera_targets_npz=camera_targets_npz if include_torso else None,
    )
    heading_tf_meta = None
    lw_c = tracks["lw_c"]
    rw_c = tracks["rw_c"]
    lw_R = tracks["lw_R"]
    rw_R = tracks["rw_R"]
    torso_c = tracks["torso_c"]
    torso_R = tracks["torso_R"]
    root_c = tracks["root_c"]
    if apply_heading_tf:
        tf = _apply_heading_delta_yup(
            root_c=tracks["root_c"],
            root_R=tracks["root_R"],
            lw_c=tracks["lw_c"],
            rw_c=tracks["rw_c"],
            lw_R=tracks["lw_R"],
            rw_R=tracks["rw_R"],
            torso_c=tracks["torso_c"],
            torso_R=tracks["torso_R"],
            target_yaw_deg=heading_target_yaw_deg,
        )
        lw_c, rw_c = tf["lw_c"], tf["rw_c"]
        lw_R, rw_R = tf["lw_R"], tf["rw_R"]
        torso_c, torso_R = tf["torso_c"], tf["torso_R"]
        root_c = tf["root_c"]
        heading_tf_meta = {
            "delta_yaw_deg": tf["delta_yaw_deg"],
            "session_yaw_deg_frame0": tf["session_yaw_deg_frame0"],
            "target_yaw_deg": tf["target_yaw_deg"],
            "note": (
                "Δyaw = target(0°) − session_pelvis_yaw; wrists rotated about Y "
                "through root into Kimodo canonical heading. Heading constraint OFF; "
                "pack joints-only onto frozen session/joint0 body_*."
            ),
        }

    T = len(times_s)
    device_t = torch.device(device)
    sk = G1Skeleton34().to(device_t)
    J = len(sk.bone_order_names)

    # Sparse tensors: only overwrite intended EE slots; rest zeros (never indexed).
    g_pos = np.zeros((T, J, 3), dtype=np.float64)
    g_rots = np.tile(np.eye(3, dtype=np.float64), (T, J, 1, 1))
    g_pos[:, sk.root_idx] = root_c

    li = sk.bone_index[WRIST_KIMODO[0]]
    ri = sk.bone_index[WRIST_KIMODO[1]]
    g_pos[:, li] = lw_c
    g_pos[:, ri] = rw_c
    g_rots[:, li] = lw_R
    g_rots[:, ri] = rw_R

    pos_names = list(WRIST_KIMODO)
    rot_names = list(WRIST_KIMODO)
    overlay_err = {
        "left_wrist": 0.0,
        "right_wrist": 0.0,
    }

    if include_torso:
        ti = sk.bone_index[TORSO_KIMODO]
        g_pos[:, ti] = torso_c
        g_rots[:, ti] = torso_R
        pos_names = [TORSO_KIMODO] + pos_names
        rot_names = [TORSO_KIMODO] + rot_names

    frame_idx = torch.tensor(kimodo_frames, dtype=torch.long, device=device_t)
    gp_t = torch.tensor(g_pos, dtype=torch.float32, device=device_t)
    gr_t = torch.tensor(g_rots, dtype=torch.float32, device=device_t)
    # Forced by API when constraining global positions (= allowed root xy).
    smooth = gp_t[:, sk.root_idx][:, [0, 2]].contiguous()

    cset = SparseEEConstraintSet(
        sk,
        frame_indices=frame_idx,
        global_joints_positions=gp_t,
        global_joints_rots=gr_t,
        smooth_root_2d=smooth,
        pos_joint_names=pos_names,
        rot_joint_names=rot_names,
        constrain_smooth_root=True,  # API-forced root xy
        constrain_root_y=False,
        # Pin Kimodo heading to canonical target after wrist Δyaw transform so
        # joints transfer cleanly onto frozen session body (+90°). Free heading
        # drifts (e.g. −40°) and FK under joint0 root blows up.
        constrain_heading=bool(apply_heading_tf),
        label=variant,
    )
    if apply_heading_tf:
        # Override hip-scaffold heading (hips are zeros in SparseEE) with target.
        ang = np.radians(float(heading_target_yaw_deg))
        head = torch.tensor(
            [[np.cos(ang), np.sin(ang)]] * T, dtype=torch.float32, device=device_t
        )
        cset.global_root_heading = head

    # Verify EE slots
    err_l = float(
        np.linalg.norm(cset.global_joints_positions[:, li].cpu().numpy() - lw_c, axis=-1).mean()
    )
    err_r = float(
        np.linalg.norm(cset.global_joints_positions[:, ri].cpu().numpy() - rw_c, axis=-1).mean()
    )
    overlay_err["left_wrist"] = err_l
    overlay_err["right_wrist"] = err_r
    if include_torso:
        ti = sk.bone_index[TORSO_KIMODO]
        overlay_err["torso"] = float(
            np.linalg.norm(
                cset.global_joints_positions[:, ti].cpu().numpy() - torso_c, axis=-1
            ).mean()
        )

    heading = cset.global_root_heading.detach().cpu().numpy()
    heading_deg = np.degrees(np.arctan2(heading[:, 1], heading[:, 0]))

    from collections import defaultdict

    dd, idd = defaultdict(list), defaultdict(list)
    cset.update_constraints(dd, idd)
    runtime_keys = sorted(dd.keys())
    pos_bones = [sk.bone_order_names[i] for i in cset.pos_indices.tolist()]
    rot_bones = [sk.bone_order_names[i] for i in cset.rot_indices.tolist()]
    assert "root_y_pos" not in runtime_keys
    assert "hand_roll" not in " ".join(pos_bones + rot_bones)
    expected_keys = ["global_joints_positions", "global_joints_rots", "smooth_root_2d"]
    if apply_heading_tf:
        expected_keys.append("global_root_heading")
        assert "global_root_heading" in runtime_keys
    else:
        assert "global_root_heading" not in runtime_keys
    assert runtime_keys == sorted(expected_keys)

    meta = {
        "variant": variant,
        "times_s": times_s,
        "session_frames": tracks["sess_frames"].tolist(),
        "kimodo_frames": kimodo_frames.tolist(),
        "n_kimodo_frames": n_kimodo,
        "root_offset_xz_kimodo": tracks["off"].tolist(),
        "overlay_ee_match_err_m": overlay_err,
        "pos_joint_names": pos_names,
        "rot_joint_names": rot_names,
        "runtime_update_keys": runtime_keys,
        "pos_bones_verified": pos_bones,
        "rot_bones_verified": rot_bones,
        "constrain_smooth_root": True,
        "constrain_root_y": False,
        "constrain_heading": bool(apply_heading_tf),
        "heading_deg_constrained": heading_deg.tolist() if apply_heading_tf else None,
        "heading_deg_scaffold_unused": None if apply_heading_tf else heading_deg.tolist(),
        "heading_transform": heading_tf_meta,
        "api_limitation": (
            "KimodoMotionRep.create_conditions requires smooth_root_2d (root xz) "
            "whenever global_joints_positions are constrained; cannot omit root xz "
            "while keeping wrist/torso positions. root_y IS omitted. "
            "wrist_heading_tf also pins global_root_heading to canonical target yaw "
            "(hips absent in SparseEE so scaffold heading is overridden)."
        ),
        "mapping": {
            "session_wrists": list(WRIST_SESSION),
            "kimodo_wrists": list(WRIST_KIMODO),
            "session_torso": TORSO_SESSION if include_torso else None,
            "kimodo_torso": TORSO_KIMODO if include_torso else None,
        },
        "pose_source": {
            "session_file": tracks["source_npz"],
            "session_fields": tracks["body_fields"],
            "wrist_ori_from": "session_umi2 body_quat_w",
            "torso_source": tracks.get("torso_source"),
            "camera_pipeline_note": (
                "Variant B torso = session_targets_zup.npz (camera→D435→torso). "
                "Not session_umi2 torso_link (standing recipe ~0.11 m offset)."
            ),
            "build_meta": str(
                Path(tracks["source_npz"]).resolve().parents[1] / "build_meta.json"
            ),
        },
        "mink_used": False,
        "no_rest_scaffold_heading": True,
        "no_hand_roll": True,
        "method": (
            f"SparseEEConstraintSet variant={variant}: listed bone pos+rot; "
            "smooth_root_2d from session pelvis xz (API-forced = root xy); "
            + (
                f"heading-Δyaw wrist transform (Δ={heading_tf_meta['delta_yaw_deg']:.3f}°) "
                f"+ pin Kimodo heading to {heading_tf_meta['target_yaw_deg']:.1f}°; "
                if heading_tf_meta
                else "NO heading transform; NO heading constraint; "
            )
            + "NO root_y; NO hand_roll; NO Mink."
        ),
    }
    return [cset], sk, meta


def _csv_frame0_to_isaac_q0(csv_path: Path) -> tuple[np.ndarray, dict]:
    """Map Kimodo G1 CSV frame-0 → Isaac 29-DoF without importing kimodo package."""
    import xml.etree.ElementTree as ET

    isaac_joints = list(json.loads(Path(ISAAC_ORDER_JSON).read_text())["joint_names"])
    xml_path = Path(os.environ.get(
        "KODA_KIMODO_G1_XML",
        KODA_HOME / "src/koda/vendor/kimodo/kimodo/assets/skeletons/g1skel34/xml/g1.xml",
    ))
    mujoco_joints = [
        j.get("name")
        for j in ET.parse(xml_path).getroot().find("worldbody").findall(".//joint")
    ]
    assert set(isaac_joints) == set(mujoco_joints)
    mujoco_to_isaac = np.array(
        [mujoco_joints.index(n) for n in isaac_joints], dtype=np.int64
    )
    qpos = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if qpos.ndim == 1:
        qpos = qpos[None, :]
    assert qpos.shape[1] == 36, qpos.shape
    q0_mujoco = qpos[0, 7:].astype(np.float32)
    q0_isaac = q0_mujoco[mujoco_to_isaac].astype(np.float32)
    info = {
        "csv": str(csv_path),
        "n_frames": int(qpos.shape[0]),
        "root_mujoco_frame0": qpos[0, :3].tolist(),
        "root_quat_wxyz_frame0": qpos[0, 3:7].tolist(),
        "q0_isaac": q0_isaac.tolist(),
        "mujoco_joint_order": mujoco_joints,
        "mapping": "csv mujoco_rest_zero=False; pack raw remapped Isaac-order angles",
    }
    return q0_isaac, info


def _offline_wrist_fk_err(
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    q_isaac: np.ndarray,
    left_target: np.ndarray,
    right_target: np.ndarray,
) -> dict:
    """MuJoCo FK wrist pos error vs session targets (no Isaac).

    Optional diagnostic only — pack must not depend on mujoco being importable
    in the current env (kimodo often lacks it; scalebfm has it).
    """
    try:
        import mujoco
    except ImportError as exc:
        return {"error": f"mujoco unavailable ({exc}); skip offline FK"}

    try:
        from package_kimodo_q0_ablation import ISAAC_ORDER_JSON

        isaac_joints = list(json.loads(Path(ISAAC_ORDER_JSON).read_text())["joint_names"])
        mjcf = Path(os.environ.get(
            "KODA_SPARSETRACK_MJCF",
            KODA_HOME / "resources/SparseTrack/assets/g1_29dof_with_hand_st_collision.xml",
        ))
        model = mujoco.MjModel.from_xml_path(str(mjcf))
        data = mujoco.MjData(model)
        lw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
        rw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
        data.qpos[:] = 0.0
        data.qpos[0:3] = np.asarray(root_pos, dtype=np.float64)
        data.qpos[3:7] = np.asarray(root_quat_wxyz, dtype=np.float64)
        for i, name in enumerate(isaac_joints):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[jid]] = float(q_isaac[i])
        mujoco.mj_forward(model, data)
        lp = data.xpos[lw_id].copy()
        rp = data.xpos[rw_id].copy()
        el = float(np.linalg.norm(lp - left_target))
        er = float(np.linalg.norm(rp - right_target))
        return {
            "left_err_m": el,
            "right_err_m": er,
            "mean_err_m": 0.5 * (el + er),
            "fk_left": lp.tolist(),
            "fk_right": rp.tolist(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _optimize_root_xy_for_wrists(
    root0: np.ndarray,
    root_quat_wxyz: np.ndarray,
    q_isaac: np.ndarray,
    left_target: np.ndarray,
    right_target: np.ndarray,
    *,
    optimize_z: bool = False,
) -> dict:
    """Nelder-Mead root xy (optional z) minimizing mean L/R wrist FK error.

    Joints + quat stay fixed (Kimodo solution). Used when CSV frame-0 xy is a
    soft-diffusion compromise rather than an exact EE root.
    """
    from scipy.optimize import minimize

    root0 = np.asarray(root0, dtype=np.float64)
    z0 = float(root0[2])

    def _eval(xyz: np.ndarray) -> dict:
        return _offline_wrist_fk_err(xyz, root_quat_wxyz, q_isaac, left_target, right_target)

    fk0 = _eval(root0)

    if optimize_z:
        def loss(xyz):
            return _eval(np.asarray(xyz, dtype=np.float64))["mean_err_m"]

        res = minimize(loss, root0.copy(), method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 800})
        root_opt = np.asarray(res.x, dtype=np.float64)
    else:
        def loss(xy):
            return _eval(np.array([xy[0], xy[1], z0], dtype=np.float64))["mean_err_m"]

        res = minimize(loss, root0[:2].copy(), method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 500})
        root_opt = np.array([res.x[0], res.x[1], z0], dtype=np.float64)

    fk_opt = _eval(root_opt)
    return {
        "optimize_z": bool(optimize_z),
        "root_before": root0.tolist(),
        "root_after": root_opt.tolist(),
        "fk_before": fk0,
        "fk_after": fk_opt,
        "improved": bool(fk_opt["mean_err_m"] < fk0["mean_err_m"] - 1e-5),
    }


def measure_kimodo_soft_wrist_residual(
    *,
    npz_path: Path,
    session_npz: Path,
    variant: str = "wrist_noyaw",
    duration_s: float = 1.5,
) -> dict:
    """Distance between Kimodo posed_joints wrists and SparseEE targets (Y-up)."""
    clst, sk, meta = build_sparse_constraint(
        session_npz=session_npz,
        variant=variant,
        duration_s=duration_s,
        device="cpu",
    )
    cset = clst[0]
    li = int(sk.bone_index[WRIST_KIMODO[0]])
    ri = int(sk.bone_index[WRIST_KIMODO[1]])
    tgt_l = cset.global_joints_positions[:, li].detach().cpu().numpy()
    tgt_r = cset.global_joints_positions[:, ri].detach().cpu().numpy()
    posed = np.asarray(np.load(npz_path)["posed_joints"], dtype=np.float64)
    kframes = np.asarray(meta["kimodo_frames"], dtype=np.int64)
    err_l = np.linalg.norm(posed[kframes, li] - tgt_l, axis=-1)
    err_r = np.linalg.norm(posed[kframes, ri] - tgt_r, axis=-1)
    d0_l = (posed[kframes[0], li] - tgt_l[0]).tolist()
    d0_r = (posed[kframes[0], ri] - tgt_r[0]).tolist()
    return {
        "runtime_update_keys": meta["runtime_update_keys"],
        "pos_bones": meta["pos_bones_verified"],
        "rot_bones": meta["rot_bones_verified"],
        "constrain_smooth_root": meta["constrain_smooth_root"],
        "constrain_root_y": meta["constrain_root_y"],
        "constrain_heading": meta["constrain_heading"],
        "frame0_left_m": float(err_l[0]),
        "frame0_right_m": float(err_r[0]),
        "mean_left_m": float(err_l.mean()),
        "mean_right_m": float(err_r.mean()),
        "frame0_delta_left_xyz_yup": d0_l,
        "frame0_delta_right_xyz_yup": d0_r,
        "per_keyframe_left_m": err_l.tolist(),
        "per_keyframe_right_m": err_r.tolist(),
    }


def pack_q0(
    *,
    csv_path: Path,
    session_npz: Path,
    out_dir: Path,
    work_dir: Path,
    stem: str,
    q0_source: str,
    optimize_root_xy: bool = True,
    optimize_root_z: bool = False,
) -> dict:
    """Pack SparseEE Kimodo q0 for fair-init play.

    Critical packing rule (SparseEE / free heading):
      Kimodo joints were solved under Kimodo CSV root pos+quat, NOT session pelvis.
      ScaleTrack reset reads motion ``body_pos_w[:,0]`` / ``body_quat_w[:,0]``, so
      frame-0 pelvis must come from Kimodo CSV root (the xy/quat that realize the
      wrist EE). Wrist and other body targets stay from session for UMI-2 tracking.

    Root xy: start from Kimodo CSV frame-0, then optionally re-optimize xy under
    fixed joints+quat to minimize mean wrist FK error vs session targets.
    """
    q0, qinfo = _csv_frame0_to_isaac_q0(csv_path)
    (work_dir / "q0_extract_meta.json").write_text(json.dumps(qinfo, indent=2))
    np.save(work_dir / "q0_isaac.npy", q0)

    kimodo_root = np.asarray(qinfo["root_mujoco_frame0"], dtype=np.float64)
    kimodo_quat = np.asarray(qinfo["root_quat_wxyz_frame0"], dtype=np.float64)

    base = np.load(session_npz, allow_pickle=True)
    n = int(base["joint_pos"].shape[0])
    n_j = int(base["joint_pos"].shape[1])
    assert q0.shape == (n_j,), f"q0 {q0.shape} vs {n_j}"
    bn = [str(x) for x in base["body_names"]]
    pel_i = bn.index("pelvis") if "pelvis" in bn else 0
    lw_i = bn.index("left_wrist_yaw_link")
    rw_i = bn.index("right_wrist_yaw_link")

    body_pos = np.asarray(base["body_pos_w"], dtype=np.float32).copy()
    body_quat = np.asarray(base["body_quat_w"], dtype=np.float32).copy()
    sess_pel0 = body_pos[0, pel_i].astype(np.float64).copy()
    sess_quat0 = body_quat[0, pel_i].astype(np.float64).copy()
    sess_lw0 = body_pos[0, lw_i].astype(np.float64)
    sess_rw0 = body_pos[0, rw_i].astype(np.float64)

    # Offline FK: bad packing (session pelvis+quat) vs Kimodo CSV root.
    fk_bad = _offline_wrist_fk_err(sess_pel0, sess_quat0, q0, sess_lw0, sess_rw0)
    fk_csv = _offline_wrist_fk_err(kimodo_root, kimodo_quat, q0, sess_lw0, sess_rw0)

    root_opt_meta = None
    pack_root = kimodo_root.copy()
    if optimize_root_xy:
        root_opt_meta = _optimize_root_xy_for_wrists(
            kimodo_root,
            kimodo_quat,
            q0,
            sess_lw0,
            sess_rw0,
            optimize_z=optimize_root_z,
        )
        if root_opt_meta["improved"] or root_opt_meta["fk_after"]["mean_err_m"] <= fk_csv["mean_err_m"]:
            pack_root = np.asarray(root_opt_meta["root_after"], dtype=np.float64)

    fk_good = _offline_wrist_fk_err(pack_root, kimodo_quat, q0, sess_lw0, sess_rw0)

    # Robot-init root = optimized (or Kimodo) root under Kimodo quat.
    body_pos[0, pel_i] = pack_root.astype(np.float32)
    body_quat[0, pel_i] = kimodo_quat.astype(np.float32)

    jp = np.zeros((n, n_j), dtype=np.float32)
    jp[0] = q0.astype(np.float32)
    jv = np.zeros((n, n_j), dtype=np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / f"{stem}.npz"
    payload = {k: base[k] for k in base.files}
    payload["body_pos_w"] = body_pos
    payload["body_quat_w"] = body_quat
    payload["joint_pos"] = jp
    payload["joint_vel"] = jv
    payload["source"] = np.asarray(f"{stem}_hold")
    payload["base_motion"] = np.asarray(str(session_npz))
    payload["q0_source"] = np.asarray(q0_source)
    payload["pack_root_source"] = np.asarray(
        "kimodo_csv_then_wrist_fk_xy" if optimize_root_xy else "kimodo_csv_frame0"
    )
    # Document prior session arm-IK xy (not applied): heading-locked IK, wrong for SparseEE.
    prior_reset = Path(out_dir) / "session_umi2_reset_xy_q0.npz"
    prior_xy = None
    if prior_reset.is_file():
        pr = np.load(prior_reset, allow_pickle=True)
        if "reset_root_delta_xy" in pr.files:
            prior_xy = np.asarray(pr["reset_root_delta_xy"], dtype=np.float64).tolist()
    if "reset_root_delta_xy" in payload:
        del payload["reset_root_delta_xy"]
    np.savez_compressed(out_npz, **payload)
    chk = np.load(out_npz, allow_pickle=True)
    # Wrist / non-frame0 bodies unchanged; only frame-0 pelvis rewritten for robot init.
    assert np.allclose(chk["body_pos_w"][1:], base["body_pos_w"][1:])
    assert np.allclose(chk["body_quat_w"][1:], base["body_quat_w"][1:])
    assert np.allclose(chk["body_pos_w"][0, lw_i], base["body_pos_w"][0, lw_i])
    assert np.allclose(chk["body_pos_w"][0, rw_i], base["body_pos_w"][0, rw_i])
    assert np.allclose(chk["body_pos_w"][0, pel_i], pack_root.astype(np.float32))
    assert np.allclose(chk["body_quat_w"][0, pel_i], kimodo_quat.astype(np.float32))
    assert np.allclose(chk["joint_pos"][1:], 0.0)
    assert np.allclose(chk["joint_vel"], 0.0)
    # With session heading ON, Kimodo quat ≈ session quat so session-root packing is
    # no longer catastrophic; only require optimized/CSV root beats or matches CSV.
    assert fk_good["mean_err_m"] <= fk_csv["mean_err_m"] + 1e-6, (
        f"root xy opt worsened FK: good={fk_good} csv={fk_csv}"
    )
    _write_yaml(out_dir / f"{stem}.yaml", stem, out_npz)

    def _yaw_deg_wxyz(q: np.ndarray) -> float:
        w, x, y, z = [float(v) for v in q]
        return float(np.degrees(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))))

    yaw_kimodo = _yaw_deg_wxyz(kimodo_quat)
    yaw_session = _yaw_deg_wxyz(sess_quat0)

    soft_residual = None
    soft_npz = work_dir / "kimodo_gen_out.npz"
    if soft_npz.is_file():
        try:
            soft_residual = measure_kimodo_soft_wrist_residual(
                npz_path=soft_npz,
                session_npz=session_npz,
                variant="cam_torso_wrists" if "cam_torso" in stem else "wrist_noyaw",
            )
        except Exception as exc:  # noqa: BLE001 — pack must not fail on soft metric
            soft_residual = {"error": str(exc)}

    summary = {
        "stem": stem,
        "npz": str(out_npz),
        "yaml": str(out_dir / f"{stem}.yaml"),
        "session_npz": str(session_npz),
        "q0_l2": float(np.linalg.norm(q0)),
        "mink_used_in_constraints": False,
        "play_note": (
            "UMI-2 mode_index=1, 500 steps, --hold_motion_q0; "
            "frame0 pelvis = Kimodo root (+ optional xy/z wrist FK opt); "
            "wrist body targets remain session"
        ),
        "q0_extract": qinfo,
        "q0_source": q0_source,
        "kimodo_soft_wrist_residual": soft_residual,
        "pack_root_fix": {
            "bug": (
                "Previously SparseEE left heading free → Kimodo yaw ~−14°…−28° vs "
                "session ~+90° → sideways robot; packing under session quat then "
                "blew wrists ~0.5m; packing under free Kimodo quat fixed FK but "
                "looked unnatural."
            ),
            "fix": (
                "Re-generate with session heading constrained; pack Kimodo CSV "
                "frame0 quat + (optional) wrist-FK-optimized root xy."
            ),
            "root_xy_source": (
                "kimodo_csv_frame0 then Nelder-Mead xy"
                + ("+z" if optimize_root_z else "")
                + " minimizing mean wrist FK vs session"
                if optimize_root_xy
                else "kimodo_csv_frame0"
            ),
            "yaw_deg_kimodo_frame0": yaw_kimodo,
            "yaw_deg_session_pelvis": yaw_session,
            "yaw_delta_deg_kimodo_minus_session": yaw_kimodo - yaw_session,
            "kimodo_root_xyz": kimodo_root.tolist(),
            "pack_root_xyz": pack_root.tolist(),
            "kimodo_quat_wxyz": kimodo_quat.tolist(),
            "session_pelvis_xyz_unused_for_init": sess_pel0.tolist(),
            "session_pelvis_quat_unused_for_init": sess_quat0.tolist(),
            "prior_session_arm_ik_reset_root_delta_xy": prior_xy,
            "prior_session_arm_ik_note": (
                "session_umi2_reset_xy delta [-0.22,-0.145] is a legacy arm-IK "
                "xy under session heading; superseded by wrist-FK root xy search "
                "on Kimodo joints+quat."
            ),
            "root_xy_optimize": root_opt_meta,
            "offline_fk_wrist_err_bad_session_root": fk_bad,
            "offline_fk_wrist_err_kimodo_csv_root": fk_csv,
            "offline_fk_wrist_err_fixed_kimodo_root": fk_good,
        },
    }
    mink_path = out_dir / "session_umi2_init_mink_q0.npz"
    joint0_path = out_dir / "session_umi2_init_joint0.npz"
    if mink_path.is_file():
        mq = np.asarray(np.load(mink_path)["joint_pos"][0], dtype=np.float32)
        summary["q0_vs_mink_l2"] = float(np.linalg.norm(q0 - mq))
    if joint0_path.is_file():
        jq = np.asarray(np.load(joint0_path)["joint_pos"][0], dtype=np.float32)
        summary["q0_vs_joint0_l2"] = float(np.linalg.norm(q0 - jq))
    (work_dir / "pack_meta.json").write_text(json.dumps(summary, indent=2))
    (out_dir / f"{stem}_pack_meta.json").write_text(json.dumps(summary, indent=2))
    return summary


def pack_q0_joints_only_frozen_body(
    *,
    csv_path: Path,
    body_ref_npz: Path,
    out_dir: Path,
    work_dir: Path,
    stem: str,
    q0_source: str,
    session_npz: Path | None = None,
) -> dict:
    """Fair pack: copy body_* from joint0/mink reference; only joint_pos[0] from Kimodo.

    Never rewrites pelvis xyz/quat. Asserts body_pos_w / body_quat_w allclose to ref.
    """
    q0, qinfo = _csv_frame0_to_isaac_q0(csv_path)
    (work_dir / "q0_extract_meta.json").write_text(json.dumps(qinfo, indent=2))
    np.save(work_dir / "q0_isaac.npy", q0)

    ref = np.load(body_ref_npz, allow_pickle=True)
    n = int(ref["joint_pos"].shape[0])
    n_j = int(ref["joint_pos"].shape[1])
    assert q0.shape == (n_j,), f"q0 {q0.shape} vs {n_j}"

    bn = [str(x) for x in ref["body_names"]]
    pel_i = bn.index("pelvis") if "pelvis" in bn else 0
    lw_i = bn.index("left_wrist_yaw_link")
    rw_i = bn.index("right_wrist_yaw_link")

    body_pos = np.asarray(ref["body_pos_w"], dtype=np.float32).copy()
    body_quat = np.asarray(ref["body_quat_w"], dtype=np.float32).copy()
    sess_pel0 = body_pos[0, pel_i].astype(np.float64)
    sess_quat0 = body_quat[0, pel_i].astype(np.float64)
    sess_lw0 = body_pos[0, lw_i].astype(np.float64)
    sess_rw0 = body_pos[0, rw_i].astype(np.float64)

    # Offline FK under FROZEN session/joint0 root (fair-init transfer check).
    # Optional: _offline_wrist_fk_err never raises if mujoco missing.
    kimodo_root = np.asarray(qinfo["root_mujoco_frame0"], dtype=np.float64)
    kimodo_quat = np.asarray(qinfo["root_quat_wxyz_frame0"], dtype=np.float64)
    fk_fair = _offline_wrist_fk_err(sess_pel0, sess_quat0, q0, sess_lw0, sess_rw0)
    fk_kimodo_root = _offline_wrist_fk_err(
        kimodo_root, kimodo_quat, q0, sess_lw0, sess_rw0
    )

    jp = np.zeros((n, n_j), dtype=np.float32)
    jp[0] = q0.astype(np.float32)
    jv = np.zeros((n, n_j), dtype=np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / f"{stem}.npz"
    payload = {k: ref[k] for k in ref.files}
    payload["body_pos_w"] = body_pos
    payload["body_quat_w"] = body_quat
    payload["joint_pos"] = jp
    payload["joint_vel"] = jv
    payload["source"] = np.asarray(f"{stem}_hold")
    payload["base_motion"] = np.asarray(str(body_ref_npz))
    payload["q0_source"] = np.asarray(q0_source)
    payload["pack_root_source"] = np.asarray("frozen_joint0_body")
    if "reset_root_delta_xy" in payload:
        del payload["reset_root_delta_xy"]
    np.savez_compressed(out_npz, **payload)

    chk = np.load(out_npz, allow_pickle=True)
    assert np.allclose(chk["body_pos_w"], ref["body_pos_w"]), "body_pos_w must stay frozen"
    assert np.allclose(chk["body_quat_w"], ref["body_quat_w"]), "body_quat_w must stay frozen"
    assert np.allclose(chk["joint_pos"][1:], 0.0)
    assert np.allclose(chk["joint_vel"], 0.0)
    assert np.allclose(chk["joint_pos"][0], q0)
    _write_yaml(out_dir / f"{stem}.yaml", stem, out_npz)

    def _yaw_deg_wxyz(q: np.ndarray) -> float:
        w, x, y, z = [float(v) for v in q]
        return float(np.degrees(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))))

    soft_residual = None
    soft_npz = work_dir / "kimodo_gen_out.npz"
    if soft_npz.is_file():
        try:
            soft_residual = measure_kimodo_soft_wrist_residual(
                npz_path=soft_npz,
                session_npz=session_npz or body_ref_npz,
                variant="wrist_heading_tf",
            )
        except Exception as exc:  # noqa: BLE001
            soft_residual = {"error": str(exc)}

    summary = {
        "stem": stem,
        "npz": str(out_npz),
        "yaml": str(out_dir / f"{stem}.yaml"),
        "body_ref_npz": str(body_ref_npz),
        "session_npz": str(session_npz) if session_npz else None,
        "q0_l2": float(np.linalg.norm(q0)),
        "mink_used_in_constraints": False,
        "fair_init": True,
        "body_pos_frozen": True,
        "body_quat_frozen": True,
        "play_note": (
            "UMI-2 mode_index=1, 500 steps, --hold_motion_q0; "
            "body_* identical to joint0/mink; only joint_pos[0] from Kimodo "
            "heading-Δyaw wrist SparseEE."
        ),
        "q0_extract": qinfo,
        "q0_source": q0_source,
        "pack_root_source": "frozen_joint0_body",
        "kimodo_soft_wrist_residual": soft_residual,
        "yaw_deg_reference_pelvis": _yaw_deg_wxyz(sess_quat0),
        "yaw_deg_kimodo_csv_frame0": _yaw_deg_wxyz(kimodo_quat),
        "offline_fk_wrist_err_under_frozen_session_root": fk_fair,
        "offline_fk_wrist_err_under_kimodo_csv_root": fk_kimodo_root,
        "protocol": (
            "heading-Δyaw wrist transform → Kimodo SparseEE → pack joints-only; "
            "body_pos_w/body_quat_w allclose to joint0"
        ),
    }
    mink_path = out_dir / "session_umi2_init_mink_q0.npz"
    joint0_path = out_dir / "session_umi2_init_joint0.npz"
    if mink_path.is_file():
        mq = np.asarray(np.load(mink_path)["joint_pos"][0], dtype=np.float32)
        mbody = np.load(mink_path)
        summary["q0_vs_mink_l2"] = float(np.linalg.norm(q0 - mq))
        summary["body_match_mink"] = bool(
            np.allclose(chk["body_pos_w"], mbody["body_pos_w"])
            and np.allclose(chk["body_quat_w"], mbody["body_quat_w"])
        )
    if joint0_path.is_file():
        jq = np.asarray(np.load(joint0_path)["joint_pos"][0], dtype=np.float32)
        jbody = np.load(joint0_path)
        summary["q0_vs_joint0_l2"] = float(np.linalg.norm(q0 - jq))
        summary["body_match_joint0"] = bool(
            np.allclose(chk["body_pos_w"], jbody["body_pos_w"])
            and np.allclose(chk["body_quat_w"], jbody["body_quat_w"])
        )
        assert summary["body_match_joint0"], "fair-init body_* must match joint0"
    (work_dir / "pack_meta.json").write_text(json.dumps(summary, indent=2))
    (out_dir / f"{stem}_pack_meta.json").write_text(json.dumps(summary, indent=2))
    return summary


VARIANT_CFG = {
    "wrist_noyaw": {
        "work_subdir": "session_kimodo_wrist_noyaw",
        "stem": "session_umi2_init_kimodo_wrist_noyaw",
        # Artifact name kept; INVALID for fair-init if pack rewrites body_pos.
        "q0_source": "kimodo_wrist_posori_session_heading_norooty",
        "pack_mode": "rewrite_pelvis",  # INVALID fair-init
    },
    "cam_torso_wrists": {
        "work_subdir": "session_kimodo_cam_torso_wrists",
        "stem": "session_umi2_init_kimodo_cam_torso_wrists",
        "q0_source": "kimodo_cam_torso_wrists_posori_session_heading",
        "pack_mode": "rewrite_pelvis",  # INVALID fair-init
    },
    "wrist_heading_tf": {
        "work_subdir": "session_kimodo_wrist_heading_tf",
        "stem": "session_umi2_init_kimodo_wrist_heading_tf",
        "q0_source": "kimodo_wrist_heading_dyaw_sparse_ee_joints_only",
        "pack_mode": "joints_only_frozen_body",
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANT_CFG) + ["all"], required=True)
    ap.add_argument(
        "--mode",
        choices=["build_constraints", "generate", "pack_from_csv", "all"],
        required=True,
    )
    ap.add_argument("--session_npz", type=Path, default=DEFAULT_SESSION)
    ap.add_argument("--camera_targets_npz", type=Path, default=DEFAULT_CAMERA_TARGETS)
    ap.add_argument("--ablation_dir", type=Path, default=DEFAULT_ABLATION)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--body_ref_npz", type=Path, default=DEFAULT_JOINT0)
    ap.add_argument("--duration", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--diffusion_steps", type=int, default=100)
    ap.add_argument(
        "--heading_target_yaw_deg",
        type=float,
        default=0.0,
        help="Kimodo-canonical heading target for wrist_heading_tf (default 0°).",
    )
    ap.add_argument(
        "--cfg_weight",
        type=float,
        nargs=2,
        metavar=("TEXT_W", "CONSTRAINT_W"),
        default=[2.0, 5.0],
        help="Separated CFG [text, constraint]. Default 2.0 5.0 (stronger wrist/heading).",
    )
    ap.add_argument(
        "--optimize_root_xy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Nelder-Mead root xy minimizing mean wrist FK (legacy rewrite_pelvis packs)",
    )
    ap.add_argument(
        "--optimize_root_z",
        action="store_true",
        help="Also optimize root z with xy (legacy rewrite_pelvis packs)",
    )
    args = ap.parse_args()

    variants = list(VARIANT_CFG) if args.variant == "all" else [args.variant]
    session = args.session_npz.resolve()
    cfg_weight = [float(args.cfg_weight[0]), float(args.cfg_weight[1])]
    body_ref = args.body_ref_npz.resolve()

    for v in variants:
        cfg = VARIANT_CFG[v]
        work = (args.ablation_dir / cfg["work_subdir"]).resolve()
        work.mkdir(parents=True, exist_ok=True)

        if args.mode in ("build_constraints", "generate", "all"):
            clst, sk, meta = build_sparse_constraint(
                session_npz=session,
                variant=v,
                duration_s=args.duration,
                device="cpu",
                camera_targets_npz=args.camera_targets_npz.resolve(),
                heading_target_yaw_deg=float(args.heading_target_yaw_deg),
            )
            meta["cfg_weight"] = cfg_weight
            meta["diffusion_steps"] = int(args.diffusion_steps)
            (work / "constraints_meta.json").write_text(json.dumps(meta, indent=2))
            (work / "targets_yup.json").write_text(json.dumps(meta, indent=2))
            print(
                json.dumps(
                    {
                        "variant": v,
                        **{
                            k: meta[k]
                            for k in (
                                "overlay_ee_match_err_m",
                                "pos_joint_names",
                                "rot_joint_names",
                                "runtime_update_keys",
                                "constrain_heading",
                                "constrain_root_y",
                                "heading_transform",
                                "api_limitation",
                            )
                            if k in meta
                        },
                    },
                    indent=2,
                ),
                flush=True,
            )

            if args.mode in ("generate", "all"):
                for name in ("kimodo_gen_out.csv", "kimodo_gen_out.npz", "meta.json"):
                    src = work / name
                    if src.is_file():
                        bak = work / f"{name}.prev.bak"
                        if not bak.is_file():
                            bak.write_bytes(src.read_bytes())
                csv_path = run_kimodo_generate(
                    constraint_lst=clst,
                    work_dir=work,
                    duration_s=args.duration,
                    diffusion_steps=args.diffusion_steps,
                    seed=args.seed,
                    cfg_weight=cfg_weight,
                )
                soft = measure_kimodo_soft_wrist_residual(
                    npz_path=work / "kimodo_gen_out.npz",
                    session_npz=session,
                    variant=v,
                    duration_s=args.duration,
                )
                (work / "soft_wrist_residual.json").write_text(json.dumps(soft, indent=2))
                print(json.dumps({"soft_wrist_residual": soft}, indent=2), flush=True)
            else:
                csv_path = None
        else:
            csv_path = None

        if args.mode in ("pack_from_csv", "all"):
            csv_path = (csv_path or (work / "kimodo_gen_out.csv")).resolve()
            if not csv_path.is_file():
                raise SystemExit(f"CSV missing: {csv_path}")
            if cfg.get("pack_mode") == "joints_only_frozen_body":
                if not body_ref.is_file():
                    raise SystemExit(f"body_ref missing: {body_ref}")
                summary = pack_q0_joints_only_frozen_body(
                    csv_path=csv_path,
                    body_ref_npz=body_ref,
                    out_dir=args.out_dir.resolve(),
                    work_dir=work,
                    stem=cfg["stem"],
                    q0_source=cfg["q0_source"],
                    session_npz=session,
                )
            else:
                summary = pack_q0(
                    csv_path=csv_path,
                    session_npz=session,
                    out_dir=args.out_dir.resolve(),
                    work_dir=work,
                    stem=cfg["stem"],
                    q0_source=cfg["q0_source"],
                    optimize_root_xy=bool(args.optimize_root_xy),
                    optimize_root_z=bool(args.optimize_root_z),
                )
            print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
