# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assemble Kimodo NPZ-compatible motion dicts from local rotations + root trajectory."""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Tuple

import numpy as np
import torch

from kimodo.geometry import matrix_to_quaternion, quaternion_to_matrix
from kimodo.motion_rep.feature_utils import compute_heading_angle, compute_vel_xyz
from kimodo.motion_rep.feet import foot_detect_from_pos_and_vel
from kimodo.motion_rep.smooth_root import get_smooth_root_pos
from kimodo.skeleton import SkeletonBase
from kimodo.skeleton.registry import build_skeleton
from kimodo.tools import to_numpy

# Default motion rate for Kimodo NPZ produced by format conversion (matches common model FPS).
KIMODO_CONVERT_TARGET_FPS = 30.0


def _quaternion_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation; ``q0``, ``q1`` (..., 4) wxyz; ``t`` broadcastable to (...,
    1)."""
    if t.dim() < q0.dim():
        t = t.unsqueeze(-1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = torch.abs(dot).clamp(-1.0, 1.0)
    theta_0 = torch.acos(dot)
    sin_theta = torch.sin(theta_0)
    s0 = torch.sin((1.0 - t) * theta_0) / sin_theta.clamp(min=1e-8)
    s1 = torch.sin(t * theta_0) / sin_theta.clamp(min=1e-8)
    q = s0 * q0 + s1 * q1
    return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=1e-8)


def resample_motion_dict_to_kimodo_fps(
    motion_dict: Dict[str, torch.Tensor],
    skeleton: SkeletonBase,
    source_fps: float,
    target_fps: float = KIMODO_CONVERT_TARGET_FPS,
) -> Tuple[Dict[str, torch.Tensor], bool]:
    """Resample a Kimodo motion dict to ``target_fps``.

    When the fps ratio is close to an integer (e.g. 120 / 30 = 4), the faster
    stepping method is used (take every *step*-th frame).  Otherwise falls back
    to linear interp (root) + quaternion slerp (joints).

    Re-runs :func:`complete_motion_dict` at the target rate so derived channels stay consistent.

    Returns:
        The motion dict and ``True`` if time resampling was applied, else ``False`` (already at
        ``target_fps`` with matching frame count; only re-derived via FK).
    """
    local_rot_mats = motion_dict["local_rot_mats"]
    root_positions = motion_dict["root_positions"]
    local_rot_mats, root_positions = _coerce_time_local_root(local_rot_mats, root_positions)
    t_in = int(local_rot_mats.shape[0])
    if t_in < 1:
        raise ValueError("Motion must have at least one frame.")
    if source_fps <= 0:
        raise ValueError(f"source_fps must be positive; got {source_fps}")

    t_out = max(1, int(round(t_in * target_fps / source_fps)))
    if t_out == t_in and abs(float(source_fps) - float(target_fps)) < 1e-3:
        return complete_motion_dict(local_rot_mats, root_positions, skeleton, float(target_fps)), False

    ratio = source_fps / target_fps
    step = round(ratio)
    if step >= 2 and abs(ratio - step) < 0.05:
        local_out = local_rot_mats[::step]
        root_out = root_positions[::step]
    else:
        device = local_rot_mats.device
        dtype = local_rot_mats.dtype
        u = torch.linspace(0, t_in - 1, t_out, device=device, dtype=dtype)
        i0 = u.floor().long().clamp(0, t_in - 1)
        i1 = torch.minimum(i0 + 1, torch.tensor(t_in - 1, device=device))
        tau_1d = (u - i0.float()).unsqueeze(-1)
        rp0 = root_positions[i0]
        rp1 = root_positions[i1]
        root_out = (1.0 - tau_1d) * rp0 + tau_1d * rp1

        quats = matrix_to_quaternion(local_rot_mats)
        q0 = quats[i0]
        q1 = quats[i1]
        tau_q = (u - i0.float()).view(t_out, 1, 1)
        quat_out = _quaternion_slerp(q0, q1, tau_q)
        local_out = quaternion_to_matrix(quat_out)

    return complete_motion_dict(local_out, root_out, skeleton, float(target_fps)), True


def warn_kimodo_npz_framerate(source_fps: float, t_before: int, t_after: int) -> None:
    """Emit a warning after time resampling for Kimodo NPZ (linear root, quaternion slerp per
    joint)."""
    warnings.warn(
        f"Resampled motion to {KIMODO_CONVERT_TARGET_FPS:.0f} Hz for Kimodo NPZ "
        f"(source ~{source_fps:.4g} Hz, {t_before} input frames → {t_after} output frames). "
        "Pass --source-fps if the detected source rate is wrong.",
        UserWarning,
        stacklevel=3,
    )


def _coerce_time_local_root(
    local_rot_mats: torch.Tensor,
    root_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize to shapes (T, J, 3, 3) and (T, 3)."""
    if local_rot_mats.dim() == 5:
        if int(local_rot_mats.shape[0]) != 1:
            raise ValueError(f"local_rot_mats batch size must be 1 for single clip; got {local_rot_mats.shape[0]}")
        local_rot_mats = local_rot_mats[0]
    if root_positions.dim() == 3:
        if int(root_positions.shape[0]) != 1:
            raise ValueError(f"root_positions batch size must be 1; got {root_positions.shape[0]}")
        root_positions = root_positions[0]
    if local_rot_mats.dim() != 4:
        raise ValueError(f"local_rot_mats must be (T,J,3,3); got {tuple(local_rot_mats.shape)}")
    if root_positions.dim() != 2 or int(root_positions.shape[-1]) != 3:
        raise ValueError(f"root_positions must be (T,3); got {tuple(root_positions.shape)}")
    if int(local_rot_mats.shape[0]) != int(root_positions.shape[0]):
        raise ValueError("local_rot_mats and root_positions must have the same number of frames")
    return local_rot_mats, root_positions


def complete_motion_dict(
    local_rot_mats: torch.Tensor,
    root_positions: torch.Tensor,
    skeleton: SkeletonBase,
    fps: float,
) -> Dict[str, torch.Tensor]:
    """Build the Kimodo motion output dict from local rotations and root positions.

    Matches keys written by CLI generation (see docs/source/user_guide/output_formats.md).

    Args:
        local_rot_mats: (T, J, 3, 3) or (1, T, J, 3, 3) local rotation matrices.
        root_positions: (T, 3) or (1, T, 3) root / pelvis world positions (meters).
        skeleton: Skeleton instance (SOMA77, G1, SMPL-X, etc.).
        fps: Sampling rate (Hz).

    Returns:
        Dict with tensors ``posed_joints``, ``global_rot_mats``, ``local_rot_mats``,
        ``foot_contacts``, ``smooth_root_pos``, ``root_positions``, ``global_root_heading``.
    """
    device = local_rot_mats.device
    dtype = local_rot_mats.dtype
    local_rot_mats, root_positions = _coerce_time_local_root(
        local_rot_mats.to(device=device, dtype=dtype),
        root_positions.to(device=device, dtype=dtype),
    )

    global_rot_mats, posed_joints, _ = skeleton.fk(local_rot_mats, root_positions)

    smooth_root_pos = get_smooth_root_pos(root_positions.unsqueeze(0)).squeeze(0)

    lengths = torch.tensor([posed_joints.shape[0]], device=device)
    velocities = compute_vel_xyz(posed_joints.unsqueeze(0), fps, lengths=lengths).squeeze(0)

    heading_angle = compute_heading_angle(posed_joints.unsqueeze(0), skeleton).squeeze(0)
    global_root_heading = torch.stack([torch.cos(heading_angle), torch.sin(heading_angle)], dim=-1)

    foot_contacts = foot_detect_from_pos_and_vel(
        posed_joints.unsqueeze(0),
        velocities.unsqueeze(0),
        skeleton,
        0.15,
        0.10,
    ).squeeze(0)

    return {
        "posed_joints": posed_joints,
        "global_rot_mats": global_rot_mats,
        "local_rot_mats": local_rot_mats,
        "foot_contacts": foot_contacts,
        "smooth_root_pos": smooth_root_pos,
        "root_positions": root_positions,
        "global_root_heading": global_root_heading,
    }


def motion_dict_to_numpy(d: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Convert motion dict values to numpy arrays for ``np.savez``."""
    out: Dict[str, np.ndarray] = {}
    for k, v in d.items():
        if hasattr(v, "detach"):
            out[k] = to_numpy(v)
        elif isinstance(v, np.ndarray):
            out[k] = v
        else:
            out[k] = np.asarray(v)
    return out


def save_kimodo_npz(path: str, motion_dict: Dict[str, Any]) -> None:
    """Save a Kimodo-compatible motion dict to ``.npz`` (numpy arrays)."""
    np.savez(path, **motion_dict_to_numpy(motion_dict))


def load_kimodo_npz(path: str) -> Dict[str, np.ndarray]:
    """Load arrays from a Kimodo ``.npz`` file."""
    with np.load(path, allow_pickle=False) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def load_g1_csv(
    path: str,
    source_fps: float = KIMODO_CONVERT_TARGET_FPS,
    *,
    mujoco_rest_zero: bool = False,
) -> Dict[str, torch.Tensor]:
    """Load a G1 MuJoCo ``qpos`` CSV (``(T, 36)``) into a Kimodo motion dict.

    Args:
        path: CSV path (comma-separated, no header).
        source_fps: Source frame rate (Hz) of the CSV data.
        mujoco_rest_zero: Must match how the CSV was written (see :class:`MujocoQposConverter`).
    """
    from kimodo.exports.mujoco import MujocoQposConverter

    qpos = np.loadtxt(path, delimiter=",")
    if qpos.ndim != 2 or qpos.shape[-1] != 36:
        raise ValueError(f"Expected G1 CSV with shape (T, 36); got {qpos.shape}")
    sk = build_skeleton(34)
    converter = MujocoQposConverter(sk)
    return converter.qpos_to_motion_dict(qpos, float(source_fps), mujoco_rest_zero=mujoco_rest_zero)


def load_amass_npz(
    path: str,
    source_fps: float | None = None,
    *,
    z_up: bool = True,
) -> Dict[str, torch.Tensor]:
    """Load an AMASS-style SMPL-X ``.npz`` into a Kimodo motion dict (22 joints).

    Args:
        path: NPZ with ``trans``, ``root_orient``, ``pose_body``, etc.
        source_fps: Source frame rate (Hz); if ``None``, uses ``mocap_frame_rate``
            from the file when present, else 30 Hz.
        z_up: If ``True``, apply AMASS Z-up to Kimodo Y-up transform (same as CLI).
    """
    from kimodo.exports.smplx import amass_npz_to_kimodo_motion

    sk = build_skeleton(22)
    return amass_npz_to_kimodo_motion(path, sk, source_fps=source_fps, z_up=z_up)


def load_kimodo_npz_as_torch(
    path: str,
    source_fps: float = KIMODO_CONVERT_TARGET_FPS,
    *,
    ensure_complete: bool = True,
) -> tuple[Dict[str, torch.Tensor], int]:
    """Load a Kimodo NPZ and return all arrays as torch tensors on the skeleton device.

    Args:
        path: Kimodo NPZ file path.
        source_fps: Source frame rate (Hz) used for derived channels when
            ``ensure_complete=True``.
        ensure_complete: If ``True`` and the NPZ lacks derived channels
            (``posed_joints``, ``global_rot_mats``, …), run :func:`complete_motion_dict`
            to fill them from ``local_rot_mats`` + ``root_positions``.
            If ``False``, load all arrays verbatim (requires ``local_rot_mats``).

    Returns:
        ``(tensor_dict, num_joints)``
    """
    raw = load_kimodo_npz(path)
    if "local_rot_mats" in raw:
        j = int(raw["local_rot_mats"].shape[1])
    elif "posed_joints" in raw:
        j = int(raw["posed_joints"].shape[1])
    else:
        raise ValueError("Kimodo NPZ must contain 'local_rot_mats' or 'posed_joints'.")
    sk = build_skeleton(j)
    device = sk.neutral_joints.device
    dtype = torch.float32

    if not ensure_complete:
        if "local_rot_mats" not in raw:
            raise ValueError("Kimodo NPZ must contain 'local_rot_mats' (and typically 'root_positions').")
        out: Dict[str, torch.Tensor] = {}
        for k, v in raw.items():
            out[k] = torch.from_numpy(np.asarray(v)).to(device=device, dtype=dtype)
        return out, j

    if "posed_joints" in raw and "global_rot_mats" in raw:
        out = {}
        for k, v in raw.items():
            out[k] = torch.from_numpy(np.asarray(v)).to(device=device, dtype=dtype)
        return out, j

    if "local_rot_mats" not in raw or "root_positions" not in raw:
        raise ValueError("Kimodo NPZ must contain posed_joints+global_rot_mats, or local_rot_mats+root_positions.")
    local = torch.from_numpy(np.asarray(raw["local_rot_mats"])).to(device=device, dtype=dtype)
    root = torch.from_numpy(np.asarray(raw["root_positions"])).to(device=device, dtype=dtype)
    return complete_motion_dict(local, root, sk, float(source_fps)), j


def save_kimodo_npz_at_target_fps(
    motion: Dict[str, torch.Tensor],
    skeleton: SkeletonBase,
    source_fps: float,
    output_path: str,
    target_fps: float = KIMODO_CONVERT_TARGET_FPS,
) -> None:
    """Resample a motion dict to ``target_fps`` when needed, then save Kimodo NPZ."""
    t_before = int(motion["local_rot_mats"].shape[0])
    motion, did_resample = resample_motion_dict_to_kimodo_fps(motion, skeleton, source_fps, target_fps)
    t_after = int(motion["local_rot_mats"].shape[0])
    if did_resample:
        warn_kimodo_npz_framerate(source_fps, t_before, t_after)
    save_kimodo_npz(output_path, motion)


def kimodo_npz_to_bytes(motion_dict: Dict[str, Any]) -> bytes:
    """Serialize a Kimodo motion dict to in-memory NPZ bytes."""
    import io

    buf = io.BytesIO()
    np.savez(buf, **motion_dict_to_numpy(motion_dict))
    return buf.getvalue()


def g1_csv_to_bytes(motion_dict: Dict[str, Any], skeleton: SkeletonBase, device: Any) -> bytes:
    """Convert a motion dict to G1 MuJoCo CSV bytes via :class:`MujocoQposConverter`."""
    import io

    from kimodo.exports.mujoco import MujocoQposConverter

    converter = MujocoQposConverter(skeleton)
    qpos = converter.dict_to_qpos(
        {k: v for k, v in motion_dict.items() if k in ("local_rot_mats", "root_positions")},
        device,
        numpy=True,
    )
    buf = io.StringIO()
    np.savetxt(buf, qpos, delimiter=",")
    return buf.getvalue().encode("utf-8")


def amass_npz_to_bytes(motion_dict: Dict[str, Any], skeleton: SkeletonBase, fps: float) -> bytes:
    """Convert a motion dict to AMASS NPZ bytes via :class:`AMASSConverter`."""
    import io

    from kimodo.exports.smplx import AMASSConverter

    converter = AMASSConverter(skeleton=skeleton, fps=fps)
    buf = io.BytesIO()
    converter.convert_save_npz(
        {k: v for k, v in motion_dict.items() if k in ("local_rot_mats", "root_positions")},
        buf,
    )
    return buf.getvalue()


def _read_amass_source_fps(path: str) -> float:
    """Read the source frame rate from an AMASS NPZ, defaulting to 30 Hz."""
    with np.load(path, allow_pickle=True) as z:
        if "mocap_frame_rate" in z.files:
            return float(z["mocap_frame_rate"])
    return 30.0


def load_motion_file(
    path: str,
    source_fps: float | None = None,
    target_fps: float | None = None,
    *,
    z_up: bool = True,
    mujoco_rest_zero: bool = False,
) -> tuple[Dict[str, torch.Tensor], int]:
    """Load a motion file and return a Kimodo motion dict plus joint count.

    Supports SOMA BVH (``.bvh``), G1 MuJoCo CSV (``.csv``), Kimodo NPZ, and AMASS SMPL-X NPZ
    (``.npz``).

    The motion is loaded at its native (or overridden) source rate, then
    resampled to ``target_fps`` when they differ.

    Args:
        path: Path to ``.bvh``, ``.csv``, or ``.npz``.
        source_fps: Source frame rate (Hz).  If provided, trusted as-is.
            If ``None``, auto-detected per format: BVH ``Frame Time`` header,
            AMASS ``mocap_frame_rate``, or :data:`KIMODO_CONVERT_TARGET_FPS`
            (30 Hz) for CSV / Kimodo NPZ.
        target_fps: Desired output frame rate (Hz).  Defaults to
            :data:`KIMODO_CONVERT_TARGET_FPS` (30 Hz).  The motion is
            resampled when ``source_fps`` and ``target_fps`` differ.
        z_up: AMASS NPZ only; passed to :func:`load_amass_npz`.
        mujoco_rest_zero: G1 CSV only; passed to :func:`load_g1_csv`.

    Returns:
        ``(motion_dict, num_joints)`` with the same keys as :func:`complete_motion_dict`.
    """
    from kimodo.exports.motion_formats import infer_npz_kind

    if target_fps is None:
        target_fps = KIMODO_CONVERT_TARGET_FPS

    ext = os.path.splitext(path)[1].lower()
    if ext == ".bvh":
        from kimodo.exports.bvh import bvh_to_kimodo_motion

        motion_dict, bvh_fps = bvh_to_kimodo_motion(path)
        effective_source = source_fps if source_fps is not None else bvh_fps
        num_joints = int(motion_dict["local_rot_mats"].shape[1])
    elif ext == ".csv":
        effective_source = source_fps if source_fps is not None else KIMODO_CONVERT_TARGET_FPS
        motion_dict = load_g1_csv(path, source_fps=effective_source, mujoco_rest_zero=mujoco_rest_zero)
        num_joints = 34
    elif ext == ".npz":
        kind = infer_npz_kind(path)
        if kind == "amass":
            effective_source = source_fps if source_fps is not None else _read_amass_source_fps(path)
            motion_dict = load_amass_npz(path, source_fps=effective_source, z_up=z_up)
            num_joints = 22
        else:
            effective_source = source_fps if source_fps is not None else KIMODO_CONVERT_TARGET_FPS
            motion_dict, num_joints = load_kimodo_npz_as_torch(path, source_fps=effective_source)
    else:
        raise ValueError(f"Unsupported motion file {path!r}; expected .bvh, .csv, or .npz")

    if abs(effective_source - target_fps) > 0.5:
        sk = build_skeleton(num_joints)
        motion_dict, did_resample = resample_motion_dict_to_kimodo_fps(motion_dict, sk, effective_source, target_fps)
        if did_resample:
            t_out = int(motion_dict["local_rot_mats"].shape[0])
            warnings.warn(
                f"Resampled motion from {effective_source:.4g} Hz to " f"{target_fps:.0f} Hz ({t_out} frames).",
                UserWarning,
                stacklevel=2,
            )

    return motion_dict, num_joints
