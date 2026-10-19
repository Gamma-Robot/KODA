# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Library API for converting between Kimodo NPZ, AMASS NPZ, SOMA BVH, and G1 MuJoCo CSV."""

from __future__ import annotations

import warnings

import numpy as np

from kimodo.exports.bvh import bvh_to_kimodo_motion, save_motion_bvh
from kimodo.exports.motion_formats import (
    infer_source_format_from_path,
    infer_target_format_from_path,
    resolve_source_fps,
)
from kimodo.exports.motion_io import (
    load_amass_npz,
    load_g1_csv,
    load_kimodo_npz_as_torch,
    save_kimodo_npz_at_target_fps,
)
from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.exports.smplx import AMASSConverter
from kimodo.skeleton.registry import build_skeleton


def convert_motion_files(
    input_path: str,
    output_path: str,
    *,
    from_fmt: str | None = None,
    to_fmt: str | None = None,
    source_fps: float | None = None,
    z_up: bool = True,
    mujoco_rest_zero: bool = False,
    bvh_standard_tpose: bool = False,
) -> None:
    """Convert a motion file between Kimodo-supported formats.

    Supported pairs (hub-and-spoke through Kimodo NPZ):

    - amass <-> kimodo
    - soma-bvh <-> kimodo
    - g1-csv <-> kimodo

    Args:
        input_path: Source file (``.npz``, ``.bvh``, or ``.csv``).
        output_path: Destination file.
        from_fmt: Source format; inferred from extension/contents when ``None``.
        to_fmt: Target format; inferred from extension when ``None``.
        source_fps: Source motion frame rate (Hz).  If provided, trusted as-is.
            If ``None``, auto-detected from BVH ``Frame Time``, AMASS
            ``mocap_frame_rate``, or default 30.
        z_up: For AMASS conversions, apply the Z-up <-> Kimodo Y-up transform.
        mujoco_rest_zero: For G1 CSV, joint angles relative to MuJoCo rest pose.
        bvh_standard_tpose: If input or output is BVH: the BVH file uses the standard T-pose 
            as its rest pose instead of the BONES-SEED rest pose.
    """
    from_fmt = from_fmt or infer_source_format_from_path(input_path)
    to_fmt = to_fmt or infer_target_format_from_path(output_path, from_fmt)

    _validate_output_extension(to_fmt, output_path)

    pair = (from_fmt, to_fmt)

    if pair == ("amass", "kimodo"):
        sk = build_skeleton(22)
        effective_source = source_fps
        if effective_source is None:
            with np.load(input_path, allow_pickle=True) as z:
                effective_source = float(z["mocap_frame_rate"]) if "mocap_frame_rate" in z.files else 30.0
        motion = load_amass_npz(input_path, source_fps=effective_source, z_up=z_up)
        save_kimodo_npz_at_target_fps(motion, sk, effective_source, output_path)
        return

    if pair == ("kimodo", "amass"):
        data, J = load_kimodo_npz_as_torch(input_path, ensure_complete=False)
        if J != 22:
            raise ValueError(f"Kimodo→AMASS requires 22 joints (SMPL-X); this file has J={J}.")
        sk = build_skeleton(22)
        effective_source = resolve_source_fps(source_fps, "kimodo", input_path, None)
        converter = AMASSConverter(fps=effective_source, skeleton=sk)
        converter.convert_save_npz(data, output_path, z_up=z_up)
        return

    if pair == ("soma-bvh", "kimodo"):
        sk = build_skeleton(77)
        motion, bvh_fps = bvh_to_kimodo_motion(input_path, skeleton=sk, standard_tpose=bvh_standard_tpose)
        effective_source = source_fps if source_fps is not None else bvh_fps
        save_kimodo_npz_at_target_fps(motion, sk, effective_source, output_path)
        return

    if pair == ("kimodo", "soma-bvh"):
        data, J = load_kimodo_npz_as_torch(input_path, ensure_complete=False)
        if J == 30:
            warnings.warn(
                f"Input has 30 joints (somaskel30); expanding to somaskel77 for BVH export.",
                UserWarning,
                stacklevel=2,
            )
            sk = build_skeleton(30)
        elif J == 77:
            sk = build_skeleton(77)
        else:
            raise ValueError(f"Kimodo→BVH requires a SOMA skeleton (30 or 77 joints); this file has J={J}.")
        effective_source = resolve_source_fps(source_fps, "kimodo", input_path, None)
        save_motion_bvh(
            output_path,
            data["local_rot_mats"],
            data["root_positions"],
            skeleton=sk,
            fps=effective_source,
            standard_tpose=bvh_standard_tpose,
        )
        return

    if pair == ("g1-csv", "kimodo"):
        sk = build_skeleton(34)
        effective_source = resolve_source_fps(source_fps, "g1-csv", input_path, None)
        motion = load_g1_csv(input_path, source_fps=effective_source, mujoco_rest_zero=mujoco_rest_zero)
        save_kimodo_npz_at_target_fps(motion, sk, effective_source, output_path)
        return

    if pair == ("kimodo", "g1-csv"):
        data, J = load_kimodo_npz_as_torch(input_path, ensure_complete=False)
        if J != 34:
            raise ValueError(f"Kimodo→CSV requires G1 with 34 joints; this file has J={J}.")
        sk = build_skeleton(34)
        effective_source = resolve_source_fps(source_fps, "kimodo", input_path, None)
        converter = MujocoQposConverter(sk)
        qpos = converter.dict_to_qpos(
            {k: v for k, v in data.items() if k in ("local_rot_mats", "root_positions")},
            device=str(sk.neutral_joints.device),
            numpy=True,
            mujoco_rest_zero=mujoco_rest_zero,
        )
        converter.save_csv(qpos, output_path)
        return

    raise ValueError(
        f"Unsupported conversion {from_fmt!r} → {to_fmt!r}. "
        "Supported: amass↔kimodo (SMPL-X NPZ), soma-bvh↔kimodo, g1-csv↔kimodo."
    )


def _validate_output_extension(to_fmt: str, output_path: str) -> None:
    lower = output_path.lower()
    if to_fmt == "kimodo" and lower.endswith(".npz"):
        return
    if to_fmt == "amass":
        if not lower.endswith(".npz"):
            raise ValueError("AMASS output must use a .npz path.")
    elif to_fmt == "soma-bvh":
        if not lower.endswith(".bvh"):
            raise ValueError("SOMA BVH output must use a .bvh path.")
    elif to_fmt == "g1-csv":
        if not lower.endswith(".csv"):
            raise ValueError("G1 CSV output must use a .csv path.")
