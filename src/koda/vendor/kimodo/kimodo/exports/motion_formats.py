# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Infer motion file formats from paths and NPZ contents."""

from __future__ import annotations

import os
from typing import Literal

import numpy as np

MotionSourceFormat = Literal["amass", "kimodo", "soma-bvh", "g1-csv"]
MotionTargetFormat = Literal["amass", "kimodo", "soma-bvh", "g1-csv"]
NpzMotionKind = Literal["amass", "kimodo"]


def infer_npz_kind(path: str) -> NpzMotionKind:
    """Classify a ``.npz`` as AMASS SMPL-X or Kimodo from required array keys."""
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
    if "trans" in keys and "pose_body" in keys and "root_orient" in keys:
        return "amass"
    if "local_rot_mats" in keys or "posed_joints" in keys:
        return "kimodo"
    raise ValueError(
        f"Unrecognized NPZ {path!r}: expected AMASS keys (trans, pose_body, ...) "
        "or Kimodo keys (local_rot_mats, posed_joints, ...)."
    )


def infer_source_format_from_path(path: str) -> MotionSourceFormat:
    """Infer converter input format from file extension and NPZ contents when needed."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".bvh":
        return "soma-bvh"
    if ext == ".csv":
        return "g1-csv"
    if ext == ".npz":
        return infer_npz_kind(path)  # type: ignore[return-value]
    raise ValueError(f"Cannot infer format from extension of {path!r}")


def infer_target_format_from_path(path: str, from_fmt: MotionSourceFormat) -> MotionTargetFormat:
    """Infer converter output format from destination path and source format."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".bvh":
        return "soma-bvh"
    if ext == ".csv":
        return "g1-csv"
    if ext == ".npz":
        if from_fmt == "amass":
            return "kimodo"
        if from_fmt == "kimodo":
            return "amass"
        if from_fmt in ("g1-csv", "soma-bvh"):
            return "kimodo"
        raise ValueError(
            "Ambiguous .npz output: set --to to 'kimodo' or 'amass' when the input format is not amass/kimodo."
        )
    raise ValueError(f"Cannot infer output format from extension of {path!r}")


def resolve_source_fps(
    fps: float | None,
    from_kind: str,
    input_path: str,
    data: dict | None,
) -> float:
    """Resolve source frame rate (Hz) for conversion when ``fps`` is not overridden."""
    if fps is not None:
        return float(fps)
    if data is not None and "mocap_frame_rate" in data:
        return float(np.asarray(data["mocap_frame_rate"]).item())
    if from_kind == "soma-bvh":
        from kimodo.exports.bvh import read_bvh_frame_time_seconds

        return 1.0 / read_bvh_frame_time_seconds(input_path)
    return 30.0
