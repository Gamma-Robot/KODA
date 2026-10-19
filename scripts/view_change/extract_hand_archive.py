#!/usr/bin/env python3
"""Extract the 50 Hz hand archive from a full G1 tracking kinematics file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--full-kinematics", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--require-hand-qvel",
        action="store_true",
        help="Reject inputs without the synchronized tracking hand_qvel field.",
    )
    p.add_argument(
        "--hand-dof",
        type=int,
        default=14,
        help="Expected hand width: 14 for Dex3, 12 for the BrainCo hands.",
    )
    a = p.parse_args()
    with np.load(a.full_kinematics.expanduser().resolve(), allow_pickle=False) as z:
        data = {k: np.asarray(z[k]) for k in z.files}
    qpos = np.asarray(data.get("hand_qpos"), dtype=np.float32)
    names = np.asarray(data.get("hand_joint_names"), dtype=str)
    timestamps = np.asarray(data["timestamps"], dtype=np.float64)
    fps = float(np.asarray(data.get("fps", 50.0)).reshape(()))
    if qpos.ndim != 2 or qpos.shape[1] != a.hand_dof or names.shape != (a.hand_dof,):
        raise ValueError(
            f"invalid {a.hand_dof}-DoF hand contract "
            f"qpos={qpos.shape} names={names.shape}"
        )
    qvel = np.asarray(data.get("hand_qvel"), dtype=np.float32) if "hand_qvel" in data else None
    if a.require_hand_qvel and (qvel is None or qvel.shape != qpos.shape):
        raise ValueError(
            "tracking kinematics must contain hand_qvel with the same shape as hand_qpos"
        )
    if qvel is None or qvel.shape != qpos.shape:
        # Older whole-body releases carried qpos but omitted hand_qvel.  The
        # 50 Hz archive remains deterministic when velocity is reconstructed
        # from its native timestamps instead of silently dropping the hand.
        qvel = np.zeros_like(qpos, dtype=np.float32)
        if len(qpos) > 1:
            dt = np.diff(timestamps)
            dt = np.maximum(dt, 1.0e-6)
            qvel[1:-1] = ((qpos[2:] - qpos[:-2]) / (timestamps[2:, None] - timestamps[:-2, None])).astype(np.float32)
            qvel[0] = ((qpos[1] - qpos[0]) / dt[0]).astype(np.float32)
            qvel[-1] = ((qpos[-1] - qpos[-2]) / dt[-1]).astype(np.float32)
    out = a.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        qpos=qpos,
        qvel=qvel,
        joint_names=names,
        hand_joint_names=names,
        timestamps=timestamps - timestamps[0],
        fps=np.asarray(fps),
        source_method=np.asarray("tracking_kinematics_hand_qpos"),
    )
    print(f"[extract-hand] wrote {out} frames={len(qpos)}")


if __name__ == "__main__":
    main()
