#!/usr/bin/env python3
"""Rewrite an existing episode's ``/hand-pose`` using Stera's HaWoR backend.

Inference lives in ``stera.models.hawor``. This file only dumps ``rgb.mp4``
1:1, calls ``HandTracker(model="hawor").detect_sequence``, and patches hdf5.
"""
from __future__ import annotations

import json
from pathlib import Path
import os

import h5py
import numpy as np
from stera.data.export import _hands_to_arrays
from stera.models import HandTracker

DEFAULT_HAWOR_DIR = Path(os.environ.get("KODA_HAWOR_DIR", str(Path(os.environ.get("KODA_ASSETS", "assets")) / "models/hawor")))
DEFAULT_YOLO_CONF = 0.2


def run_hawor_detect_sequence(
    rgb_source: str | Path,
    rgb_k: np.ndarray,
    *,
    hawor_dir: Path = DEFAULT_HAWOR_DIR,
    yolo_conf: float = DEFAULT_YOLO_CONF,
):
    """Same HaWoR call used by rewrite, process_session, and batch.

    ``rgb_source`` is ``rgb.mp4`` or a 1:1 JPEG directory. ``rgb_k`` is the
    RGB camera matrix, never depth_K. No LiDAR pin, no W0+MCP, no SLAM.
    """
    rgb_k = np.asarray(rgb_k, dtype=np.float64).reshape(3, 3)
    if "hawor" not in HandTracker.SUPPORTED_MODELS or not callable(
        getattr(HandTracker, "detect_sequence", None)
    ):
        raise RuntimeError(
            "This pipeline needs the vendored HaWoR-enabled stera-sdk. "
            "Install the KODA CUDA environment and provide KODA_HAWOR_DIR."
        )
    tracker = HandTracker(
        model="hawor",
        model_path=str(Path(hawor_dir)),
        yolo_conf=float(yolo_conf),
    )
    seq = tracker.detect_sequence(str(rgb_source), intrinsics=rgb_k)
    stats = dict(getattr(tracker._backend, "last_stats", {}) or {})
    return seq, stats, tracker


_HAND_POSE_ATTR_KEYS = ("backend", "coord_frame", "joint_layout")


def normalize_hand_pose_hdf5_schema(h5_path: Path) -> None:
    """Keep ``/hand-pose`` attrs identical to the original WiLoR export."""
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "a") as h5f:
        if "hand-pose" not in h5f:
            return
        hp = h5f["hand-pose"]
        for key in list(hp.attrs.keys()):
            if key not in _HAND_POSE_ATTR_KEYS:
                del hp.attrs[key]


def write_hand_poses_hdf5(
    h5_path: Path,
    poses_by_idx: dict[int, list],
    *,
    n_frames: int | None = None,
    timestamps: np.ndarray | None = None,
) -> dict:
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "a") as h5f:
        if n_frames is None:
            if "hand-pose" in h5f and "left_joints" in h5f["hand-pose"]:
                n_frames = int(h5f["hand-pose/left_joints"].shape[0])
            elif "metadata" in h5f and "num_rgb_frames" in h5f["metadata"].attrs:
                n_frames = int(h5f["metadata"].attrs["num_rgb_frames"])
            else:
                n_frames = (max(poses_by_idx) + 1) if poses_by_idx else 0
        if timestamps is None and "hand-pose" in h5f and "timestamps" in h5f["hand-pose"]:
            timestamps = np.asarray(h5f["hand-pose/timestamps"], dtype=np.float64)
        if timestamps is None:
            timestamps = np.arange(int(n_frames), dtype=np.float64)
        timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if len(timestamps) != int(n_frames):
            if len(timestamps) > int(n_frames):
                timestamps = timestamps[: int(n_frames)]
            else:
                pad = np.full(int(n_frames) - len(timestamps), np.nan, dtype=np.float64)
                timestamps = np.concatenate([timestamps, pad], axis=0)
        arrs = _hands_to_arrays(poses_by_idx, n_rgb=int(n_frames))
        if "hand-pose" in h5f:
            del h5f["hand-pose"]
        hp = h5f.create_group("hand-pose")
        hp.create_dataset("timestamps", data=timestamps)
        hp.create_dataset("left_joints", data=arrs["left_joints"])
        hp.create_dataset("right_joints", data=arrs["right_joints"])
        hp.create_dataset("left_valid", data=arrs["left_valid"])
        hp.create_dataset("right_valid", data=arrs["right_valid"])
        hp.create_dataset("left_confidence", data=arrs["left_confidence"])
        hp.create_dataset("right_confidence", data=arrs["right_confidence"])
        hp.attrs["coord_frame"] = "camera_3d" if arrs["has_3d"] else "image_2d"
        hp.attrs["joint_layout"] = (
            "wrist, 5×[mcp,pip,dip,tip] (thumb,index,middle,ring,pinky)"
        )
        hp.attrs["backend"] = arrs.get("backend") or "hawor"
        compress_keys = {"left_mano_vertices", "right_mano_vertices"}
        for key in (
            "left_kpts_2d_rgb",
            "right_kpts_2d_rgb",
            "left_mano_vertices",
            "right_mano_vertices",
            "left_mano_global_orient",
            "right_mano_global_orient",
            "left_mano_hand_pose",
            "right_mano_hand_pose",
            "left_mano_betas",
            "right_mano_betas",
            "left_pred_cam",
            "right_pred_cam",
            "left_pred_cam_t",
            "right_pred_cam_t",
            "left_cam_t",
            "right_cam_t",
            "left_focal_length",
            "right_focal_length",
        ):
            buf = arrs.get(key)
            if buf is None:
                continue
            kw = (
                {
                    "chunks": (1,) + buf.shape[1:],
                    "compression": "gzip",
                    "compression_opts": 4,
                }
                if key in compress_keys
                else {}
            )
            hp.create_dataset(key, data=buf, **kw)
        backend = str(hp.attrs.get("backend") or "hawor")
    return {
        "n_frames": int(n_frames),
        "n_left": int(np.count_nonzero(arrs["left_valid"])),
        "n_right": int(np.count_nonzero(arrs["right_valid"])),
        "backend": backend,
    }


def rewrite_hands_from_export(
    out_dir: Path,
    *,
    hawor_dir: Path = DEFAULT_HAWOR_DIR,
    yolo_conf: float = DEFAULT_YOLO_CONF,
) -> dict:
    out_dir = Path(out_dir).expanduser().resolve()
    rgb = out_dir / "rgb.mp4"
    h5_path = out_dir / "annotation.hdf5"
    k_path = out_dir / "calibrations" / "rgb_K.npy"
    if not rgb.is_file():
        raise FileNotFoundError(rgb)
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)
    if not k_path.is_file():
        raise FileNotFoundError(k_path)
    rgb_k = np.load(k_path)
    seq, stats, _tracker = run_hawor_detect_sequence(
        rgb, rgb_k, hawor_dir=hawor_dir, yolo_conf=yolo_conf
    )
    poses_by_idx = {i: hands for i, hands in enumerate(seq) if hands}
    written = write_hand_poses_hdf5(h5_path, poses_by_idx, n_frames=len(seq))
    normalize_hand_pose_hdf5_schema(h5_path)
    payload = {**stats, **written, "out_dir": str(out_dir)}
    work = out_dir / "_hawor_work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "hawor_run.json").write_text(json.dumps(payload, indent=2))
    print("[hawor]", json.dumps(payload, indent=2))
    return payload


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hawor-dir", type=Path, default=DEFAULT_HAWOR_DIR)
    p.add_argument("--yolo-conf", type=float, default=DEFAULT_YOLO_CONF)
    args = p.parse_args(argv)
    rewrite_hands_from_export(
        args.out, hawor_dir=args.hawor_dir, yolo_conf=args.yolo_conf
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
