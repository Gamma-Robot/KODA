#!/usr/bin/env python3
"""Process a Stera MCAP into an episode directory.

Follows https://www.fpvlabs.ai/stera/docs/process/get-started/quickstart
and Episode export guide. Writes visualization.rrd for offline Rerun viewing.

By default, drops the prefix/suffix until the first/last frame where both
hands are detected, and exports only that span (rgb / hdf5 / rrd).

This file lives in the KODA repository. A thin shim remains at
``<DATA4>/zzb/ego_anywhere/process_session_noviz.py`` for old commands.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import h5py
import numpy as np
import stera
from stera.data import MCAPReader
from stera.models import FaceBlurrer, HandTracker, UpperBodyEstimator
from stera.viz import Visualizer

_EGO2 = Path(__file__).resolve().parent
if str(_EGO2) not in sys.path:
    sys.path.insert(0, str(_EGO2))


def _has_both_hands(hands) -> bool:
    sides = {getattr(h, "hand_side", None) for h in (hands or [])}
    return "left" in sides and "right" in sides


def both_hands_frame_bounds(
    hand_poses: dict[int, list], n_frames: int
) -> tuple[int, int] | None:
    """Inclusive [start, end] from first to last frame with both hands visible."""
    if n_frames <= 0:
        return None
    mask = [_has_both_hands(hand_poses.get(i, [])) for i in range(n_frames)]
    if not any(mask):
        return None
    start = next(i for i, ok in enumerate(mask) if ok)
    end = n_frames - 1 - next(i for i, ok in enumerate(reversed(mask)) if ok)
    return start, end


def dump_session_rgb_jpegs(session, img_dir: Path) -> list[float]:
    """Write ``frame.index``-named JPEGs and return timestamps in iterator order."""
    img_dir.mkdir(parents=True, exist_ok=True)
    timestamps: list[float] = []
    expected = int(getattr(session, "num_rgb_frames", 0) or 0)
    existing = list(img_dir.glob("*.jpg"))
    need_write = expected <= 0 or len(existing) != expected
    for frame in session.frames():
        timestamps.append(float(frame.timestamp))
        if not need_write:
            continue
        rgb = np.asarray(frame.rgb)
        bgr = rgb[:, :, ::-1] if rgb.ndim == 3 and rgb.shape[2] == 3 else rgb
        path = img_dir / f"{int(frame.index):06d}.jpg"
        if not cv2.imwrite(str(path), bgr):
            raise RuntimeError(f"failed to write {path}")
    return timestamps


def restrict_session_to_frame_range(
    session,
    start: int,
    end: int,
    *,
    frame_timestamps: list[float] | None = None,
) -> dict:
    """In-place restrict session export/iteration to inclusive [start, end].

    Remaps ``hand_poses`` keys to 0..N-1, filters cam-pose / IMU by the kept
    timestamp window, and patches ``frames`` / duration / frame counts so
    ``session.export`` and Rerun only see the trimmed clip.
    """
    if end < start:
        raise ValueError(f"invalid trim range [{start}, {end}]")

    orig_n = int(session.num_rgb_frames)
    orig_duration = float(session.duration)
    if start < 0 or end >= orig_n:
        raise ValueError(
            f"trim [{start}, {end}] out of range for {orig_n} rgb frames"
        )

    if frame_timestamps is None:
        frame_timestamps = []
        for frame in session.frames():
            frame_timestamps.append(float(frame.timestamp))
    if len(frame_timestamps) != orig_n:
        raise RuntimeError(
            f"frame_timestamps length {len(frame_timestamps)} != num_rgb_frames {orig_n}"
        )
    t0 = float(frame_timestamps[start])
    t1 = float(frame_timestamps[end])
    n = end - start + 1
    fps = orig_n / max(orig_duration, 1e-3)
    new_duration = n / fps

    _orig_frames = session.frames
    _orig_poses = session.all_camera_poses
    _orig_imu = session.all_imu_samples
    _orig_get_summary = session._get_summary

    def frames(max_depth_dt: float = 0.1, max_pose_dt: float = 0.1):
        for frame in _orig_frames(max_depth_dt=max_depth_dt, max_pose_dt=max_pose_dt):
            if frame.index < start:
                continue
            if frame.index > end:
                break
            yield replace(frame, index=frame.index - start)

    def all_camera_poses():
        return [
            (ts, pose)
            for ts, pose in _orig_poses()
            if (t0 - 1e-6) <= float(ts) <= (t1 + 1e-6)
        ]

    def all_imu_samples():
        return [
            (ts, sample)
            for ts, sample in _orig_imu()
            if (t0 - 1e-6) <= float(ts) <= (t1 + 1e-6)
        ]

    def _get_summary():
        summary = dict(_orig_get_summary())
        topics = dict(summary.get("topics", {}))
        topics[session._topics.rgb] = n
        # Depth is nearest-neighbor matched to RGB; export writes one depth per RGB.
        if getattr(session, "_topics", None) is not None and hasattr(
            session._topics, "depth"
        ):
            topics[session._topics.depth] = n
        summary["topics"] = topics
        summary["duration_sec"] = float(new_duration)
        return summary

    session.frames = frames
    session.all_camera_poses = all_camera_poses
    session.all_imu_samples = all_imu_samples
    session._get_summary = _get_summary
    session._summary = None
    # Invalidate timestamp caches so filtered pose/imu paths rebuild.
    if hasattr(session, "_pose_timestamps"):
        session._pose_timestamps = None
    if hasattr(session, "_imu_timestamps"):
        session._imu_timestamps = None
    if hasattr(session, "_depth_timestamps"):
        session._depth_timestamps = None
    if hasattr(session, "_cached_poses"):
        session._cached_poses = None

    old_hp = dict(getattr(session, "_hand_poses", {}) or {})
    remapped: dict[int, list] = {}
    for i in range(start, end + 1):
        hands = list(old_hp.get(i, []))
        new_i = i - start
        for hp in hands:
            if hasattr(hp, "frame_idx"):
                hp.frame_idx = new_i
        remapped[new_i] = hands
    session._hand_poses = remapped

    info = {
        "method": "trim_prefix_suffix_until_both_hands",
        "start_frame": int(start),
        "end_frame": int(end),
        "n_frames_kept": int(n),
        "n_frames_original": int(orig_n),
        "n_frames_dropped_prefix": int(start),
        "n_frames_dropped_suffix": int(orig_n - 1 - end),
        "t0": float(t0),
        "t1": float(t1),
        "duration_s_original": float(orig_duration),
        "duration_s_kept": float(new_duration),
        "applied_to": [
            "rgb.mp4 (streamed trimmed frames only)",
            "annotation.hdf5:/depth",
            "annotation.hdf5:/cam-pose (timestamp window)",
            "annotation.hdf5:/imu (timestamp window)",
            "annotation.hdf5:/hand-pose",
            "annotation.hdf5:/metadata",
            "thumbnail.jpg",
            "visualization.rrd",
        ],
        "not_trimmed": [
            "mesh.ply (spatial map)",
            "calibrations/",
        ],
    }
    return info


def _read_binary_ply(path: Path):
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header.append(line)
            if line.startswith("end_header"):
                break
        n_verts = 0
        props = []
        section = None
        for line in header:
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
                section = "vertex"
            elif line.startswith("element face"):
                section = "face"
            elif line.startswith("property") and section == "vertex":
                props.append(line.split()[1:])
        typemap = {
            "float": "f4",
            "double": "f8",
            "uchar": "u1",
            "uint8": "u1",
            "int": "i4",
            "uint": "u4",
            "ushort": "u2",
        }
        dtype = np.dtype(
            {"names": [n for _, n in props], "formats": [typemap[t] for t, _ in props]}
        )
        verts = np.fromfile(f, dtype=dtype, count=n_verts)
        rest = f.read()
    return header, verts, rest


def _fit_lowest_horizontal_plane(
    xyz: np.ndarray,
    *,
    thresh: float = 0.015,
    min_up: float = 0.9,
    min_inliers: int = 200,
    max_rounds: int = 6,
    max_iteration: int = 2000,
) -> dict:
    """RANSAC-extract near-horizontal planes; return the lowest one by inlier mean Y."""
    import pyransac3d as pyrsc

    remaining = np.asarray(xyz, dtype=np.float64)
    candidates: list[dict] = []
    rng = np.random.default_rng(0)

    for _ in range(max_rounds):
        if remaining.shape[0] < max(min_inliers, 100):
            break
        # subsample for speed on dense meshes
        if remaining.shape[0] > 20000:
            sel = rng.choice(remaining.shape[0], 20000, replace=False)
            fit_pts = remaining[sel]
        else:
            fit_pts = remaining
            sel = None

        eq, inliers_local = pyrsc.Plane().fit(
            fit_pts, thresh=thresh, maxIteration=max_iteration
        )
        eq = np.asarray(eq, dtype=np.float64).reshape(4)
        n = eq[:3]
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-8:
            break
        n = n / n_norm
        eq = eq / n_norm
        if n[1] < 0:
            n = -n
            eq = -eq

        if sel is None:
            inlier_idx = np.asarray(inliers_local, dtype=np.int64)
            inlier_pts = remaining[inlier_idx]
        else:
            # map subsample inliers back approximately: take all remaining
            # points within thresh of this plane
            dist = np.abs(remaining @ eq[:3] + eq[3])
            inlier_idx = np.where(dist <= thresh)[0]
            inlier_pts = remaining[inlier_idx]

        up = float(abs(n[1]))
        height = float(np.mean(inlier_pts[:, 1])) if inlier_pts.size else float("inf")
        if up >= min_up and inlier_pts.shape[0] >= min_inliers:
            candidates.append(
                {
                    "equation_axbyczd": eq.tolist(),
                    "normal": n.tolist(),
                    "up_alignment": up,
                    "height_mean_inlier_y": height,
                    "n_inliers": int(inlier_pts.shape[0]),
                }
            )

        if inlier_idx.size == 0:
            break
        mask = np.ones(remaining.shape[0], dtype=bool)
        mask[inlier_idx] = False
        remaining = remaining[mask]

    if not candidates:
        # Fallback: RANSAC only on points below median Y (bias to floor).
        y_med = float(np.median(xyz[:, 1]))
        low = xyz[xyz[:, 1] <= y_med]
        if low.shape[0] < 100:
            low = xyz
        eq, inliers = pyrsc.Plane().fit(
            low, thresh=thresh, minPoints=min(100, max(3, low.shape[0] // 5)),
            maxIteration=max_iteration,
        )
        eq = np.asarray(eq, dtype=np.float64).reshape(4)
        n = eq[:3]
        n = n / (np.linalg.norm(n) + 1e-12)
        eq = eq / (np.linalg.norm(eq[:3]) + 1e-12)
        if n[1] < 0:
            n = -n
            eq = -eq
        inlier_pts = low[np.asarray(inliers, dtype=np.int64)]
        candidates.append(
            {
                "equation_axbyczd": eq.tolist(),
                "normal": n.tolist(),
                "up_alignment": float(abs(n[1])),
                "height_mean_inlier_y": float(np.mean(inlier_pts[:, 1])),
                "n_inliers": int(inlier_pts.shape[0]),
                "fallback": "bottom_half_ransac",
            }
        )

    candidates.sort(key=lambda c: c["height_mean_inlier_y"])
    best = candidates[0]
    best["n_horizontal_candidates"] = len(candidates)
    return best


def estimate_ground_from_xyz(
    xyz: np.ndarray,
    *,
    plane_thresh: float = 0.015,
    min_up: float = 0.9,
) -> dict:
    """Fit lowest near-horizontal plane; return plane dict + ground_y_offset."""
    plane = _fit_lowest_horizontal_plane(
        np.asarray(xyz, dtype=np.float64),
        thresh=plane_thresh,
        min_up=min_up,
    )
    ground_y = float(plane["height_mean_inlier_y"])
    return {
        "ground_axis": "y",
        "ground_y_offset": ground_y,
        "method": "ransac_lowest_horizontal_plane",
        "plane": plane,
    }


def estimate_ground_from_session(
    session,
    *,
    plane_thresh: float = 0.015,
    min_up: float = 0.9,
) -> dict | None:
    """Estimate ground Y from the session map mesh (before any shift)."""
    mesh_data = session.mesh()
    if mesh_data is None:
        return None
    verts = np.asarray(mesh_data[0], dtype=np.float64)
    if verts.size == 0:
        return None
    return estimate_ground_from_xyz(
        verts, plane_thresh=plane_thresh, min_up=min_up
    )


def patch_session_ground_y(session, ground_y: float) -> None:
    """Shift world-frame Y on mesh / poses / point clouds used by viz+export.

    Hand poses stay camera_3d (unchanged). Depth / IMU / calibrations unchanged.
    """
    from stera.core.types import Pose6D

    ground_y = float(ground_y)
    if abs(ground_y) < 1e-12:
        return

    # Invalidate pose cache so the wrapped reader reloads from topic once.
    if hasattr(session, "_cached_poses"):
        session._cached_poses = None

    _mesh = session.mesh
    _poses = session.all_camera_poses
    _pc = session.point_cloud
    _dense = session.dense_point_cloud

    def mesh():
        data = _mesh()
        if data is None:
            return None
        verts, faces, colors = data
        verts = np.asarray(verts, dtype=np.float64).copy()
        verts[:, 1] -= ground_y
        return verts.astype(np.float32), faces, colors

    def all_camera_poses():
        out = []
        for ts, pose in _poses():
            t = np.asarray(pose.translation, dtype=np.float64).copy()
            t[1] -= ground_y
            out.append(
                (
                    ts,
                    Pose6D(
                        rotation=pose.rotation,
                        translation=t.astype(np.float32),
                        timestamp=pose.timestamp,
                    ),
                )
            )
        return out

    def point_cloud(source: str = "auto"):
        xyz, rgb = _pc(source=source)
        if len(xyz) == 0:
            return xyz, rgb
        xyz = np.asarray(xyz, dtype=np.float64).copy()
        xyz[:, 1] -= ground_y
        return xyz.astype(np.float32), rgb

    def dense_point_cloud(*args, **kwargs):
        xyz, rgb = _dense(*args, **kwargs)
        if len(xyz) == 0:
            return xyz, rgb
        xyz = np.asarray(xyz, dtype=np.float64).copy()
        xyz[:, 1] -= ground_y
        return xyz.astype(np.float32), rgb

    session.mesh = mesh
    session.all_camera_poses = all_camera_poses
    session.point_cloud = point_cloud
    session.dense_point_cloud = dense_point_cloud


def write_ground_align_meta(out_dir: Path, info: dict) -> dict:
    """Stamp ground-align metadata onto exported episode (already shifted)."""
    out_dir = Path(out_dir)
    h5_path = out_dir / "annotation.hdf5"
    cam_y = None
    mesh_y = None
    mesh_path = out_dir / "mesh.ply"
    if mesh_path.is_file():
        _, verts, _ = _read_binary_ply(mesh_path)
        y = verts["y"].astype(np.float64)
        mesh_y = [float(y.min()), float(y.max())]
    if h5_path.is_file():
        with h5py.File(h5_path, "a") as f:
            if "cam-pose" in f and "translations" in f["cam-pose"]:
                t = f["cam-pose"]["translations"][:]
                cam_y = [float(t[:, 1].min()), float(t[:, 1].max())]
                f["cam-pose"].attrs["ground_y_offset"] = float(info["ground_y_offset"])
                f["cam-pose"].attrs["ground_axis"] = "y"
                if "plane" in info and "equation_axbyczd" in info["plane"]:
                    f["cam-pose"].attrs["ground_plane_eq"] = np.asarray(
                        info["plane"]["equation_axbyczd"], dtype=np.float64
                    )
            if "metadata" in f:
                f["metadata"].attrs["ground_y_offset"] = float(info["ground_y_offset"])
                f["metadata"].attrs["ground_axis"] = "y"

    full = {
        **info,
        "mesh_y_after": mesh_y,
        "cam_pose_y_after": cam_y,
        "applied_to": [
            "session.mesh / mesh.ply (pre-export)",
            "session camera poses / annotation.hdf5:/cam-pose/translations",
            "session point clouds",
            "visualization.rrd (world mesh, poses, skeleton, hand world)",
        ],
        "not_applied": [
            "hand-pose (camera_3d)",
            "depth",
            "imu",
            "calibrations",
        ],
    }
    (out_dir / "ground_align.json").write_text(json.dumps(full, indent=2))
    return full


def _undo_previous_ground_align(out_dir: Path) -> float | None:
    """If ground_align.json exists, add back the previous Y offset so we can re-fit."""
    meta_path = Path(out_dir) / "ground_align.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text())
    offset = meta.get("ground_y_offset", meta.get("ground_y_before"))
    if offset is None:
        return None
    offset = float(offset)
    mesh_path = Path(out_dir) / "mesh.ply"
    if not mesh_path.is_file():
        return None
    header, verts, rest = _read_binary_ply(mesh_path)
    verts = verts.copy()
    verts["y"] = (verts["y"].astype(np.float64) + offset).astype(np.float32)
    with open(mesh_path, "wb") as f:
        for line in header:
            f.write((line + "\n").encode("ascii"))
        f.write(verts.tobytes())
        f.write(rest)
    h5_path = Path(out_dir) / "annotation.hdf5"
    if h5_path.is_file():
        with h5py.File(h5_path, "a") as f:
            if "cam-pose" in f and "translations" in f["cam-pose"]:
                t = f["cam-pose"]["translations"][:]
                t[:, 1] += np.float32(offset)
                f["cam-pose"]["translations"][:] = t
    return offset


def align_ground_y0(
    out_dir: Path,
    *,
    plane_thresh: float = 0.015,
    min_up: float = 0.9,
    undo_previous: bool = True,
) -> dict:
    """Fallback: estimate + shift already-exported mesh.ply / cam-pose on disk."""
    out_dir = Path(out_dir)
    mesh_path = out_dir / "mesh.ply"
    h5_path = out_dir / "annotation.hdf5"
    if not mesh_path.is_file():
        raise FileNotFoundError(f"missing mesh: {mesh_path}")

    undone = None
    if undo_previous:
        undone = _undo_previous_ground_align(out_dir)

    header, verts, rest = _read_binary_ply(mesh_path)
    xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float64)
    info = estimate_ground_from_xyz(xyz, plane_thresh=plane_thresh, min_up=min_up)
    ground_y = float(info["ground_y_offset"])

    xyz[:, 1] -= ground_y
    new_verts = verts.copy()
    new_verts["x"] = xyz[:, 0].astype(np.float32)
    new_verts["y"] = xyz[:, 1].astype(np.float32)
    new_verts["z"] = xyz[:, 2].astype(np.float32)
    with open(mesh_path, "wb") as f:
        for line in header:
            f.write((line + "\n").encode("ascii"))
        f.write(new_verts.tobytes())
        f.write(rest)

    if h5_path.is_file():
        with h5py.File(h5_path, "a") as f:
            if "cam-pose" in f and "translations" in f["cam-pose"]:
                t = f["cam-pose"]["translations"][:]
                t[:, 1] -= np.float32(ground_y)
                f["cam-pose"]["translations"][:] = t

    info["undone_previous_offset"] = undone
    info["applied_via"] = "post_export_file_rewrite"
    return write_ground_align_meta(out_dir, info)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mcap", type=Path, default=None, help="Input Stera MCAP")
    p.add_argument("--out", type=Path, required=True, help="Episode output directory")
    p.add_argument(
        "--wilor-dir",
        type=Path,
        default=Path(os.environ.get("KODA_WILOR_DIR", str(Path(os.environ.get("KODA_ASSETS", "assets")) / "models/wilor"))),
        help="Local WiLoR repo root (contains wilor/ + pretrained_models/)",
    )
    p.add_argument(
        "--hand-model",
        choices=("wilor", "hawor"),
        default="hawor",
        help="Hand backend. hawor is camera-space HaWoR (no LiDAR / W0 pin)",
    )
    p.add_argument(
        "--hawor-dir",
        type=Path,
        default=Path(os.environ.get("KODA_HAWOR_DIR", str(Path(os.environ.get("KODA_ASSETS", "assets")) / "models/hawor"))),
        help="Local HaWoR repo root (https://github.com/ThunderVVV/HaWoR)",
    )
    p.add_argument(
        "--hawor-yolo-conf",
        type=float,
        default=0.2,
        help="YOLO confidence for HaWoR (same as hawor_hand_tracker, default 0.2)",
    )
    p.add_argument(
        "--rewrite-hands",
        action="store_true",
        help=(
            "Re-run only the hand backend on an existing --out episode "
            "(rgb.mp4 + hdf5) and replace annotation.hdf5:/hand-pose"
        ),
    )
    p.add_argument(
        "--blur",
        action="store_true",
        help="Enable MediaPipe face blur (off by default)",
    )
    p.add_argument("--no-hands", action="store_true", help="Skip hand tracking")
    p.add_argument(
        "--no-trim-hands",
        action="store_true",
        help="Keep full timeline (do not drop prefix/suffix without both hands)",
    )
    p.add_argument("--no-skeleton", action="store_true", help="Skip upper-body estimate")
    p.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip Rerun visualization.rrd export",
    )
    p.add_argument(
        "--map-3d",
        default="both",
        choices=["auto", "mesh", "mesh_cloud", "point_cloud", "both", "none"],
        help="3D map mode logged into visualization.rrd (default: both)",
    )
    p.add_argument(
        "--mesh-color-speed",
        type=float,
        default=0.5,
        help="MeshRefiner color_speed for viz mesh (0=fast, 1=full; default 0.5)",
    )
    p.add_argument(
        "--no-ground-align",
        action="store_true",
        help="Do not shift world so mesh lowest Y-plane is y=0",
    )
    p.add_argument(
        "--ground-plane-thresh",
        type=float,
        default=0.015,
        help="RANSAC inlier distance threshold (meters) for ground plane",
    )
    p.add_argument(
        "--ground-min-up",
        type=float,
        default=0.9,
        help="Min |normal·Y| to treat a RANSAC plane as horizontal ground",
    )
    args = p.parse_args()
    if args.rewrite_hands:
        if args.out is None:
            p.error("--out is required with --rewrite-hands")
    elif args.mcap is None or args.out is None:
        p.error("--mcap and --out are required")

    stera.setup_logging()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.rewrite_hands:
        from hawor_hand_tracker import rewrite_hands_from_export

        info = rewrite_hands_from_export(
            args.out,
            hawor_dir=args.hawor_dir,
            yolo_conf=args.hawor_yolo_conf,
        )
        print("[rewrite-hands]", json.dumps(info, indent=2, default=str))
        return

    print(f"[open] {args.mcap}")
    session = MCAPReader(str(args.mcap))
    print(
        f"[session] rgb_frames≈{getattr(session, 'num_rgb_frames', '?')} "
        f"intrinsics={session.rgb_intrinsics is not None}"
    )

    ground_info_early = None
    if not args.no_ground_align:
        ground_info_early = estimate_ground_from_session(
            session,
            plane_thresh=args.ground_plane_thresh,
            min_up=args.ground_min_up,
        )
        if ground_info_early is not None:
            patch_session_ground_y(session, ground_info_early["ground_y_offset"])
            print(
                "[ground] pre-shift session "
                f"Δy={ground_info_early['ground_y_offset']:.6f} "
                f"(inliers={ground_info_early['plane']['n_inliers']})"
            )
        else:
            print("[ground] no session mesh yet; will fall back to post-export align")

    hawor_tracker = None
    hawor_stats = None
    frame_timestamps: list[float] = []
    if args.no_hands:
        hands = None
    elif args.hand_model == "hawor":
        from hawor_hand_tracker import run_hawor_detect_sequence
        from natsort import natsorted

        print(
            "[hands] hawor_hand_tracker.run_hawor_detect_sequence "
            "(RGB K, no LiDAR pin, no W0+index-MCP, no SLAM/infiller)"
        )
        img_dir = args.out / "_hawor_work" / "extracted_images"
        print("[hands] pass 1/2: dump RGB then HaWoR detect_sequence")
        frame_timestamps = dump_session_rgb_jpegs(session, img_dir)
        if session.rgb_intrinsics is None:
            raise SystemExit("HaWoR needs RGB intrinsics from the MCAP")
        rgb_k = np.asarray(session.rgb_intrinsics.K, dtype=np.float64)
        seq, hawor_stats, hands = run_hawor_detect_sequence(
            img_dir,
            rgb_k,
            hawor_dir=args.hawor_dir,
            yolo_conf=args.hawor_yolo_conf,
        )
        hawor_tracker = hands
        imgfiles = [Path(p) for p in natsorted(img_dir.glob("*.jpg"))]
        for path, hand_poses in zip(imgfiles, seq):
            if not hand_poses:
                continue
            try:
                frame_idx = int(path.stem)
            except ValueError:
                continue
            session.add_hand_pose(int(frame_idx), hand_poses)
        work = args.out / "_hawor_work"
        work.mkdir(parents=True, exist_ok=True)
        (work / "hawor_run.json").write_text(
            json.dumps(hawor_stats, indent=2, default=str)
        )
    else:
        hands = HandTracker(model="wilor", model_path=str(args.wilor_dir))

    # Pass 1: hand tracking over the full session (needed to find trim bounds).
    do_trim = hands is not None and not args.no_trim_hands
    if hawor_tracker is None and hands is not None:
        print("[hands] pass 1/2: detect over full session")
        for frame in session.frames():
            hand_poses = hands.detect_hands(frame)
            session.add_hand_pose(frame.index, hand_poses)
            frame_timestamps.append(float(frame.timestamp))

    trim_info = None
    if do_trim:
        bounds = both_hands_frame_bounds(session.hand_poses, session.num_rgb_frames)
        if bounds is None:
            print(
                "[trim] WARNING: no frame with both hands; keeping full session"
            )
        else:
            start, end = bounds
            if start == 0 and end == session.num_rgb_frames - 1:
                print("[trim] both-hands span already covers full session")
                trim_info = {
                    "method": "trim_prefix_suffix_until_both_hands",
                    "start_frame": 0,
                    "end_frame": int(end),
                    "n_frames_kept": int(session.num_rgb_frames),
                    "n_frames_original": int(session.num_rgb_frames),
                    "n_frames_dropped_prefix": 0,
                    "n_frames_dropped_suffix": 0,
                    "skipped": "already_full_span",
                }
            else:
                trim_info = restrict_session_to_frame_range(
                    session,
                    start,
                    end,
                    frame_timestamps=frame_timestamps,
                )
                print(
                    "[trim] keep frames "
                    f"[{trim_info['start_frame']}, {trim_info['end_frame']}] "
                    f"({trim_info['n_frames_kept']}/{trim_info['n_frames_original']}); "
                    f"drop prefix={trim_info['n_frames_dropped_prefix']} "
                    f"suffix={trim_info['n_frames_dropped_suffix']}"
                )
    elif args.no_hands:
        print("[trim] skipped (--no-hands)")
    else:
        print("[trim] skipped (--no-trim-hands)")

    blur = FaceBlurrer(model="mediapipe") if args.blur else None
    skeleton = None if args.no_skeleton else UpperBodyEstimator(session=session)
    visualizer = None
    if not args.no_viz:
        visualizer = Visualizer(
            session,
            map_3d=args.map_3d,
            mesh_refine={"color_speed": args.mesh_color_speed},
        )

    # Pass 2: blur / skeleton / rgb / viz only on the (possibly trimmed) range.
    print(
        f"[process] pass 2/2: export loop over {session.num_rgb_frames} frames"
    )
    for frame in session.frames():
        rgb = frame.rgb if hasattr(frame, "rgb") else frame
        if blur is not None:
            rgb_out = blur.blur(frame)
        else:
            rgb_out = rgb
        hand_poses = session.hand_poses.get(frame.index, []) if hands is not None else []
        body = None
        if skeleton is not None:
            body = skeleton.estimate(frame, hands=hand_poses)
        session.add_rgb_frame(frame.index, rgb_out)
        if visualizer is not None:
            visualizer.log_frame(frame, hands=hand_poses, skeleton=body)

    print(f"[export] → {args.out}")
    # If we overwrite an old episode that was already ground-aligned on disk,
    # undo that shift first only when using the post-export fallback path.
    manifest = session.export(str(args.out), visualizer=visualizer)
    if args.hand_model == "hawor" and not args.no_hands:
        from hawor_hand_tracker import normalize_hand_pose_hdf5_schema

        normalize_hand_pose_hdf5_schema(args.out / "annotation.hdf5")
        if hawor_stats:
            manifest["hawor"] = hawor_stats
    if trim_info is not None:
        manifest["hand_trim"] = trim_info
        (args.out / "hand_trim.json").write_text(json.dumps(trim_info, indent=2))
        print("[trim]", json.dumps(trim_info, indent=2))
    if not args.no_ground_align:
        if ground_info_early is not None:
            ground_info = write_ground_align_meta(args.out, ground_info_early)
        else:
            ground_info = align_ground_y0(
                args.out,
                plane_thresh=args.ground_plane_thresh,
                min_up=args.ground_min_up,
            )
        manifest["ground_align"] = ground_info
        print("[ground]", json.dumps(ground_info, indent=2))
    manifest_path = args.out / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps(manifest, indent=2, default=str))
    print(f"[done] wrote manifest {manifest_path}")
    rrd = args.out / "visualization.rrd"
    if rrd.is_file():
        print(f"[viz] {rrd}  ({rrd.stat().st_size / 1e6:.1f} MB)")
        print("[viz] open with:  rerun", str(rrd))


if __name__ == "__main__":
    main()
