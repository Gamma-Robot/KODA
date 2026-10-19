#!/usr/bin/env python3
"""Source RGB-D depth warp to the target robot camera.

Emits four aligned maps per frame under --output-dir:
  rgb/%06d.png
  depth/%06d.npy   (metric meters; 0 = invalid)
  valid/%06d.png   (255 = has warped content)
  hole/%06d.png    (255 = hole / invalid)
plus preview mp4s.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R, Slerp
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Cache3D is an optional compatibility backend. The production single-frame
# path below uses the torch projection implementation and has no data_alignment
# checkout dependency.
Cache3D_Buffer = None  # type: ignore[assignment,misc]
_CACHE3D_IMPORT_ERROR: Exception | None = None
try:
    from viewport_transform.cache_3d import Cache3D_Buffer  # type: ignore
except ImportError as exc:
    _CACHE3D_IMPORT_ERROR = exc

try:
    from causal_surfel_cache import (
        CausalSurfelBackgroundCache,
        PROVENANCE_FUSED,
        PROVENANCE_INVALID,
        PROVENANCE_OBSERVED,
    )
    from frame_volume import as_mask, as_rgb, create_volume, is_packed_stream, open_volume
    from geometry_gate import GeometryGateThresholds, evaluate_geometry_gate, write_gate_report
except ImportError:  # package-style import
    from .causal_surfel_cache import (
        CausalSurfelBackgroundCache,
        PROVENANCE_FUSED,
        PROVENANCE_INVALID,
        PROVENANCE_OBSERVED,
    )
    from .frame_volume import as_mask, as_rgb, create_volume, is_packed_stream, open_volume
    from .geometry_gate import GeometryGateThresholds, evaluate_geometry_gate, write_gate_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video_file", type=Path, required=True, help="Original source RGB video")
    p.add_argument(
        "--source-rgb-dir",
        type=Path,
        default=None,
        help="Optional packed OpenCV RGB volume; preferred over --video_file decode.",
    )
    p.add_argument(
        "--warp-chunk",
        type=int,
        default=16,
        help="Independent warps per GPU batch before sequential geometry hold.",
    )
    p.add_argument("--h5_file", type=Path, required=True, help="Source camera h5 (transforms/camera)")
    p.add_argument("--gt_depth_dir", type=Path, required=True, help="Source metric depth_m npy dir")
    p.add_argument("--target_poses_npz", type=Path, required=True, help="ST kinematics.npz")
    p.add_argument(
        "--target_intrinsics_json", type=Path, required=True,
        help="Target camera_K.json or root meta.json containing model.d435.K",
    )
    p.add_argument("--target-width", type=int, default=None)
    p.add_argument("--target-height", type=int, default=None)
    p.add_argument(
        "--target-time-match", choices=["nearest", "index", "interpolate"], default="nearest",
        help="nearest is the formal 30 Hz source -> 50 Hz target mapping",
    )
    p.add_argument(
        "--target-time-domain", choices=["relative", "absolute"], default="relative",
        help="D435 timestamps are relative to the first source frame",
    )
    p.add_argument("--timeline-manifest", type=Path, default=None)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument(
        "--alignment-manifest", "--manifest", dest="alignment_manifest", type=Path, default=None,
        help="Stage-0 alignment_manifest.json used for strict geometry preflight",
    )
    p.add_argument(
        "--gate-mode", choices=["strict", "fallback", "legacy_debug", "off"], default="off",
        help="off skips the full-sequence pose-convention preflight (the default). "
        "fallback keeps low-coverage frames but still stops on global calibration/convention errors",
    )
    p.add_argument("--geometry-gate-report", type=Path, default=None)
    p.add_argument("--depth-confidence-dir", type=Path, default=None)
    p.add_argument("--depth-edge-invalid-dir", type=Path, default=None)
    p.add_argument("--dynamic-mask-dir", "--human-mask-dir", dest="dynamic_mask_dir", type=Path, default=None)
    p.add_argument(
        "--fusion-backend", choices=["single", "surfel"], default="single",
        help="single keeps the historical one-frame warp; surfel is causal multi-frame fusion",
    )
    p.add_argument("--history-size", type=int, default=8)
    p.add_argument("--voxel-size-m", type=float, default=0.005)
    p.add_argument("--confidence-threshold", type=float, default=0.2)
    p.add_argument("--max-cache-points", type=int, default=300_000, help="deterministic per-frame surfel sample cap")
    p.add_argument("--cache-checkpoint", type=Path, default=None)
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--resume-cache", action="store_true")
    p.add_argument(
        "--preview-video",
        action="store_true",
        help="Encode novel_rgb.mp4. Packed rgb_geometry is the formal RGB contract.",
    )
    p.add_argument("--gate-sample-stride", type=int, default=1)
    p.add_argument("--gate-max-frames", type=int, default=0, help="0 means all frames")
    p.add_argument(
        "--gate-max-points", type=int, default=50_000,
        help="deterministic cap per preflight frame after stride; 0 means no cap",
    )
    p.add_argument(
        "--allow-depth-resize", action="store_true",
        help="legacy compatibility: resize mismatched depth to RGB; calibrated inputs must already be registered",
    )
    p.add_argument("--world_up_axis", type=str, default="y", choices=["y", "z"])
    p.add_argument(
        "--target_pose_convention",
        type=str,
        default="sparsetrack_mujoco_opengl",
        choices=["sparsetrack_mujoco_opengl", "opencv"],
    )
    p.add_argument(
        "--st-extra-Rz180",
        action="store_true",
        default=False,
        help=(
            "Left-multiply ARKit R by 180° about Z after MuJoCo→ARKit. "
            "Opt in for legacy Stera mounts; current hands_zed poses already include the flip."
        ),
    )
    p.add_argument(
        "--no-st-extra-Rz180",
        action="store_false",
        dest="st_extra_Rz180",
        help="Use plain MuJoCo-to-EgoDex conversion (the default).",
    )
    p.add_argument("--splat-radius-px", type=int, default=1)
    return p.parse_args()


def _load_K_json(path: Path, out_w: int, out_h: int) -> np.ndarray:
    meta = json.loads(path.read_text())
    if isinstance(meta.get("model"), dict) and isinstance(meta["model"].get("d435"), dict):
        meta = meta["model"]["d435"]
    if "K" in meta:
        K = np.asarray(meta["K"], dtype=np.float64).reshape(3, 3)
        kw = int(meta.get("width", meta.get("native_width", out_w)))
        kh = int(meta.get("height", meta.get("native_height", out_h)))
    else:
        fx = float(meta["fx"])
        fy = float(meta["fy"])
        cx = float(meta.get("cx", out_w * 0.5))
        cy = float(meta.get("cy", out_h * 0.5))
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        kw = int(meta.get("width", out_w))
        kh = int(meta.get("height", out_h))
    if kw != out_w or kh != out_h:
        sx, sy = out_w / float(kw), out_h / float(kh)
        K = K.copy()
        K[0, 0] *= sx
        K[1, 1] *= sy
        # Match cv2.resize pixel-center coordinates.  In the formal pipeline
        # target dimensions equal the calibration dimensions, so this branch
        # is only a compatibility path and must not silently use cx*s.
        K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
        K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return K


def _target_native_size(path: Path) -> tuple[int | None, int | None]:
    meta = json.loads(path.read_text())
    if isinstance(meta.get("model"), dict) and isinstance(meta["model"].get("d435"), dict):
        meta = meta["model"]["d435"]
    width = meta.get("width", meta.get("native_width"))
    height = meta.get("height", meta.get("native_height"))
    return (
        int(width) if width is not None else None,
        int(height) if height is not None else None,
    )


def _resolve_video_index_mode(
    video_count: int,
    frame_ids: list[int],
    start: int,
    end: int,
) -> str:
    """Decide whether an RGB video is a full recording or a stage-local slice."""
    if video_count <= 0:
        raise ValueError("RGB video frame count must be known for indexed rendering")
    if end <= start:
        raise ValueError(f"invalid frame range [{start}, {end})")

    ids = [int(value) for value in frame_ids]
    if ids and len(ids) != end - start:
        raise ValueError(
            "RGB manifest frame count differs from the selected range: "
            f"ids={len(ids)} range={end - start}"
        )
    if ids and ids != list(range(start, end)):
        raise ValueError("RGB manifest frame_indices do not match the selected range")

    # A video with exactly one frame per declared manifest ID is a sliced,
    # stage-local artifact. Otherwise the selected global frame IDs must be
    # directly addressable in the original full recording.
    if ids and video_count == len(ids) and (start != 0 or end != video_count):
        return "local"
    if end <= video_count:
        return "global"
    if video_count >= end - start:
        return "local"
    raise ValueError(
        "RGB video cannot address the requested range: "
        f"video_count={video_count} range=[{start}, {end})"
    )


class PackedRgbReader:
    """Random-access packed RGB volume with the SequentialVideoReader interface."""

    def __init__(self, directory: Path) -> None:
        self.volume = open_volume(directory, color_space="rgb")

    def get_rgb(self, index: int) -> np.ndarray:
        return as_rgb(self.volume.frame_at(int(index), int(index)))

    def close(self) -> None:
        return None


class SequentialVideoReader:
    """Sequential RGB decode with a get_data fallback for rare backward seeks."""

    def __init__(self, reader) -> None:
        self.reader = reader
        self._iter = reader.iter_data()
        self._next = 0

    def get_rgb(self, index: int) -> np.ndarray:
        want = int(index)
        if want < self._next:
            return np.asarray(self.reader.get_data(want))[:, :, :3]
        skipped = 0
        while self._next < want:
            try:
                next(self._iter)
            except StopIteration:
                return np.asarray(self.reader.get_data(want))[:, :, :3]
            self._next += 1
            skipped += 1
            if skipped > 10_000:
                return np.asarray(self.reader.get_data(want))[:, :, :3]
        try:
            frame = np.asarray(next(self._iter))[:, :, :3]
        except StopIteration:
            return np.asarray(self.reader.get_data(want))[:, :, :3]
        self._next += 1
        return frame

    def close(self) -> None:
        self.reader.close()


def _nearest_indices(query: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Absolute-nearest timestamp map; exact ties select the earlier sample."""
    q = np.asarray(query, dtype=np.float64).reshape(-1)
    s = np.asarray(samples, dtype=np.float64).reshape(-1)
    if not len(s) or np.any(~np.isfinite(s)) or np.any(np.diff(s) < 0):
        raise ValueError("target timestamps must be finite, non-empty, and monotonic")
    right = np.clip(np.searchsorted(s, q, side="left"), 0, len(s) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(q - s[left]) <= np.abs(s[right] - q)
    return np.where(choose_left, left, right).astype(np.int64)


# From viewport_transform_batch_video.py (pyc restored): MuJoCo↔ARKit and OpenCV↔OpenGL.
_ARKIT_TO_MUJOCO = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)
_OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def _c2w_from_st(
    pos_w: np.ndarray,
    quat_wxyz: np.ndarray,
    *,
    extra_R_arkit: np.ndarray | None = None,
) -> np.ndarray:
    """SparseTrack MuJoCo/OpenGL camera → EgoDex ARKit OpenCV c2w.

    Matches ``mujoco_opengl_pose_to_arkit_opencv_c2w`` in viewport_transform_batch_video.
    Optional ``extra_R_arkit`` left-multiplies the rotation (e.g. 180° about Z) when the
    Stera/robot mount needs an additional flip vs EgoDex.
    """
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    R_gl = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    t_m = np.asarray(pos_w, dtype=np.float64).reshape(3)
    r_cv_mj = R_gl @ _OPENCV_TO_OPENGL
    p_inv = _ARKIT_TO_MUJOCO.T
    r_arkit = p_inv @ r_cv_mj
    t_arkit = p_inv @ t_m
    if extra_R_arkit is not None:
        r_arkit = np.asarray(extra_R_arkit, dtype=np.float64) @ r_arkit
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = r_arkit
    T[:3, 3] = t_arkit
    return T


def _relative_w2c(src_c2w: np.ndarray, tgt_c2w: np.ndarray) -> np.ndarray:
    """Target w2c expressed in source camera frame (Cache3D convention)."""
    src_w2c = np.linalg.inv(src_c2w)
    tgt_w2c = np.linalg.inv(tgt_c2w)
    # Points in source cam → world → target cam:
    # X_t = tgt_w2c @ src_c2w @ X_s = (tgt_w2c @ inv(src_w2c)) @ X_s
    return tgt_w2c @ src_c2w


def _target_c2w(
    pos: np.ndarray,
    quat_wxyz: np.ndarray,
    convention: str,
    extra_rz180: bool = False,
) -> np.ndarray:
    """Build a target camera c2w using the explicitly selected convention."""
    extra = R.from_euler("z", 180.0, degrees=True).as_matrix() if extra_rz180 else None
    if convention == "sparsetrack_mujoco_opengl":
        return _c2w_from_st(pos, quat_wxyz, extra_R_arkit=extra)
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    if extra is not None:
        T[:3, :3] = extra @ T[:3, :3]
    return T


def _interpolate_target_pose(
    positions: np.ndarray,
    quats_wxyz: np.ndarray,
    timestamps_s: np.ndarray,
    query_s: float,
    convention: str,
    extra_rz180: bool = False,
) -> np.ndarray:
    """Interpolate target camera pose at a source timestamp (causal-safe).

    Position interpolation is linear; orientation uses shortest-path SLERP and
    clamps outside the recorded target range.  The pose convention conversion is
    applied only after interpolation, so a legacy Rz toggle cannot be hidden by
    the interpolation step.
    """
    ts = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    pos = np.asarray(positions, dtype=np.float64)
    quat = np.asarray(quats_wxyz, dtype=np.float64)
    if len(ts) < 1 or len(pos) != len(ts) or len(quat) != len(ts):
        raise ValueError("target pose timestamps and arrays must have equal non-zero length")
    if len(ts) == 1 or query_s <= ts[0]:
        return _target_c2w(pos[0], quat[0], convention, extra_rz180)
    if query_s >= ts[-1]:
        return _target_c2w(pos[-1], quat[-1], convention, extra_rz180)
    hi = int(np.searchsorted(ts, query_s, side="right"))
    hi = min(max(1, hi), len(ts) - 1)
    lo = hi - 1
    alpha = float((query_s - ts[lo]) / max(ts[hi] - ts[lo], 1e-12))
    p = (1.0 - alpha) * pos[lo] + alpha * pos[hi]
    rots = R.from_quat(np.asarray(quat[:, [1, 2, 3, 0]], dtype=np.float64))
    interp = Slerp(np.asarray(ts, dtype=np.float64), rots)([float(query_s)])[0]
    q_xyzw = interp.as_quat()
    q_wxyz = np.asarray([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
    return _target_c2w(p, q_wxyz, convention, extra_rz180)


def _read_target_timestamps(kin: np.lib.npyio.NpzFile, n_poses: int) -> tuple[np.ndarray | None, str | None]:
    """Read a target-camera timeline without guessing its epoch."""
    for key in (
        "camera_timestamps_s", "camera_timestamps", "timestamps_s",
        "timestamps", "time_s", "t_sec",
    ):
        if key not in kin:
            continue
        candidate = np.asarray(kin[key], dtype=np.float64).reshape(-1)
        if len(candidate) == n_poses and np.all(np.isfinite(candidate)):
            if len(candidate) > 1 and np.any(np.diff(candidate) < 0):
                continue
            return candidate, key
    return None, None


def _align_target_timestamps(
    raw_timestamps_s: np.ndarray | None,
    timestamp_key: str | None,
    *,
    source_anchor_s: float | None,
    target_anchor_index: int,
) -> tuple[np.ndarray | None, dict[str, object]]:
    """Map target-relative ``t_sec`` into the source recording time domain.

    SparseTrack writes ``t_sec`` relative to its first tracked frame.  Source
    RGB/depth timestamps are typically recording-epoch seconds, so comparing
    them directly would clamp every interpolation to a target endpoint.  The
    mapping is anchored to the exact frame pair used by this render and written
    to the manifest for auditability.
    """
    if raw_timestamps_s is None or timestamp_key is None:
        return None, {
            "available": False,
            "reason": "target_timestamp_missing",
        }
    raw = np.asarray(raw_timestamps_s, dtype=np.float64).reshape(-1)
    if timestamp_key != "t_sec":
        return raw, {
            "available": True,
            "timestamp_key": timestamp_key,
            "mapping": "absolute_or_shared_epoch",
            "offset_s": 0.0,
        }
    idx = int(target_anchor_index)
    if source_anchor_s is None or not np.isfinite(source_anchor_s):
        return None, {
            "available": False,
            "timestamp_key": timestamp_key,
            "reason": "source_timestamp_anchor_missing",
        }
    if idx < 0 or idx >= len(raw):
        return None, {
            "available": False,
            "timestamp_key": timestamp_key,
            "reason": "target_timestamp_anchor_out_of_range",
            "target_anchor_index": idx,
        }
    offset = float(source_anchor_s - raw[idx])
    return raw + offset, {
        "available": True,
        "timestamp_key": timestamp_key,
        "mapping": "relative_t_sec_anchored_to_source_rgb_frame",
        "source_anchor_s": float(source_anchor_s),
        "target_anchor_index": idx,
        "target_anchor_t_sec": float(raw[idx]),
        "offset_s": offset,
    }


def _indexed_files(directory: Path, suffix: str = ".npy") -> dict[int, Path]:
    """Return numeric frame files without relying on lexicographic gaps."""
    result: dict[int, Path] = {}
    if directory is None or not directory.is_dir():
        return result
    for path in directory.glob(f"*{suffix}"):
        try:
            result[int(path.stem)] = path
        except ValueError:
            continue
    return result


def _frame_file(mapping: dict[int, Path], global_index: int, local_index: int) -> Path | None:
    """Resolve either global or stage-local frame numbering."""
    return mapping.get(global_index) or mapping.get(local_index)


def _load_mask(path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None or not path.is_file():
        return np.zeros(shape, dtype=bool)
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"failed to read mask {path}")
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 127


def _load_float_map(path: Path | None, shape: tuple[int, int], default: float = 1.0) -> np.ndarray:
    if path is None or not path.is_file():
        return np.full(shape, default, dtype=np.float32)
    if path.suffix.lower() == ".npy":
        a = np.asarray(np.load(path), dtype=np.float32)
    else:
        a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if a is None:
            raise RuntimeError(f"failed to read map {path}")
        a = np.asarray(a, dtype=np.float32)
        if a.max(initial=0) > 1.5:
            a /= 255.0
    if a.shape != shape:
        a = cv2.resize(a, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.nan_to_num(a, nan=0.0).astype(np.float32)


def _gate_frame_metrics(
    depth: np.ndarray,
    src_c2w: np.ndarray,
    tgt_c2w: np.ndarray,
    src_K: np.ndarray,
    tgt_K: np.ndarray,
    image_hw: tuple[int, int],
    *,
    target_image_hw: tuple[int, int] | None = None,
    confidence: np.ndarray | None = None,
    edge_invalid: np.ndarray | None = None,
    dynamic_mask: np.ndarray | None = None,
    sample_stride: int = 8,
    max_points: int = 50_000,
    frame_index: int = 0,
    timing_offset_ms: float | None = None,
) -> dict[str, object]:
    """Compute target-camera preflight metrics from a sampled source depth map."""
    H, W = image_hw
    target_H, target_W = target_image_hw or image_hw
    d = np.asarray(depth, dtype=np.float32)
    if d.shape != (H, W):
        return {
            "frame": frame_index, "valid_ratio": 0.0, "front_ratio": 0.0,
            "in_bounds_ratio": 0.0, "bad": True, "depth_resolution_mismatch": True,
            "timing_offset_ms": timing_offset_ms,
        }
    conf = np.ones_like(d, dtype=np.float32) if confidence is None else np.asarray(confidence, dtype=np.float32)
    edge = np.zeros_like(d, dtype=bool) if edge_invalid is None else np.asarray(edge_invalid, dtype=bool)
    dyn = np.zeros_like(d, dtype=bool) if dynamic_mask is None else np.asarray(dynamic_mask, dtype=bool)
    valid = np.isfinite(d) & (d > 1e-4) & (d < 100.0) & (conf >= 0.2) & ~edge & ~dyn
    denom = max(1, int((~dyn).sum()))
    ys, xs = np.nonzero(valid)
    stride = max(1, int(sample_stride))
    if len(xs) > 0:
        take = np.arange(0, len(xs), stride, dtype=np.int64)
        ys, xs = ys[take], xs[take]
    uncapped_n = len(xs)
    cap = max(0, int(max_points))
    if cap and uncapped_n > cap:
        # Keep spatial coverage without random state. ``//`` produces unique,
        # monotonically increasing samples whenever cap < uncapped_n, and the
        # selection is convention-independent so selected/Rz diagnostics compare
        # precisely the same source pixels.
        take = (np.arange(cap, dtype=np.int64) * uncapped_n) // cap
        ys, xs = ys[take], xs[take]
    n = len(xs)
    row: dict[str, object] = {
        "frame": int(frame_index),
        "valid_ratio": float(valid.sum() / denom),
        "front_ratio": 0.0,
        "in_bounds_ratio": 0.0,
        "timing_offset_ms": timing_offset_ms,
        "gate_sample_count": int(n),
        "gate_uncapped_sample_count": int(uncapped_n),
        "gate_max_points": int(cap),
        "bad": False,
    }
    if n == 0:
        row["bad"] = True
        return row
    z = d[ys, xs].astype(np.float64)
    pix = np.stack([xs.astype(np.float64), ys.astype(np.float64), np.ones(n)], axis=1)
    rays = pix @ np.linalg.inv(src_K).T
    points_s = rays * z[:, None]
    points_h = np.concatenate([points_s, np.ones((n, 1))], axis=1)
    points_t = (points_h @ _relative_w2c(src_c2w, tgt_c2w).T)[:, :3]
    tz = points_t[:, 2]
    front = np.isfinite(tz) & (tz > 1e-6)
    row["front_ratio"] = float(front.mean())
    if not front.any():
        row["bad"] = True
        return row
    q = points_t[front] @ tgt_K.T
    uv = q[:, :2] / np.maximum(q[:, 2:3], 1e-9)
    inside = (
        (uv[:, 0] >= 0) & (uv[:, 0] < target_W) &
        (uv[:, 1] >= 0) & (uv[:, 1] < target_H)
    )
    row["in_bounds_ratio"] = float(inside.mean())
    # Duplicate target pixels provide a useful local depth consistency check.
    if inside.any():
        ui = np.clip(np.rint(uv[inside, 0]), 0, target_W - 1).astype(np.int64)
        vi = np.clip(np.rint(uv[inside, 1]), 0, target_H - 1).astype(np.int64)
        zz = tz[front][inside]
        key = vi * target_W + ui
        order = np.argsort(key)
        key_s, z_s = key[order], zz[order]
        splits = np.r_[0, np.flatnonzero(key_s[1:] != key_s[:-1]) + 1, len(key_s)]
        rel_err = []
        occluded_points = 0
        visible_points = 0
        for a, b in zip(splits[:-1], splits[1:]):
            if b - a > 1:
                group = z_s[a:b]
                mn = max(float(np.min(group)), 1e-6)
                # A target pixel may legitimately receive points from a front
                # and a rear surface. Treat points farther than the configured
                # 15% overlap tolerance as occluded, rather than as calibration
                # error; only repeated samples of the visible surface contribute
                # to the consistency statistic.
                visible = group <= mn * 1.15
                visible_points += int(np.count_nonzero(visible))
                occluded_points += int(np.count_nonzero(~visible))
                if np.count_nonzero(visible) > 1:
                    visible_group = group[visible]
                    rel_err.append(float((np.max(visible_group) - mn) / mn))
        if rel_err:
            row["depth_relative_error"] = float(np.percentile(rel_err, 95))
        if visible_points or occluded_points:
            row["occluded_duplicate_ratio"] = float(
                occluded_points / max(1, visible_points + occluded_points)
            )
    # Round-trip source reprojection is a numerical sanity check; non-zero values
    # expose malformed pose matrices without assuming a target RGB correspondence.
    back = np.concatenate([points_t, np.ones((n, 1))], axis=1) @ np.linalg.inv(_relative_w2c(src_c2w, tgt_c2w)).T
    back = back[:, :3]
    q_back = back @ src_K.T
    uv_back = q_back[:, :2] / np.maximum(q_back[:, 2:3], 1e-9)
    err = np.linalg.norm(uv_back - pix[:, :2], axis=1)
    err = err[np.isfinite(err)]
    if err.size:
        row["reprojection_error_px"] = float(np.percentile(err, 50))
        row["reprojection_error_p50_px"] = float(np.percentile(err, 50))
        row["reprojection_error_p95_px"] = float(np.percentile(err, 95))
    row["bad"] = bool(
        row["valid_ratio"] < 0.30 or row["front_ratio"] < 0.85 or row["in_bounds_ratio"] < 0.30
    )
    return row


def _read_manifest(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(f"invalid alignment manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"alignment manifest must be an object: {path}")
    return value


def warp_frame(
    image_rgb: np.ndarray,
    depth_m: np.ndarray,
    src_K: np.ndarray,
    rel_w2c: np.ndarray,
    tgt_K: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns warped_rgb uint8, warped_depth float32, valid_mask uint8(0/255)."""
    if Cache3D_Buffer is None:
        raise RuntimeError(
            "single-frame fusion requires the Cache3D/NVIDIA Warp dependency"
        ) from _CACHE3D_IMPORT_ERROR
    H, W = image_rgb.shape[:2]
    depth = torch.as_tensor(depth_m, dtype=torch.float32, device=device)
    depth = torch.nan_to_num(depth, nan=0.0)
    # Invalid source depth → far so they don't splat usefully; mask will catch holes
    invalid = depth <= 1e-4
    depth = depth.clone()
    depth[invalid] = 1e4

    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    image_bchw = (image_tensor.unsqueeze(0) * 2 - 1).to(device)
    depth_b1hw = depth.unsqueeze(0).unsqueeze(0)
    # Zero-out invalid in a mask channel via depth clamp already; also pass input mask
    src_mask = (~invalid).float().unsqueeze(0).unsqueeze(0).to(device)

    K_s = torch.as_tensor(src_K, dtype=torch.float32, device=device).unsqueeze(0)
    K_t = torch.as_tensor(tgt_K, dtype=torch.float32, device=device).view(1, 1, 3, 3)
    w2c0 = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
    rel = torch.as_tensor(rel_w2c, dtype=torch.float32, device=device).view(1, 1, 4, 4)

    gen = torch.Generator(device=device).manual_seed(42)
    cache = Cache3D_Buffer(
        frame_buffer_max=2,
        generator=gen,
        noise_aug_strength=0.0,
        input_image=image_bchw,
        input_depth=depth_b1hw,
        input_w2c=w2c0,
        input_intrinsics=K_s,
        input_mask=src_mask,
        filter_points_threshold=0.05,
        foreground_masking=False,
        device=device,
    )
    rendered, masks = cache.render_cache(rel, K_t, render_depth=False)
    warped = rendered[0, 0, 0]  # C,H,W in [-1,1]
    mask = masks[0, 0, 0]
    if mask.dim() == 3:
        mask = mask[0]
    rgb = ((warped.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    valid = (mask.detach().cpu().numpy() > 0.5)

    # Second pass for depth
    depth_img, masks2 = cache.render_cache(rel, K_t, render_depth=True)
    # When render_depth=True, Cache3D_Base replaces pixels with depth map
    dmap = depth_img[0, 0, 0]
    if dmap.dim() == 3:
        # unexpected channel dim
        dmap = dmap[0]
    d_np = dmap.detach().cpu().numpy().astype(np.float32)
    valid2 = masks2[0, 0, 0]
    if valid2.dim() == 3:
        valid2 = valid2[0]
    valid2 = valid2.detach().cpu().numpy() > 0.5
    valid = valid & valid2 & np.isfinite(d_np) & (d_np > 1e-4) & (d_np < 1e3)
    d_out = np.where(valid, d_np, 0.0).astype(np.float32)
    rgb = np.where(valid[..., None], rgb, 0).astype(np.uint8)
    return rgb, d_out, (valid.astype(np.uint8) * 255)


def warp_frames_to_target(
    image_rgb: np.ndarray,
    depth_m: np.ndarray,
    src_K: np.ndarray,
    rel_w2c: np.ndarray,
    tgt_K: np.ndarray,
    target_hw: tuple[int, int],
    device: str,
    *,
    confidence: np.ndarray | None = None,
    splat_radius_px: int = 1,
    src_K_t: torch.Tensor | None = None,
    tgt_K_t: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project a stack of source RGB-D frames into independently sized targets.

    ``rel_w2c`` may differ per frame. The z-buffer is one ``scatter_reduce`` over
    ``N * Ht * Wt`` so a chunk is one H2D / D2H round-trip. Splat offsets stay
    sequential so a 1080p source does not materialize a 9x point cloud.
    """
    rgb_np = np.ascontiguousarray(image_rgb[..., :3], dtype=np.uint8)
    squeeze = rgb_np.ndim == 3
    if squeeze:
        rgb_np = rgb_np[None]
    if rgb_np.ndim != 4:
        raise ValueError(f"RGB must be (H,W,3) or (N,H,W,3); got {rgb_np.shape}")
    n, hs, ws = (int(rgb_np.shape[0]), int(rgb_np.shape[1]), int(rgb_np.shape[2]))
    depth_np = np.ascontiguousarray(depth_m, dtype=np.float32)
    if depth_np.ndim == 2:
        depth_np = depth_np[None]
    if depth_np.shape != (n, hs, ws):
        raise ValueError(f"RGB/depth shape mismatch: {rgb_np.shape[:3]} vs {depth_np.shape}")
    if confidence is None:
        confidence_np = np.ones((n, hs, ws), dtype=np.float32)
    else:
        confidence_np = np.ascontiguousarray(confidence, dtype=np.float32)
        if confidence_np.ndim == 2:
            confidence_np = confidence_np[None]
        if confidence_np.shape != (n, hs, ws):
            raise ValueError("confidence must match source depth")
    rel = np.ascontiguousarray(rel_w2c, dtype=np.float32)
    if rel.ndim == 2:
        rel = rel[None]
    if rel.shape != (n, 4, 4):
        raise ValueError(f"rel_w2c must be (4,4) or (N,4,4); got {rel.shape}")

    ht, wt = map(int, target_hw)
    if ht <= 0 or wt <= 0:
        raise ValueError(f"invalid target size {target_hw}")
    empty = (
        np.zeros((n, ht, wt, 3), np.uint8),
        np.zeros((n, ht, wt), np.float32),
        np.zeros((n, ht, wt), np.uint8),
        np.zeros((n, ht, wt), np.float32),
    )

    def _squeeze(
        rgb_o: np.ndarray, depth_o: np.ndarray, valid_o: np.ndarray, conf_o: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if squeeze:
            return rgb_o[0], depth_o[0], valid_o[0], conf_o[0]
        return rgb_o, depth_o, valid_o, conf_o

    torch_device = torch.device(device)
    depth = torch.as_tensor(depth_np, device=torch_device)
    conf = torch.as_tensor(confidence_np, device=torch_device)
    source_valid = torch.isfinite(depth) & (depth > 1e-4) & (depth < 100.0) & (conf > 0.0)
    if not bool(source_valid.any()):
        return _squeeze(*empty)

    yy = torch.arange(hs, device=torch_device)
    xx = torch.arange(ws, device=torch_device)
    z_src = depth
    k_s = src_K_t if src_K_t is not None else torch.as_tensor(src_K, dtype=torch.float32, device=torch_device)
    x_src = (xx.to(torch.float32) - k_s[0, 2]) * z_src / k_s[0, 0]
    y_src = (yy.to(torch.float32).view(1, hs, 1) - k_s[1, 2]) * z_src / k_s[1, 1]
    points = torch.stack((x_src, y_src.expand_as(z_src), z_src), dim=-1)
    transform = torch.as_tensor(rel, dtype=torch.float32, device=torch_device)
    points_t = torch.einsum("nhwj,nij->nhwi", points, transform[:, :3, :3]) + transform[:, None, None, :3, 3]
    z = points_t[..., 2]
    keep = source_valid & torch.isfinite(points_t).all(dim=-1) & (z > 1e-4) & (z < 100.0)
    if not bool(keep.any()):
        return _squeeze(*empty)

    k_t = tgt_K_t if tgt_K_t is not None else torch.as_tensor(tgt_K, dtype=torch.float32, device=torch_device)
    projected = points_t @ k_t.T
    u0 = torch.round(projected[..., 0] / projected[..., 2]).to(torch.int64)
    v0 = torch.round(projected[..., 1] / projected[..., 2]).to(torch.int64)
    frame_id = torch.arange(n, device=torch_device).view(n, 1, 1).expand(n, hs, ws)
    ys = yy.view(1, hs, 1).expand(n, hs, ws)
    xs = xx.view(1, 1, ws).expand(n, hs, ws)
    source_flat = frame_id * (hs * ws) + ys * ws + xs
    keep_i = keep.reshape(-1)
    u0 = u0.reshape(-1)[keep_i]
    v0 = v0.reshape(-1)[keep_i]
    z = z.reshape(-1)[keep_i]
    source_flat = source_flat.reshape(-1)[keep_i]
    frame_id = frame_id.reshape(-1)[keep_i]
    if z.numel() == 0:
        return _squeeze(*empty)

    radius = max(0, int(splat_radius_px))
    offsets = [
        (dx, dy, max(0.2, 1.0 - 0.2 * max(abs(dx), abs(dy)) / max(float(radius), 1.0)))
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
    ]
    hw_t = ht * wt
    base = frame_id * hw_t
    z_buffer = torch.full((n * hw_t,), torch.inf, dtype=torch.float32, device=torch_device)
    for dx, dy, _ in offsets:
        u, v = u0 + dx, v0 + dy
        inside = (u >= 0) & (u < wt) & (v >= 0) & (v < ht)
        if bool(inside.any()):
            pix = base[inside] + v[inside] * wt + u[inside]
            z_buffer.scatter_reduce_(0, pix, z[inside], reduce="amin", include_self=True)

    sentinel = int(n * hs * ws * max(1, len(offsets)) + 1)
    chosen = torch.full((n * hw_t,), sentinel, dtype=torch.int64, device=torch_device)
    tolerance_scale = 1e-4
    for offset_id, (dx, dy, _) in enumerate(offsets):
        u, v = u0 + dx, v0 + dy
        inside = (u >= 0) & (u < wt) & (v >= 0) & (v < ht)
        if not bool(inside.any()):
            continue
        pix = base[inside] + v[inside] * wt + u[inside]
        z_here = z[inside]
        winner = torch.abs(z_here - z_buffer[pix]) <= torch.maximum(
            torch.full_like(z_here, 1e-5), tolerance_scale * z_buffer[pix]
        )
        if bool(winner.any()):
            encoded = source_flat[inside][winner] * len(offsets) + offset_id
            chosen.scatter_reduce_(0, pix[winner], encoded, reduce="amin", include_self=True)

    valid_flat = chosen < sentinel
    output_rgb = torch.zeros((n * hw_t, 3), dtype=torch.uint8, device=torch_device)
    output_conf = torch.zeros((n * hw_t,), dtype=torch.float32, device=torch_device)
    if bool(valid_flat.any()):
        encoded = chosen[valid_flat]
        source_index = torch.div(encoded, len(offsets), rounding_mode="floor")
        offset_index = torch.remainder(encoded, len(offsets))
        colors = torch.as_tensor(rgb_np.reshape(-1, 3), dtype=torch.uint8, device=torch_device)
        conf_flat = torch.as_tensor(confidence_np.reshape(-1), device=torch_device)
        falloff = torch.as_tensor([item[2] for item in offsets], dtype=torch.float32, device=torch_device)
        output_rgb[valid_flat] = colors[source_index]
        output_conf[valid_flat] = conf_flat[source_index] * falloff[offset_index]

    rgb_out = output_rgb.reshape(n, ht, wt, 3).cpu().numpy()
    depth_out = torch.where(torch.isfinite(z_buffer), z_buffer, 0.0).reshape(n, ht, wt).cpu().numpy().astype(np.float32)
    valid_out = (valid_flat.reshape(n, ht, wt).cpu().numpy().astype(np.uint8) * 255)
    conf_out = output_conf.reshape(n, ht, wt).cpu().numpy().astype(np.float32)
    return _squeeze(rgb_out, depth_out, valid_out, conf_out)


def warp_frame_to_target(
    image_rgb: np.ndarray,
    depth_m: np.ndarray,
    src_K: np.ndarray,
    rel_w2c: np.ndarray,
    tgt_K: np.ndarray,
    target_hw: tuple[int, int],
    device: str,
    *,
    confidence: np.ndarray | None = None,
    splat_radius_px: int = 1,
    src_K_t: torch.Tensor | None = None,
    tgt_K_t: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project one source RGB-D frame into an independently sized target image."""
    return warp_frames_to_target(
        image_rgb,
        depth_m,
        src_K,
        rel_w2c,
        tgt_K,
        target_hw,
        device,
        confidence=confidence,
        splat_radius_px=splat_radius_px,
        src_K_t=src_K_t,
        tgt_K_t=tgt_K_t,
    )


def main() -> None:
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    manifest_path = args.alignment_manifest
    if manifest_path is None:
        candidate = args.h5_file.parent / "alignment_manifest.json"
        manifest_path = candidate if candidate.is_file() else None
    manifest = _read_manifest(manifest_path)

    # Probe RGB size from packed OpenCV volume when present; otherwise one mp4 frame.
    packed_source = (
        args.source_rgb_dir
        if args.source_rgb_dir is not None and is_packed_stream(args.source_rgb_dir)
        else None
    )
    if packed_source is not None:
        packed_probe = open_volume(packed_source, color_space="rgb")
        H0, W0 = (int(value) for value in packed_probe.frame_shape[:2])
        video_count = len(packed_probe)
        fps = float(args.fps or 30.0)
        meta_file = packed_source / "volume_meta.json"
        if meta_file.is_file():
            try:
                fps = float(json.loads(meta_file.read_text(encoding="utf-8")).get("fps") or fps)
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
            if args.fps:
                fps = float(args.fps)
    else:
        probe_reader = imageio.get_reader(str(args.video_file))
        video_meta = probe_reader.get_meta_data()
        try:
            video_count = int(probe_reader.count_frames())
        except Exception:
            video_count = 0
        try:
            first_rgb = np.asarray(probe_reader.get_data(0))[:, :, :3]
        except Exception as exc:
            probe_reader.close()
            raise RuntimeError(f"failed to read first RGB frame: {args.video_file}") from exc
        probe_reader.close()
        H0, W0 = first_rgb.shape[:2]
        fps = float(args.fps or video_meta.get("fps") or 30.0)

    with h5py.File(args.h5_file, "r") as h5:
        src_K = np.asarray(h5["camera/intrinsic"], dtype=np.float64).reshape(3, 3)
        src_c2w = np.asarray(h5["transforms/camera"], dtype=np.float64)

    kin = np.load(args.target_poses_npz)
    cam_pos = np.asarray(kin["camera_pos_w"], dtype=np.float64)
    cam_quat = np.asarray(kin["camera_quat_wxyz"], dtype=np.float64)
    if len(cam_pos) != len(cam_quat):
        raise ValueError("camera_pos_w and camera_quat_wxyz must have equal length")
    target_ts_raw, target_ts_key = _read_target_timestamps(kin, len(cam_pos))
    target_frame_ids = (
        np.asarray(kin["egodex_frame"], dtype=np.int64).reshape(-1)
        if "egodex_frame" in kin else np.empty(0, dtype=np.int64)
    )
    if len(target_frame_ids) not in (0, len(cam_pos)):
        raise ValueError("egodex_frame and target camera pose arrays must have equal length")
    target_lookup = (
        {int(frame_id): index for index, frame_id in enumerate(target_frame_ids)}
        if len(target_frame_ids) else {}
    )
    depth_vol = open_volume(args.gt_depth_dir)
    conf_vol = open_volume(args.depth_confidence_dir) if args.depth_confidence_dir else None
    edge_vol = open_volume(args.depth_edge_invalid_dir) if args.depth_edge_invalid_dir else None
    dynamic_vol = open_volume(args.dynamic_mask_dir) if args.dynamic_mask_dir else None

    manifest_frame_start = int((manifest or {}).get("frame_range", {}).get("start", 0))
    camera_ids = (
        (manifest or {}).get("streams", {}).get("camera_pose", {}).get("frame_indices", [])
        if isinstance((manifest or {}).get("streams", {}), dict)
        else []
    )
    rgb_ids = (
        (manifest or {}).get("streams", {}).get("rgb", {}).get("frame_indices", [])
        if isinstance((manifest or {}).get("streams", {}), dict)
        else []
    )
    camera_lookup = {int(fid): j for j, fid in enumerate(camera_ids)} if camera_ids else {}
    source_pose_n = len(src_c2w)
    target_pose_n = len(cam_pos)
    # Adapter outputs may be stage-local (camera.h5 has only the requested
    # slice) while target kinematics retain global frame numbering.  Use the
    # manifest IDs to recover the global range in that case.
    if camera_lookup and int(args.start_frame) == 0:
        start = min(camera_lookup)
        available_n = min(video_count if video_count > 0 else len(camera_lookup), len(camera_lookup))
    else:
        start = max(0, int(args.start_frame))
        available_n = min(video_count if video_count > 0 else source_pose_n, source_pose_n)
    end = (start + available_n) if args.max_frames is None else min(start + available_n, start + int(args.max_frames))
    if end <= start:
        raise SystemExit(f"empty frame range [{start},{end})")
    video_index_mode = _resolve_video_index_mode(
        video_count,
        [int(value) for value in rgb_ids] if rgb_ids else list(range(start, end)),
        start,
        end,
    )
    native_target_w, native_target_h = _target_native_size(args.target_intrinsics_json)
    target_W = int(args.target_width or native_target_w or W0)
    target_H = int(args.target_height or native_target_h or H0)
    tgt_K = _load_K_json(args.target_intrinsics_json, target_W, target_H)

    def source_pose_index(global_i: int, local_i: int) -> int:
        if global_i in camera_lookup:
            return camera_lookup[global_i]
        return global_i if global_i < source_pose_n else local_i

    target_index_by_frame: dict[int, int] = {}

    def target_pose_index(global_i: int, local_i: int) -> int:
        if global_i in target_index_by_frame:
            return target_index_by_frame[global_i]
        if global_i in target_lookup:
            return target_lookup[global_i]
        if global_i < target_pose_n:
            return global_i
        if local_i < target_pose_n:
            return local_i
        raise IndexError(
            f"target pose unavailable for global/local frame {global_i}/{local_i} "
            f"(n={target_pose_n})"
        )

    def load_depth(global_i: int, local_i: int) -> tuple[np.ndarray, bool]:
        d = np.asarray(depth_vol.frame_at(global_i, local_i), dtype=np.float32)
        resized = d.shape != (H0, W0)
        if resized:
            if not args.allow_depth_resize and args.gate_mode == "strict":
                return d, True
            d = cv2.resize(d, (W0, H0), interpolation=cv2.INTER_NEAREST)
        return d, resized

    def load_conf(global_i: int, local_i: int) -> np.ndarray:
        if conf_vol is None:
            return np.ones((H0, W0), np.float32)
        a = np.asarray(conf_vol.frame_at(global_i, local_i), dtype=np.float32)
        if a.shape != (H0, W0):
            a = cv2.resize(a, (W0, H0), interpolation=cv2.INTER_NEAREST)
        return np.nan_to_num(a, nan=0.0).astype(np.float32)

    def load_edge(global_i: int, local_i: int) -> np.ndarray:
        if edge_vol is None:
            return np.zeros((H0, W0), dtype=bool)
        return as_mask(edge_vol.frame_at(global_i, local_i))

    def load_dynamic(global_i: int, local_i: int) -> np.ndarray:
        if dynamic_vol is None:
            return np.zeros((H0, W0), dtype=bool)
        mask = as_mask(dynamic_vol.frame_at(global_i, local_i))
        if mask.shape != (H0, W0):
            mask = cv2.resize(mask.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST) > 0
        return mask

    # Timing offsets come from the adapter's explicit map, never from an assumed
    # zero offset.  Missing values remain unavailable and fail strict mode.
    timing_by_frame: dict[int, float] = {}
    rgb_ts_by_frame: dict[int, float] = {}
    if isinstance(manifest, dict):
        for row in manifest.get("rgb_depth_frame_map", []):
            try:
                timing_by_frame[int(row["rgb_index"])] = float(row["time_offset_ms"])
            except (KeyError, TypeError, ValueError):
                continue
        rgb_stream = manifest.get("streams", {}).get("rgb", {})
        if isinstance(rgb_stream, dict):
            ids = rgb_stream.get("frame_indices", [])
            ts_values = rgb_stream.get("timestamps_s", [])
            for fid, ts in zip(ids, ts_values):
                try:
                    rgb_ts_by_frame[int(fid)] = float(ts)
                except (TypeError, ValueError):
                    pass

    selected_global_indices = list(range(start, end))
    target_ts = target_ts_raw
    source_time_origin_s: float | None = None
    source_query_times: np.ndarray | None = None
    signed_target_offsets_s: np.ndarray | None = None
    if args.target_time_match == "nearest":
        if target_ts_raw is None:
            raise SystemExit("nearest target matching requires target timestamps")
        missing_source_times = [i for i in selected_global_indices if i not in rgb_ts_by_frame]
        if missing_source_times:
            raise SystemExit(
                "nearest target matching requires measured source timestamps; missing "
                + ", ".join(map(str, missing_source_times[:8]))
            )
        all_source_times = np.asarray(list(rgb_ts_by_frame.values()), dtype=np.float64)
        source_time_origin_s = float(np.min(all_source_times))
        source_query_times = np.asarray(
            [rgb_ts_by_frame[i] for i in selected_global_indices], dtype=np.float64
        )
        if args.target_time_domain == "relative":
            source_query_times = source_query_times - source_time_origin_s
            target_match_times = np.asarray(target_ts_raw, dtype=np.float64)
        else:
            target_match_times = np.asarray(target_ts_raw, dtype=np.float64)
        matched = _nearest_indices(source_query_times, target_match_times)
        target_index_by_frame.update(
            {global_i: int(target_i) for global_i, target_i in zip(selected_global_indices, matched)}
        )
        signed_target_offsets_s = target_match_times[matched] - source_query_times
        target_time_alignment = {
            "available": True,
            "mapping": "source_to_absolute_nearest_target",
            "source_time_domain": args.target_time_domain,
            "source_time_origin_s": source_time_origin_s,
            "tie_break": "earlier_target",
        }
    elif args.target_time_match == "interpolate":
        target_anchor_index = target_pose_index(start, 0)
        target_ts, target_time_alignment = _align_target_timestamps(
            target_ts_raw,
            target_ts_key,
            source_anchor_s=rgb_ts_by_frame.get(start),
            target_anchor_index=target_anchor_index,
        )
    else:
        target_time_alignment = {
            "available": target_ts_raw is not None,
            "mapping": "index_aligned",
        }

    def target_pose_for(global_i: int, local_i: int) -> np.ndarray:
        if args.target_time_match == "interpolate" and target_ts is not None and global_i in rgb_ts_by_frame:
            return _interpolate_target_pose(
                cam_pos, cam_quat, target_ts, rgb_ts_by_frame[global_i],
                args.target_pose_convention, bool(args.st_extra_Rz180),
            )
        ti = target_pose_index(global_i, local_i)
        return _target_c2w(
            cam_pos[ti], cam_quat[ti], args.target_pose_convention, bool(args.st_extra_Rz180)
        )

    def gate_rows(extra_rz: bool) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        stride = max(1, int(args.gate_sample_stride))
        gate_indices = list(range(start, end, stride))
        if args.gate_max_frames and len(gate_indices) > args.gate_max_frames:
            gate_indices = gate_indices[: int(args.gate_max_frames)]
        for global_i in gate_indices:
            local_i = global_i - start
            d, resized = load_depth(global_i, local_i)
            conf = load_conf(global_i, local_i)
            edge = load_edge(global_i, local_i)
            dyn = load_dynamic(global_i, local_i)
            si = source_pose_index(global_i, local_i)
            tgt = (
                _interpolate_target_pose(
                    cam_pos, cam_quat, target_ts, rgb_ts_by_frame[global_i],
                    args.target_pose_convention, extra_rz,
                )
                if args.target_time_match == "interpolate" and target_ts is not None and global_i in rgb_ts_by_frame
                else _target_c2w(
                    cam_pos[target_pose_index(global_i, local_i)],
                    cam_quat[target_pose_index(global_i, local_i)],
                    args.target_pose_convention, extra_rz,
                )
            )
            row = _gate_frame_metrics(
                d, src_c2w[si], tgt, src_K, tgt_K, (H0, W0),
                target_image_hw=(target_H, target_W), confidence=conf,
                edge_invalid=edge, dynamic_mask=dyn, sample_stride=1,
                max_points=args.gate_max_points,
                frame_index=global_i, timing_offset_ms=timing_by_frame.get(global_i),
            )
            row["depth_resolution_mismatch"] = bool(resized)
            rows.append(row)
        return rows

    if args.gate_mode == "off":
        selected_rows = []
        alternate_rows = []
        print("[gt_warp] geometry gate skipped (gate-mode=off)", flush=True)
    else:
        selected_rows = gate_rows(bool(args.st_extra_Rz180))
        alternate_rows = gate_rows(not bool(args.st_extra_Rz180))
    # Materialize the unified alignment contract at the render root so the
    # target-camera timeline is recorded alongside Stage-0 RGB/depth/source
    # timestamps without mutating the original adapter artifact.
    render_manifest = dict(manifest) if isinstance(manifest, dict) else {"schema_version": 2, "mode": "legacy_debug"}
    render_streams = dict(render_manifest.get("streams", {}))
    target_indices = [target_pose_index(i, i - start) for i in range(start, end)]
    render_streams["target_camera_pose"] = {
        "source": str(args.target_poses_npz),
        "timestamps_s": (target_ts_raw[target_indices].tolist() if target_ts_raw is not None else None),
        "raw_timestamps_s": (target_ts_raw[target_indices].tolist() if target_ts_raw is not None else None),
        "timestamp_source": (
            f"kinematics_npz:{target_ts_key}" if target_ts is not None and target_ts_key is not None
            else "index_aligned_unavailable"
        ),
        "time_alignment": target_time_alignment,
        "frame_indices": target_indices,
        "pose_convention": args.target_pose_convention,
        "extra_Rz180": bool(args.st_extra_Rz180),
        "pose_direction": "camera_to_world",
        "external_pose": "scale_tracked_d435_camera_pose",
        "image_size": [target_W, target_H],
    }
    render_manifest["streams"] = render_streams
    render_manifest["render_alignment"] = {
        "source_manifest": str(manifest_path) if manifest_path is not None else None,
        "rgb_video_index_mode": video_index_mode,
        "rgb_video_frame_count": int(video_count),
        "target_pose_convention": args.target_pose_convention,
        "extra_Rz180": bool(args.st_extra_Rz180),
        "target_time_alignment": target_time_alignment,
        "source_to_target": [
            {
                "source_index": int(global_i),
                "rgb_video_index": int(global_i if video_index_mode == "global" else local_i),
                "source_timestamp_s": float(rgb_ts_by_frame[global_i]),
                "source_relative_timestamp_s": (
                    float(source_query_times[local_i]) if source_query_times is not None else None
                ),
                "target_index": int(target_indices[local_i]),
                "target_timestamp_s": (
                    float(target_ts_raw[target_indices[local_i]]) if target_ts_raw is not None else None
                ),
                "signed_offset_ms": (
                    float(signed_target_offsets_s[local_i] * 1000.0)
                    if signed_target_offsets_s is not None else None
                ),
            }
            for local_i, global_i in enumerate(selected_global_indices)
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "alignment_manifest.json").write_text(json.dumps(render_manifest, indent=2) + "\n")
    timeline_path = args.timeline_manifest or (args.output_dir / "timeline_manifest.json")
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    timeline_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_video": str(args.video_file),
                "rgb_video_index_mode": video_index_mode,
                "rgb_video_frame_count": int(video_count),
                "target_poses": str(args.target_poses_npz),
                "source_time_origin_s": source_time_origin_s,
                "target_time_domain": args.target_time_domain,
                "match": "absolute_nearest" if args.target_time_match == "nearest" else args.target_time_match,
                "tie_break": "earlier_target",
                "source_to_target": render_manifest["render_alignment"]["source_to_target"],
            },
            indent=2,
        )
        + "\n"
    )
    calibration_complete = bool(
        manifest and manifest.get("calibration_status") == "complete"
    )
    timing_available = bool(
        manifest and manifest.get("timing", {}).get("measured", False)
    )
    if args.gate_mode == "off":
        selected_gate = {
            "status": "bypassed", "mode": "off", "failed_checks": [],
            "frame_count": int(end - start), "thresholds": GeometryGateThresholds().__dict__,
        }
    else:
        selected_gate = evaluate_geometry_gate(
            selected_rows, calibration_complete=calibration_complete,
            timing_available=timing_available, mode=args.gate_mode,
            rz_alternate={
                "tested": True,
                "selected_extra_Rz180": bool(args.st_extra_Rz180),
                "selected": selected_rows,
                "alternate_extra_Rz180": not bool(args.st_extra_Rz180),
                "alternate": alternate_rows,
            },
        )
    gate_report = dict(selected_gate)
    gate_report["calibration_status"] = "complete" if calibration_complete else "incomplete_legacy_debug"
    gate_report["timestamp_status"] = "measured" if timing_available else "incomplete_legacy_debug"
    gate_report["selected_pose_convention"] = args.target_pose_convention
    gate_report["selected_extra_Rz180"] = bool(args.st_extra_Rz180)
    gate_report["sampling"] = {
        "frame_stride": max(1, int(args.gate_sample_stride)),
        "max_frames": int(args.gate_max_frames),
        "max_points": max(0, int(args.gate_max_points)),
        "point_selection": "evenly_spaced_deterministic_after_stride",
    }
    convention_meta = manifest.get("coordinate_convention", {}) if isinstance(manifest, dict) else {}
    expected_extra = convention_meta.get("target_extra_Rz180") if isinstance(convention_meta, dict) else None
    expected_convention = convention_meta.get("target_pose_convention") if isinstance(convention_meta, dict) else None
    convention_failures = []
    if expected_extra is not None and bool(expected_extra) != bool(args.st_extra_Rz180):
        convention_failures.append("target_extra_Rz180_mismatch")
    if expected_convention is not None and str(expected_convention) != str(args.target_pose_convention):
        convention_failures.append("target_pose_convention_mismatch")
    if convention_failures:
        gate_report.setdefault("failed_checks", []).extend(convention_failures)
        gate_report["failed_checks"] = sorted(set(gate_report["failed_checks"]))
        gate_report["pose_convention_check"] = {
            "expected_convention": expected_convention,
            "expected_extra_Rz180": expected_extra,
            "selected_convention": args.target_pose_convention,
            "selected_extra_Rz180": bool(args.st_extra_Rz180),
            "pass": False,
        }
        if gate_report.get("status") == "pass":
            gate_report["status"] = "fail" if args.gate_mode == "strict" else "legacy_debug"
    else:
        gate_report["pose_convention_check"] = {
            "pass": True, "expected_convention": expected_convention,
            "expected_extra_Rz180": expected_extra,
        }
    gate_report["alternate_summary"] = (
        evaluate_geometry_gate(
            alternate_rows, calibration_complete=calibration_complete,
            timing_available=timing_available, mode="legacy_debug",
        )
        if args.gate_mode != "off" else {"status": "diagnostic_only"}
    )
    gate_path = args.geometry_gate_report or (args.output_dir / "quality" / "geometry_gate.json")
    hard_fallback_failures: list[str] = []
    if args.gate_mode == "fallback":
        checks = gate_report.get("checks", {})
        front_check = checks.get("target_front_ratio", {}) if isinstance(checks, dict) else {}
        bounds_check = checks.get("in_bounds_ratio", {}) if isinstance(checks, dict) else {}
        reproj50_check = checks.get("reprojection_p50_px", {}) if isinstance(checks, dict) else {}
        reproj95_check = checks.get("reprojection_p95_px", {}) if isinstance(checks, dict) else {}
        if not calibration_complete:
            hard_fallback_failures.append("calibration_complete")
        if not timing_available:
            hard_fallback_failures.append("timing_available")
        if float(front_check.get("value") or 0.0) < GeometryGateThresholds().target_front_ratio:
            hard_fallback_failures.append("target_front_ratio")
        # Some individual frames legitimately have little overlap. A near-zero
        # episode mean instead indicates a coordinate/convention failure.
        if float(bounds_check.get("value") or 0.0) < 0.05:
            hard_fallback_failures.append("global_in_bounds_ratio")
        if reproj50_check and not bool(reproj50_check.get("pass", True)):
            hard_fallback_failures.append("reprojection_p50_px")
        if reproj95_check and not bool(reproj95_check.get("pass", True)):
            hard_fallback_failures.append("reprojection_p95_px")
        hard_fallback_failures.extend(convention_failures)
        gate_report["fallback_policy"] = {
            "allowed_failed_checks": ["bad_frame_ratio", "max_bad_frame_run", "in_bounds_ratio"],
            "hard_failures": sorted(set(hard_fallback_failures)),
        }
        gate_report["status"] = "fail" if hard_fallback_failures else (
            "pass" if selected_gate.get("status") == "pass" else "fallback"
        )
    write_gate_report(gate_path, gate_report)
    if args.gate_mode == "strict" and (selected_gate.get("status") != "pass" or convention_failures):
        raise SystemExit(
            "geometry gate failed before rendering: "
            + ", ".join(sorted(set(selected_gate.get("failed_checks", []) + convention_failures)))
        )
    if args.gate_mode == "fallback" and hard_fallback_failures:
        raise SystemExit(
            "geometry gate hard failure before fallback rendering: "
            + ", ".join(sorted(set(hard_fallback_failures)))
        )

    out = args.output_dir
    chunk_frames = end - start
    rgb_out = create_volume(
        out / "rgb_geometry",
        chunk_frames,
        (target_H, target_W, 3),
        np.uint8,
        fill=0,
        color_space="rgb",
    )
    depth_out = create_volume(out / "depth", chunk_frames, (target_H, target_W), np.float32, fill=0)
    valid_out = create_volume(out / "valid", chunk_frames, (target_H, target_W), np.uint8, fill=0)
    hole_out = create_volume(out / "hole", chunk_frames, (target_H, target_W), np.uint8, fill=255)
    conf_out_vol = create_volume(out / "confidence", chunk_frames, (target_H, target_W), np.float32, fill=0)
    source_count_out = create_volume(
        out / "source_count", chunk_frames, (target_H, target_W), np.uint32, fill=0
    )
    provenance_out = create_volume(out / "provenance", chunk_frames, (target_H, target_W), np.uint8, fill=0)

    frame_metrics_path = out / "quality" / "frame_metrics.jsonl"
    frame_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    resume_active = bool(args.resume_cache)
    if resume_active and args.fusion_backend != "surfel":
        raise SystemExit("--resume-cache is only supported with --fusion-backend surfel")
    if resume_active and (args.cache_checkpoint is None or not args.cache_checkpoint.is_file()):
        raise SystemExit("--resume-cache requires an existing --cache-checkpoint")

    cache: CausalSurfelBackgroundCache | None = None
    if args.fusion_backend == "surfel":
        if args.resume_cache and args.cache_checkpoint and args.cache_checkpoint.is_file():
            cache = CausalSurfelBackgroundCache.load(args.cache_checkpoint)
            cache.render_device = device if device.startswith("cuda") else None
        else:
            cache = CausalSurfelBackgroundCache(
                voxel_size=args.voxel_size_m,
                history_size=args.history_size,
                confidence_threshold=args.confidence_threshold,
                max_points_per_frame=args.max_cache_points,
                render_device=device if device.startswith("cuda") else None,
            )
    if resume_active:
        assert cache is not None
        for attr, expected in (
            ("history_size", int(args.history_size)),
            ("max_points_per_frame", int(args.max_cache_points)),
        ):
            if int(getattr(cache, attr)) != expected:
                raise SystemExit(
                    f"resume cache {attr}={getattr(cache, attr)} does not match requested {expected}"
                )
        if abs(float(cache.voxel_size) - float(args.voxel_size_m)) > 1e-9:
            raise SystemExit("resume cache voxel size does not match requested configuration")
        if abs(float(cache.confidence_threshold) - float(args.confidence_threshold)) > 1e-9:
            raise SystemExit("resume cache confidence threshold does not match requested configuration")

    # A checkpoint counts frames relative to the original chunk start.  Existing
    # local outputs are required before skipping them; this catches a stale or
    # partially copied cache instead of silently misaligning RGB/depth labels.
    resumed_frames = int(cache.processed_frame_count) if resume_active and cache is not None else 0
    if resumed_frames < 0 or resumed_frames > chunk_frames:
        raise SystemExit(
            f"resume cache contains {resumed_frames} frames, but requested chunk has {chunk_frames}"
        )
    if resumed_frames:
        if len(rgb_out) < resumed_frames or len(depth_out) < resumed_frames:
            raise SystemExit("resume cache has no matching packed output prefix")
    render_start = start + resumed_frames

    existing_rows: dict[int, dict[str, object]] = {}
    if resumed_frames and frame_metrics_path.is_file():
        for line in frame_metrics_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                existing_rows[int(item["frame"])] = dict(item)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

    writer = None
    if bool(getattr(args, "preview_video", False)) and render_start < end:
        video_path = out / ("novel_rgb_resume.mp4" if resumed_frames else "novel_rgb.mp4")
        writer = imageio.get_writer(
            str(video_path), fps=fps, codec="libx264", quality=8,
            macro_block_size=1, pixelformat="yuv420p",
        )
    quality_rows: list[dict[str, object]] = []
    hole_fracs: list[float] = []
    depth_stats: list[dict[str, object]] = []
    frame_status_rows: list[dict[str, object]] = []
    gate_row_by_frame = {int(row["frame"]): dict(row) for row in selected_rows}
    last_valid_geometry: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    src_K_t = torch.as_tensor(src_K, dtype=torch.float32, device=device)
    tgt_K_t = torch.as_tensor(tgt_K, dtype=torch.float32, device=device)
    warp_chunk = max(1, int(getattr(args, "warp_chunk", 8) or 8))
    if packed_source is not None:
        reader = PackedRgbReader(packed_source) if render_start < end else None
    else:
        reader = SequentialVideoReader(imageio.get_reader(str(args.video_file))) if render_start < end else None

    def _read_rgb(global_i: int, local_i: int) -> np.ndarray:
        assert reader is not None
        try:
            video_index = global_i if video_index_mode == "global" else local_i
            rgb = reader.get_rgb(video_index)
        except Exception:
            rgb = reader.get_rgb(local_i)
        if rgb.shape[:2] != (H0, W0):
            rgb = cv2.resize(rgb, (W0, H0), interpolation=cv2.INTER_LINEAR)
        return rgb

    def _prepare_independent(global_i: int, local_i: int) -> dict[str, object]:
        rgb = _read_rgb(global_i, local_i)
        d, resized = load_depth(global_i, local_i)
        if resized and d.shape != (H0, W0):
            d = cv2.resize(d, (W0, H0), interpolation=cv2.INTER_NEAREST)
        conf = load_conf(global_i, local_i)
        edge = load_edge(global_i, local_i)
        dyn = load_dynamic(global_i, local_i)
        si = source_pose_index(global_i, local_i)
        tgt_c2w = target_pose_for(global_i, local_i)
        src_pose = src_c2w[si]
        source_valid = np.isfinite(d) & (d > 1e-4) & (conf >= args.confidence_threshold) & ~edge & ~dyn
        return {
            "global_i": global_i,
            "local_i": local_i,
            "rgb": rgb,
            "depth": d,
            "conf": conf,
            "src_pose": src_pose,
            "tgt_c2w": tgt_c2w,
            "source_valid": source_valid,
        }

    def _warp_independent(global_i: int, local_i: int) -> dict[str, object]:
        prepared = _prepare_independent(global_i, local_i)
        rgb = prepared["rgb"]
        d = prepared["depth"]
        conf = prepared["conf"]
        source_valid = prepared["source_valid"]
        src_pose = prepared["src_pose"]
        tgt_c2w = prepared["tgt_c2w"]
        if args.fusion_backend == "surfel":
            assert cache is not None
            cache.add_frame(
                rgb, d, src_K, src_pose, confidence=conf,
                dynamic_mask=load_dynamic(global_i, local_i), edge_invalid=load_edge(global_i, local_i),
                timestamp=rgb_ts_by_frame.get(global_i, float(global_i) / max(fps, 1e-6)),
            )
            rendered = cache.render(tgt_K, tgt_c2w, image_size=(target_H, target_W))
            warped_rgb = rendered["rgb"]
            warped_d = rendered["depth"]
            valid = rendered["valid"]
            conf_out = rendered["confidence"]
            source_count = rendered["source_count"]
            provenance = rendered["provenance"]
        else:
            masked_d = np.where(source_valid, d, 0.0).astype(np.float32)
            rel = _relative_w2c(src_pose, tgt_c2w)
            warped_rgb, warped_d, valid_u8, conf_out = warp_frame_to_target(
                rgb, masked_d, src_K, rel, tgt_K, (target_H, target_W), device,
                confidence=np.where(source_valid, conf, 0.0).astype(np.float32),
                splat_radius_px=args.splat_radius_px,
                src_K_t=src_K_t,
                tgt_K_t=tgt_K_t,
            )
            valid = valid_u8 > 127
            source_count = valid.astype(np.uint32)
            provenance = np.where(valid, PROVENANCE_OBSERVED, PROVENANCE_INVALID).astype(np.uint8)
        return {
            "global_i": global_i,
            "local_i": local_i,
            "warped_rgb": warped_rgb,
            "warped_d": warped_d,
            "valid": valid,
            "conf_out": conf_out,
            "source_count": source_count,
            "provenance": provenance,
            "source_valid": source_valid,
        }

    def _warp_independent_batch(global_indices: list[int]) -> list[dict[str, object]]:
        n_chunk = len(global_indices)
        local0 = int(global_indices[0] - start)
        locals_are_contiguous = global_indices == list(
            range(int(global_indices[0]), int(global_indices[0]) + n_chunk)
        )

        def _slice_volume(vol, loader, default):
            if vol is None:
                return default
            if locals_are_contiguous and vol.packed:
                return np.asarray(vol.array[local0:local0 + n_chunk])
            return np.stack([loader(int(g), int(g) - start) for g in global_indices])

        if (
            packed_source is not None
            and reader is not None
            and getattr(reader, "volume", None) is not None
            and reader.volume.packed
            and locals_are_contiguous
        ):
            video0 = int(global_indices[0] if video_index_mode == "global" else local0)
            rgb_b = np.asarray(as_rgb(reader.volume.array[video0:video0 + n_chunk]))
            if tuple(rgb_b.shape[1:3]) != (H0, W0):
                rgb_b = np.stack(
                    [cv2.resize(frame, (W0, H0), interpolation=cv2.INTER_LINEAR) for frame in rgb_b]
                )
        else:
            rgb_b = np.stack([_read_rgb(int(g), int(g) - start) for g in global_indices])

        d_b = np.asarray(
            _slice_volume(depth_vol, lambda g, loc: load_depth(g, loc)[0], None),
            dtype=np.float32,
        )
        if d_b.ndim == 4:
            d_b = d_b[..., 0]
        if tuple(d_b.shape[1:]) != (H0, W0):
            d_b = np.stack(
                [cv2.resize(frame, (W0, H0), interpolation=cv2.INTER_NEAREST) for frame in d_b]
            )
        conf_b = np.asarray(
            _slice_volume(
                conf_vol,
                load_conf,
                np.ones((n_chunk, H0, W0), np.float32),
            ),
            dtype=np.float32,
        )
        if tuple(conf_b.shape[1:]) != (H0, W0):
            conf_b = np.stack(
                [cv2.resize(frame, (W0, H0), interpolation=cv2.INTER_NEAREST) for frame in conf_b]
            )
        conf_b = np.nan_to_num(conf_b, nan=0.0).astype(np.float32, copy=False)
        edge_b = as_mask(
            _slice_volume(
                edge_vol,
                lambda g, loc: load_edge(g, loc).astype(np.uint8) * 255,
                np.zeros((n_chunk, H0, W0), np.uint8),
            )
        )
        if tuple(edge_b.shape[1:]) != (H0, W0):
            edge_b = np.stack(
                [
                    cv2.resize(frame.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST) > 0
                    for frame in edge_b
                ]
            )
        dyn_b = as_mask(
            _slice_volume(
                dynamic_vol,
                lambda g, loc: load_dynamic(g, loc).astype(np.uint8) * 255,
                np.zeros((n_chunk, H0, W0), np.uint8),
            )
        )
        if tuple(dyn_b.shape[1:]) != (H0, W0):
            dyn_b = np.stack(
                [
                    cv2.resize(frame.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST) > 0
                    for frame in dyn_b
                ]
            )
        source_valid = (
            np.isfinite(d_b)
            & (d_b > 1e-4)
            & (conf_b >= args.confidence_threshold)
            & ~np.asarray(edge_b, dtype=bool)
            & ~np.asarray(dyn_b, dtype=bool)
        )
        depth_masked = np.where(source_valid, d_b, 0.0).astype(np.float32)
        conf_masked = np.where(source_valid, conf_b, 0.0).astype(np.float32)
        rel_b = np.stack(
            [
                _relative_w2c(
                    src_c2w[source_pose_index(int(g), int(g) - start)],
                    target_pose_for(int(g), int(g) - start),
                )
                for g in global_indices
            ]
        ).astype(np.float32)
        warped_rgb, warped_d, valid_u8, conf_out = warp_frames_to_target(
            rgb_b, depth_masked, src_K, rel_b, tgt_K, (target_H, target_W), device,
            confidence=conf_masked,
            splat_radius_px=args.splat_radius_px,
            src_K_t=src_K_t,
            tgt_K_t=tgt_K_t,
        )
        out: list[dict[str, object]] = []
        for i, global_i in enumerate(global_indices):
            valid = valid_u8[i] > 127
            out.append(
                {
                    "global_i": int(global_i),
                    "local_i": int(global_i) - start,
                    "warped_rgb": warped_rgb[i],
                    "warped_d": warped_d[i],
                    "valid": valid,
                    "conf_out": conf_out[i],
                    "source_count": valid.astype(np.uint32),
                    "provenance": np.where(valid, PROVENANCE_OBSERVED, PROVENANCE_INVALID).astype(np.uint8),
                    "source_valid": source_valid[i],
                }
            )
        return out

    def _commit_geometry(item: dict[str, object]) -> None:
        nonlocal last_valid_geometry
        global_i = int(item["global_i"])
        local_i = int(item["local_i"])
        warped_rgb = item["warped_rgb"]
        warped_d = item["warped_d"]
        valid = item["valid"]
        conf_out = item["conf_out"]
        source_count = item["source_count"]
        provenance = item["provenance"]
        source_valid = item["source_valid"]
        gate_row = gate_row_by_frame.get(global_i, {})
        frame_status = "low_coverage" if bool(gate_row.get("bad", False)) else "normal"
        if not bool(np.any(valid)) and last_valid_geometry is not None:
            warped_rgb, warped_d, valid, conf_out, source_count, provenance = (
                value.copy() for value in last_valid_geometry
            )
            frame_status = "background_hold"
        elif bool(np.any(valid)):
            last_valid_geometry = (
                np.asarray(warped_rgb).copy(), np.asarray(warped_d).copy(),
                np.asarray(valid).copy(), np.asarray(conf_out).copy(),
                np.asarray(source_count).copy(), np.asarray(provenance).copy(),
            )
        else:
            frame_status = "leading_empty_geometry"

        valid_u8 = (np.asarray(valid, dtype=bool).astype(np.uint8) * 255)
        hole_u8 = (255 - valid_u8).astype(np.uint8)
        rgb_out[local_i] = np.asarray(warped_rgb, dtype=np.uint8)
        depth_out[local_i] = np.asarray(warped_d, np.float32)
        valid_out[local_i] = valid_u8
        hole_out[local_i] = hole_u8
        conf_out_vol[local_i] = np.asarray(conf_out, np.float32)
        source_count_out[local_i] = np.asarray(source_count, np.uint32)
        provenance_out[local_i] = np.asarray(provenance, np.uint8)
        if writer is not None:
            writer.append_data(warped_rgb)

        hole_frac = float((hole_u8 > 127).mean())
        hole_fracs.append(hole_frac)
        vd = np.asarray(warped_d)[valid]
        row = dict(gate_row)
        row.update({
            "frame": global_i, "hole_ratio": hole_frac,
            "fusion_backend": args.fusion_backend,
            "valid_projection_ratio": float(np.mean(valid)),
            "source_valid_ratio": float(np.mean(source_valid)),
            "provenance_fraction": {
                "observed": float(np.mean(provenance == PROVENANCE_OBSERVED)),
                "fused": float(np.mean(provenance == PROVENANCE_FUSED)),
                "invalid": float(np.mean(provenance == PROVENANCE_INVALID)),
            },
            "depth_p50_m": float(np.median(vd)) if vd.size else None,
            "depth_p90_m": float(np.percentile(vd, 90)) if vd.size else None,
            "frame_status": frame_status,
            "target_index": int(target_pose_index(global_i, local_i)),
        })
        quality_rows.append(row)
        frame_status_rows.append({
            "source_index": int(global_i),
            "processed_index": int(local_i),
            "target_index": int(target_pose_index(global_i, local_i)),
            "source_timestamp_s": (
                float(rgb_ts_by_frame[global_i]) if global_i in rgb_ts_by_frame else None
            ),
            "target_timestamp_s": (
                float(target_ts_raw[target_pose_index(global_i, local_i)])
                if target_ts_raw is not None else None
            ),
            "status": frame_status,
            "valid_projection_ratio": float(np.mean(valid)),
            "hole_ratio": hole_frac,
        })
        depth_stats.append({"frame": global_i, "hole_frac": hole_frac, "d_p50": row["depth_p50_m"], "d_p90": row["depth_p90_m"]})
        if cache is not None and args.cache_checkpoint and args.checkpoint_every > 0 and (local_i + 1) % args.checkpoint_every == 0:
            args.cache_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            cache.save(args.cache_checkpoint)

    try:
        step = 1 if args.fusion_backend == "surfel" else warp_chunk
        pbar = tqdm(total=max(0, end - render_start), desc=f"gt_warp[{args.fusion_backend}]")
        cursor = render_start
        while cursor < end:
            chunk_end = min(end, cursor + step)
            independent = (
                [
                    _warp_independent(global_i, global_i - start)
                    for global_i in range(cursor, chunk_end)
                ]
                if args.fusion_backend == "surfel"
                else _warp_independent_batch(list(range(cursor, chunk_end)))
            )
            for item in independent:
                _commit_geometry(item)
            pbar.update(chunk_end - cursor)
            cursor = chunk_end
        pbar.close()
    finally:
        if reader is not None:
            reader.close()
        if writer is not None:
            writer.close()
    rgb_out.flush()
    depth_out.flush()
    valid_out.flush()
    hole_out.flush()
    conf_out_vol.flush()
    source_count_out.flush()
    provenance_out.flush()
    if cache is not None and args.cache_checkpoint:
        args.cache_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        cache.save(args.cache_checkpoint)
    # Merge rows from a resumed prefix and keep one record per global frame.
    merged_rows = dict(existing_rows)
    merged_rows.update({int(r["frame"]): r for r in quality_rows})
    all_quality_rows = [merged_rows[key] for key in sorted(merged_rows)]
    frame_metrics_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in all_quality_rows))
    (out / "frame_status.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in frame_status_rows)
    )
    if resumed_frames and bool(getattr(args, "preview_video", False)):
        # Re-encode the complete preview from the packed RGB volume.
        full_writer = imageio.get_writer(
            str(out / "novel_rgb.mp4"), fps=fps, codec="libx264", quality=8,
            macro_block_size=1, pixelformat="yuv420p",
        )
        try:
            for local_i in range(chunk_frames):
                full_writer.append_data(np.asarray(rgb_out[local_i], dtype=np.uint8))
        finally:
            full_writer.close()
    (out / "warp_meta.json").write_text(
        json.dumps(
            {
                "schema_version": 2, "n_frames": end - start, "start": start, "end": end,
                "hole_frac_mean": (
                    float(np.mean([float(r["hole_ratio"]) for r in all_quality_rows if r.get("hole_ratio") is not None]))
                    if any(r.get("hole_ratio") is not None for r in all_quality_rows) else None
                ),
                "depth_stats_sample": depth_stats[:5], "target_K": tgt_K.tolist(),
                "source_K": src_K.tolist(), "source_size": [W0, H0],
                "size": [target_W, target_H], "target_size": [target_W, target_H],
                "fusion_backend": args.fusion_backend, "history_size": args.history_size,
                "voxel_size_m": args.voxel_size_m, "geometry_gate": str(gate_path),
                "max_cache_points": args.max_cache_points,
                "render_backend": cache.last_render_backend if cache is not None else "single_frame_torch_zbuffer",
                "target_time_match": args.target_time_match,
                "rgb_video_index_mode": video_index_mode,
                "rgb_video_frame_count": int(video_count),
                "qualified_geometry": gate_report.get("status") in ("pass", "fallback"),
            }, indent=2,
        )
    )
    print(
        f"[gt_warp] wrote {out} frames={end-start} hole≈{np.mean(hole_fracs) if hole_fracs else float('nan'):.3f} "
        f"backend={args.fusion_backend} gate={gate_report.get('status')}", flush=True,
    )


if __name__ == "__main__":
    main()
