#!/usr/bin/env python3
"""Stera episode → EgoDex-like camera h5 + RGB-aligned metric depth + world wrists.

Reads ego_anywhere / Stera export:
  rgb.mp4, annotation.hdf5 (depth mm @256x144, cam-pose, hand-pose), calibrations/

Writes under --out-dir:
  camera.h5              camera/intrinsic + transforms/{camera,hip,leftHand,...}
  depth_m_rgb/frames.npy float32 meters at RGB resolution (packed)
  depth_valid/frames.npy uint8 0/255
  depth_confidence/frames.npy float32
  depth_edge_invalid/frames.npy uint8 0/255
  wrists_world.npz       convenience dump of wrist SE3 + validity
  adapter_meta.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

try:
    from depth_splat import (
        distortion_is_zero,
        register_depth_pinhole_torch,
        splat_device_available,
        splat_offsets,
        splat_zbuffer_batch,
    )
    from frame_volume import as_rgb, create_volume, is_packed_stream, open_volume
    from geometry_gate import (
        compute_depth_confidence_batch,
        evaluate_geometry_gate,
        write_gate_report,
    )
except ImportError:  # package-style import
    from .depth_splat import (
        distortion_is_zero,
        register_depth_pinhole_torch,
        splat_device_available,
        splat_offsets,
        splat_zbuffer_batch,
    )
    from .frame_volume import as_rgb, create_volume, is_packed_stream, open_volume
    from .geometry_gate import (
        compute_depth_confidence_batch,
        evaluate_geometry_gate,
        write_gate_report,
    )


# MediaPipe / WiLoR 21-joint indices
WRIST = 0
INDEX_TIP = 8
MIDDLE_TIP = 12
PINKY_TIP = 20
THUMB_TIP = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--episode",
        type=Path,
        required=True,
        help="Stera episode dir (rgb.mp4 + annotation.hdf5 + calibrations/)",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--calibration-dir", type=Path, default=None)
    p.add_argument("--rgb-k", type=Path, default=None)
    p.add_argument("--rgb-d", type=Path, default=None)
    p.add_argument("--depth-k", type=Path, default=None)
    p.add_argument("--depth-d", type=Path, default=None)
    p.add_argument("--t-rgb-from-depth", type=Path, default=None)
    p.add_argument(
        "--depth-aligned-to-rgb", "--registered-depth-in-rgb",
        dest="depth_aligned_to_rgb",
        action="store_true",
        help=(
            "Explicitly declare that native depth already uses the RGB optical "
            "frame. This permits T_rgb_from_depth=I only after the supplied "
            "intrinsics pass the registered-depth compatibility check."
        ),
    )
    p.add_argument(
        "--rgb-frame-locked-to-camera-pose",
        action="store_true",
        help=(
            "Explicitly declare that each RGB MP4 frame is paired with the "
            "same-index cam-pose timestamp. The declaration is rejected unless "
            "the RGB/camera/depth stream counts and timestamps are consistent."
        ),
    )
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--strict-calibration", action="store_true",
                   help="fail if depth K/D or RGB-depth extrinsics are absent")
    p.add_argument(
        "--splat-device",
        default="auto",
        help="Device for the registered-depth z-buffer splat: auto, cuda, or cpu",
    )
    p.add_argument(
        "--source-rgb-dir",
        type=Path,
        default=None,
        help="Optional packed OpenCV RGB volume; skips rgb.mp4 decode when present.",
    )
    return p.parse_args()


def _optional_calibration(path: Path | None, shape: tuple[int, ...], default: np.ndarray) -> tuple[np.ndarray, bool]:
    if path is not None and path.is_file():
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text())
            for key in ("value", "K", "D", "matrix", "transform", "T"):
                if key in payload:
                    payload = payload[key]
                    break
            value = np.asarray(payload, dtype=np.float64)
        else:
            value = np.asarray(np.load(path), dtype=np.float64)
        # Distortion vectors may be 4, 5, 8, 12 or 14 coefficients.
        if len(shape) == 1 and shape[0] == 5:
            return value.reshape(-1), True
        return value.reshape(shape), True
    return np.asarray(default, dtype=np.float64).reshape(shape), False


def _first_existing(paths: list[Path]) -> Path | None:
    return next((p for p in paths if p.is_file()), None)


def _registered_depth_identity_check(
    rgb_K: np.ndarray,
    depth_K: np.ndarray,
    rgb_hw: tuple[int, int],
    depth_hw: tuple[int, int],
    *,
    rtol: float = 2e-3,
    atol: float = 2e-3,
) -> dict[str, object]:
    """Verify the signature of depth already registered to RGB optical rays.

    The native depth image may be lower resolution, but for a registered depth
    product its intrinsics must be the RGB intrinsics scaled into that image
    grid.  This check is deliberately only a guard for an explicit caller
    declaration; it never infers an identity depth-to-RGB transform by itself.
    """
    rgb_h, rgb_w = (int(rgb_hw[0]), int(rgb_hw[1]))
    depth_h, depth_w = (int(depth_hw[0]), int(depth_hw[1]))
    if min(rgb_h, rgb_w, depth_h, depth_w) <= 0:
        return {
            "pass": False,
            "reason": "non_positive_image_size",
            "expected_depth_K": None,
            "max_abs_error": None,
        }
    expected = np.asarray(rgb_K, dtype=np.float64).reshape(3, 3).copy()
    sx = depth_w / float(rgb_w)
    sy = depth_h / float(rgb_h)
    expected[0, 0] *= sx
    expected[0, 1] *= sx
    expected[0, 2] *= sx
    expected[1, 0] *= sy
    expected[1, 1] *= sy
    expected[1, 2] *= sy
    supplied = np.asarray(depth_K, dtype=np.float64).reshape(3, 3)
    close = np.allclose(supplied, expected, rtol=rtol, atol=atol)
    return {
        "pass": bool(close),
        "reason": "scaled_rgb_intrinsics_match" if close else "scaled_rgb_intrinsics_mismatch",
        "rgb_to_depth_scale": [float(sx), float(sy)],
        "expected_depth_K": expected.tolist(),
        "max_abs_error": float(np.max(np.abs(supplied - expected))),
        "rtol": float(rtol),
        "atol": float(atol),
    }


def _frame_locked_rgb_check(
    *,
    rgb_count: int,
    camera_count: int,
    depth_count: int,
    rgb_timestamps: np.ndarray,
    camera_timestamps: np.ndarray,
    depth_timestamps: np.ndarray,
) -> dict[str, object]:
    """Validate an explicit RGB-to-camera frame-lock declaration."""
    n = int(rgb_count)
    same_counts = n > 0 and n == int(camera_count) and n == int(depth_count)
    rgb_t = np.asarray(rgb_timestamps, dtype=np.float64).reshape(-1)
    cam_t = np.asarray(camera_timestamps, dtype=np.float64).reshape(-1)
    depth_t = np.asarray(depth_timestamps, dtype=np.float64).reshape(-1)
    enough = len(rgb_t) >= n and len(cam_t) >= n and len(depth_t) >= n
    if same_counts and enough:
        cam_err = float(np.max(np.abs(rgb_t[:n] - cam_t[:n]))) if n else 0.0
        depth_err = float(np.max(np.abs(rgb_t[:n] - depth_t[:n]))) if n else 0.0
    else:
        cam_err = None
        depth_err = None
    passed = bool(same_counts and enough and cam_err is not None and depth_err is not None and cam_err <= 1e-6 and depth_err <= 1e-6)
    return {
        "pass": passed,
        "reason": "frame_counts_and_timestamps_match" if passed else "frame_counts_or_timestamps_mismatch",
        "rgb_count": n,
        "camera_count": int(camera_count),
        "depth_count": int(depth_count),
        "max_abs_rgb_camera_s": cam_err,
        "max_abs_rgb_depth_s": depth_err,
    }


def _normalise_timestamps(values: np.ndarray) -> np.ndarray:
    """Convert common timestamp units to seconds without changing the epoch."""
    raw = np.asarray(values)
    a = raw.astype(np.float64, copy=False).reshape(-1)
    if not a.size:
        return a
    finite = np.abs(a[np.isfinite(a)])
    scale = 1.0
    # Unix nanoseconds/microseconds are by far the most common integer forms.
    # Relative timestamps (including the Stera epoch around 5e4 s) stay intact.
    max_abs = float(np.max(finite)) if finite.size else 0.0
    # Integer clocks conventionally encode ns/us/ms.  Floating-point clocks
    # around 1e9 are often Unix *seconds* and must not be divided by 1e3.
    integer_clock = np.issubdtype(raw.dtype, np.integer)
    if max_abs > 1e17 or (integer_clock and max_abs > 1e14):
        scale = 1e-9
    elif max_abs > 1e11 or (integer_clock and max_abs > 1e8):
        scale = 1e-6
    elif integer_clock and max_abs > 1e5:
        scale = 1e-3
    return a * scale


def _stream_timestamps(
    h5: h5py.File,
    stream: str,
    n: int,
    *,
    fallback: np.ndarray | None = None,
) -> tuple[np.ndarray, str | None]:
    """Read timestamps for one stream and return ``(values, source)``.

    Stream-specific datasets are deliberately preferred over a root-level
    ``timestamps`` dataset.  Returning the source lets the manifest distinguish
    measured synchronization from an index/fps estimate.
    """
    candidates = {
        "rgb": ("rgb/timestamps", "rgb/timestamp", "video/timestamps"),
        "depth": ("depth/timestamps", "depth/timestamp"),
        "camera": ("cam-pose/timestamps", "cam-pose/timestamp", "camera/timestamps"),
        "hand": ("hand-pose/timestamps", "hand-pose/timestamp"),
    }
    for key in candidates.get(stream, ()) + ("timestamps", "timestamp", "frame_timestamps"):
        if key in h5:
            a = _normalise_timestamps(np.asarray(h5[key]).reshape(-1))
            if len(a) >= n:
                return a[:n], key
    if fallback is not None and len(fallback) >= n:
        return np.asarray(fallback[:n], dtype=np.float64), "derived_from_camera"
    return np.arange(n, dtype=np.float64) / 30.0, None


def _nearest_time_map(query: np.ndarray, source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map each query timestamp to the nearest source sample and report offsets."""
    q = np.asarray(query, dtype=np.float64).reshape(-1)
    s = np.asarray(source, dtype=np.float64).reshape(-1)
    if not len(s):
        raise ValueError("cannot map timestamps against an empty source stream")
    finite_source = np.isfinite(s)
    if not np.any(finite_source):
        raise ValueError("source timestamps contain no finite values")
    source_indices = np.flatnonzero(finite_source)
    order = source_indices[np.argsort(s[finite_source], kind="stable")]
    sorted_s = s[order]
    # Searchsorted is O(n log m), deterministic, and handles unequal stream rates.
    right = np.searchsorted(sorted_s, q, side="left")
    right = np.clip(right, 0, len(sorted_s) - 1)
    left = np.clip(right - 1, 0, len(s) - 1)
    choose_left = np.abs(q - sorted_s[left]) <= np.abs(q - sorted_s[right])
    sorted_idx = np.where(choose_left, left, right).astype(np.int64)
    idx = order[sorted_idx]
    return idx.astype(np.int64), (s[idx] - q).astype(np.float64)


# Cap the registered z-buffer at ~1.5 GiB (depth + confidence float32).
_REGISTER_MAX_OUTPUT_BYTES = 1_500_000_000


def _as_n_hw(
    arr: np.ndarray | None,
    n: int,
    hw: tuple[int, int],
    name: str,
) -> np.ndarray | None:
    if arr is None:
        return None
    a = np.asarray(arr)
    h, w = hw
    if a.ndim == 2:
        a = a[None]
    if a.shape != (n, h, w):
        raise ValueError(f"{name} shape {tuple(a.shape)} != {(n, h, w)}")
    return a


def _register_chunk_size(n: int, out_hw: tuple[int, int]) -> int:
    h, w = int(out_hw[0]), int(out_hw[1])
    bytes_per = max(1, h * w * 8)
    return max(1, min(int(n), _REGISTER_MAX_OUTPUT_BYTES // bytes_per))


def _register_depth_chunk(
    depth: np.ndarray,
    K_depth: np.ndarray,
    D_depth: np.ndarray,
    K_rgb: np.ndarray,
    D_rgb: np.ndarray,
    T_rgb_from_depth: np.ndarray,
    out_hw: tuple[int, int],
    confidence: np.ndarray | None,
    source_valid: np.ndarray | None,
    source_edge_invalid: np.ndarray | None,
    splat_radius_px: int | None,
    splat_device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Register one contiguous stack of native-depth frames into RGB."""
    n, h, w = (int(depth.shape[0]), int(depth.shape[1]), int(depth.shape[2]))
    H, W = int(out_hw[0]), int(out_hw[1])
    if (
        str(splat_device).startswith("cuda")
        and distortion_is_zero(D_depth)
        and distortion_is_zero(D_rgb)
    ):
        try:
            return register_depth_pinhole_torch(
                depth,
                K_depth,
                K_rgb,
                T_rgb_from_depth,
                out_hw,
                confidence=confidence,
                source_valid=source_valid,
                source_edge_invalid=source_edge_invalid,
                splat_radius_px=splat_radius_px,
                device=splat_device,
            )
        except Exception:
            pass
    empty_depth = np.zeros((n, H, W), np.float32)
    empty_conf = np.zeros((n, H, W), np.float32)
    empty_edge = np.ones((n, H, W), bool)
    yy, xx = np.indices((h, w), dtype=np.float32)
    pixels = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    z = np.asarray(depth, dtype=np.float64).reshape(n, -1)
    valid = np.isfinite(z) & (z > 1e-6)
    if source_valid is not None:
        valid &= np.asarray(source_valid, dtype=bool).reshape(n, -1)
    if source_edge_invalid is not None:
        se = np.asarray(source_edge_invalid, dtype=bool)
        # One-pixel dilation is in native-depth coordinates. It prevents the
        # splat footprint from leaking the remaining side of a depth edge.
        kernel = np.ones((3, 3), np.uint8)
        dilated = np.empty((n, h, w), dtype=bool)
        for i in range(n):
            dilated[i] = cv2.dilate(se[i].astype(np.uint8), kernel, iterations=1) > 0
        valid &= ~dilated.reshape(n, -1)
    fi, pi = np.nonzero(valid)
    if fi.size == 0:
        return empty_depth, empty_conf, empty_edge
    und = cv2.undistortPoints(
        pixels[pi, None, :].astype(np.float64), K_depth, D_depth
    ).reshape(-1, 2)
    z_p = z[fi, pi]
    pts = np.column_stack([und[:, 0] * z_p, und[:, 1] * z_p, z_p])
    T = np.asarray(T_rgb_from_depth, dtype=np.float64).reshape(4, 4)
    pts_rgb = (T[:3, :3] @ pts.T).T + T[:3, 3]
    front = pts_rgb[:, 2] > 1e-6
    pts_rgb = pts_rgb[front]
    fi = fi[front]
    if confidence is None:
        src_c = np.ones(len(pts_rgb), dtype=np.float32)
    else:
        src_c = np.asarray(confidence, dtype=np.float32).reshape(n, -1)[valid][front]
    if pts_rgb.size == 0:
        return empty_depth, empty_conf, empty_edge
    proj, _ = cv2.projectPoints(pts_rgb, np.zeros(3), np.zeros(3), K_rgb, D_rgb)
    uv = np.rint(proj.reshape(-1, 2)).astype(np.int64)
    inside = (
        (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        & np.isfinite(pts_rgb[:, 2]) & (src_c >= 0.2)
    )
    uv = uv[inside]
    z_rgb = pts_rgb[inside, 2].astype(np.float32)
    src_c = np.clip(src_c[inside], 0.0, 1.0).astype(np.float32)
    frame_id = fi[inside].astype(np.int64)
    if not len(uv):
        return empty_depth, empty_conf, empty_edge
    if splat_radius_px is None:
        splat_radius_px = max(0, int(np.ceil(0.5 * max(W / max(w, 1), H / max(h, 1)))))
    radius = max(0, int(splat_radius_px))
    return splat_zbuffer_batch(
        uv,
        z_rgb,
        src_c,
        frame_id,
        n,
        H,
        W,
        splat_offsets(radius),
        device=splat_device,
    )


def _register_depth_batch(
    depth: np.ndarray,
    K_depth: np.ndarray,
    D_depth: np.ndarray,
    K_rgb: np.ndarray,
    D_rgb: np.ndarray,
    T_rgb_from_depth: np.ndarray,
    out_hw: tuple[int, int],
    confidence: np.ndarray | None = None,
    source_valid: np.ndarray | None = None,
    source_edge_invalid: np.ndarray | None = None,
    splat_radius_px: int | None = None,
    splat_device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate a native-depth stack into RGB with one z-buffer per frame."""
    depth_b = np.asarray(depth)
    if depth_b.ndim == 2:
        depth_b = depth_b[None]
    elif depth_b.ndim != 3:
        raise ValueError(f"depth must be (H, W) or (N, H, W); got {depth_b.shape}")
    n, h, w = (int(depth_b.shape[0]), int(depth_b.shape[1]), int(depth_b.shape[2]))
    conf_b = _as_n_hw(confidence, n, (h, w), "confidence")
    valid_b = _as_n_hw(source_valid, n, (h, w), "source_valid")
    edge_b = _as_n_hw(source_edge_invalid, n, (h, w), "source_edge_invalid")
    chunk = _register_chunk_size(n, out_hw)
    if n <= chunk:
        return _register_depth_chunk(
            depth_b,
            K_depth,
            D_depth,
            K_rgb,
            D_rgb,
            T_rgb_from_depth,
            out_hw,
            conf_b,
            valid_b,
            edge_b,
            splat_radius_px,
            splat_device,
        )
    depths, confs, edges = [], [], []
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        d, c, e = _register_depth_chunk(
            depth_b[start:stop],
            K_depth,
            D_depth,
            K_rgb,
            D_rgb,
            T_rgb_from_depth,
            out_hw,
            None if conf_b is None else conf_b[start:stop],
            None if valid_b is None else valid_b[start:stop],
            None if edge_b is None else edge_b[start:stop],
            splat_radius_px,
            splat_device,
        )
        depths.append(d)
        confs.append(c)
        edges.append(e)
    return np.concatenate(depths), np.concatenate(confs), np.concatenate(edges)


def _register_depth(
    depth: np.ndarray,
    K_depth: np.ndarray,
    D_depth: np.ndarray,
    K_rgb: np.ndarray,
    D_rgb: np.ndarray,
    T_rgb_from_depth: np.ndarray,
    out_hw: tuple[int, int],
    confidence: np.ndarray | None = None,
    source_valid: np.ndarray | None = None,
    source_edge_invalid: np.ndarray | None = None,
    splat_radius_px: int | None = None,
    splat_device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate native depth into RGB with a confidence-aware metric z-buffer.

    Native RGB-D sensors commonly have a much lower depth resolution than RGB.
    After the calibrated projection, each high-confidence depth sample covers a
    small target-pixel footprint and competes through a nearest-depth z-buffer.
    This is deliberately not a depth ``resize``: every accepted target value
    originated in a de-distorted depth ray transformed by ``T_rgb_from_depth``.
    Depth edges are excluded before splatting so the footprint does not bridge
    foreground/background discontinuities.
    """
    d, c, e = _register_depth_batch(
        np.asarray(depth),
        K_depth,
        D_depth,
        K_rgb,
        D_rgb,
        T_rgb_from_depth,
        out_hw,
        confidence,
        source_valid,
        source_edge_invalid,
        splat_radius_px=splat_radius_px,
        splat_device=splat_device,
    )
    return d[0], c[0], e[0]


def _se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def _rot_from_points(origin: np.ndarray, x_axis_pt: np.ndarray, y_hint: np.ndarray) -> np.ndarray:
    """Build orthonormal R with X toward x_axis_pt, Y near y_hint."""
    x = np.asarray(x_axis_pt, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    xn = np.linalg.norm(x)
    if xn < 1e-8:
        return np.eye(3, dtype=np.float64)
    x = x / xn
    y = np.asarray(y_hint, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    # Gram-Schmidt
    y = y - x * np.dot(y, x)
    yn = np.linalg.norm(y)
    if yn < 1e-8:
        # arbitrary perpendicular
        tmp = np.array([0.0, 1.0, 0.0]) if abs(x[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        y = tmp - x * np.dot(tmp, x)
        yn = np.linalg.norm(y)
    y = y / max(yn, 1e-12)
    z = np.cross(x, y)
    zn = np.linalg.norm(z)
    z = z / max(zn, 1e-12)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def _hand_se3_cam(joints: np.ndarray, tip_idx: int) -> np.ndarray:
    """Camera-frame SE3 for a hand link from 21 joints (meters)."""
    j = np.asarray(joints, dtype=np.float64).reshape(21, 3)
    origin = j[WRIST]
    tip = j[tip_idx]
    # palm normal hint: wrist -> middle knuckle-ish (joint 9)
    hint = j[9] if tip_idx != 9 else j[5]
    R = _rot_from_points(origin, tip if tip_idx != WRIST else j[MIDDLE_TIP], hint)
    if tip_idx == WRIST:
        t = origin
    else:
        t = j[tip_idx]
        # keep rotation from wrist orientation
    return _se3(R, t)


def _optical_to_world_se3(
    T_opt: np.ndarray,
    R_link_c2w: np.ndarray,
    t_c2w: np.ndarray,
    R_o2l: np.ndarray,
) -> np.ndarray:
    """Lift an optical-frame SE3 into world using link-frame cam-pose + R_o2l.

    Stera convention (see stera.core.transforms.optical_to_world):
      p_world = R_link_c2w @ R_o2l @ p_optical + t
    so T_world = T_link_c2w @ T_o2l @ T_optical.
    """
    T_link = _se3(R_link_c2w, t_c2w)
    T_o2l = _se3(R_o2l, np.zeros(3))
    return T_link @ T_o2l @ np.asarray(T_opt, dtype=np.float64)


def _optical_c2w(R_link_c2w: np.ndarray, t_c2w: np.ndarray, R_o2l: np.ndarray) -> np.ndarray:
    """OpenCV optical camera-to-world from Stera link cam-pose."""
    return _se3(R_link_c2w, t_c2w) @ _se3(R_o2l, np.zeros(3))


def _probe_source_rgb(
    episode: Path,
    source_rgb_dir: Path | None,
) -> tuple[int, int, int, float, object | None]:
    """Return (count, height, width, fps, packed volume or None)."""
    if source_rgb_dir is not None and is_packed_stream(source_rgb_dir):
        volume = open_volume(source_rgb_dir, color_space="rgb")
        height, width = (int(value) for value in volume.frame_shape[:2])
        fps = 30.0
        meta_file = source_rgb_dir / "volume_meta.json"
        if meta_file.is_file():
            try:
                fps = float(json.loads(meta_file.read_text(encoding="utf-8")).get("fps") or 30.0)
            except (TypeError, ValueError, json.JSONDecodeError):
                fps = 30.0
        return len(volume), height, width, fps, volume
    cap = cv2.VideoCapture(str(episode / "rgb.mp4"))
    ok, frame0 = cap.read()
    if not ok or frame0 is None:
        cap.release()
        raise RuntimeError(f"failed to read {episode / 'rgb.mp4'}")
    height, width = frame0.shape[:2]
    rgb_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    rgb_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return rgb_count, height, width, rgb_fps, None


def main() -> None:
    args = parse_args()
    ep = args.episode.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    depth_dir = out / "depth_m_rgb"
    valid_dir = out / "depth_valid"
    confidence_dir = out / "depth_confidence"
    edge_invalid_dir = out / "depth_edge_invalid"
    depth_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    confidence_dir.mkdir(parents=True, exist_ok=True)
    edge_invalid_dir.mkdir(parents=True, exist_ok=True)

    calib = (args.calibration_dir or (ep / "calibrations")).resolve()
    rgb_k_path = args.rgb_k or _first_existing([calib / "rgb_K.npy", calib / "K_rgb.npy"])
    if rgb_k_path is None:
        raise FileNotFoundError(f"missing RGB intrinsics under {calib}")
    rgb_K = np.asarray(np.load(rgb_k_path), dtype=np.float64).reshape(3, 3)
    rgb_d_path = args.rgb_d or _first_existing([calib / "rgb_D.npy", calib / "D_rgb.npy"])
    rgb_D, has_rgb_D = _optional_calibration(rgb_d_path, (5,), np.zeros(5)) if rgb_d_path else (np.zeros(5), False)
    R_o2l_path = calib / "R_optical_to_link.npy"
    if not R_o2l_path.is_file():
        raise FileNotFoundError(f"missing {R_o2l_path}")
    R_o2l = np.load(R_o2l_path).astype(np.float64).reshape(3, 3)
    ann = ep / "annotation.hdf5"
    # Probe RGB size and frame count before selecting HDF5 samples.  Packed
    # OpenCV RGB is preferred; the MP4 is only opened when that volume is absent.
    rgb_count, H, W, rgb_fps, packed_rgb = _probe_source_rgb(ep, args.source_rgb_dir)

    with h5py.File(ann, "r") as f:
        n_cam_all = int(f["cam-pose/translations"].shape[0])
        cam_ts_all, cam_ts_source = _stream_timestamps(f, "camera", n_cam_all)
        n_depth_all = int(f["depth/frames"].shape[0])
        depth_ts_all, depth_ts_source = _stream_timestamps(
            f, "depth", n_depth_all, fallback=cam_ts_all
        )
        n_hand_all = int(f["hand-pose/left_joints"].shape[0])
        hand_ts_all, hand_ts_source = _stream_timestamps(
            f, "hand", n_hand_all, fallback=cam_ts_all
        )
        rgb_ts_all, rgb_ts_source = _stream_timestamps(
            f, "rgb", n_cam_all, fallback=cam_ts_all
        )
        n_stream = min(n_cam_all, rgb_count if rgb_count > 0 else n_cam_all)
        rgb_ts_all = rgb_ts_all[:n_stream]
        # Stera export writes exactly one nearest-depth sample for each RGB
        # frame.  The MP4 PTS are encoding-relative and are not a calibrated
        # recording clock, so retain the matched depth clock instead.
        if rgb_ts_source == "derived_from_camera" and len(depth_ts_all) >= n_stream:
            rgb_ts_all = np.asarray(depth_ts_all[:n_stream], dtype=np.float64)
            rgb_ts_source = "matched_depth_timestamp_from_export"
        # Map each RGB sample to the nearest camera/depth/hand sample.  The
        # maps and signed offsets are persisted in alignment_manifest.json.
        cam_map_all, cam_offset_all = _nearest_time_map(rgb_ts_all, cam_ts_all)
        depth_map_all, depth_offset_all = _nearest_time_map(rgb_ts_all, depth_ts_all)
        hand_map_all, hand_offset_all = _nearest_time_map(rgb_ts_all, hand_ts_all)
        start = max(0, int(args.start_frame))
        end = n_stream if args.max_frames is None else min(n_stream, start + int(args.max_frames))
        n = end - start
        if n <= 0:
            raise SystemExit(f"empty frame range [{start},{end})")
        sel = np.arange(start, end, dtype=np.int64)
        cam_sel = cam_map_all[sel]
        depth_sel = depth_map_all[sel]
        hand_sel = hand_map_all[sel]
        cam_R = np.asarray(f["cam-pose/rotations"], dtype=np.float64)[cam_sel]
        cam_t = np.asarray(f["cam-pose/translations"], dtype=np.float64)[cam_sel]
        depth_mm = np.asarray(f["depth/frames"])[depth_sel]  # uint16
        depth_frame_valid = np.asarray(f["depth/valid"], dtype=bool)[depth_sel]
        depth_sensor_confidence = None
        for conf_key in ("depth/confidence", "depth/conf", "depth/confidence_map"):
            if conf_key in f:
                raw_conf = np.asarray(f[conf_key])[depth_sel]
                if raw_conf.shape == depth_mm.shape:
                    depth_sensor_confidence = raw_conf
                break
        left_j = np.asarray(f["hand-pose/left_joints"], dtype=np.float64)[hand_sel]
        right_j = np.asarray(f["hand-pose/right_joints"], dtype=np.float64)[hand_sel]
        left_valid = np.asarray(f["hand-pose/left_valid"], dtype=bool)[hand_sel]
        right_valid = np.asarray(f["hand-pose/right_valid"], dtype=bool)[hand_sel]
        left_k2d = np.asarray(f["hand-pose/left_kpts_2d_rgb"], dtype=np.float32)[hand_sel]
        right_k2d = np.asarray(f["hand-pose/right_kpts_2d_rgb"], dtype=np.float32)[hand_sel]

    # Use the camera clock for RGB when the MP4 lacks embedded per-frame
    # timestamps.  A strict run may accept this only with an explicit
    # frame-lock declaration validated against the synchronized HDF5 streams.
    frame_lock_check = _frame_locked_rgb_check(
        rgb_count=rgb_count,
        camera_count=n_cam_all,
        depth_count=n_depth_all,
        rgb_timestamps=rgb_ts_all,
        camera_timestamps=cam_ts_all,
        depth_timestamps=depth_ts_all,
    )
    rgb_frame_lock_declared = bool(args.rgb_frame_locked_to_camera_pose)
    if rgb_frame_lock_declared and not frame_lock_check["pass"]:
        raise RuntimeError(
            "RGB frame-lock declaration rejected: "
            + str(frame_lock_check["reason"])
        )
    if rgb_ts_source not in (
        None,
        "derived_from_camera",
        "matched_depth_timestamp_from_export",
    ):
        rgb_ts_measured = True
        rgb_timestamp_status = "measured"
    elif rgb_frame_lock_declared:
        rgb_ts_measured = True
        rgb_ts_source = "verified_exported_rgb_depth_pairing"
        rgb_timestamp_status = "declared_exported_rgb_depth_pairing_verified"
    else:
        rgb_ts_measured = False
        rgb_timestamp_status = "matched_depth_timestamp_unverified"

    # Prefer explicit depth calibration. An identity depth-to-RGB transform is
    # accepted only for an explicitly declared registered-depth product after
    # its scaled-intrinsics signature has been verified.
    depth_native_hw = tuple(int(x) for x in depth_mm.shape[1:])
    dh, dw = depth_native_hw
    depth_K_default = rgb_K.copy()
    depth_K_default[0, 0] *= dw / float(W)
    depth_K_default[1, 1] *= dh / float(H)
    depth_K_default[0, 2] *= dw / float(W)
    depth_K_default[1, 2] *= dh / float(H)
    depth_k_path = args.depth_k or _first_existing([calib / "depth_K.npy", calib / "K_depth.npy"])
    depth_d_path = args.depth_d or _first_existing([calib / "depth_D.npy", calib / "D_depth.npy"])
    t_path = args.t_rgb_from_depth or _first_existing(
        [calib / "T_rgb_from_depth.npy", calib / "T_rgb_from_depth_m.npy", calib / "depth_to_rgb.npy"]
    )
    depth_K, has_depth_K = _optional_calibration(depth_k_path, (3, 3), depth_K_default) if depth_k_path else (depth_K_default, False)
    depth_D, has_depth_D = _optional_calibration(depth_d_path, (5,), np.zeros(5)) if depth_d_path else (np.zeros(5), False)
    T_rgb_from_depth, has_depth_extrinsics = _optional_calibration(t_path, (4, 4), np.eye(4)) if t_path else (np.eye(4), False)
    has_depth_extrinsics_file = bool(has_depth_extrinsics)
    depth_identity_check: dict[str, object] | None = None
    if has_depth_extrinsics:
        depth_extrinsics_source = "file:" + str(t_path)
        depth_registration_status = "measured_transform_file"
    elif args.depth_aligned_to_rgb:
        depth_identity_check = _registered_depth_identity_check(
            rgb_K, depth_K, (H, W), depth_native_hw,
        )
        if not depth_identity_check["pass"]:
            raise RuntimeError(
                "registered depth-to-RGB identity declaration rejected: "
                + str(depth_identity_check["reason"])
            )
        T_rgb_from_depth = np.eye(4, dtype=np.float64)
        has_depth_extrinsics = True
        depth_extrinsics_source = "declared_registered_depth_in_rgb_optical_frame"
        depth_registration_status = "inferred_registered_same_optical_grid"
    else:
        depth_extrinsics_source = "missing"
        depth_registration_status = "missing"
    calibration_complete = bool(has_rgb_D and has_depth_K and has_depth_D and has_depth_extrinsics)
    if args.strict_calibration and not calibration_complete:
        missing = []
        for ok, name in ((has_rgb_D, "rgb_D.npy"), (has_depth_K, "depth_K.npy"),
                         (has_depth_D, "depth_D.npy"), (has_depth_extrinsics, "T_rgb_from_depth.npy or --depth-aligned-to-rgb")):
            if not ok:
                missing.append(name)
        raise RuntimeError("strict calibration requested; missing " + ", ".join(missing))

    # depth mm → m, then calibrated RGB registration (nearest-depth z-buffer).
    depth_m = depth_mm.astype(np.float32) / 1000.0
    depth_m[depth_m <= 0] = 0.0

    transforms = {
        "camera": np.zeros((n, 4, 4), dtype=np.float64),
        "hip": np.zeros((n, 4, 4), dtype=np.float64),
        "leftHand": np.zeros((n, 4, 4), dtype=np.float64),
        "rightHand": np.zeros((n, 4, 4), dtype=np.float64),
        "leftShoulder": np.zeros((n, 4, 4), dtype=np.float64),
        "rightShoulder": np.zeros((n, 4, 4), dtype=np.float64),
        "leftIndexFingerTip": np.zeros((n, 4, 4), dtype=np.float64),
        "rightIndexFingerTip": np.zeros((n, 4, 4), dtype=np.float64),
        "leftMiddleFingerTip": np.zeros((n, 4, 4), dtype=np.float64),
        "rightMiddleFingerTip": np.zeros((n, 4, 4), dtype=np.float64),
        "leftLittleFingerTip": np.zeros((n, 4, 4), dtype=np.float64),
        "rightLittleFingerTip": np.zeros((n, 4, 4), dtype=np.float64),
        # Extra EgoDex finger keys DexPilot may read (copy tips / wrist as fallback)
        "leftThumbTip": np.zeros((n, 4, 4), dtype=np.float64),
        "rightThumbTip": np.zeros((n, 4, 4), dtype=np.float64),
        "leftRingFingerTip": np.zeros((n, 4, 4), dtype=np.float64),
        "rightRingFingerTip": np.zeros((n, 4, 4), dtype=np.float64),
    }

    proj_errs = []
    frame_metrics = []
    rgb_depth_offsets_ms = (depth_offset_all[sel] * 1000.0).astype(np.float64)
    rgb_camera_offsets_ms = (cam_offset_all[sel] * 1000.0).astype(np.float64)
    splat_device = splat_device_available(args.splat_device)
    depth_vol = create_volume(depth_dir, n, (H, W), np.float32, fill=0)
    valid_vol = create_volume(valid_dir, n, (H, W), np.uint8, fill=0)
    conf_vol = create_volume(confidence_dir, n, (H, W), np.float32, fill=0)
    edge_vol = create_volume(edge_invalid_dir, n, (H, W), np.uint8, fill=0)
    skip_rgb_edges = depth_sensor_confidence is not None
    rgb_native_stack = None
    rgb_present = None
    rgb_cap = None
    if not skip_rgb_edges:
        dh, dw = int(depth_m.shape[1]), int(depth_m.shape[2])
        rgb_native_stack = np.zeros((n, dh, dw, 3), dtype=np.uint8)
        rgb_present = np.zeros((n,), dtype=bool)
        if packed_rgb is None:
            rgb_cap = cv2.VideoCapture(str(ep / "rgb.mp4"))
            if start:
                rgb_cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for i in range(n):
            rgb_native = None
            if packed_rgb is not None:
                packed_index = i if len(packed_rgb) == n else start + i
                rgb = as_rgb(packed_rgb.frame_at(packed_index, packed_index))
                rgb_native = cv2.resize(
                    rgb, (dw, dh), interpolation=cv2.INTER_AREA
                )
            elif rgb_cap is not None:
                ok_rgb, rgb_bgr = rgb_cap.read()
                if ok_rgb and rgb_bgr is not None:
                    rgb_native = cv2.resize(
                        rgb_bgr, (dw, dh), interpolation=cv2.INTER_AREA
                    )
            if rgb_native is not None:
                rgb_native_stack[i] = rgb_native
                rgb_present[i] = True
        if rgb_cap is not None:
            rgb_cap.release()
            rgb_cap = None
    confidence_native, valid_native, edge_native = compute_depth_confidence_batch(
        depth_m,
        rgb=rgb_native_stack,
        sensor_confidence=depth_sensor_confidence,
        rgb_present=rgb_present,
        use_temporal=not skip_rgb_edges,
    )
    d_all, d_conf_all, d_edge_all = _register_depth_batch(
        depth_m, depth_K, depth_D, rgb_K, rgb_D, T_rgb_from_depth,
        (H, W), confidence_native, valid_native, edge_native,
        splat_device=splat_device,
    )
    if np.any(~np.asarray(depth_frame_valid, dtype=bool)):
        bad = ~np.asarray(depth_frame_valid, dtype=bool)
        d_all[bad] = 0.0
        d_conf_all[bad] = 0.0
        d_edge_all[bad] = True
    valid_all = (d_all > 1e-4) & np.isfinite(d_all) & (d_conf_all >= 0.2) & ~d_edge_all
    try:
        depth_vol.array[:] = d_all.astype(np.float32, copy=False)
        conf_vol.array[:] = d_conf_all.astype(np.float32, copy=False)
        valid_vol.array[:] = (valid_all.astype(np.uint8) * 255)
        edge_vol.array[:] = (d_edge_all.astype(np.uint8) * 255)
    except TypeError:
        for i in range(n):
            depth_vol[i] = d_all[i].astype(np.float32)
            valid_vol[i] = (valid_all[i].astype(np.uint8) * 255)
            conf_vol[i] = d_conf_all[i].astype(np.float32)
            edge_vol[i] = (d_edge_all[i].astype(np.uint8) * 255)
    for i in range(n):
        R_link = cam_R[i]
        t = cam_t[i]
        # EgoDex/ST expect OpenCV optical c2w (not Stera link-frame pose).
        T_opt_c2w = _optical_c2w(R_link, t, R_o2l)
        R_opt = T_opt_c2w[:3, :3]
        transforms["camera"][i] = T_opt_c2w
        frame_metrics.append({
            "frame": i,
            "valid_ratio": float(valid_all[i].mean()),
            # Target-camera front/in-bounds ratios are only meaningful after
            # applying the target pose and are computed by viewport preflight.
            "front_ratio": None,
            "in_bounds_ratio": None,
            "timing_offset_ms": float(rgb_depth_offsets_ms[i]),
            "bad": bool(valid_all[i].mean() < 0.30),
        })

        # Hands: joints are camera optical 3D (meters); lift with R_o2l.
        for side, joints, tip_map in (
            (
                "left",
                left_j[i],
                {
                    "Hand": WRIST,
                    "IndexFingerTip": INDEX_TIP,
                    "MiddleFingerTip": MIDDLE_TIP,
                    "LittleFingerTip": PINKY_TIP,
                    "ThumbTip": THUMB_TIP,
                    "RingFingerTip": 16,
                },
            ),
            (
                "right",
                right_j[i],
                {
                    "Hand": WRIST,
                    "IndexFingerTip": INDEX_TIP,
                    "MiddleFingerTip": MIDDLE_TIP,
                    "LittleFingerTip": PINKY_TIP,
                    "ThumbTip": THUMB_TIP,
                    "RingFingerTip": 16,
                },
            ),
        ):
            for name, jidx in tip_map.items():
                T_opt = _hand_se3_cam(joints, jidx)
                transforms[f"{side}{name}"][i] = _optical_to_world_se3(
                    T_opt, R_link, t, R_o2l
                )

        # Shoulders: approximate offsets in optical frame, then to world.
        # OpenCV optical: +X right, +Y down, +Z forward
        shoulder_off_L = np.array([-0.18, 0.12, 0.05])
        shoulder_off_R = np.array([0.18, 0.12, 0.05])
        transforms["leftShoulder"][i] = _se3(R_opt, R_opt @ shoulder_off_L + t)
        transforms["rightShoulder"][i] = _se3(R_opt, R_opt @ shoulder_off_R + t)

        # Hip: below camera in world Y-up; yaw from optical forward projected to XZ.
        hip_t = t.copy()
        hip_t[1] = max(0.05, float(t[1]) - 0.72)
        forward = R_opt @ np.array([0.0, 0.0, 1.0])
        yaw = np.arctan2(forward[0], forward[2])
        c, s = np.cos(yaw), np.sin(yaw)
        R_hip = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
        transforms["hip"][i] = _se3(R_hip, hip_t)

        # Projection unit check (joints vs kpts) — X should match; Y often OOB for wrists
        if left_valid[i]:
            j0 = left_j[i, 0]
            z = max(float(j0[2]), 1e-6)
            u = rgb_K[0, 0] * j0[0] / z + rgb_K[0, 2]
            v = rgb_K[1, 1] * j0[1] / z + rgb_K[1, 2]
            proj_errs.append(
                {
                    "frame": i,
                    "k2d": left_k2d[i, 0].tolist(),
                    "proj": [float(u), float(v)],
                    "err": [float(u - left_k2d[i, 0, 0]), float(v - left_k2d[i, 0, 1])],
                }
            )
            frame_metrics[-1]["hand_reprojection_error_px"] = float(
                np.hypot(u - left_k2d[i, 0, 0], v - left_k2d[i, 0, 1])
            )

    if rgb_cap is not None:
        rgb_cap.release()
    depth_vol.flush()
    valid_vol.flush()
    conf_vol.flush()
    edge_vol.flush()
    depth_vol.flush()
    valid_vol.flush()
    conf_vol.flush()
    edge_vol.flush()

    h5_path = out / "camera.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("camera/intrinsic", data=rgb_K.astype(np.float32))
        for k, arr in transforms.items():
            h5.create_dataset(f"transforms/{k}", data=arr.astype(np.float32))
        # DexPilot often expects many finger intermediate frames — fill with tip/wrist copies
        extras = [
            "leftIndexFingerKnuckle",
            "leftIndexFingerIntermediateBase",
            "leftIndexFingerIntermediateTip",
            "leftIndexFingerMetacarpal",
            "leftMiddleFingerKnuckle",
            "leftMiddleFingerIntermediateBase",
            "leftMiddleFingerIntermediateTip",
            "leftMiddleFingerMetacarpal",
            "leftLittleFingerKnuckle",
            "leftLittleFingerIntermediateBase",
            "leftLittleFingerIntermediateTip",
            "leftLittleFingerMetacarpal",
            "leftRingFingerKnuckle",
            "leftRingFingerIntermediateBase",
            "leftRingFingerIntermediateTip",
            "leftRingFingerMetacarpal",
            "leftThumbKnuckle",
            "leftThumbIntermediateBase",
            "leftThumbIntermediateTip",
            "leftThumbTip",
            "rightIndexFingerKnuckle",
            "rightIndexFingerIntermediateBase",
            "rightIndexFingerIntermediateTip",
            "rightIndexFingerMetacarpal",
            "rightMiddleFingerKnuckle",
            "rightMiddleFingerIntermediateBase",
            "rightMiddleFingerIntermediateTip",
            "rightMiddleFingerMetacarpal",
            "rightLittleFingerKnuckle",
            "rightLittleFingerIntermediateBase",
            "rightLittleFingerIntermediateTip",
            "rightLittleFingerMetacarpal",
            "rightRingFingerKnuckle",
            "rightRingFingerIntermediateBase",
            "rightRingFingerIntermediateTip",
            "rightRingFingerMetacarpal",
            "rightThumbKnuckle",
            "rightThumbIntermediateBase",
            "rightThumbIntermediateTip",
            "rightThumbTip",
            "leftArm",
            "leftForearm",
            "rightArm",
            "rightForearm",
        ]
        for name in extras:
            if name in transforms:
                continue
            # copy nearest tip / hand
            if "Thumb" in name:
                src = "leftThumbTip" if name.startswith("left") else "rightThumbTip"
            elif "Index" in name:
                src = "leftIndexFingerTip" if name.startswith("left") else "rightIndexFingerTip"
            elif "Middle" in name:
                src = "leftMiddleFingerTip" if name.startswith("left") else "rightMiddleFingerTip"
            elif "Little" in name:
                src = "leftLittleFingerTip" if name.startswith("left") else "rightLittleFingerTip"
            elif "Ring" in name:
                src = "leftRingFingerTip" if name.startswith("left") else "rightRingFingerTip"
            elif name.endswith("Arm") or name.endswith("Forearm"):
                src = "leftHand" if name.startswith("left") else "rightHand"
            else:
                src = "leftHand" if name.startswith("left") else "rightHand"
            h5.create_dataset(f"transforms/{name}", data=transforms[src].astype(np.float32))

    np.savez_compressed(
        out / "wrists_world.npz",
        leftHand=transforms["leftHand"].astype(np.float32),
        rightHand=transforms["rightHand"].astype(np.float32),
        camera=transforms["camera"].astype(np.float32),
        left_valid=left_valid,
        right_valid=right_valid,
        left_kpts_2d=left_k2d,
        right_kpts_2d=right_k2d,
    )

    # Symlink / copy rgb for downstream
    rgb_link = out / "rgb.mp4"
    if not rgb_link.exists():
        try:
            rgb_link.symlink_to(ep / "rgb.mp4")
        except OSError:
            import shutil

            shutil.copy2(ep / "rgb.mp4", rgb_link)

    meta = {
        "episode": str(ep),
        "n_frames": n,
        "start_frame": start,
        "end_frame": end,
        "rgb_size": [W, H],
        "depth_native": list(depth_mm.shape[1:]),
        "depth_unit": "mm_in_h5_converted_to_meters",
        "camera_h5": str(h5_path),
        "depth_m_rgb_dir": str(depth_dir),
        "depth_confidence_dir": str(confidence_dir),
        "depth_edge_invalid_dir": str(edge_invalid_dir),
        "depth_to_rgb_registration": {
            "T_rgb_from_depth_source": depth_extrinsics_source,
            "status": depth_registration_status,
            "registered_depth_identity_check": depth_identity_check,
        },
        "rgb_frame_lock": {
            "declared": rgb_frame_lock_declared,
            "timestamp_status": rgb_timestamp_status,
            "check": frame_lock_check,
        },
        "R_optical_to_link": R_o2l.tolist(),
        "proj_check_left_wrist_first": proj_errs[:3],
        "note": (
            "Hand SE3: WiLoR optical joints lifted via "
            "T_world = T_link_c2w @ T_o2l @ T_optical (Stera optical_to_world). "
            "transforms/camera stored as OpenCV optical c2w = T_link @ T_o2l. "
            "World is Y-up (ground-aligned)."
        ),
    }
    (out / "adapter_meta.json").write_text(json.dumps(meta, indent=2))

    # Stable, self-describing input contract consumed by downstream offline stages.
    frame_ts = rgb_ts_all[sel]
    depth_ts_sel = depth_ts_all[depth_sel]
    cam_ts_sel = cam_ts_all[cam_sel]
    hand_ts_sel = hand_ts_all[hand_sel]
    timing_complete = bool(
        rgb_ts_measured
        and cam_ts_source is not None
        and depth_ts_source is not None
        and hand_ts_source is not None
    )
    timestamp_status = (
        "measured_or_verified_frame_locked"
        if timing_complete else "incomplete_legacy_debug"
    )
    calibration_version = None
    calib_meta = calib / "meta.json"
    if calib_meta.is_file():
        try:
            calibration_version = json.loads(calib_meta.read_text()).get("version")
        except Exception:
            calibration_version = None
    alignment_manifest = {
        "schema_version": 2,
        "mode": "calibrated" if calibration_complete and timing_complete else "legacy_debug",
        "episode": str(ep),
        "frame_range": {"start": start, "end": end, "count": n},
        "streams": {
            "rgb": {
                "path": str(ep / "rgb.mp4"),
                "timestamps_s": frame_ts.tolist(),
                "timestamp_source": rgb_ts_source,
                "frame_indices": list(range(start, end)),
                "size": [W, H],
                "fps": rgb_fps if rgb_fps > 0 else None,
            },
            "depth": {
                "source": str(ann), "dataset": "depth/frames",
                "timestamps_s": depth_ts_sel.tolist(), "timestamp_source": depth_ts_source,
                "frame_indices": depth_sel.astype(int).tolist(),
                "native_size": [dw, dh], "unit": "mm",
            },
            "camera_pose": {
                "source": str(ann), "dataset": "cam-pose/{rotations,translations}",
                "timestamps_s": cam_ts_sel.tolist(), "timestamp_source": cam_ts_source,
                "frame_indices": cam_sel.astype(int).tolist(),
                "pose_direction": "camera_link_to_world",
                "external_pose": True,
            },
            "hand_pose": {
                "source": str(ann), "dataset": "hand-pose/*",
                "timestamps_s": hand_ts_sel.tolist(), "timestamp_source": hand_ts_source,
                "frame_indices": hand_sel.astype(int).tolist(),
            },
        },
        "rgb_depth_frame_map": [
            {
                "rgb_index": int(start + i),
                "depth_index": int(depth_sel[i]),
                "rgb_timestamp_s": float(frame_ts[i]),
                "depth_timestamp_s": float(depth_ts_sel[i]),
                "time_offset_ms": float(rgb_depth_offsets_ms[i]),
            }
            for i in range(n)
        ],
        "rgb_camera_frame_map": [
            {
                "rgb_index": int(start + i),
                "camera_index": int(cam_sel[i]),
                "rgb_timestamp_s": float(frame_ts[i]),
                "camera_timestamp_s": float(cam_ts_sel[i]),
                "time_offset_ms": float(rgb_camera_offsets_ms[i]),
            }
            for i in range(n)
        ],
        "calibration": {
            "rgb": {"K": rgb_K.tolist(), "D": rgb_D.tolist(), "distortion_model": "opencv"},
            "depth": {"K": depth_K.tolist(), "D": depth_D.tolist(), "distortion_model": "opencv"},
            "T_rgb_from_depth": T_rgb_from_depth.tolist(),
            "T_rgb_from_depth_source": depth_extrinsics_source,
            "T_rgb_from_depth_status": depth_registration_status,
            "registered_depth_identity_check": depth_identity_check,
            "version": calibration_version,
            "files_present": {"rgb_D": has_rgb_D, "depth_K": has_depth_K, "depth_D": has_depth_D,
                              "T_rgb_from_depth": has_depth_extrinsics,
                              "T_rgb_from_depth_file": has_depth_extrinsics_file},
        },
        "coordinate_convention": {
            "camera_frame": "opencv_optical_x_right_y_down_z_forward",
            "world_frame": "stera_y_up",
            "pose_direction": "camera_to_world",
            "source_camera_pose_external": "annotation.hdf5:/cam-pose transformed from link to OpenCV optical",
            "R_optical_to_link": R_o2l.tolist(),
            # The offline SparseTrack target recorder currently emits the plain
            # convention.  Keeping this explicit lets Stage 3 reject a legacy
            # Rz(180°) toggle instead of guessing from visual coverage.
            "target_pose_convention": "sparsetrack_mujoco_opengl",
            "target_extra_Rz180": False,
            "alternate_diagnostic": "apply_Rz_180_only_for_geometry_diagnostics_never_silent",
        },
        "depth_processing": {
            "registration": "calibrated_zbuffer_depth_aware_splat",
            "confidence_source": "sensor" if depth_sensor_confidence is not None else "derived_valid_range_gradient",
            "invalid_edge_artifact": str(edge_invalid_dir),
            "splat_device": splat_device,
            "batched_registration": True,
            "pinhole_gpu": bool(
                str(splat_device).startswith("cuda")
                and distortion_is_zero(depth_D)
                and distortion_is_zero(rgb_D)
            ),
        },
        "calibration_status": "complete" if calibration_complete else "incomplete_legacy_debug",
        "timestamp_status": timestamp_status,
        "timing": {
            "rgb_depth_offset_ms": rgb_depth_offsets_ms.tolist(),
            "rgb_camera_offset_ms": rgb_camera_offsets_ms.tolist(),
            "p95_abs_rgb_depth_ms": float(np.percentile(np.abs(rgb_depth_offsets_ms), 95)) if n else None,
            "max_abs_rgb_depth_ms": float(np.max(np.abs(rgb_depth_offsets_ms))) if n else None,
            "measured": timing_complete,
            "rgb_frame_lock": {
                "declared": rgb_frame_lock_declared,
                "timestamp_status": rgb_timestamp_status,
                "check": frame_lock_check,
            },
        },
    }
    (out / "alignment_manifest.json").write_text(json.dumps(alignment_manifest, indent=2) + "\n")
    # Adapter-level output intentionally has no target-camera metrics. The
    # strict target geometry gate runs in viewport_gt_depth_warp before Stage 3;
    # do not present unavailable target checks here as a failed source adapter.
    gate = evaluate_geometry_gate(
        frame_metrics,
        calibration_complete=calibration_complete,
        timing_available=timing_complete,
        mode="legacy_debug",
        rz_alternate={"tested": False, "reason": "target-camera pose is evaluated by viewport stage"},
    )
    if calibration_complete and timing_complete:
        gate["status"] = "deferred"
        gate["mode"] = "deferred_to_target_viewport"
        gate["deferred_checks"] = [
            "target_front_ratio",
            "in_bounds_ratio",
            "reprojection_p50_px",
            "reprojection_p95_px",
            "depth_relative_p95",
        ]
        gate["failed_checks"] = []
    gate["calibration_status"] = alignment_manifest["calibration_status"]
    gate["timestamp_status"] = alignment_manifest["timestamp_status"]
    write_gate_report(out / "quality" / "geometry_gate.json", gate)
    quality_dir = out / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    (quality_dir / "frame_metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in frame_metrics)
    )
    print(f"[adapter] wrote {h5_path} n={n} depth→m @ {W}x{H} splat={splat_device}", flush=True)
    if proj_errs:
        e0 = proj_errs[0]["err"]
        print(f"[adapter] left wrist proj err sample dx={e0[0]:.1f} dy={e0[1]:.1f}", flush=True)


if __name__ == "__main__":
    main()
