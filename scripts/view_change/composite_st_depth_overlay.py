#!/usr/bin/env python3
"""Composite SparseTrack RGB onto novel-view / clean final via depth z-buffer.

Default (legacy) depth rules:
  - DA3 on final, scale-aligned to MuJoCo with median(d_st / d_da3) on non-hole pixels.
  - Hole pixels: d_nv = ``--hole-depth`` (large but finite).
  - MuJoCo non-robot pixels: d_st = ``--st-bg-depth``.
  - Robot/hand keeps real MuJoCo depth; paste if d_st < d_nv.

Metric mode (``--no-scale --no-hole``):
  - Use DA3 depth as-is (no median scale) and MuJoCo robot depth as-is.
  - No hole override; z-buffer is pure closer-wins on real depths.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_ROOT = PROJECT_ROOT / "resources"
DATA_ROOT = PROJECT_ROOT / "data"

try:
    from frame_volume import (
        FrameVolume,
        as_mask,
        as_rgb,
        create_volume,
        is_packed_stream,
        open_volume,
    )
    from paste_mode import paste_mask_mode_ab
    from quality_report import (
        PROVENANCE_FUSED,
        PROVENANCE_INVALID,
        PROVENANCE_OBSERVED,
        PROVENANCE_ROBOT,
    )
except ImportError:  # pragma: no cover - standalone invocation fallback
    from .frame_volume import (
        FrameVolume,
        as_mask,
        as_rgb,
        create_volume,
        is_packed_stream,
        open_volume,
    )
    from .paste_mode import paste_mask_mode_ab
    PROVENANCE_INVALID = np.uint8(0)
    PROVENANCE_OBSERVED = np.uint8(1)
    PROVENANCE_FUSED = np.uint8(2)
    PROVENANCE_ROBOT = np.uint8(3)


def _json_socket_request(
    sock_path: str,
    payload: dict,
    *,
    timeout_s: float = 7200.0,
) -> dict:
    encoded = (json.dumps(payload) + "\n").encode("utf-8")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(float(timeout_s))
        sock.connect(sock_path)
        sock.sendall(encoded)
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError("DA3 worker closed the connection")
            buf += chunk
        reply = json.loads(buf.split(b"\n", 1)[0].decode("utf-8"))
        if not isinstance(reply, dict):
            raise ValueError("DA3 worker reply must be a JSON object")
        return reply
    finally:
        sock.close()


def _da3_worker_infer_volume(
    *,
    rgb_dir: Path,
    out_dir: Path,
    process_res: int,
    fx: float | None,
    fy: float | None,
    metric_focal_div: float | None,
    start: int,
    end: int,
) -> dict:
    sock_path = str(os.environ.get("DA3_WORKER_SOCK", "")).strip()
    if not sock_path:
        raise RuntimeError("DA3_WORKER_SOCK is empty")
    if not os.path.exists(sock_path):
        raise FileNotFoundError(f"DA3 worker socket is missing: {sock_path}")
    print(f"[composite] using persistent DA3 worker {sock_path}", flush=True)
    reply = _json_socket_request(
        sock_path,
        {
            "op": "infer_volume",
            "rgb_dir": str(Path(rgb_dir).resolve()),
            "out_dir": str(Path(out_dir).resolve()),
            "process_res": int(process_res),
            "fx": None if fx is None else float(fx),
            "fy": None if fy is None else float(fy),
            "metric_focal_div": (
                None if metric_focal_div is None else float(metric_focal_div)
            ),
            "start": int(start),
            "end": int(end),
        },
    )
    if not reply.get("ok"):
        raise RuntimeError(reply.get("error") or "DA3 worker infer_volume failed")
    return reply


def _stub_da3_optional_deps() -> None:
    """Avoid hard deps (moviepy/gsplat/pycolmap) pulled by DA3 export utils."""
    import types

    stubs = {
        "moviepy": types.ModuleType("moviepy"),
        "moviepy.editor": types.ModuleType("moviepy.editor"),
        "pycolmap": types.ModuleType("pycolmap"),
        "gsplat": types.ModuleType("gsplat"),
        "plyfile": types.ModuleType("plyfile"),
    }
    stubs["moviepy.editor"].VideoFileClip = object
    stubs["moviepy.editor"].ImageSequenceClip = object
    stubs["plyfile"].PlyData = object
    stubs["plyfile"].PlyElement = object
    for name, mod in stubs.items():
        sys.modules.setdefault(name, mod)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--final-mp4",
        type=Path,
        default=DATA_ROOT / "view_change/legacy/final_0.mp4",
    )
    p.add_argument(
        "--st-rgb-dir",
        type=Path,
        default=DATA_ROOT / "view_change/legacy/robot_rgb",
    )
    p.add_argument(
        "--st-depth-dir",
        type=Path,
        default=DATA_ROOT / "view_change/legacy/robot_depth",
    )
    p.add_argument(
        "--hole-masks",
        type=Path,
        default=DATA_ROOT / "view_change/legacy/hole_masks",
        help="Optional hole / hand masks. Ignored when --no-hole.",
    )
    p.add_argument(
        "--no-hole",
        action="store_true",
        help="Do not override any pixel with --hole-depth; use DA3 depth everywhere for bg.",
    )
    p.add_argument(
        "--no-scale",
        action="store_true",
        help="Do not median-scale DA3 to MuJoCo; treat DA3 depth as metric meters.",
    )
    p.add_argument(
        "--depth-vis-out",
        type=Path,
        default=None,
        help="Optional side-by-side depth viz mp4 of z-buffer inputs (MuJoCo|DA3).",
    )
    p.add_argument(
        "--depth-vis-vmin",
        type=float,
        default=0.024,
        help="Near clip for depth viz colormap (meters).",
    )
    p.add_argument(
        "--depth-vis-vmax",
        type=float,
        default=0.37,
        help="Far clip for depth viz colormap (meters).",
    )
    p.add_argument(
        "--da3-model-dir",
        type=str,
        default=str(
            Path(os.environ.get("KODA_ASSETS", str(PROJECT_ROOT / "assets")))
            / "models/depth_anything_3"
        ),
        help="Local snapshot path or HF id (e.g. depth-anything/DA3METRIC-LARGE)",
    )
    p.add_argument(
        "--da3-src",
        type=Path,
        default=RESOURCE_ROOT / "Depth-Anything-3/src",
        help="Add this to sys.path for depth_anything_3 imports",
    )
    p.add_argument(
        "--metric-focal-div",
        type=float,
        default=300.0,
        help=(
            "DA3METRIC-LARGE: d_m = d_raw * ((fx+fy)/2) / this, using process-res focal. "
            "Set <=0 to disable (for DA3-BASE relative depth)."
        ),
    )
    p.add_argument(
        "--camera-k-json",
        type=Path,
        default=None,
        help=(
            "Optional camera_K.json with fx/fy (and optional width/height). "
            "If omitted with --metric-focal-div>0, uses EgoDex native fx=fy from "
            "cx≈W/2 style K scaled to the clean video resolution."
        ),
    )
    p.add_argument(
        "--reuse-da3-dir",
        type=Path,
        default=None,
        help="If set, load per-frame DA3 depth from this dir (XXXXXX.npy) instead of running DA3",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional preview mp4. Formal path writes packed --rgb-out-dir only.",
    )
    p.add_argument("--dump-dir", type=Path, default=None, help="Optional dump for depths/masks")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--process-res", type=int, default=504)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--fps", type=float, default=30.0)
    # Depth policy
    p.add_argument(
        "--robot-max-depth",
        type=float,
        default=1.0,
        help="MuJoCo pixels with depth<=this (m) count as robot/hand; rest treated as background",
    )
    p.add_argument(
        "--robot-mask-dir",
        type=Path,
        default=None,
        help="Optional per-frame robot/object-id masks (PNG; any non-zero value is robot). Overrides depth threshold.",
    )
    p.add_argument(
        "--robot-mask-id-dir",
        type=Path,
        default=None,
        help="Alias for --robot-mask-dir, useful for MuJoCo object-id masks.",
    )
    p.add_argument(
        "--st-bg-depth",
        type=float,
        default=1e9,
        help="Assigned depth for non-robot MuJoCo pixels (must be > --hole-depth)",
    )
    p.add_argument(
        "--hole-depth",
        type=float,
        default=1e6,
        help="Assigned d_nv for original novel-view hole pixels (large, but < --st-bg-depth)",
    )
    p.add_argument(
        "--bg-depth-dir",
        type=Path,
        default=None,
        help=(
            "Optional metric background depth dir (XXXXXX.npy), e.g. warped GT depth. "
            "When set, skips DA3 and uses these depths as d_bg (meters)."
        ),
    )
    p.add_argument(
        "--paste-mode",
        type=str,
        default="legacy",
        choices=["legacy", "A", "B"],
        help=(
            "A/B hole paste (requires --bg-depth-dir): "
            "A=in_hole & d_rob valid → paste; "
            "B=also d_rob < neighborhood d_bg. "
            "Non-hole always metric z-buffer d_rob < d_bg. "
            "legacy=previous DA3/hole-depth behaviour."
        ),
    )
    p.add_argument(
        "--neighbor-radius",
        type=int,
        default=3,
        help="Paste mode B: radius (px) for valid d_bg neighborhood median around holes.",
    )
    p.add_argument(
        "--paste-chunk",
        type=int,
        default=16,
        help="Batched paste-mode A/B frames per kernel when --bg-depth-dir is set.",
    )
    p.add_argument(
        "--final-rgb-dir",
        type=Path,
        default=None,
        help="Optional RGB frame dir instead of --final-mp4 (XXXXXX.png).",
    )
    p.add_argument(
        "--rgb-out-dir",
        "--out-rgb-dir",
        type=Path,
        default=None,
        help="Optional directory for composited RGB frames (XXXXXX.png).",
    )
    p.add_argument(
        "--geometry-provenance-dir",
        "--input-provenance-dir",
        type=Path,
        default=None,
        help=(
            "Optional geometry provenance PNGs (0 invalid, 1 observed, 2 fused). "
            "Non-hole pixels default to observed when omitted."
        ),
    )
    p.add_argument(
        "--robot-ids",
        type=str,
        default=None,
        help="Comma-separated positive IDs to treat as robot in an object-id mask; default is all nonzero IDs.",
    )
    p.add_argument(
        "--timeline-manifest",
        type=Path,
        default=None,
        help=(
            "Measured 30 Hz -> 50 Hz timeline. When supplied, source frame i "
            "reads robot RGB/depth/ID frame source_to_target[i]."
        ),
    )
    p.add_argument(
        "--require-robot-id",
        action="store_true",
        help="Fail instead of silently using the legacy robot_max_depth heuristic.",
    )
    p.add_argument(
        "--robot-instance-json",
        type=Path,
        default=None,
        help="Optional robot_instance.json to validate alongside the ID directory.",
    )
    return p.parse_args()


def list_indexed(dir_path: Path, suffix: str) -> list[Path]:
    files = sorted(
        dir_path.glob(f"*{suffix}"),
        key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem),
    )
    if not files:
        raise FileNotFoundError(f"No *{suffix} in {dir_path}")
    return files


def _load_source_to_target_map(path: Path | None) -> tuple[np.ndarray | None, dict[str, object]]:
    """Load the formal source->nearest-D435 map, preserving its audit trail."""
    if path is None:
        return None, {"policy": "index_aligned_legacy"}
    if not path.is_file():
        raise FileNotFoundError(f"timeline manifest not found: {path}")
    payload = json.loads(path.read_text())
    try:
        values = np.asarray(payload["source_to_target"]["target_indices"], dtype=np.int64)
        source_count = int(payload["source"]["frame_count"])
        target_count = int(payload["target"]["frame_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid source_to_target timeline contract: {path}") from exc
    if values.ndim != 1 or len(values) != source_count:
        raise ValueError(
            f"timeline source_to_target length {len(values)} != source count {source_count}"
        )
    if np.any((values < 0) | (values >= target_count)):
        raise ValueError("timeline source_to_target contains an out-of-range target index")
    if np.any(np.diff(values) < 0):
        # A nearest map should be monotonic for monotonic clocks. Rejecting a
        # malformed manifest is preferable to mixing robot bundles.
        raise ValueError("timeline source_to_target map is not monotonic")
    return values, {
        "policy": payload.get("source_to_target", {}).get("policy", "absolute_nearest_tie_earlier"),
        "manifest": str(path.resolve()),
        "source_count": source_count,
        "target_count": target_count,
    }


def _load_robot_instance_contract(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"robot instance manifest not found: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"robot instance manifest must be an object: {path}")
    if not bool(payload.get("formal_dataset_allowed", False)):
        raise ValueError("robot instance manifest is not marked formal_dataset_allowed")
    if payload.get("legacy_robot_mask", False):
        raise ValueError("legacy robot mask cannot be used for formal compositing")
    return payload


def _parse_ids(spec: str | None) -> set[int] | None:
    """Parse an optional object-id allowlist, rejecting malformed values."""
    if spec is None or not spec.strip():
        return None
    ids: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token, 0)
        except ValueError as exc:
            raise SystemExit(f"invalid --robot-ids token: {token!r}") from exc
        if value <= 0:
            raise SystemExit("--robot-ids values must be positive")
        ids.add(value)
    return ids or None


def _load_robot_mask_array(
    raw: np.ndarray,
    shape: tuple[int, int],
    ids: set[int] | None = None,
) -> np.ndarray:
    raw = np.asarray(raw)
    if raw.ndim == 3:
        raw = raw.max(axis=2)
    if raw.shape[:2] != shape:
        raw = cv2.resize(raw, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    if ids is None:
        return np.asarray(raw > 0, dtype=bool)
    return np.isin(np.asarray(raw, dtype=np.int64), np.fromiter(ids, dtype=np.int64))


def _load_robot_mask(
    path: Path,
    shape: tuple[int, int],
    ids: set[int] | None = None,
) -> np.ndarray:
    """Load an 8/16-bit MuJoCo object-id mask without truncating IDs."""
    if path.suffix.lower() == ".npy":
        raw = np.asarray(np.load(path))
    else:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"failed to read robot mask {path}")
    return _load_robot_mask_array(raw, shape, ids)


def _indexed_mask_files(dir_path: Path) -> list[Path]:
    """List PNG/NPY masks in one deterministic frame order."""
    files = sorted(
        [*dir_path.glob("*.png"), *dir_path.glob("*.npy")],
        key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem),
    )
    if not files:
        raise FileNotFoundError(f"No *.png or *.npy masks in {dir_path}")
    return files


def load_da3(model_dir: str | Path, da3_src: Path, device: str):
    """Load DA3. Prefer conda env ``da3`` where deps are installed."""
    src = str(da3_src.resolve())
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from depth_anything_3.api import DepthAnything3
    except Exception:
        _stub_da3_optional_deps()
        import types

        export_pkg = types.ModuleType("depth_anything_3.utils.export")
        export_pkg.export = lambda *a, **k: None  # type: ignore[attr-defined]
        sys.modules["depth_anything_3.utils.export"] = export_pkg
        for sub in ("gs", "colmap", "glb", "npz", "depth_vis", "feat_vis"):
            sys.modules.setdefault(
                f"depth_anything_3.utils.export.{sub}", types.ModuleType(sub)
            )
        pose_align = types.ModuleType("depth_anything_3.utils.pose_align")
        pose_align.align_poses_umeyama = lambda *a, **k: None  # type: ignore[attr-defined]
        sys.modules["depth_anything_3.utils.pose_align"] = pose_align
        sys.modules.setdefault("evo", types.ModuleType("evo"))
        sys.modules.setdefault("evo.core", types.ModuleType("evo.core"))
        traj = types.ModuleType("evo.core.trajectory")
        traj.PosePath3D = object  # type: ignore[attr-defined]
        sys.modules.setdefault("evo.core.trajectory", traj)
        from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(str(model_dir))
    model = model.to(device)
    model.eval()
    return model


def estimate_scale(d_da3: np.ndarray, d_st: np.ndarray, hole: np.ndarray) -> float | None:
    """Median scale s so that s * d_da3 ≈ d_st on non-hole valid pixels."""
    valid = (
        (~hole)
        & np.isfinite(d_da3)
        & np.isfinite(d_st)
        & (d_da3 > 1e-6)
        & (d_st > 1e-6)
        & (d_st < 1e5)
    )
    if valid.sum() < 64:
        return None
    ratios = d_st[valid] / d_da3[valid]
    return float(np.median(ratios))


def resize_depth(depth: np.ndarray, h: int, w: int) -> np.ndarray:
    if depth.shape[:2] == (h, w):
        return depth.astype(np.float32)
    return cv2.resize(depth.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def _load_fx_fy(k_json: Path | None, frame_w: int, frame_h: int) -> tuple[float, float]:
    """Return fx, fy in pixels for the given frame size."""
    if k_json is not None and k_json.is_file():
        meta = json.loads(k_json.read_text())
        fx = float(meta.get("fx", meta["K"][0][0]))
        fy = float(meta.get("fy", meta["K"][1][1]))
        kw = int(meta.get("width", meta.get("native_width", frame_w)))
        kh = int(meta.get("height", meta.get("native_height", frame_h)))
        if kw != frame_w or kh != frame_h:
            fx *= frame_w / float(kw)
            fy *= frame_h / float(kh)
        return fx, fy
    # Fallback: EgoDex declutter trim native 1920x1080 fx=fy≈736.63
    fx0, fy0, w0, h0 = 736.6339111328125, 736.6339111328125, 1920, 1080
    return fx0 * (frame_w / w0), fy0 * (frame_h / h0)


@torch.inference_mode()
def da3_depth(
    model,
    image_rgb: np.ndarray,
    process_res: int,
    *,
    fx: float | None = None,
    fy: float | None = None,
    metric_focal_div: float | None = None,
) -> np.ndarray:
    pred = model.inference([image_rgb], process_res=process_res)
    d = np.asarray(pred.depth[0], dtype=np.float32)
    if metric_focal_div is not None and metric_focal_div > 0:
        if fx is None or fy is None:
            raise ValueError("metric_focal_div requires fx/fy")
        h_full, w_full = image_rgb.shape[:2]
        h_p, w_p = d.shape[:2]
        fx_p = float(fx) * (w_p / float(w_full))
        fy_p = float(fy) * (h_p / float(h_full))
        focal = 0.5 * (fx_p + fy_p)
        d = (d * (focal / float(metric_focal_div))).astype(np.float32)
    return d


def _colorize_depth(d: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    try:
        import matplotlib

        cmap = matplotlib.colormaps["turbo"]
    except Exception:
        cmap = None
    x = np.clip((d.astype(np.float64) - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    far = ~np.isfinite(d) | (d >= 5.0) | (d <= 0)
    if cmap is not None:
        rgb = (cmap(x)[..., :3] * 255).astype(np.uint8)
    else:
        rgb = (np.stack([x, x, x], axis=-1) * 255).astype(np.uint8)
    rgb[far] = 0
    return rgb


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 12 + 9 * len(text)), 28), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _neighbor_dbg_median(
    d_bg: np.ndarray,
    hole: np.ndarray,
    radius: int,
    device: str | None = None,
) -> np.ndarray:
    """For hole pixels, median of valid non-hole d_bg in a local window.

    Accepts ``HxW`` or batched ``BxHxW``. Only hole pixels are reduced; the
    unfold still covers the image, but ``nanquantile`` runs on the hole subset
    and on CUDA when available.
    """
    d_bg = np.asarray(d_bg, dtype=np.float32)
    hole = np.asarray(hole, dtype=bool)
    batched = d_bg.ndim == 3
    if d_bg.ndim == 2:
        d_bg = d_bg[None]
        hole = hole[None]
    if d_bg.ndim != 3 or hole.shape != d_bg.shape:
        raise ValueError(f"d_bg/hole must be HxW or BxHxW, got {d_bg.shape}/{hole.shape}")
    batch, h, w = d_bg.shape
    out = np.full((batch, h, w), np.nan, dtype=np.float32)
    r = max(1, int(radius))
    k = 2 * r + 1
    valid = np.isfinite(d_bg) & (d_bg > 1e-6) & (d_bg < 1e5) & (~hole)
    if not np.any(hole):
        return out[0] if not batched else out
    requested = "cpu" if device is None else str(device)
    if requested.startswith("cuda") and torch.cuda.is_available():
        torch_device = torch.device(requested if requested != "cuda" else "cuda")
    elif device is None and torch.cuda.is_available():
        torch_device = torch.device("cuda")
    else:
        torch_device = torch.device("cpu")
    try:
        filled = np.where(valid, d_bg, np.nan).astype(np.float32, copy=False)
        t = torch.from_numpy(np.ascontiguousarray(filled)).to(torch_device)[:, None]
        hole_t = torch.from_numpy(np.ascontiguousarray(hole)).to(torch_device)
        t = torch.nn.functional.pad(t, (r, r, r, r), value=float("nan"))
        win = torch.nn.functional.unfold(t, kernel_size=k)
        vals = win.permute(0, 2, 1)
        hole_vals = vals[hole_t.reshape(batch, -1)]
        if hole_vals.numel() == 0:
            return out[0] if not batched else out
        try:
            med = torch.nanquantile(hole_vals, 0.5, dim=1)
        except (AttributeError, RuntimeError):  # older torch
            finite_count = torch.isfinite(hole_vals).sum(dim=1)
            ordered = torch.sort(hole_vals, dim=1).values
            lo = torch.clamp((finite_count - 1) // 2, min=0)
            hi = torch.clamp(finite_count // 2, min=0)
            med = 0.5 * (
                ordered.gather(1, lo[:, None]).squeeze(1)
                + ordered.gather(1, hi[:, None]).squeeze(1)
            )
            med[finite_count == 0] = float("nan")
        out[hole] = med.detach().cpu().numpy().astype(np.float32, copy=False)
    except Exception:
        for b_index in range(batch):
            rows, cols = np.where(hole[b_index])
            for y, x in zip(rows, cols):
                y0, y1 = max(0, y - r), min(h, y + r + 1)
                x0, x1 = max(0, x - r), min(w, x + r + 1)
                patch = d_bg[b_index, y0:y1, x0:x1]
                mask = valid[b_index, y0:y1, x0:x1]
                if mask.any():
                    out[b_index, y, x] = float(np.median(patch[mask]))
    out[~hole] = np.nan
    return out[0] if not batched else out


def _read_rgb_png(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class _Cv2VideoReader:
    """Small imageio-compatible reader used when imageio-ffmpeg is absent."""

    def __init__(self, path: Path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"failed to open video {path}")

    def get_meta_data(self) -> dict[str, float]:
        return {
            "fps": float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0),
            "nframes": float(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0),
        }

    def count_frames(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def get_data(self, index: int) -> np.ndarray:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, bgr = self.cap.read()
        if not ok or bgr is None:
            raise IndexError(f"video has no frame {index}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.cap.release()


class _Cv2VideoWriter:
    def __init__(self, path: Path, fps: float):
        self.path = path
        self.fps = float(fps)
        self.writer = None

    def append_data(self, frame: np.ndarray) -> None:
        rgb = np.asarray(frame)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError(f"video frame must be RGB, got {rgb.shape}")
        if self.writer is None:
            h, w = rgb.shape[:2]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (w, h))
            if not self.writer.isOpened():
                raise RuntimeError(f"failed to create video {self.path}")
        self.writer.write(cv2.cvtColor(rgb[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()


def _open_video_reader(path: Path):
    try:
        candidate = imageio.get_reader(str(path))
        # imageio 2.33 may select tifffile for MP4 when imageio-ffmpeg is
        # unavailable; it cannot provide video semantics.
        plugin = getattr(candidate, "instance", None)
        if plugin is not None and "tifffile" in type(plugin).__module__.lower():
            candidate.close()
            raise RuntimeError("imageio selected tifffile for video")
        return candidate
    except Exception:
        return _Cv2VideoReader(path)


def _open_video_writer(path: Path, fps: float, *, quality: int = 8):
    try:
        candidate = imageio.get_writer(
            str(path),
            fps=fps,
            codec="libx264",
            quality=quality,
            macro_block_size=1,
            pixelformat="yuv420p",
        )
        plugin = getattr(candidate, "instance", None)
        if plugin is not None and "tifffile" in type(plugin).__module__.lower():
            candidate.close()
            raise RuntimeError("imageio selected tifffile for video")
        return candidate
    except Exception:
        return _Cv2VideoWriter(path, fps)


def _open_final_source(args: argparse.Namespace):
    """Open final RGB lazily, avoiding a full-episode frame list."""
    if args.final_rgb_dir is not None:
        try:
            return open_volume(args.final_rgb_dir, color_space="rgb"), float(args.fps or 30.0)
        except FileNotFoundError:
            paths = list_indexed(args.final_rgb_dir, ".png")
            return paths, float(args.fps or 30.0)
    reader = _open_video_reader(args.final_mp4)
    meta = reader.get_meta_data()
    return reader, float(args.fps or meta.get("fps") or 30.0)


def _final_source_length(source, fallback: int) -> int:
    if isinstance(source, (list, FrameVolume)):
        return len(source)
    try:
        count = int(source.count_frames())
        if count > 0:
            return count
    except Exception:
        pass
    return int(fallback)


def _get_final_frame(source, index: int) -> np.ndarray:
    if isinstance(source, FrameVolume):
        return as_rgb(source[index])
    if isinstance(source, list):
        return _read_rgb_png(source[index])
    return np.asarray(source.get_data(index))


def _close_final_source(source) -> None:
    if isinstance(source, FrameVolume):
        source.flush()
        return
    if not isinstance(source, list):
        source.close()


def main() -> None:
    args = parse_args()
    use_metric_ab = args.paste_mode in ("A", "B")
    if use_metric_ab and args.bg_depth_dir is None:
        raise SystemExit("--paste-mode A/B requires --bg-depth-dir")
    if not use_metric_ab and not args.no_hole and not (args.hole_depth < args.st_bg_depth):
        raise SystemExit(
            f"--hole-depth ({args.hole_depth}) must be < --st-bg-depth ({args.st_bg_depth})"
        )

    st_rgb_vol = open_volume(args.st_rgb_dir, color_space="rgb")
    st_depth_vol = open_volume(args.st_depth_dir)
    robot_mask_root = args.robot_mask_dir or args.robot_mask_id_dir
    if args.require_robot_id and robot_mask_root is None:
        raise SystemExit("--require-robot-id requires --robot-mask-dir/--robot-mask-id-dir")
    robot_id_vol: FrameVolume | None = None
    if robot_mask_root is not None:
        robot_id_vol = open_volume(robot_mask_root)
    robot_instance_contract = _load_robot_instance_contract(args.robot_instance_json)
    if args.require_robot_id and robot_instance_contract is None:
        raise SystemExit("--require-robot-id also requires --robot-instance-json")
    source_to_target, timeline_contract = _load_source_to_target_map(args.timeline_manifest)
    robot_ids = _parse_ids(args.robot_ids)
    hole_vol: FrameVolume | None
    if args.no_hole and not use_metric_ab:
        hole_vol = None
    else:
        hole_vol = open_volume(args.hole_masks) if args.hole_masks is not None else None
        if use_metric_ab and hole_vol is None:
            raise SystemExit("paste-mode A/B requires --hole-masks")

    if args.final_rgb_dir is None and args.final_mp4 is None:
        raise SystemExit("one of --final-mp4 or --final-rgb-dir is required")
    final_source, fps = _open_final_source(args)
    geometry_count = _final_source_length(final_source, len(hole_vol) if hole_vol is not None else len(st_rgb_vol))
    n = geometry_count
    if hole_vol is not None and len(hole_vol) < n:
        raise SystemExit(f"frame count mismatch final={n} holes={len(hole_vol)}")

    bg_vol: FrameVolume | None = None
    if args.bg_depth_dir is not None:
        bg_vol = open_volume(args.bg_depth_dir)
        if len(bg_vol) < n:
            raise SystemExit(f"bg-depth-dir has {len(bg_vol)} < final {n}")
    geom_prov_vol: FrameVolume | None = None
    if args.geometry_provenance_dir is not None:
        geom_prov_vol = open_volume(args.geometry_provenance_dir)
        if len(geom_prov_vol) < n:
            raise SystemExit(
                f"geometry provenance dir has {len(geom_prov_vol)} < final {n}"
            )

    start = max(0, args.start_frame)
    end = n if args.max_frames is None else min(n, start + args.max_frames)
    if start >= end:
        raise SystemExit(f"empty range [{start}, {end})")
    if source_to_target is not None and end > len(source_to_target):
        raise SystemExit(
            f"timeline has {len(source_to_target)} source frames but compositor needs {end}"
        )
    selected_target_indices = (
        source_to_target[start:end]
        if source_to_target is not None
        else np.arange(start, end, dtype=np.int64)
    )
    if selected_target_indices.size:
        max_target = int(np.max(selected_target_indices))
        for label, files in (
            ("robot RGB", st_rgb_vol),
            ("robot depth", st_depth_vol),
            ("robot ID", robot_id_vol),
        ):
            if files is not None and max_target >= len(files):
                raise SystemExit(
                    f"{label} has {len(files)} frames but mapping selects target {max_target}"
                )

    dump_dir = args.dump_dir
    if dump_dir is None:
        if args.rgb_out_dir is not None:
            dump_dir = args.rgb_out_dir.parent / "composite_debug"
        elif args.out is not None:
            dump_dir = args.out.parent / f"{args.out.stem}_debug"
        else:
            raise SystemExit("--dump-dir or --rgb-out-dir is required when --out is omitted")
    da3_dir = dump_dir / "da3_depth"
    paste_dir = dump_dir / "paste_mask"
    robot_dir = dump_dir / "robot_mask"
    provenance_dir = dump_dir / "provenance"

    model = None
    reuse_da3_vol = None
    worker_da3 = False
    metric_div = (
        float(args.metric_focal_div)
        if args.metric_focal_div is not None and args.metric_focal_div > 0
        else None
    )
    first_rgb = _get_final_frame(final_source, 0)
    if first_rgb.ndim != 3 or first_rgb.shape[2] < 3:
        _close_final_source(final_source)
        raise SystemExit("final RGB source must contain at least three channels")
    h0, w0 = first_rgb.shape[:2]
    fx, fy = _load_fx_fy(args.camera_k_json, w0, h0)
    da3_sock = str(os.environ.get("DA3_WORKER_SOCK", "")).strip()
    if bg_vol is not None:
        print(f"Using metric bg depth from {args.bg_depth_dir} paste_mode={args.paste_mode}", flush=True)
    elif args.reuse_da3_dir is not None:
        print(f"Reusing DA3 depths from {args.reuse_da3_dir}", flush=True)
        if is_packed_stream(args.reuse_da3_dir):
            reuse_da3_vol = open_volume(args.reuse_da3_dir)
    elif (
        da3_sock
        and args.final_rgb_dir is not None
        and os.path.exists(da3_sock)
    ):
        try:
            _da3_worker_infer_volume(
                rgb_dir=args.final_rgb_dir,
                out_dir=da3_dir,
                process_res=int(args.process_res),
                fx=float(fx),
                fy=float(fy),
                metric_focal_div=metric_div,
                start=int(start),
                end=int(end),
            )
            worker_da3 = True
        except Exception as error:
            print(
                f"[composite] DA3 worker failed ({error}); loading in-process",
                flush=True,
            )
            device = args.device
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
            print(
                f"Loading DA3 from {args.da3_model_dir} on {device} "
                f"metric_focal_div={metric_div} fx={fx:.3f} fy={fy:.3f} frame={w0}x{h0}",
                flush=True,
            )
            model = load_da3(args.da3_model_dir, args.da3_src, device)
    else:
        device = args.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        print(
            f"Loading DA3 from {args.da3_model_dir} on {device} "
            f"metric_focal_div={metric_div} fx={fx:.3f} fy={fy:.3f} frame={w0}x{h0}",
            flush=True,
        )
        model = load_da3(args.da3_model_dir, args.da3_src, device)

    writer = None
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        writer = _open_video_writer(args.out, fps, quality=8)
    rgb_out_dir = args.rgb_out_dir
    rgb_out_vol = None
    if rgb_out_dir is None and args.out is None:
        raise SystemExit("one of --out or --rgb-out-dir is required")
    if rgb_out_dir is not None:
        rgb_out_vol = create_volume(
            rgb_out_dir, n, (h0, w0, 3), np.uint8, fill=0, color_space="rgb"
        )
    if worker_da3:
        da3_vol = open_volume(da3_dir, writable=True)
    else:
        da3_vol = None
    paste_vol = create_volume(paste_dir, n, (h0, w0), np.uint8, fill=0)
    robot_mask_vol = create_volume(robot_dir, n, (h0, w0), np.uint8, fill=0)
    provenance_vol = create_volume(provenance_dir, n, (h0, w0), np.uint8, fill=0)

    scales: list[dict] = []
    last_scale: float | None = None
    # h0, w0 already set above from first frame

    depth_vis_writer = None
    if args.depth_vis_out is not None:
        args.depth_vis_out.parent.mkdir(parents=True, exist_ok=True)
        depth_vis_writer = _open_video_writer(args.depth_vis_out, fps, quality=7)

    print(
        f"policy: paste_mode={args.paste_mode} no_scale={args.no_scale} no_hole={args.no_hole} "
        f"robot_max_depth={args.robot_max_depth} "
        f"hole_depth={args.hole_depth} st_bg_depth={args.st_bg_depth}",
        flush=True,
    )

    paste_chunk = max(1, int(getattr(args, "paste_chunk", 8) or 8))
    paste_device = args.device if str(args.device).startswith("cuda") else "cpu"

    def _load_composite_geometry(i: int) -> dict:
        target_i = int(source_to_target[i]) if source_to_target is not None else i
        final_rgb = _get_final_frame(final_source, i)
        if final_rgb.ndim != 3 or final_rgb.shape[2] < 3:
            raise RuntimeError(f"final frame {i} is not an RGB image: {final_rgb.shape}")
        if final_rgb.shape[2] >= 4:
            final_rgb = final_rgb[:, :, :3]
        st_rgb_i = as_rgb(st_rgb_vol[target_i])
        if st_rgb_i.shape[:2] != (h0, w0):
            st_rgb_i = cv2.resize(st_rgb_i, (w0, h0), interpolation=cv2.INTER_LINEAR)
        d_st_raw = np.asarray(st_depth_vol[target_i], dtype=np.float32).squeeze()
        if d_st_raw.ndim != 2:
            raise RuntimeError(
                f"SparseTrack depth target frame {target_i} for source {i} must be HxW, "
                f"got {d_st_raw.shape}"
            )
        if d_st_raw.shape[:2] != (h0, w0):
            d_st_raw = resize_depth(d_st_raw, h0, w0)
        valid_st = np.isfinite(d_st_raw) & (d_st_raw > 1e-6)
        if robot_id_vol is not None:
            if target_i >= len(robot_id_vol):
                raise SystemExit(
                    f"robot mask dir has {len(robot_id_vol)} < target frame {target_i}"
                )
            robot = _load_robot_mask_array(robot_id_vol[target_i], (h0, w0), robot_ids) & valid_st
        else:
            robot = valid_st & (d_st_raw <= args.robot_max_depth)
        if hole_vol is None:
            hole_bool = np.zeros((h0, w0), dtype=bool)
        else:
            hole_bool = as_mask(hole_vol[i])
            if hole_bool.shape[:2] != (h0, w0):
                hole_bool = cv2.resize(
                    hole_bool.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST
                ) > 0
        d_bg = None
        if bg_vol is not None:
            d_bg = np.asarray(bg_vol[i], dtype=np.float32).squeeze()
            if d_bg.ndim != 2:
                raise RuntimeError(f"background depth frame {i} must be HxW, got {d_bg.shape}")
            d_bg = resize_depth(d_bg, h0, w0)
        return {
            "i": i,
            "target_i": target_i,
            "final_rgb": final_rgb,
            "st_rgb_i": st_rgb_i,
            "d_st_raw": d_st_raw,
            "robot": robot,
            "hole_bool": hole_bool,
            "d_bg": d_bg,
        }

    def _emit_composite_frame(
        item: dict,
        *,
        use_st: np.ndarray,
        d_nv: np.ndarray,
        d_rel: np.ndarray,
        used_scale,
    ) -> None:
        i = int(item["i"])
        robot = item["robot"]
        hole_bool = item["hole_bool"]
        d_st_raw = item["d_st_raw"]
        out = np.where(use_st[..., None], item["st_rgb_i"], item["final_rgb"]).astype(np.uint8)
        if writer is not None:
            writer.append_data(out)
        if rgb_out_vol is not None:
            rgb_out_vol[i] = out
        if depth_vis_writer is not None:
            d_mj_vis = np.where(robot, d_st_raw, np.nan).astype(np.float32)
            d_da3_vis = d_nv.copy()
            d_da3_vis[~np.isfinite(d_da3_vis) | (d_da3_vis > 5.0)] = np.nan
            c_mj = _label(
                _colorize_depth(d_mj_vis, args.depth_vis_vmin, args.depth_vis_vmax),
                "MuJoCo(zbuf)",
            )
            da3_tag = "DA3METRIC(zbuf)" if metric_div else "DA3(zbuf)"
            c_da = _label(
                _colorize_depth(d_da3_vis, args.depth_vis_vmin, args.depth_vis_vmax),
                da3_tag,
            )
            depth_vis_writer.append_data(np.concatenate([c_mj, c_da], axis=1))
        if da3_vol is not None:
            da3_vol[i] = np.asarray(d_rel, dtype=np.float32)
        paste_vol[i] = (use_st.astype(np.uint8) * 255)
        robot_mask_vol[i] = (robot.astype(np.uint8) * 255)
        provenance = np.full((h0, w0), PROVENANCE_INVALID, dtype=np.uint8)
        if geom_prov_vol is not None:
            raw_prov = np.asarray(geom_prov_vol[i])
            if raw_prov.ndim == 3:
                raw_prov = raw_prov[..., 0]
            if raw_prov.shape[:2] != (h0, w0):
                raw_prov = cv2.resize(raw_prov, (w0, h0), interpolation=cv2.INTER_NEAREST)
            known = np.isin(raw_prov, np.array([0, 1, 2, 3, 4], dtype=raw_prov.dtype))
            provenance[known] = raw_prov[known].astype(np.uint8)
            provenance[(~hole_bool) & (provenance == PROVENANCE_INVALID)] = PROVENANCE_OBSERVED
        else:
            provenance[~hole_bool] = PROVENANCE_OBSERVED
        provenance[use_st] = PROVENANCE_ROBOT
        provenance_vol[i] = provenance
        scales.append(
            {
                "frame": i,
                "source_index": i,
                "target_index": int(item["target_i"]),
                "scale": used_scale,
                "st_frac": float(use_st.mean()),
                "robot_frac": float(robot.mean()),
                "hole_frac": float(hole_bool.mean()),
                "st_in_hole_frac": float(use_st[hole_bool].mean()) if hole_bool.any() else None,
                "st_outside_hole_frac": float(use_st[~hole_bool].mean())
                if (~hole_bool).any()
                else None,
            }
        )

    if use_metric_ab:
        for chunk_start in tqdm(range(start, end, paste_chunk), desc="composite"):
            chunk_end = min(end, chunk_start + paste_chunk)
            items = [_load_composite_geometry(i) for i in range(chunk_start, chunk_end)]
            d_bg_b = np.stack([item["d_bg"] for item in items]).astype(np.float32)
            d_st_b = np.stack([item["d_st_raw"] for item in items]).astype(np.float32)
            robot_b = np.stack([item["robot"] for item in items]).astype(bool)
            hole_b = np.stack([item["hole_bool"] for item in items]).astype(bool)
            d_nb = None
            if args.paste_mode == "B":
                d_nb = _neighbor_dbg_median(
                    d_bg_b, hole_b, args.neighbor_radius, device=paste_device
                )
            use_st_b = paste_mask_mode_ab(
                d_st_b,
                d_bg_b,
                robot_b,
                hole_b,
                paste_mode=args.paste_mode,
                neighbor_depth=d_nb,
                st_bg_depth=args.st_bg_depth,
                device=paste_device,
            )
            for local, item in enumerate(items):
                d_bg = item["d_bg"]
                d_nv = np.where(
                    np.isfinite(d_bg) & (d_bg > 1e-6),
                    d_bg,
                    args.st_bg_depth,
                ).astype(np.float32)
                _emit_composite_frame(
                    item,
                    use_st=np.asarray(use_st_b[local]),
                    d_nv=d_nv,
                    d_rel=d_bg,
                    used_scale=1.0,
                )
    else:
        for i in tqdm(range(start, end), desc="composite"):
            item = _load_composite_geometry(i)
            d_st_raw = item["d_st_raw"]
            robot = item["robot"]
            hole_bool = item["hole_bool"]
            final_rgb = item["final_rgb"]
            d_st = np.where(robot, d_st_raw, args.st_bg_depth).astype(np.float32)
            used_scale = 1.0
            if bg_vol is not None:
                d_bg = item["d_bg"]
                d_rel = d_bg
                if args.no_scale:
                    used_scale = 1.0
                    d_nv = d_bg.astype(np.float32)
                else:
                    s = estimate_scale(d_bg, d_st_raw, hole_bool)
                    if s is None and last_scale is not None:
                        s = last_scale
                    if s is None:
                        d_nv = np.full((h0, w0), 1e3, dtype=np.float32)
                        used_scale = None
                    else:
                        last_scale = s
                        used_scale = s
                        d_nv = (s * d_bg).astype(np.float32)
                if not args.no_hole:
                    d_nv[hole_bool] = args.hole_depth
                use_st = d_st < d_nv
            else:
                if reuse_da3_vol is not None:
                    d_rel = np.asarray(reuse_da3_vol[i], dtype=np.float32).squeeze()
                    if d_rel.ndim != 2:
                        raise RuntimeError(f"reused depth frame {i} must be HxW, got {d_rel.shape}")
                elif args.reuse_da3_dir is not None:
                    d_rel = np.asarray(
                        np.load(args.reuse_da3_dir / f"{i:06d}.npy"), dtype=np.float32
                    ).squeeze()
                    if d_rel.ndim != 2:
                        raise RuntimeError(f"reused depth frame {i} must be HxW, got {d_rel.shape}")
                elif worker_da3:
                    d_rel = np.asarray(da3_vol[i], dtype=np.float32).squeeze()
                    if d_rel.ndim != 2:
                        raise RuntimeError(f"worker depth frame {i} must be HxW, got {d_rel.shape}")
                else:
                    d_rel = da3_depth(
                        model,
                        final_rgb,
                        args.process_res,
                        fx=fx,
                        fy=fy,
                        metric_focal_div=metric_div,
                    )
                d_rel = resize_depth(d_rel, h0, w0)
                if args.no_scale:
                    used_scale = 1.0
                    d_nv = d_rel.astype(np.float32)
                else:
                    s = estimate_scale(d_rel, d_st_raw, hole_bool)
                    if s is None and last_scale is not None:
                        s = last_scale
                    if s is None:
                        d_nv = np.full((h0, w0), 1e3, dtype=np.float32)
                        used_scale = None
                    else:
                        last_scale = s
                        used_scale = s
                        d_nv = (s * d_rel).astype(np.float32)
                if not args.no_hole:
                    d_nv[hole_bool] = args.hole_depth
                use_st = d_st < d_nv
            _emit_composite_frame(
                item, use_st=use_st, d_nv=d_nv, d_rel=d_rel, used_scale=used_scale
            )

    if writer is not None:
        writer.close()
    if rgb_out_vol is not None:
        rgb_out_vol.flush()
    if da3_vol is not None:
        da3_vol.flush()
    paste_vol.flush()
    robot_mask_vol.flush()
    provenance_vol.flush()
    if depth_vis_writer is not None:
        depth_vis_writer.close()
        print(f"depth vis: {args.depth_vis_out}", flush=True)
    _close_final_source(final_source)
    (dump_dir / "scale.json").write_text(
        json.dumps(
            {
                "policy": {
                    "no_scale": bool(args.no_scale),
                    "no_hole": bool(args.no_hole),
                    "da3_model_dir": str(args.da3_model_dir),
                    "metric_focal_div": metric_div,
                    "fx": fx,
                    "fy": fy,
                    "robot_max_depth": args.robot_max_depth,
                    "robot_mask_dir": str(robot_mask_root) if robot_mask_root is not None else None,
                    "robot_mask_source": "object_id" if robot_mask_root is not None else "legacy_robot_mask",
                    "require_robot_id": bool(args.require_robot_id),
                    "robot_instance_json": (
                        str(args.robot_instance_json.resolve())
                        if args.robot_instance_json is not None else None
                    ),
                    "robot_ids": sorted(robot_ids) if robot_ids is not None else None,
                    "source_to_target": timeline_contract,
                    "provenance_codes": {
                        "invalid": int(PROVENANCE_INVALID),
                        "observed": int(PROVENANCE_OBSERVED),
                        "fused": int(PROVENANCE_FUSED),
                        "robot": int(PROVENANCE_ROBOT),
                        "synthetic": 4,
                    },
                    "hole_depth": args.hole_depth,
                    "st_bg_depth": args.st_bg_depth,
                    "depth_vis_vmin": args.depth_vis_vmin,
                    "depth_vis_vmax": args.depth_vis_vmax,
                },
                "frames": scales,
            },
            indent=2,
        )
    )
    (dump_dir / "robot_instance.json").write_text(
        json.dumps(
            {
                "mask_source": str(robot_mask_root) if robot_mask_root is not None else None,
                "mask_mode": "object_id" if robot_mask_root is not None else "legacy_robot_mask",
                "robot_ids": sorted(robot_ids) if robot_ids is not None else None,
                "depth_threshold_m": None if robot_mask_root is not None else float(args.robot_max_depth),
                "source_instance_manifest": (
                    str(args.robot_instance_json.resolve())
                    if args.robot_instance_json is not None else None
                ),
                "source_to_target": timeline_contract,
                "formal_dataset_allowed": bool(
                    robot_mask_root is not None
                    and (not args.require_robot_id or robot_instance_contract is not None)
                ),
            },
            indent=2,
        )
        + "\n"
    )
    if args.out is not None:
        print(f"wrote {args.out}", flush=True)
    print(f"debug: {dump_dir}", flush=True)


if __name__ == "__main__":
    main()
