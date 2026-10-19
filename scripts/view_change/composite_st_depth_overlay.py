#!/usr/bin/env python3
"""Composite robot RGB over the warped source RGB-D view using metric depth.

Observed pixels use a metric z-buffer. Hole pixels use paste mode A (valid
robot geometry) or mode B (also compare against neighboring observed depth).
No estimated depth or depth-scale fitting is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

try:
    import torch
except ImportError:  # CPU metric compositing does not require PyTorch.
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"

try:
    from frame_volume import (
        FrameVolume,
        as_mask,
        as_rgb,
        create_volume,
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
        open_volume,
    )
    from .paste_mode import paste_mask_mode_ab
    PROVENANCE_INVALID = np.uint8(0)
    PROVENANCE_OBSERVED = np.uint8(1)
    PROVENANCE_FUSED = np.uint8(2)
    PROVENANCE_ROBOT = np.uint8(3)


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
        help="Hole masks for the warped source RGB-D frames.",
    )
    p.add_argument(
        "--depth-vis-out",
        type=Path,
        default=None,
        help="Optional side-by-side depth preview (MuJoCo | source RGB-D).",
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
        "--out",
        type=Path,
        default=None,
        help="Optional preview mp4. Formal path writes packed --rgb-out-dir only.",
    )
    p.add_argument("--dump-dir", type=Path, default=None, help="Optional dump for depths/masks")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=None)
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
        help="Background sentinel for invalid depth, in meters.",
    )
    p.add_argument(
        "--bg-depth-dir",
        type=Path,
        required=True,
        help="Metric source RGB-D depth warped into the target camera (meters).",
    )
    p.add_argument(
        "--paste-mode",
        type=str,
        default="B",
        choices=["A", "B"],
        help=(
            "A/B hole paste (requires --bg-depth-dir): "
            "A=in_hole & d_rob valid → paste; "
            "B=also d_rob < neighborhood d_bg. "
            "Non-hole always metric z-buffer d_rob < d_bg."
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


def resize_depth(depth: np.ndarray, h: int, w: int) -> np.ndarray:
    if depth.shape[:2] == (h, w):
        return depth.astype(np.float32)
    return cv2.resize(depth.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


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
    try:
        if torch is None:
            raise ImportError("PyTorch unavailable; use NumPy median")
        if requested.startswith("cuda") and torch.cuda.is_available():
            torch_device = torch.device(requested if requested != "cuda" else "cuda")
        elif device is None and torch.cuda.is_available():
            torch_device = torch.device("cuda")
        else:
            torch_device = torch.device("cpu")
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
    hole_vol = open_volume(args.hole_masks)

    if args.final_rgb_dir is None and args.final_mp4 is None:
        raise SystemExit("one of --final-mp4 or --final-rgb-dir is required")
    final_source, fps = _open_final_source(args)
    geometry_count = _final_source_length(final_source, len(hole_vol))
    n = geometry_count
    if len(hole_vol) < n:
        raise SystemExit(f"frame count mismatch final={n} holes={len(hole_vol)}")

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
    paste_dir = dump_dir / "paste_mask"
    robot_dir = dump_dir / "robot_mask"
    provenance_dir = dump_dir / "provenance"

    first_rgb = _get_final_frame(final_source, 0)
    if first_rgb.ndim != 3 or first_rgb.shape[2] < 3:
        _close_final_source(final_source)
        raise SystemExit("final RGB source must contain at least three channels")
    h0, w0 = first_rgb.shape[:2]
    print(f"Using metric bg depth from {args.bg_depth_dir} paste_mode={args.paste_mode}", flush=True)

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
    paste_vol = create_volume(paste_dir, n, (h0, w0), np.uint8, fill=0)
    robot_mask_vol = create_volume(robot_dir, n, (h0, w0), np.uint8, fill=0)
    provenance_vol = create_volume(provenance_dir, n, (h0, w0), np.uint8, fill=0)

    scales: list[dict] = []
    # h0, w0 already set above from first frame

    depth_vis_writer = None
    if args.depth_vis_out is not None:
        args.depth_vis_out.parent.mkdir(parents=True, exist_ok=True)
        depth_vis_writer = _open_video_writer(args.depth_vis_out, fps, quality=7)

    print(
        f"policy: paste_mode={args.paste_mode} depth_source=source_rgbd_meters "
        f"robot_max_depth={args.robot_max_depth} st_bg_depth={args.st_bg_depth}",
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
                f"Robot depth target frame {target_i} for source {i} must be HxW, "
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
        hole_bool = as_mask(hole_vol[i])
        if hole_bool.shape[:2] != (h0, w0):
            hole_bool = cv2.resize(
                hole_bool.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST
            ) > 0
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
            d_bg_vis = d_nv.copy()
            d_bg_vis[~np.isfinite(d_bg_vis) | (d_bg_vis > 5.0)] = np.nan
            c_mj = _label(
                _colorize_depth(d_mj_vis, args.depth_vis_vmin, args.depth_vis_vmax),
                "MuJoCo(zbuf)",
            )
            c_bg = _label(
                _colorize_depth(d_bg_vis, args.depth_vis_vmin, args.depth_vis_vmax),
                "Source RGB-D(zbuf)",
            )
            depth_vis_writer.append_data(np.concatenate([c_mj, c_bg], axis=1))
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
                "scale": 1.0,
                "st_frac": float(use_st.mean()),
                "robot_frac": float(robot.mean()),
                "hole_frac": float(hole_bool.mean()),
                "st_in_hole_frac": float(use_st[hole_bool].mean()) if hole_bool.any() else None,
                "st_outside_hole_frac": float(use_st[~hole_bool].mean())
                if (~hole_bool).any()
                else None,
            }
        )

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
            )
    if writer is not None:
        writer.close()
    if rgb_out_vol is not None:
        rgb_out_vol.flush()
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
                    "no_scale": True,
                    "depth_source": "source_rgbd_meters",
                    "bg_depth_dir": str(args.bg_depth_dir.resolve()),
                    "paste_mode": args.paste_mode,
                    "neighbor_radius": args.neighbor_radius,
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
