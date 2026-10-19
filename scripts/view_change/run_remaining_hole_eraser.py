#!/usr/bin/env python3
"""DiffuEraser on remaining composite holes + tiny black-pixel neighbor fill.

Remaining hole = novel-view hole pixels that were NOT covered by robot paste.

The inpainting backend is retained as an offline visual completion stage.  Its
generated pixels are always marked with provenance code 4; geometric depth and
validity maps are never changed.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_ROOT = PROJECT_ROOT / "resources"
ASSET_ROOT = Path(os.environ.get("KODA_ASSETS", str(PROJECT_ROOT / "assets"))).expanduser().resolve()

try:
    from frame_volume import as_rgb, create_volume, is_packed_stream, open_volume
except ImportError:  # pragma: no cover - package-style import
    from .frame_volume import as_rgb, create_volume, is_packed_stream, open_volume

try:
    from frame_volume import as_rgb, create_volume, is_packed_stream, open_volume
except ImportError:  # pragma: no cover - package-style import
    from .frame_volume import as_rgb, create_volume, is_packed_stream, open_volume

try:
    from quality_report import (
        PROVENANCE_FUSED,
        PROVENANCE_INVALID,
        PROVENANCE_OBSERVED,
        PROVENANCE_ROBOT,
        PROVENANCE_SYNTHETIC,
    )
except ImportError:  # pragma: no cover - standalone invocation fallback
    PROVENANCE_INVALID = np.uint8(0)
    PROVENANCE_OBSERVED = np.uint8(1)
    PROVENANCE_FUSED = np.uint8(2)
    PROVENANCE_ROBOT = np.uint8(3)
    PROVENANCE_SYNTHETIC = np.uint8(4)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--composite-mp4",
        type=Path,
        default=None,
        help="Composite RGB video (mutually exclusive with --composite-rgb-dir).",
    )
    p.add_argument(
        "--composite-rgb-dir",
        type=Path,
        default=None,
        help="Composite RGB frame directory (XXXXXX.png), read lazily.",
    )
    p.add_argument("--hole-dir", type=Path, required=True, help="novel-view hole masks")
    p.add_argument("--choice-dir", type=Path, required=True, help="paste choice masks (255=robot)")
    p.add_argument(
        "--out-mp4",
        type=Path,
        default=None,
        help="Optional 30 Hz preview mp4. Formal path writes packed --rgb-final-dir only.",
    )
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--synthetic-dir", type=Path, default=None, help="Output per-frame synthetic pixel masks")
    p.add_argument(
        "--provenance-dir",
        type=Path,
        default=None,
        help="Output per-frame provenance labels (0 invalid, 1 observed, 2 fused, 3 robot, 4 synthetic)",
    )
    p.add_argument(
        "--geometry-provenance-dir",
        "--input-provenance-dir",
        type=Path,
        default=None,
        help="Optional provenance from geometry/compositor; preserves observed/fused labels.",
    )
    p.add_argument(
        "--rgb-final-dir",
        "--out-rgb-dir",
        type=Path,
        default=None,
        help="Optional output directory for final RGB frames (XXXXXX.png).",
    )
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--tiny-cc-max", type=int, default=32, help="Max connected black pixels to neighbor-fill")
    p.add_argument("--skip-diffueraser", action="store_true")
    p.add_argument(
        "--inpaint-mode", choices=["causal", "visual_only"], default="visual_only",
        help="visual_only runs native offline DiffuEraser (formal default); causal is a debug-only local fallback",
    )
    p.add_argument(
        "--require-diffueraser",
        action="store_true",
        help="Formal mode: fail on missing weights, model error, missing output, or output frame-count mismatch.",
    )
    p.add_argument(
        "--flow-dir", type=Path, default=None,
        help="Optional current-to-previous optical-flow NPYs for causal history sampling",
    )
    p.add_argument(
        "--depth-dir", type=Path, default=None,
        help="Optional geometry depth directory recorded with the causal inpaint contract",
    )
    p.add_argument(
        "--diffueraser-python",
        type=Path,
        default=os.environ.get("KODA_PYTHON", sys.executable),
    )
    p.add_argument(
        "--diffueraser-script",
        type=Path,
        default=RESOURCE_ROOT / "DiffuEraser/run_diffueraser.py",
    )
    p.add_argument(
        "--diffueraser-ckpt",
        default="Normal CFG 8-Step",
        help="Native DiffuEraser PCM checkpoint; formal default uses CFG diffusion.",
    )
    p.add_argument("--mask-dilation-iter", type=int, default=4)
    p.add_argument("--max-img-size", type=int, default=960)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--min-model-frames",
        type=int,
        default=22,
        help=(
            "Tail-pad only the native DiffuEraser input to this frame count; "
            "the published output is always trimmed to the real input length."
        ),
    )
    p.add_argument(
        "--propainter-model-dir",
        type=Path,
        default=ASSET_ROOT / "models/diffueraser/propainter",
    )
    return p.parse_args()


def _remaining_holes(hole: np.ndarray, choice: np.ndarray) -> np.ndarray:
    """255 where still a hole after paste."""
    h = hole > 127
    covered = choice > 127
    rem = h & (~covered)
    return (rem.astype(np.uint8) * 255)


def _list_rgb_frames(path: Path) -> list[Path]:
    files = sorted(path.glob("*.png"), key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem))
    if not files:
        raise FileNotFoundError(f"no RGB PNG frames in {path}")
    return files


def _frame_key(path: Path) -> int | None:
    try:
        return int(path.stem)
    except ValueError:
        return None


def _align_mask_paths(
    source_paths: list[Path] | None,
    mask_paths: list[Path],
) -> tuple[list[Path], list[str]]:
    """Align directory masks to source frame IDs when numeric stems exist."""
    if source_paths is None:
        return mask_paths, [f"{i:06d}" for i in range(len(mask_paths))]
    source_keys = [_frame_key(p) for p in source_paths]
    mask_keys = [_frame_key(p) for p in mask_paths]
    if all(k is not None for k in source_keys) and all(k is not None for k in mask_keys):
        by_key = {int(k): p for k, p in zip(mask_keys, mask_paths)}
        missing = [int(k) for k in source_keys if int(k) not in by_key]
        if missing:
            raise SystemExit(
                "mask directory is missing source frame IDs: "
                + ", ".join(f"{k:06d}" for k in missing[:8])
            )
        return [by_key[int(k)] for k in source_keys], [p.stem for p in source_paths]
    # Legacy/non-indexed exports are positional.
    return mask_paths, [p.stem for p in source_paths]


def _read_rgb_frame(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read RGB frame {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class _Cv2VideoReader:
    def __init__(self, path: Path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"failed to open video {path}")

    def get_meta_data(self) -> dict[str, float]:
        return {"fps": float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)}

    def count_frames(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def __next__(self) -> np.ndarray:
        ok, bgr = self.cap.read()
        if not ok or bgr is None:
            raise StopIteration
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
        if self.writer is None:
            h, w = rgb.shape[:2]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(
                str(self.path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (w, h),
            )
            if not self.writer.isOpened():
                raise RuntimeError(f"failed to create video {self.path}")
        self.writer.write(cv2.cvtColor(rgb[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()


def _open_video_reader(path: Path):
    try:
        candidate = imageio.get_reader(str(path))
        plugin = getattr(candidate, "instance", None)
        if plugin is not None and "tifffile" in type(plugin).__module__.lower():
            candidate.close()
            raise RuntimeError("imageio selected tifffile for video")
        return candidate
    except Exception:
        return _Cv2VideoReader(path)


def _open_video_writer(path: Path, fps: float):
    try:
        candidate = imageio.get_writer(
            str(path),
            fps=fps,
            codec="libx264",
            quality=8,
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


def _video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open generated video {path}")
    try:
        return int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()


def _prior_diffusion_metrics(
    priori_path: Path,
    diffusion_path: Path,
    remaining,
    frame_names: list[str],
) -> dict[str, float | int]:
    """Quantify that the diffusion output differs from the ProPainter prior.

    The comparison is restricted to the remaining-hole mask.  Outside that
    mask the final compositor restores the original pixels, so a global video
    difference would be both misleading and needlessly sensitive to codec
    noise.
    """
    prior = cv2.VideoCapture(str(priori_path))
    diffusion = cv2.VideoCapture(str(diffusion_path))
    if not prior.isOpened() or not diffusion.isOpened():
        prior.release()
        diffusion.release()
        raise RuntimeError("cannot open prior/diffusion videos for audit")
    remaining_vol = None if isinstance(remaining, Path) else remaining
    remaining_dir = remaining if isinstance(remaining, Path) else None
    absolute_error_sum = 0.0
    compared = 0
    changed = 0
    try:
        for index, frame_name in enumerate(frame_names):
            ok_a, a = prior.read()
            ok_b, b = diffusion.read()
            if not ok_a or not ok_b:
                raise RuntimeError("prior/diffusion video ended before expected frame count")
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
            if b.shape[:2] != a.shape[:2]:
                b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
            if remaining_vol is not None:
                mask = _as_u8_mask(remaining_vol[index])
            else:
                assert remaining_dir is not None
                mask = cv2.imread(str(remaining_dir / f"{frame_name}.png"), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"missing remaining-hole mask {frame_name}")
            if mask.shape[:2] != a.shape[:2]:
                mask = cv2.resize(mask, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
            selected = mask > 127
            if not np.any(selected):
                continue
            delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
            absolute_error_sum += float(delta[selected].sum())
            changed += int(np.count_nonzero(np.any(delta > 2, axis=2) & selected))
            compared += int(np.count_nonzero(selected))
    finally:
        prior.release()
        diffusion.release()
    return {
        "frames": int(len(frame_names)),
        "masked_pixels": int(compared),
        "mae": float(absolute_error_sum / max(1, compared * 3)),
        "changed_ratio": float(changed / max(1, compared)),
    }


def _diffueraser_weights_root() -> Path:
    override = os.environ.get("KODA_DIFFUERASER_WEIGHTS", "").strip()
    return Path(override).expanduser().resolve() if override else (ASSET_ROOT / "models/diffueraser/weights")


def _validate_diffueraser_inputs(args: argparse.Namespace) -> dict[str, object]:
    root = args.diffueraser_script.resolve().parent
    weights = _diffueraser_weights_root()
    required = [
        args.diffueraser_python,
        args.diffueraser_script,
        weights / "stable-diffusion-v1-5" / "model_index.json",
        weights / "sd-vae-ft-mse" / "config.json",
        weights / "diffuEraser" / "brushnet" / "diffusion_pytorch_model.safetensors",
        weights / "diffuEraser" / "unet_main" / "diffusion_pytorch_model.safetensors",
        args.propainter_model_dir / "raft-things.pth",
        args.propainter_model_dir / "recurrent_flow_completion.pth",
        args.propainter_model_dir / "ProPainter.pth",
    ]
    missing = [str(path) for path in required if not path.exists()]
    return {
        "root": str(root),
        "propainter_model_dir": str(args.propainter_model_dir.resolve()),
        "required_files": [str(path) for path in required],
        "missing": missing,
        "pass": not missing,
    }


def _diffueraser_worker_request(payload: dict[str, object], timeout_s: float = 7200.0) -> dict[str, object]:
    sock_path = str(os.environ.get("DIFFUERASER_WORKER_SOCK", "")).strip()
    if not sock_path:
        raise RuntimeError("DIFFUERASER_WORKER_SOCK is empty")
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
                raise ConnectionError("DiffuEraser worker closed the connection")
            buf += chunk
        reply = json.loads(buf.split(b"\n", 1)[0].decode("utf-8"))
        if not isinstance(reply, dict):
            raise ValueError("DiffuEraser worker reply must be a JSON object")
        return reply
    finally:
        sock.close()


def _as_u8_mask(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 3:
        array = array[..., 0]
    if array.dtype == np.bool_:
        return (array.astype(np.uint8) * 255)
    return array.astype(np.uint8, copy=False)


def _load_provenance(
    path: Path | np.ndarray | None,
    shape: tuple[int, int],
    hole: np.ndarray,
    choice: np.ndarray,
) -> np.ndarray:
    """Load geometry provenance and fill conservative defaults."""
    out = np.full(shape, PROVENANCE_INVALID, dtype=np.uint8)
    raw: np.ndarray | None = None
    if isinstance(path, np.ndarray):
        raw = np.asarray(path)
    elif path is not None:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise RuntimeError(f"failed to read provenance {path}")
    if raw is not None:
        if raw.ndim == 3:
            raw = raw[..., 0]
        if raw.shape[:2] != shape:
            raw = cv2.resize(raw, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        known = np.isin(raw, np.array([0, 1, 2, 3, 4], dtype=raw.dtype))
        out[known] = raw[known].astype(np.uint8)
        # An old compositor may have used zero for all ordinary pixels.
        out[(~hole) & (out == PROVENANCE_INVALID)] = PROVENANCE_OBSERVED
    else:
        out[~hole] = PROVENANCE_OBSERVED
    out[choice] = PROVENANCE_ROBOT
    return out


def _tiny_black_fill(rgb: np.ndarray, max_cc: int) -> np.ndarray:
    """Fill small connected nearly-black components via neighbor median."""
    out = rgb.copy()
    gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    black = gray < 8
    if not black.any():
        return out
    num, labels, stats, _ = cv2.connectedComponentsWithStats(black.astype(np.uint8), connectivity=8)
    h, w = gray.shape
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area <= 0 or area > max_cc:
            continue
        ys, xs = np.where(labels == lab)
        y0, y1 = max(0, ys.min() - 1), min(h, ys.max() + 2)
        x0, x1 = max(0, xs.min() - 1), min(w, xs.max() + 2)
        region = out[y0:y1, x0:x1]
        mask = labels[y0:y1, x0:x1] == lab
        neigh = region.copy()
        neigh[mask] = 0
        # sample non-black neighbors
        g = cv2.cvtColor(neigh, cv2.COLOR_RGB2GRAY)
        ok = g >= 8
        if not ok.any():
            continue
        fill = np.median(neigh[ok], axis=0).astype(np.uint8)
        out[ys, xs] = fill
    return out


def _load_backward_flow(flow_dir: Path | None, frame_name: str, shape: tuple[int, int]) -> np.ndarray | None:
    if flow_dir is None:
        return None
    path = flow_dir / f"{frame_name}.npy"
    if not path.is_file():
        return None
    flow = np.asarray(np.load(path), dtype=np.float32)
    if flow.ndim != 3 or flow.shape[2] != 2:
        return None
    if flow.shape[:2] != shape:
        old_h, old_w = flow.shape[:2]
        flow = cv2.resize(flow, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        flow[..., 0] *= shape[1] / max(1, old_w)
        flow[..., 1] *= shape[0] / max(1, old_h)
    return flow


def _causal_inpaint(
    current: np.ndarray,
    synthetic: np.ndarray,
    previous_final: np.ndarray | None,
    flow_to_previous: np.ndarray | None,
) -> np.ndarray:
    """Causal RGB completion using current image plus at most one history frame."""
    if not np.any(synthetic):
        return current.copy()
    seed = current.copy()
    if previous_final is not None:
        previous = previous_final
        if flow_to_previous is not None:
            h, w = synthetic.shape
            xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            previous = cv2.remap(
                previous_final,
                xx + flow_to_previous[..., 0], yy + flow_to_previous[..., 1],
                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            )
        seed[synthetic] = previous[synthetic]
    # Spatial completion is constrained to the remaining geometric holes.  It
    # never changes observed/fused/robot pixels; temporal history above acts as
    # the first prior whenever it is available.
    bgr = cv2.cvtColor(seed, cv2.COLOR_RGB2BGR)
    filled = cv2.inpaint(bgr, (synthetic.astype(np.uint8) * 255), 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)


def main() -> None:
    args = parse_args()
    if args.require_diffueraser and args.skip_diffueraser:
        raise SystemExit("--require-diffueraser conflicts with --skip-diffueraser")
    if args.require_diffueraser and args.inpaint_mode != "visual_only":
        raise SystemExit("--require-diffueraser requires --inpaint-mode visual_only")
    if args.out_mp4 is None and args.rgb_final_dir is None:
        raise SystemExit("one of --out-mp4 or --rgb-final-dir is required")
    if args.composite_mp4 is None and args.composite_rgb_dir is None:
        raise SystemExit("one of --composite-mp4 or --composite-rgb-dir is required")
    if args.composite_mp4 is not None and args.composite_rgb_dir is not None:
        raise SystemExit("--composite-mp4 and --composite-rgb-dir are mutually exclusive")
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    rem_dir = work / "remaining_holes"

    packed = bool(
        args.composite_rgb_dir is not None and is_packed_stream(args.composite_rgb_dir)
    )
    rgb_vol = hole_vol = choice_vol = geom_vol = None
    source_paths: list[Path] | None = None
    reader = None
    if packed:
        assert args.composite_rgb_dir is not None
        rgb_vol = open_volume(args.composite_rgb_dir, color_space="rgb")
        hole_vol = open_volume(args.hole_dir)
        choice_vol = open_volume(args.choice_dir)
        if args.geometry_provenance_dir is not None:
            geom_vol = open_volume(args.geometry_provenance_dir)
        video_len = len(rgb_vol)
        fps = float(args.fps or 30.0)
        n = min(video_len, len(hole_vol), len(choice_vol))
        if args.max_frames is not None:
            n = min(n, int(args.max_frames))
        frame_names = [f"{i:06d}" for i in range(n)]
        holes = choices = geometry_provenance = None
    else:
        hole_paths = sorted(args.hole_dir.glob("*.png"), key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem))
        choice_paths = sorted(args.choice_dir.glob("*.png"), key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem))
        if not hole_paths or not choice_paths:
            raise SystemExit("hole and choice directories must contain PNG masks")
        if args.composite_rgb_dir is not None:
            source_paths = _list_rgb_frames(args.composite_rgb_dir)
            video_len = len(source_paths)
            fps = float(args.fps or 30.0)
        else:
            reader = _open_video_reader(args.composite_mp4)
            meta = reader.get_meta_data()
            fps = float(args.fps or meta.get("fps") or 30.0)
            try:
                video_len = int(reader.count_frames())
            except Exception:
                video_len = len(hole_paths)
            if video_len <= 0:
                video_len = len(hole_paths)
        holes, frame_names = _align_mask_paths(source_paths, hole_paths)
        choices, choice_names = _align_mask_paths(source_paths, choice_paths)
        if frame_names != choice_names:
            raise SystemExit("hole and choice frame IDs do not match composite RGB frames")
        if args.geometry_provenance_dir is not None:
            prov_paths = sorted(args.geometry_provenance_dir.glob("*.png"), key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem))
            geometry_provenance, prov_names = _align_mask_paths(source_paths, prov_paths)
            if len(geometry_provenance) < min(video_len, len(holes), len(choices)):
                raise SystemExit("geometry provenance directory has too few frames")
            if prov_names[: min(len(prov_names), len(frame_names))] != frame_names[: min(len(prov_names), len(frame_names))]:
                raise SystemExit("geometry provenance frame IDs do not match composite RGB frames")
        else:
            geometry_provenance = None
        n = min(video_len, len(holes), len(choices))
        if args.max_frames is not None:
            n = min(n, int(args.max_frames))
    if n <= 0:
        if reader is not None:
            reader.close()
        raise SystemExit("no aligned frames to process")

    rem_vol = None
    if packed:
        assert hole_vol is not None
        rem_vol = create_volume(rem_dir, n, tuple(np.asarray(hole_vol[0]).shape[:2]), np.uint8, fill=0)
    else:
        rem_dir.mkdir(parents=True, exist_ok=True)

    vid_in = work / "composite_trim.mp4"
    mask_mp4 = work / "remaining_hole_mask.mp4"
    w = _open_video_writer(vid_in, fps)
    wm = _open_video_writer(mask_mp4, fps)
    last_frame = None
    last_remaining = None
    for i in tqdm(range(n), desc="remaining_holes"):
        if packed:
            frame = as_rgb(rgb_vol[i])
            h = _as_u8_mask(hole_vol[i])
            c = _as_u8_mask(choice_vol[i])
        elif source_paths is not None:
            frame = _read_rgb_frame(source_paths[i])
            h = cv2.imread(str(holes[i]), cv2.IMREAD_GRAYSCALE)
            c = cv2.imread(str(choices[i]), cv2.IMREAD_GRAYSCALE)
        else:
            try:
                frame = np.asarray(next(reader))
            except StopIteration as exc:
                raise RuntimeError(f"composite video ended before frame {i}") from exc
            h = cv2.imread(str(holes[i]), cv2.IMREAD_GRAYSCALE)
            c = cv2.imread(str(choices[i]), cv2.IMREAD_GRAYSCALE)
        frame = frame[:, :, :3]
        if h is None or c is None:
            raise RuntimeError(f"failed to read hole/choice mask for frame {i}")
        if h.shape[:2] != frame.shape[:2]:
            h = cv2.resize(h, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        if c.shape[:2] != frame.shape[:2]:
            c = cv2.resize(c, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        rem = _remaining_holes(h, c)
        if rem_vol is not None:
            rem_vol[i] = rem
        else:
            cv2.imwrite(str(rem_dir / f"{frame_names[i]}.png"), rem)
        w.append_data(frame)
        wm.append_data(np.repeat(rem[..., None], 3, axis=2))
        last_frame = frame
        last_remaining = rem
    model_frames = max(n, int(args.min_model_frames))
    if model_frames > n:
        if last_frame is None or last_remaining is None:
            raise RuntimeError("cannot pad an empty DiffuEraser input")
        for _ in range(n, model_frames):
            w.append_data(last_frame)
            wm.append_data(np.repeat(last_remaining[..., None], 3, axis=2))
    w.close()
    wm.close()
    if rem_vol is not None:
        rem_vol.flush()
    if reader is not None:
        reader.close()

    erased_mp4 = work / "diffueraser_result.mp4"
    use_visual_model = not args.skip_diffueraser and args.inpaint_mode == "visual_only"
    diffueraser_status: dict[str, object] = {
        "backend": "DiffuEraser" if use_visual_model else "causal_debug",
        "required": bool(args.require_diffueraser),
        "success": False,
        "seed": int(args.seed),
        "requested_frames": int(n),
        "model_input_frames": int(model_frames),
        "tail_padding_frames": int(model_frames - n),
        "tail_padding_policy": "repeat_last_frame",
    }
    if not use_visual_model:
        erased_mp4 = vid_in
        if not args.skip_diffueraser:
            print("[eraser] causal mode: DiffuEraser disabled (visual_only model requires explicit opt-in)", flush=True)
        diffueraser_status["success"] = not args.require_diffueraser
        diffueraser_status["reason"] = "debug_backend_selected"
    else:
        de_root = args.diffueraser_script.resolve().parent
        weights = _diffueraser_weights_root()
        preflight = _validate_diffueraser_inputs(args)
        diffueraser_status["preflight"] = preflight
        if not preflight["pass"]:
            message = "DiffuEraser preflight missing: " + ", ".join(preflight["missing"])
            if args.require_diffueraser:
                (work / "diffueraser_status.json").write_text(
                    json.dumps({**diffueraser_status, "error": message}, indent=2) + "\n"
                )
                raise SystemExit(message)
            print(f"[eraser] WARN: {message}", flush=True)
            use_visual_model = False
            erased_mp4 = vid_in
        # Native DiffuEraser interprets video_length as seconds, not frames.
        duration_s = (float(model_frames) + 0.5) / max(float(fps), 1e-6)
        cmd = [
            str(args.diffueraser_python),
            str(args.diffueraser_script),
            "--input_video",
            str(vid_in.resolve()),
            "--input_mask",
            str(mask_mp4.resolve()),
            "--save_path",
            str((work / "diffueraser_out").resolve()),
            "--mask_dilation_iter",
            str(args.mask_dilation_iter),
            "--max_img_size",
            str(args.max_img_size),
            "--video_length",
            f"{duration_s:.9f}",
            "--seed",
            str(args.seed),
            "--ckpt",
            str(args.diffueraser_ckpt),
            "--keep_priori",
            "--base_model_path",
            str(weights / "stable-diffusion-v1-5"),
            "--vae_path",
            str(weights / "sd-vae-ft-mse"),
            "--diffueraser_path",
            str(weights / "diffuEraser"),
            "--propainter_model_dir",
            str(args.propainter_model_dir.resolve()),
        ]
        diffueraser_status["duration_s"] = duration_s
        diffueraser_status["command"] = cmd
        if use_visual_model:
            worker_sock = str(os.environ.get("DIFFUERASER_WORKER_SOCK", "")).strip()
            if worker_sock and not os.path.exists(worker_sock):
                print(
                    f"[eraser] DiffuEraser worker socket missing ({worker_sock}); using subprocess",
                    flush=True,
                )
                worker_sock = ""
            print("[eraser] running:", " ".join(cmd), flush=True)
            try:
                environment = os.environ.copy()
                environment["PYTHONHASHSEED"] = str(args.seed)
                environment["KODA_DIFFUERASER_WEIGHTS"] = str(weights)
                if worker_sock:
                    print(f"[eraser] using persistent DiffuEraser worker {worker_sock}", flush=True)
                    reply = _diffueraser_worker_request(
                        {
                            "op": "inpaint",
                            "input_video": str(vid_in.resolve()),
                            "input_mask": str(mask_mp4.resolve()),
                            "save_path": str((work / "diffueraser_out").resolve()),
                            "video_length": float(duration_s),
                            "mask_dilation_iter": int(args.mask_dilation_iter),
                            "max_img_size": int(args.max_img_size),
                            "seed": int(args.seed),
                            "keep_priori": True,
                            "ref_stride": 10,
                            "neighbor_length": 10,
                            "subvideo_length": 50,
                        }
                    )
                    if not reply.get("ok"):
                        raise RuntimeError(reply.get("error") or "DiffuEraser worker inpaint failed")
                else:
                    subprocess.check_call(cmd, cwd=str(de_root), env=environment)
                candidate = work / "diffueraser_out" / "diffueraser_result.mp4"
                if not candidate.is_file():
                    cands = sorted((work / "diffueraser_out").rglob("diffueraser_result.mp4"))
                    if not cands:
                        raise RuntimeError("native DiffuEraser wrote no diffueraser_result.mp4")
                    candidate = cands[0]
                generated_count = _video_frame_count(candidate)
                diffueraser_status["generated_frames"] = generated_count
                if generated_count != model_frames:
                    raise RuntimeError(
                        "DiffuEraser output has "
                        f"{generated_count} frames, expected model input {model_frames}"
                    )
                erased_mp4 = candidate
                diffueraser_status["success"] = True
                diffueraser_status["output"] = str(candidate.resolve())
                priori_candidate = work / "diffueraser_out" / "priori.mp4"
                if not priori_candidate.is_file():
                    raise RuntimeError("native DiffuEraser did not preserve priori.mp4")
                run_manifest_path = work / "diffueraser_out" / "diffueraser_run.json"
                if not run_manifest_path.is_file():
                    raise RuntimeError("native DiffuEraser wrote no diffueraser_run.json")
                native_run = json.loads(run_manifest_path.read_text())
                if native_run.get("checkpoint") != args.diffueraser_ckpt:
                    raise RuntimeError("DiffuEraser checkpoint differs from requested contract")
                if int(native_run.get("num_inference_steps", 0)) <= 0:
                    raise RuntimeError("DiffuEraser reported no diffusion inference steps")
                if float(native_run.get("guidance_scale", -1.0)) <= 0.0:
                    raise RuntimeError("formal DiffuEraser run must use positive CFG guidance")
                diffueraser_status["priori_output"] = str(priori_candidate.resolve())
                diffueraser_status["checkpoint"] = str(args.diffueraser_ckpt)
                diffueraser_status["native_run"] = native_run
                diffueraser_status["priori_diffusion"] = _prior_diffusion_metrics(
                    priori_candidate,
                    candidate,
                    rem_vol if rem_vol is not None else rem_dir,
                    frame_names[:n],
                )
            except Exception as exc:
                diffueraser_status["error"] = f"{type(exc).__name__}: {exc}"
                (work / "diffueraser_status.json").write_text(
                    json.dumps(diffueraser_status, indent=2) + "\n"
                )
                if args.require_diffueraser:
                    raise RuntimeError(
                        f"required DiffuEraser failed: {diffueraser_status['error']}"
                    ) from exc
                print(
                    f"[eraser] WARN: DiffuEraser failed ({exc}); using debug causal fill",
                    flush=True,
                )
                use_visual_model = False
                erased_mp4 = vid_in

    rgb_final_dir = args.rgb_final_dir
    if rgb_final_dir is not None:
        rgb_final_dir.mkdir(parents=True, exist_ok=True)
    synthetic_dir = args.synthetic_dir or (work / "synthetic")
    provenance_dir = args.provenance_dir or (work / "provenance")
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if args.out_mp4 is not None:
        args.out_mp4.parent.mkdir(parents=True, exist_ok=True)
        writer = _open_video_writer(args.out_mp4, fps)
    # Read the original composite again rather than the temporary encoded
    # DiffuEraser input. This keeps observed/fused/robot RGB untouched when
    # only synthetic pixels are selected below.
    src_reader = None if (packed or source_paths is not None) else _open_video_reader(args.composite_mp4)
    out_reader = _open_video_reader(erased_mp4) if use_visual_model else None
    previous_final: np.ndarray | None = None
    rgb_final_vol = synth_vol = prov_out_vol = None
    for i in tqdm(range(n), desc="tiny_fill"):
        if packed:
            original = as_rgb(rgb_vol[i])
        elif source_paths is not None:
            original = _read_rgb_frame(source_paths[i])
        else:
            try:
                original = np.asarray(next(src_reader))[:, :, :3]
            except StopIteration as exc:
                raise RuntimeError(f"composite source ended before frame {i}") from exc
        if rem_vol is not None:
            rem = _as_u8_mask(rem_vol[i])
        else:
            rem = cv2.imread(str(rem_dir / f"{frame_names[i]}.png"), cv2.IMREAD_GRAYSCALE)
        if rem is None:
            raise RuntimeError(f"missing remaining-hole mask for frame {i}")
        if rem.shape[:2] != original.shape[:2]:
            rem = cv2.resize(rem, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
        synthetic = rem > 127
        if use_visual_model:
            assert out_reader is not None
            try:
                if hasattr(out_reader, "get_data"):
                    generated = np.asarray(out_reader.get_data(i))[:, :, :3]
                else:
                    generated = np.asarray(next(iter(out_reader)))[:, :, :3]
            except (StopIteration, IndexError):
                generated = original
            if generated.shape[:2] != original.shape[:2]:
                generated = cv2.resize(generated, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)
            generated = _tiny_black_fill(generated, args.tiny_cc_max)
        else:
            flow = _load_backward_flow(args.flow_dir, frame_names[i], original.shape[:2])
            generated = _causal_inpaint(original, synthetic, previous_final, flow)
        # DiffuEraser/tiny fill may only modify remaining holes. Observed and
        # robot-pasted pixels are copied bit-for-bit from the composite input.
        final = np.where(synthetic[..., None], generated, original).astype(np.uint8)
        previous_final = final
        if writer is not None:
            writer.append_data(final)
        if packed:
            if rgb_final_dir is not None and rgb_final_vol is None:
                rgb_final_vol = create_volume(
                    rgb_final_dir, n, final.shape, np.uint8, fill=None, color_space="rgb"
                )
            if synth_vol is None:
                synth_vol = create_volume(
                    synthetic_dir, n, original.shape[:2], np.uint8, fill=0
                )
            if prov_out_vol is None:
                prov_out_vol = create_volume(
                    provenance_dir, n, original.shape[:2], np.uint8, fill=0
                )
            if rgb_final_vol is not None:
                rgb_final_vol[i] = final
            synth_vol[i] = (synthetic.astype(np.uint8) * 255)
        elif rgb_final_dir is not None:
            cv2.imwrite(
                str(rgb_final_dir / f"{frame_names[i]}.png"),
                cv2.cvtColor(final, cv2.COLOR_RGB2BGR),
            )
        if not packed:
            cv2.imwrite(str(synthetic_dir / f"{frame_names[i]}.png"), (synthetic.astype(np.uint8) * 255))
        if packed:
            choice = _as_u8_mask(choice_vol[i])
            geom_src: Path | np.ndarray | None = (
                np.asarray(geom_vol[i]) if geom_vol is not None else None
            )
        else:
            choice = cv2.imread(str(choices[i]), cv2.IMREAD_GRAYSCALE)
            if choice is None:
                raise RuntimeError(f"failed to read choice mask {choices[i]}")
            geom_src = geometry_provenance[i] if geometry_provenance is not None else None
        if choice.shape[:2] != original.shape[:2]:
            choice = cv2.resize(choice, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
        provenance = _load_provenance(
            geom_src,
            original.shape[:2],
            synthetic | (choice > 127),
            choice > 127,
        )
        provenance[synthetic] = PROVENANCE_SYNTHETIC
        if packed:
            prov_out_vol[i] = provenance
        else:
            cv2.imwrite(str(provenance_dir / f"{frame_names[i]}.png"), provenance)
    if writer is not None:
        writer.close()
    if rgb_final_vol is not None:
        rgb_final_vol.flush()
    if synth_vol is not None:
        synth_vol.flush()
    if prov_out_vol is not None:
        prov_out_vol.flush()
    if src_reader is not None:
        src_reader.close()
    if out_reader is not None:
        out_reader.close()
    (work / "provenance_manifest.json").write_text(
        "{\n"
        '  "codes": {"invalid": 0, "observed": 1, "fused": 2, "robot": 3, "synthetic": 4},\n'
        '  "synthetic_definition": "geometric_hole AND NOT robot_paste",\n'
        '  "inpaint_mode": "' + args.inpaint_mode + '",\n'
        '  "flow_dir": ' + json.dumps(str(args.flow_dir) if args.flow_dir is not None else None) + ',\n'
        '  "depth_dir": ' + json.dumps(str(args.depth_dir) if args.depth_dir is not None else None) + ',\n'
        '  "rgb_final_written": '
        + ("true" if rgb_final_dir is not None else "false")
        + "\n}\n"
    )
    diffueraser_status["selected_backend"] = "DiffuEraser" if use_visual_model else "causal_debug"
    diffueraser_status["final_output_frames"] = int(n)
    (work / "diffueraser_status.json").write_text(
        json.dumps(diffueraser_status, indent=2) + "\n"
    )
    if args.require_diffueraser and not bool(diffueraser_status.get("success", False)):
        raise RuntimeError("required DiffuEraser did not produce a verified result")
    if args.out_mp4 is not None:
        print(f"[eraser] wrote {args.out_mp4}", flush=True)
    elif rgb_final_dir is not None:
        print(f"[eraser] wrote packed {rgb_final_dir}", flush=True)


if __name__ == "__main__":
    main()
