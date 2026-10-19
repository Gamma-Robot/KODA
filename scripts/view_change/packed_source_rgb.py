#!/usr/bin/env python3
"""Decode ``rgb.mp4`` once with OpenCV into a packed RGB memmap.

Formal consumers (EfficientSAM3, adapter confidence, warp, blue-cup HSV)
read ``frames.npy`` instead of re-decoding H264. Pixel order is RGB, matching
``cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("OpenCV is required to pack source RGB") from exc

try:
    from frame_volume import (
        as_rgb,
        create_volume,
        hardlink_packed,
        is_packed_stream,
        meta_path,
        open_volume,
        packed_path,
    )
except ImportError:
    from .frame_volume import (
        as_rgb,
        create_volume,
        hardlink_packed,
        is_packed_stream,
        meta_path,
        open_volume,
        packed_path,
    )


def probe_video(video: Path) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {video}")
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"failed to read first frame from {video}")
        height, width = int(frame.shape[0]), int(frame.shape[1])
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if count <= 0:
            count = 1
        return count, height, width, fps
    finally:
        cap.release()


def packed_rgb_matches(
    dest: Path,
    *,
    count: int,
    height: int,
    width: int,
) -> bool:
    if not is_packed_stream(dest):
        return False
    try:
        volume = open_volume(dest, color_space="rgb")
    except (OSError, ValueError, FileNotFoundError):
        return False
    shape = tuple(int(v) for v in volume.frame_shape[:2])
    return len(volume) == int(count) and shape == (int(height), int(width))


def ensure_packed_source_rgb(
    video: Path,
    dest: Path,
    *,
    start_frame: int = 0,
    max_frames: int | None = None,
    force: bool = False,
) -> Path:
    """Write or reuse ``dest/frames.npy`` from ``video`` using OpenCV."""
    video = Path(video).expanduser().resolve()
    dest = Path(dest).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    total, height, width, fps = probe_video(video)
    start = max(0, int(start_frame))
    if start >= total:
        raise ValueError(f"start_frame {start} >= video length {total}")
    available = total - start
    if max_frames is None or int(max_frames) <= 0:
        count = available
    else:
        count = min(available, int(max_frames))
    if count <= 0:
        raise ValueError(f"empty packed RGB range start={start} max_frames={max_frames}")
    if not force and packed_rgb_matches(dest, count=count, height=height, width=width):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    volume = create_volume(dest, count, (height, width, 3), np.uint8, fill=None, color_space="rgb")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to reopen {video}")
    try:
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        written = 0
        for index in range(count):
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            volume[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            written += 1
        if written != count:
            raise RuntimeError(
                f"{video} decoded {written} frames, expected {count} from {start}"
            )
        volume.flush()
    finally:
        cap.release()
    meta = json.loads(meta_path(dest).read_text(encoding="utf-8"))
    meta["source_video"] = str(video)
    meta["start_frame"] = start
    meta["fps"] = fps
    meta["decoder"] = "opencv_bgr2rgb"
    meta_path(dest).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return dest


def resolve_packed_source_rgb(
    *,
    video: Path,
    dest: Path,
    reuse_from: Path | None = None,
    start_frame: int = 0,
    max_frames: int | None = None,
    force: bool = False,
) -> Path:
    dest = Path(dest).expanduser().resolve()
    if reuse_from is not None and is_packed_stream(reuse_from) and not force:
        hardlink_packed(reuse_from, dest, force=True)
        return dest
    return ensure_packed_source_rgb(
        video,
        dest,
        start_frame=start_frame,
        max_frames=max_frames,
        force=force,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reuse-from", type=Path, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dest = resolve_packed_source_rgb(
        video=args.video,
        dest=args.out_dir,
        reuse_from=args.reuse_from,
        start_frame=args.start_frame,
        max_frames=args.max_frames if args.max_frames > 0 else None,
        force=bool(args.force),
    )
    volume = open_volume(dest, color_space="rgb")
    print(
        json.dumps(
            {
                "status": "ok",
                "out_dir": str(dest),
                "count": len(volume),
                "frame_shape": list(volume.frame_shape),
                "packed": str(packed_path(dest)),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
