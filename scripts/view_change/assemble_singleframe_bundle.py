#!/usr/bin/env python3
"""Assemble the complete processed 30 Hz bundle from warp/composite stages.

Packed streams are hard-linked (frames.npy + volume_meta.json). Legacy
per-frame directories still copy. No per-frame shutil.copy2 on the formal path.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

try:
    from frame_volume import hardlink_packed, is_packed_stream, open_volume
except ImportError:
    from .frame_volume import hardlink_packed, is_packed_stream, open_volume


GEOMETRY_STREAMS = ("depth", "valid", "hole", "confidence", "source_count")
COMPOSITE_STREAMS = ("robot_mask", "paste_mask", "provenance_geometry")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warp-dir", type=Path, required=True)
    p.add_argument("--composite-debug-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def _copy_legacy_stream(src: Path, dst: Path, n: int, force: bool) -> None:
    if not src.is_dir():
        raise FileNotFoundError(f"missing stream: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(
        [p for p in src.iterdir() if p.is_file() and p.stem.isdigit()],
        key=lambda p: int(p.stem),
    )
    expected = list(range(n))
    got = [int(p.stem) for p in files]
    if got != expected:
        raise ValueError(f"{src} does not contain exactly 0..{n - 1}: first={got[:5]} last={got[-5:]}")
    for path in files:
        target = dst / path.name
        if target.exists() or target.is_symlink():
            if not force:
                if target.is_file() and target.stat().st_size == path.stat().st_size:
                    continue
                raise FileExistsError(f"output exists: {target} (use --force)")
            target.unlink()
        shutil.copy2(path, target)


def _link_stream(src: Path, dst: Path, n: int, force: bool) -> str:
    if is_packed_stream(src):
        vol = open_volume(src)
        if len(vol) != n:
            raise ValueError(f"{src} has {len(vol)} frames, expected {n}")
        return hardlink_packed(src, dst, force=force)
    _copy_legacy_stream(src, dst, n, force)
    return "copy_legacy"


def main() -> None:
    args = _args()
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    linked: dict[str, str] = {}
    for stream in GEOMETRY_STREAMS:
        linked[stream] = _link_stream(args.warp_dir / stream, out / stream, args.count, args.force)
    linked["robot_mask"] = _link_stream(
        args.composite_debug_dir / "robot_mask", out / "robot_mask", args.count, args.force
    )
    linked["paste_mask"] = _link_stream(
        args.composite_debug_dir / "paste_mask", out / "paste_mask", args.count, args.force
    )
    linked["provenance_geometry"] = _link_stream(
        args.composite_debug_dir / "provenance",
        out / "provenance_geometry",
        args.count,
        args.force,
    )
    linked["rgb_warp"] = _link_stream(
        args.warp_dir / "rgb_geometry",
        out / "rgb_warp",
        args.count,
        args.force,
    )
    status_source = args.warp_dir / "frame_status.jsonl"
    if not status_source.is_file():
        raise FileNotFoundError(f"missing frame status: {status_source}")
    status_target = out / "frame_status.jsonl"
    if status_target.exists() and args.force:
        status_target.unlink()
    if not status_target.exists():
        shutil.copy2(status_source, status_target)
    meta = {
        "schema_version": 2,
        "frame_count": args.count,
        "layout": "packed_memmap",
        "source_rgb_geometry": str(args.warp_dir / "rgb_geometry"),
        "composite_debug": str(args.composite_debug_dir),
        "streams": list(GEOMETRY_STREAMS) + ["robot_mask", "paste_mask", "provenance_geometry", "rgb_warp"],
        "materialization": linked,
        "fused_pixels": 0,
    }
    (out / "bundle_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
