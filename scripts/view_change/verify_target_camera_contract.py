#!/usr/bin/env python3
"""Verify that rendered target streams use the configured camera contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from frame_volume import is_packed_stream, open_volume
except ImportError:  # pragma: no cover - package-style import
    from .frame_volume import is_packed_stream, open_volume

try:
    from frame_volume import is_packed_stream, open_volume
except ImportError:  # pragma: no cover - package-style import
    from .frame_volume import is_packed_stream, open_volume


def _load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = ("width", "height", "K")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{path} is missing {missing}")
    return data


def _matrix(data: dict) -> np.ndarray:
    matrix = np.asarray(data["K"], dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"invalid target K: {matrix}")
    return matrix


def _first_frame(directory: Path) -> Path:
    frames = sorted(
        (path for path in directory.iterdir() if path.is_file() and path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    if not frames:
        raise FileNotFoundError(f"no indexed frames in {directory}")
    return frames[0]


def _shape(path: Path) -> tuple[int, int]:
    if path.suffix == ".npy":
        image = np.load(path, mmap_mode="r")
    else:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"failed to decode {path}")
    if image.ndim < 2:
        raise ValueError(f"invalid image shape for {path}: {image.shape}")
    return int(image.shape[0]), int(image.shape[1])


def _stream_shape(directory: Path) -> tuple[int, int]:
    if is_packed_stream(directory):
        volume = open_volume(directory)
        return int(volume.frame_shape[0]), int(volume.frame_shape[1])
    return _shape(_first_frame(directory))




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--robot-id-dir", type=Path, required=True)
    parser.add_argument("--warp-meta", type=Path)
    parser.add_argument("--warp-rgb-dir", type=Path)
    parser.add_argument("--warp-depth-dir", type=Path)
    args = parser.parse_args()

    expected = _load(args.expected)
    actual = _load(args.actual)
    width, height = int(expected["width"]), int(expected["height"])
    if (int(actual["width"]), int(actual["height"])) != (width, height):
        raise ValueError(
            "target image size mismatch: "
            f"expected {(width, height)}, actual {(actual['width'], actual['height'])}"
        )
    if not np.allclose(_matrix(actual), _matrix(expected), rtol=0.0, atol=1e-9):
        raise ValueError("rendered target K differs from configured target K")
    for key in ("resize_convention", "projection_contract"):
        if actual.get(key) != expected.get(key):
            raise ValueError(
                f"target calibration field {key!r} mismatch: "
                f"expected {expected.get(key)!r}, actual {actual.get(key)!r}"
            )

    expected_shape = (height, width)
    streams = {
        "rgb": args.rgb_dir,
        "depth": args.depth_dir,
        "robot_id": args.robot_id_dir,
    }
    shapes = {}
    for name, directory in streams.items():
        shape = _stream_shape(directory)
        shapes[name] = list(shape)
        if shape != expected_shape:
            raise ValueError(
                f"{name} target frame has shape {shape}, expected {expected_shape}: {directory}"
            )
    warp = None
    if args.warp_meta is not None:
        if args.warp_rgb_dir is None or args.warp_depth_dir is None:
            raise ValueError("warp RGB/depth directories are required with --warp-meta")
        warp = json.loads(args.warp_meta.read_text(encoding="utf-8"))
        if tuple(map(int, warp.get("target_size", []))) != (width, height):
            raise ValueError(
                f"warp target size differs from calibration: {warp.get('target_size')}"
            )
        if not np.allclose(
            np.asarray(warp.get("target_K"), dtype=np.float64),
            _matrix(expected),
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError("warp target K differs from configured target K")
        for name, directory in {
            "warp_rgb": args.warp_rgb_dir,
            "warp_depth": args.warp_depth_dir,
        }.items():
            shape = _stream_shape(directory)
            shapes[name] = list(shape)
            if shape != expected_shape:
                raise ValueError(
                    f"{name} frame has shape {shape}, expected {expected_shape}: {directory}"
                )
    print(
        json.dumps(
            {
                "status": "pass",
                "width": width,
                "height": height,
                "K": _matrix(expected).tolist(),
                "stream_shapes": shapes,
                "warp_verified": warp is not None,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
