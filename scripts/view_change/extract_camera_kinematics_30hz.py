#!/usr/bin/env python3
"""Publish the target camera poses actually used for each 30 Hz view frame.

Robot body and Dex3 kinematics stay on the measured 50 Hz control timeline.
This sidecar records only the camera pose selected for each source RGB-D frame,
so asynchronous robot and camera streams remain explicit instead of being
forced into one frame rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-kinematics", type=Path, required=True)
    parser.add_argument("--timeline-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _manifest(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    try:
        source = payload["source"]
        target = payload["target"]
        nearest = np.asarray(
            payload["source_to_target"]["target_indices"], dtype=np.int64
        )
        source_relative = np.asarray(source["timestamps_relative_s"], dtype=np.float64)
        source_absolute = np.asarray(source["timestamps_absolute_s"], dtype=np.float64)
        target_relative = np.asarray(target["timestamps_relative_s"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid timeline manifest: {path}") from exc
    if not (
        source_relative.ndim == source_absolute.ndim == target_relative.ndim == nearest.ndim == 1
    ):
        raise ValueError("timeline arrays must be one-dimensional")
    if len(source_relative) != len(nearest):
        raise ValueError("source timestamps and nearest-target mapping differ in length")
    if np.any(nearest < 0) or np.any(nearest >= len(target_relative)):
        raise ValueError("nearest-target mapping contains an out-of-range index")
    return source_relative, source_absolute, target_relative, nearest


def extract(target_kinematics: Path, timeline_manifest: Path, output: Path) -> dict[str, Any]:
    with np.load(target_kinematics, allow_pickle=False) as archive:
        target = {key: np.asarray(archive[key]) for key in archive.files}
    required = ("timestamps", "camera_pos_w", "camera_quat_wxyz")
    missing = [key for key in required if key not in target]
    if missing:
        raise KeyError(f"target kinematics missing camera fields: {missing}")

    source_relative, source_absolute, target_relative, nearest = _manifest(timeline_manifest)
    target_timestamps = np.asarray(target["timestamps"], dtype=np.float64)
    if len(target_timestamps) != len(target_relative):
        raise ValueError("target kinematics and timeline target timestamps differ in length")

    payload = {
        "camera_pos_w": np.asarray(target["camera_pos_w"], dtype=np.float32)[nearest],
        "camera_quat_wxyz": np.asarray(target["camera_quat_wxyz"], dtype=np.float32)[nearest],
        "timestamps": source_relative,
        "timestamps_absolute_s": source_absolute,
        "target_render_timestamps": target_timestamps[nearest],
        "source_frame_indices": np.arange(len(nearest), dtype=np.int64),
        "target_render_indices": nearest,
        "fps": np.asarray(30, dtype=np.int64),
        "source_fps": np.asarray(30, dtype=np.int64),
        "target_fps": np.asarray(50, dtype=np.int64),
        "alignment_policy": np.asarray("absolute_nearest_target_per_source_frame"),
        "target_kinematics_source": np.asarray(str(target_kinematics.resolve())),
        "timeline_manifest": np.asarray(str(timeline_manifest.resolve())),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(output)
    return {
        "output": str(output.resolve()),
        "frames": int(len(nearest)),
        "fps": 30,
        "target_frames": int(len(target_timestamps)),
        "unique_target_render_frames": int(np.unique(nearest).size),
    }


def main() -> None:
    args = _args()
    print(
        json.dumps(
            extract(
                args.target_kinematics.expanduser().resolve(),
                args.timeline_manifest.expanduser().resolve(),
                args.output.expanduser().resolve(),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
