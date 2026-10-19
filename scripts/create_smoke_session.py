#!/usr/bin/env python3
"""Create a short, self-contained RGB-D session for full-pipeline smoke tests."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np


FRAME_PREFIXES = ("cam-pose/", "depth/", "hand-pose/")


def copy_attrs(source: h5py.AttributeManager, target: h5py.AttributeManager) -> None:
    for key, value in source.items():
        target[key] = value


def copy_annotation(source: Path, target: Path, frames: int) -> dict[str, int]:
    with h5py.File(source, "r") as src, h5py.File(target, "w") as dst:
        source_frames = int(src["cam-pose/timestamps"].shape[0])
        if not 5 <= frames <= source_frames:
            raise ValueError(f"frames must be in 5..{source_frames}, got {frames}")
        end_time = float(src["cam-pose/timestamps"][frames - 1])
        imu_frames = int(
            np.searchsorted(
                np.asarray(src["imu/timestamps"], dtype=np.float64),
                end_time,
                side="right",
            )
        )
        copy_attrs(src.attrs, dst.attrs)

        def visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if isinstance(obj, h5py.Group):
                group = dst.require_group(name)
                copy_attrs(obj.attrs, group.attrs)
                return
            if obj.ndim > 0 and name.startswith(FRAME_PREFIXES):
                values = obj[:frames]
            elif obj.ndim > 0 and name.startswith("imu/"):
                values = obj[:imu_frames]
            else:
                values = obj[()]
            dataset = dst.create_dataset(name, data=values)
            copy_attrs(obj.attrs, dataset.attrs)

        src.visititems(visit)
    return {"source_frames": source_frames, "frames": frames, "imu_frames": imu_frames}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-session", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--session-id", default="session_smoke_template_palm_position")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = args.source_session.expanduser().resolve()
    output = args.output_root.expanduser().resolve() / args.session_id
    required = [
        source / "rgb.mp4",
        source / "annotation.hdf5",
        source / "calibrations/rgb_K.npy",
        source / "calibrations/R_optical_to_link.npy",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() and not args.force:
        manifest = output / "smoke_manifest.json"
        if manifest.is_file():
            print(manifest.read_text(), end="")
            return
        raise FileExistsError(output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    annotation_summary = copy_annotation(
        source / "annotation.hdf5", output / "annotation.hdf5", args.frames
    )
    shutil.copytree(source / "calibrations", output / "calibrations")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source / "rgb.mp4"),
            "-frames:v",
            str(args.frames),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output / "rgb.mp4"),
        ],
        check=True,
    )
    manifest = {
        "format": "ego2humanoid_full_pipeline_smoke_v1",
        "source_session": str(source),
        "output_session": str(output),
        **annotation_summary,
    }
    (output / "smoke_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
