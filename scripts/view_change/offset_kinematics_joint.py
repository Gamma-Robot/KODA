#!/usr/bin/env python3
"""Apply a constant joint-position offset to a named kinematics trajectory."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--joint-name", default=None, help="Legacy single-joint interface.")
    parser.add_argument("--offset-deg", type=float, default=None, help="Legacy single-joint interface.")
    parser.add_argument(
        "--joint-offset",
        action="append",
        default=[],
        metavar="NAME=DEG",
        help="Repeatable named offset, for example left_wrist_yaw_joint=2.",
    )
    parser.add_argument("--min-rad", type=float, default=None)
    parser.add_argument("--max-rad", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _requested_offsets(args: argparse.Namespace) -> list[tuple[str, float]]:
    if (args.joint_name is None) != (args.offset_deg is None):
        raise ValueError("--joint-name and --offset-deg must be supplied together")
    requested: list[tuple[str, float]] = []
    if args.joint_name is not None:
        requested.append((str(args.joint_name), float(args.offset_deg)))
    for value in args.joint_offset:
        if "=" not in value:
            raise ValueError(f"invalid --joint-offset {value!r}; expected NAME=DEG")
        name, degrees = value.rsplit("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("joint offset name cannot be empty")
        requested.append((name, float(degrees)))
    if not requested:
        raise ValueError("at least one joint offset is required")
    names = [name for name, _ in requested]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate joint offsets: {duplicates}")
    if not all(math.isfinite(degrees) for _, degrees in requested):
        raise ValueError("joint offsets must be finite")
    return requested


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = _args()
    source_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_path.exists() and not args.force:
        raise FileExistsError(f"output exists; use --force: {output_path}")
    if (args.min_rad is None) != (args.max_rad is None):
        raise ValueError("--min-rad and --max-rad must be supplied together")
    with np.load(source_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if "joint_pos" not in arrays or "joint_names" not in arrays:
        raise KeyError("kinematics requires joint_pos and joint_names")
    names = [str(value) for value in arrays["joint_names"].reshape(-1).tolist()]
    requested = _requested_offsets(args)
    for joint_name, _ in requested:
        if names.count(joint_name) != 1:
            raise ValueError(f"expected one {joint_name!r}, found {names.count(joint_name)}")
    joint_pos = np.asarray(arrays["joint_pos"])
    if joint_pos.ndim != 2 or joint_pos.shape[1] != len(names):
        raise ValueError("joint_pos and joint_names shape mismatch")
    adjusted = joint_pos.copy()
    reports = []
    for joint_name, offset_deg in requested:
        index = names.index(joint_name)
        original_values = np.asarray(joint_pos[:, index], dtype=np.float64)
        if not np.all(np.isfinite(original_values)):
            raise ValueError(f"{joint_name} contains non-finite values")
        offset_rad = math.radians(offset_deg)
        adjusted[:, index] = adjusted[:, index] + offset_rad
        values = np.asarray(adjusted[:, index], dtype=np.float64)
        if args.min_rad is not None:
            if not float(args.min_rad) < float(args.max_rad):
                raise ValueError("invalid joint limits")
            violations = (values < float(args.min_rad)) | (values > float(args.max_rad))
            if np.any(violations):
                bad = np.flatnonzero(violations)[:12].tolist()
                raise ValueError(
                    f"offset violates {joint_name} limits at frames {bad}; "
                    f"range=[{values.min():.9g}, {values.max():.9g}]"
                )
        reports.append(
            {
                "joint_name": joint_name,
                "joint_index": index,
                "offset_deg": offset_deg,
                "offset_rad": offset_rad,
                "original_range_rad": [
                    float(np.min(original_values)),
                    float(np.max(original_values)),
                ],
                "adjusted_range_rad": [float(np.min(values)), float(np.max(values))],
                "joint_limit_rad": (
                    [float(args.min_rad), float(args.max_rad)]
                    if args.min_rad is not None
                    else None
                ),
            }
        )
    arrays["joint_pos"] = adjusted
    arrays["joint_offset_names"] = np.asarray([row["joint_name"] for row in reports])
    arrays["joint_offset_degrees"] = np.asarray(
        [row["offset_deg"] for row in reports], dtype=np.float64
    )
    arrays["joint_offset_radians"] = np.asarray(
        [row["offset_rad"] for row in reports], dtype=np.float64
    )
    arrays["joint_offset_source"] = np.asarray(str(source_path))
    for name in ("joint_offset_name", "joint_offset_deg", "joint_offset_rad"):
        arrays.pop(name, None)
    if len(reports) == 1:
        arrays["joint_offset_name"] = np.asarray(reports[0]["joint_name"])
        arrays["joint_offset_deg"] = np.asarray(reports[0]["offset_deg"], dtype=np.float64)
        arrays["joint_offset_rad"] = np.asarray(reports[0]["offset_rad"], dtype=np.float64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_npz(output_path, arrays)
    report = {
        "status": "pass",
        "source": str(source_path),
        "output": str(output_path),
        "offsets": reports,
        "frames": int(joint_pos.shape[0]),
        "joint_vel_unchanged": True,
    }
    if len(reports) == 1:
        report.update(reports[0])
    (output_path.parent / "joint_offsets.json").write_text(json.dumps(report, indent=2) + "\n")
    if len(reports) == 1:
        (output_path.parent / "joint_offset.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
