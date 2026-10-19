# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI entry-point for motion format conversion.

Library conversion logic lives in :mod:`kimodo.exports.motion_convert_lib`.
Format detection utilities live in :mod:`kimodo.exports.motion_formats`.
"""

from __future__ import annotations

import argparse
import sys

from kimodo.exports.motion_convert_lib import convert_motion_files


def run_convert(
    input_path: str,
    output_path: str,
    from_fmt: str | None,
    to_fmt: str | None,
    source_fps: float | None,
    z_up: bool,
    mujoco_rest_zero: bool,
    bvh_standard_tpose: bool = False,
) -> None:
    """Thin wrapper kept for backward compatibility; delegates to :func:`convert_motion_files`."""
    convert_motion_files(
        input_path,
        output_path,
        from_fmt=from_fmt,
        to_fmt=to_fmt,
        source_fps=source_fps,
        z_up=z_up,
        mujoco_rest_zero=mujoco_rest_zero,
        bvh_standard_tpose=bvh_standard_tpose,
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert Kimodo NPZ, AMASS NPZ, SOMA BVH, and G1 MuJoCo CSV.",
    )
    p.add_argument("input", help="Input file path")
    p.add_argument("output", help="Output file path")
    p.add_argument(
        "--from",
        dest="from_fmt",
        choices=("amass", "kimodo", "soma-bvh", "g1-csv"),
        default=None,
        help="Input format (default: infer from file contents/extension)",
    )
    p.add_argument(
        "--to",
        dest="to_fmt",
        choices=("kimodo", "amass", "soma-bvh", "g1-csv"),
        default=None,
        help="Output format (default: infer from output extension)",
    )
    p.add_argument(
        "--source-fps",
        "--fps",
        dest="source_fps",
        type=float,
        default=None,
        help=(
            "Source motion frame rate in Hz (default: auto-detected from "
            "BVH Frame Time / AMASS mocap_frame_rate, or 30 Hz). "
            "Kimodo NPZ output is always resampled to 30 Hz."
        ),
    )
    p.add_argument(
        "--no-z-up",
        action="store_true",
        help="For AMASS paths: disable Z-up transform (treat trans/orient as already Kimodo Y-up).",
    )
    p.add_argument(
        "--mujoco-rest-zero",
        action="store_true",
        default=False,
        help="For G1 CSV: joint angles relative to MuJoCo rest (must match export).",
    )
    p.add_argument(
        "--bvh_standard_tpose",
        action="store_true",
        default=False,
        help="If input or output is BVH: the BVH file uses the standard T-pose as its rest pose instead of the BONES-SEED rest pose.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    try:
        convert_motion_files(
            args.input,
            args.output,
            from_fmt=args.from_fmt,
            to_fmt=args.to_fmt,
            source_fps=args.source_fps,
            z_up=not args.no_z_up,
            mujoco_rest_zero=args.mujoco_rest_zero,
            bvh_standard_tpose=args.bvh_standard_tpose,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
