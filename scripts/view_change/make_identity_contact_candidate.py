#!/usr/bin/env python3
"""Create an explicit identity contact candidate for a no-contact episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--wrist-method", default="template_palm_kabsch")
    args = parser.parse_args()
    experiment = args.experiment_dir.expanduser().resolve()
    with np.load(experiment / "wrist_candidates_native.npz", allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    position_key = f"{args.wrist_method}__{args.side}_position"
    rotation_key = f"{args.wrist_method}__{args.side}_rotation"
    if position_key not in data or rotation_key not in data:
        raise KeyError(f"missing {position_key}/{rotation_key}")
    position = np.asarray(data[position_key], dtype=np.float64)
    rotation = np.asarray(data[rotation_key], dtype=np.float64)
    timestamps = np.asarray(data["timestamps"], dtype=np.float64)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.asarray(1, dtype=np.int32),
        timestamps=timestamps,
        frame_indices=np.asarray([0], dtype=np.int64),
        base_position=position,
        base_rotation=rotation,
        causal_position=position.copy(),
        causal_rotation=rotation.copy(),
        status=np.asarray("no_contact_detected"),
        reason=np.asarray(args.reason),
        optimization_space=np.asarray(f"{args.side}_wrist_yaw_link_SE3"),
        not_joint_yaw_offset=np.asarray(True),
    )
    output.with_name("contact_candidate_status.json").write_text(
        json.dumps({"status": "no_contact_detected", "reason": args.reason}, indent=2)
        + "\n"
    )
    print(str(output))


if __name__ == "__main__":
    main()
