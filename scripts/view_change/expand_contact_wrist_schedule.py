#!/usr/bin/env python3
"""Expand contact-frame wrist candidates into a causal full-timeline SE(3) schedule.

This is intentionally independent of the robot joint representation.  The
schedule stores a residual for ``*_wrist_yaw_link`` (world translation plus
base-local SO(3) rotation), so it can be audited before the tracking pipeline
consumes it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-json", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.candidates.expanduser().resolve(), allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    required = ("frame_indices", "base_position", "base_rotation", "causal_position", "causal_rotation")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"candidate missing {missing}")
    base_position = np.asarray(data["base_position"], dtype=np.float64)
    base_rotation = np.asarray(data["base_rotation"], dtype=np.float64)
    causal_position = np.asarray(data["causal_position"], dtype=np.float64)
    causal_rotation = np.asarray(data["causal_rotation"], dtype=np.float64)
    count = len(base_position)
    if any(value.shape[0] != count for value in (base_rotation, causal_position, causal_rotation)):
        raise ValueError("candidate arrays do not share a timeline")
    contact = np.asarray(data["frame_indices"], dtype=np.int64).reshape(-1)
    if not len(contact) or np.any(np.diff(contact) != 1):
        raise ValueError("frame_indices must be a non-empty contiguous contact interval")
    if contact[0] < 0 or contact[-1] >= count:
        raise ValueError("contact frame indices outside candidate timeline")

    # The candidate already contains the optimized causal pose.  Convert it to
    # explicit per-frame SE(3) residuals and hold the edge residuals causally.
    delta_world = causal_position - base_position
    delta_local = np.asarray([
        Rotation.from_matrix(base_rotation[i].T @ causal_rotation[i]).as_rotvec()
        for i in range(count)
    ])
    schedule_world = np.empty_like(delta_world)
    schedule_local = np.empty_like(delta_local)
    schedule_world[: contact[0]] = delta_world[contact[0]]
    schedule_local[: contact[0]] = delta_local[contact[0]]
    schedule_world[contact[-1] + 1 :] = delta_world[contact[-1]]
    schedule_local[contact[-1] + 1 :] = delta_local[contact[-1]]
    schedule_world[contact[0] : contact[-1] + 1] = delta_world[contact[0] : contact[-1] + 1]
    schedule_local[contact[0] : contact[-1] + 1] = delta_local[contact[0] : contact[-1] + 1]
    schedule_rotation = np.asarray([
        base_rotation[i] @ Rotation.from_rotvec(schedule_local[i]).as_matrix()
        for i in range(count)
    ])
    schedule_position = base_position + schedule_world
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.expanduser().resolve(),
        target_indices=np.arange(count, dtype=np.int64),
        contact_frame_indices=contact,
        first_contact=np.asarray(contact[0], dtype=np.int64),
        last_contact=np.asarray(contact[-1], dtype=np.int64),
        translation_world_m=schedule_world,
        rotation_local_rotvec_rad=schedule_local,
        scheduled_position=schedule_position,
        scheduled_rotation=schedule_rotation,
        optimization_space=np.asarray(f"{args.side}_wrist_yaw_link_SE3"),
        not_joint_yaw_offset=np.asarray(True),
        edge_hold_policy=np.asarray("first_contact / per_contact / last_contact"),
    )
    payload = {
        "schema_version": 1,
        "contact_frames": contact.tolist(),
        "first_contact": int(contact[0]),
        "last_contact": int(contact[-1]),
        "timeline_frames": count,
        "optimization_space": f"{args.side}_wrist_yaw_link_SE3",
        "not_joint_yaw_offset": True,
        "policy": {
            "before": "first_contact_offset",
            "contact": "per_frame_offset",
            "after": "last_contact_offset",
        },
        "candidate": str(args.candidates.expanduser().resolve()),
        "schedule": str(args.output.expanduser().resolve()),
    }
    args.contact_json.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.contact_json.expanduser().resolve().write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
