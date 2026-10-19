#!/usr/bin/env python3
"""Write EgoHumanoid open/close as contact_frames.json and optional kinematics flags."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
VIEW_CHANGE_DIR = Path(__file__).resolve().parent
for path in (SCRIPTS_DIR, VIEW_CHANGE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from egohumanoid_hand_status import compute_native_close_wave
from migrate_old_brainco_contact import _arrays_from_contact_json, _write_npz

DETECTOR = "egohumanoid_finger_curvature"


def _closed_frames(wave: np.ndarray) -> list[int]:
    return np.flatnonzero(np.asarray(wave) >= 0.5).astype(np.int64).tolist()


def load_native_hands(annotation: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(annotation, "r") as archive:
        timestamps = np.asarray(archive["hand-pose/timestamps"], dtype=np.float64)
        left = np.asarray(archive["hand-pose/left_joints"], dtype=np.float64)
        right = np.asarray(archive["hand-pose/right_joints"], dtype=np.float64)
    if left.ndim != 3 or left.shape[1:] != (21, 3):
        raise ValueError(f"{annotation}: left_joints must be (T, 21, 3), got {left.shape}")
    if right.shape != left.shape:
        raise ValueError(f"{annotation}: right_joints {right.shape} != left {left.shape}")
    if timestamps.shape != (left.shape[0],):
        raise ValueError(f"{annotation}: timestamps {timestamps.shape} != {left.shape[0]}")
    return left, right, timestamps - timestamps[0]


def build_contact_payload(left_frames: list[int], right_frames: list[int]) -> dict[str, Any]:
    union = sorted(set(left_frames) | set(right_frames))
    return {
        "schema_version": 1,
        "contact_frames": union,
        "left_contact_frames": list(left_frames),
        "right_contact_frames": list(right_frames),
        "first_contact": union[0] if union else None,
        "last_contact": union[-1] if union else None,
        "optimization_space": "per_hand_wrist_yaw_link_SE3",
        "not_joint_yaw_offset": True,
        "policy": {
            "before": "first_contact_offset",
            "contact": "per_frame_offset",
            "after": "last_contact_offset",
        },
        "status": "egohumanoid_hand_closed" if union else "no_contact_detected",
        "contact_detector": DETECTOR,
        "method": "egohumanoid_native_3d_mano21_square_wave_010",
    }


def patch_kinematics(path: Path, payload: dict[str, Any], source_time: np.ndarray) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    body_time = np.asarray(arrays["timestamps"], dtype=np.float64)
    contact = _arrays_from_contact_json(
        payload,
        n_source=int(source_time.shape[0]),
        body_time=body_time,
        source_time=source_time,
        detector=DETECTOR,
    )
    arrays.update(contact)
    _write_npz(path, arrays)
    return {
        "path": str(path),
        "left_50hz": int(np.asarray(contact["left_contact"]).sum()),
        "right_50hz": int(np.asarray(contact["right_contact"]).sum()),
        "left_30hz": int(np.asarray(contact["left_contact_30hz"]).sum()),
        "right_30hz": int(np.asarray(contact["right_contact_30hz"]).sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--kinematics", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation = args.session_dir / "annotation.hdf5"
    if not annotation.is_file():
        raise FileNotFoundError(annotation)
    left_joints, right_joints, source_time = load_native_hands(annotation)
    left_frames = _closed_frames(compute_native_close_wave(left_joints))
    right_frames = _closed_frames(compute_native_close_wave(right_joints))
    payload = build_contact_payload(left_frames, right_frames)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    reports = []
    seen: set[Path] = set()
    for kinematics in args.kinematics:
        if not kinematics.is_file():
            continue
        resolved = kinematics.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        reports.append(patch_kinematics(resolved, payload, source_time))
    print(
        json.dumps(
            {
                "annotation": str(annotation),
                "contact_json": str(args.output_json),
                "left_frames": len(left_frames),
                "right_frames": len(right_frames),
                "status": payload["status"],
                "kinematics": reports,
            }
        )
    )


if __name__ == "__main__":
    main()
