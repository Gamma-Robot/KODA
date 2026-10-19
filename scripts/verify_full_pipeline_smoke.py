#!/usr/bin/env python3
"""Verify tracking and view-change products from the full-pipeline smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def scalar(archive: dict[str, np.ndarray], key: str) -> str:
    if key not in archive:
        raise KeyError(key)
    return str(np.asarray(archive[key]).reshape(()))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def video_frames(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--kinematics-root", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    kinematics_dir = args.kinematics_root / args.session_id
    view_dir = args.view_root / args.session_id
    body = load_npz(kinematics_dir / "kinematics.npz")
    hand_candidates = (
        view_dir / "dex3_kinematics/hand_dex3_50hz.npz",
        view_dir / "dex3_kinematics/hand_from_tracking_50hz.npz",
    )
    hand_path = next((path for path in hand_candidates if path.is_file()), None)
    if hand_path is None:
        raise FileNotFoundError(hand_candidates[0])
    hand = load_npz(hand_path)
    composed = load_npz(view_dir / "dex3_kinematics/kinematics.npz")
    final_video = view_dir / "final_50hz/rgb_final_50hz.mp4"
    diffueraser_status_path = (
        view_dir / "processed_30hz/eraser/diffueraser_status.json"
    )
    structural_path = view_dir / "quality/structural_acceptance.json"
    for path in (diffueraser_status_path, structural_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    diffueraser = json.loads(diffueraser_status_path.read_text())
    structural = json.loads(structural_path.read_text())

    body_frames = int(body["joint_pos"].shape[0])
    hand_frames = int(hand["qpos"].shape[0])
    composed_frames = int(composed["joint_pos"].shape[0])
    hand_contract = (
        scalar(hand, "retarget_type")
        if "retarget_type" in hand
        else scalar(hand, "source_method")
    )
    hand_contract_ok = hand_contract in {"position", "tracking_kinematics_hand_qpos"}
    checks = {
        "wrist_reference_template_palm_kabsch": (
            scalar(body, "wrist_reference_method") == "template_palm_kabsch"
        ),
        "dex3_hand_contract": hand_contract_ok,
        "frame_counts_match": body_frames == hand_frames == composed_frames,
        "diffueraser_succeeded": bool(diffueraser.get("success", False)),
        "structural_acceptance": structural.get("status") == "pass",
        "final_video_nonempty": video_frames(final_video) == body_frames,
    }
    report = {
        "session_id": args.session_id,
        "body_frames": body_frames,
        "hand_frames": hand_frames,
        "composed_frames": composed_frames,
        "final_video_frames": video_frames(final_video),
        "wrist_reference_method": scalar(body, "wrist_reference_method"),
        "dex3_hand_contract": hand_contract,
        "hand_archive": str(hand_path),
        "checks": checks,
        "passed": all(checks.values()),
        "final_video": str(final_video),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
