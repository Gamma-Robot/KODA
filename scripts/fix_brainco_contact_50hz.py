#!/usr/bin/env python3
"""Rewrite 50Hz contact from nearest-neighbor 30Hz in BrainCo kinematics.npz."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np

ROOT = (
    Path(__file__).resolve().parents[1]
    / "data/whole_body_data/walk_pick_human_brainco"
)
KINEMATICS_RELATIVE = Path("final_50hz") / "kinematics.npz"


def nearest_index(source_time: np.ndarray, body_time: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source_time, body_time, side="left")
    right = np.clip(right, 0, len(source_time) - 1)
    left = np.clip(right - 1, 0, len(source_time) - 1)
    use_right = np.abs(source_time[right] - body_time) < np.abs(
        body_time - source_time[left]
    )
    return np.where(use_right, right, left)


def upsample_contact(path: Path) -> str:
    with np.load(path, allow_pickle=False) as payload:
        data = {name: np.asarray(payload[name]) for name in payload.files}
    required = (
        "left_contact_30hz",
        "right_contact_30hz",
        "contact_30hz_timestamps",
        "timestamps",
        "left_contact",
        "right_contact",
        "contact_50hz",
    )
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"missing {missing}")
    body_time = np.asarray(data["timestamps"], dtype=np.float64).reshape(-1)
    source_time = np.asarray(data["contact_30hz_timestamps"], dtype=np.float64).reshape(
        -1
    )
    left_30 = np.asarray(data["left_contact_30hz"], dtype=np.int8).reshape(-1)
    right_30 = np.asarray(data["right_contact_30hz"], dtype=np.int8).reshape(-1)
    if left_30.shape != source_time.shape or right_30.shape != source_time.shape:
        raise ValueError(
            f"30Hz contact/timestamp length mismatch: "
            f"left={left_30.shape} right={right_30.shape} time={source_time.shape}"
        )
    if not np.all(np.isin(left_30, (0, 1))) or not np.all(np.isin(right_30, (0, 1))):
        raise ValueError("30Hz contact is not binary")
    nearest = nearest_index(source_time, body_time)
    left_50 = left_30[nearest].astype(np.int8)
    right_50 = right_30[nearest].astype(np.int8)
    contact_50 = np.maximum(left_50, right_50).astype(np.int8)
    if (
        np.array_equal(data["left_contact"], left_50)
        and np.array_equal(data["right_contact"], right_50)
        and np.array_equal(data["contact_50hz"], contact_50)
    ):
        return "unchanged"
    data["left_contact"] = left_50
    data["right_contact"] = right_50
    data["contact_50hz"] = contact_50
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **data)
    tmp = path.with_name("kinematics.rewrite.npz")
    tmp.write_bytes(buffer.getvalue())
    tmp.replace(path)
    return "rewritten"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    counts = {"rewritten": 0, "unchanged": 0, "skipped": 0}
    failures: list[str] = []
    for session in sorted(path for path in root.iterdir() if path.is_dir()):
        kinematics = session / KINEMATICS_RELATIVE
        if not kinematics.is_file():
            counts["skipped"] += 1
            continue
        try:
            status = upsample_contact(kinematics)
        except Exception as exc:
            failures.append(f"{session.name}: {type(exc).__name__}: {exc}")
            continue
        counts[status] += 1
    print(
        f"root={root} rewritten={counts['rewritten']} "
        f"unchanged={counts['unchanged']} skipped={counts['skipped']} "
        f"failures={len(failures)}"
    )
    for row in failures:
        print(row)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
