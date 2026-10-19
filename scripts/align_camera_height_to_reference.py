#!/usr/bin/env python3
"""Match early camera height across exported egocentric sessions.

Some conversions never see the floor, so RANSAC lifts a table/shelf to y=0 and
the headset starts too low. A session that does see the floor keeps a valid
early camera Y; other sessions are shifted in world Y so their first frames
match that height.

Edits mesh.ply and annotation.hdf5:/cam-pose/translations only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import trimesh

SCHEMA_VERSION = 1
DEFAULT_EARLY_FRAMES = 5
RECORD_KEY = "camera_height_reference_alignment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-session", type=str, required=True)
    parser.add_argument("--early-frames", type=int, default=DEFAULT_EARLY_FRAMES)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Shift mesh.ply and cam-pose translations",
    )
    return parser.parse_args()


def session_dirs(root: Path) -> list[Path]:
    if root.name.startswith("session_"):
        return [root]
    sessions = sorted(path for path in root.glob("session_*") if path.is_dir())
    if not sessions:
        raise SystemExit(f"no sessions under {root}")
    return sessions


def early_camera_y(session: Path, n_frames: int) -> tuple[float, np.ndarray]:
    with h5py.File(session / "annotation.hdf5", "r") as archive:
        translations = np.asarray(archive["cam-pose/translations"], dtype=np.float64)
    if translations.ndim != 2 or translations.shape[1] < 2:
        raise ValueError(f"{session.name}: invalid cam-pose/translations")
    n = min(int(n_frames), int(translations.shape[0]))
    if n <= 0:
        raise ValueError(f"{session.name}: no camera frames")
    return float(np.mean(translations[:n, 1])), translations[:, 1].copy()


def mesh_y_range(session: Path) -> list[float] | None:
    mesh_path = session / "mesh.ply"
    if not mesh_path.is_file():
        return None
    mesh = trimesh.load(mesh_path, process=False)
    y = np.asarray(mesh.vertices, dtype=np.float64)[:, 1]
    return [float(y.min()), float(y.max())]


def measure(session: Path, n_frames: int) -> dict[str, object]:
    cam_y, all_y = early_camera_y(session, n_frames)
    ground_path = session / "ground_align.json"
    ground = json.loads(ground_path.read_text()) if ground_path.is_file() else {}
    offset = ground.get("ground_y_offset")
    existing = ground.get(RECORD_KEY)
    return {
        "session": session.name,
        "n_frames": int(min(n_frames, all_y.size)),
        "early_cam_y_m": cam_y,
        "cam_y_min_m": float(all_y.min()),
        "cam_y_max_m": float(all_y.max()),
        "mesh_y_after": mesh_y_range(session),
        "ground_y_offset": None if offset is None else float(offset),
        "already_shifted_m": (
            None
            if not existing
            else float(existing.get("world_y_shift_m", 0.0))
        ),
    }


def shift_source(session: Path, shift_y: float) -> None:
    mesh_path = session / "mesh.ply"
    mesh = trimesh.load(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    vertices[:, 1] += shift_y
    mesh.vertices = vertices
    mesh.export(mesh_path)
    with h5py.File(session / "annotation.hdf5", "a") as archive:
        translations = archive["cam-pose/translations"]
        values = translations[:]
        values[:, 1] += np.asarray(shift_y, dtype=values.dtype)
        translations[:] = values
        group = translations.parent
        group.attrs["camera_height_last_shift_m"] = float(shift_y)


def write_record(session: Path, record: dict[str, object]) -> None:
    mesh_y = mesh_y_range(session)
    with h5py.File(session / "annotation.hdf5", "r") as archive:
        cam = np.asarray(archive["cam-pose/translations"][:, 1], dtype=np.float64)
    cam_y = [float(cam.min()), float(cam.max())]
    for name in ("ground_align.json", "export_manifest.json"):
        path = session / name
        payload = json.loads(path.read_text()) if path.is_file() else {}
        if name == "export_manifest.json":
            ground = payload.setdefault("ground_align", {})
        else:
            ground = payload
        ground[RECORD_KEY] = record
        ground["mesh_y_after"] = mesh_y
        ground["cam_pose_y_after"] = cam_y
        path.write_text(json.dumps(payload, indent=2) + "\n")


def summarize(rows: list[dict[str, object]], target: float) -> None:
    print(
        f"reference_cam_y_m={target:.6f} sessions={len(rows)}"
    )
    print(
        "session\tstatus\tearly_cam_y_m\tshift_m\tcam_min\tcam_max"
    )
    for row in rows:
        print(
            f"{row['session']}\t{row['status']}\t{float(row['early_cam_y_m']):.6f}\t"
            f"{float(row.get('world_y_shift_m', 0.0)):.6f}\t"
            f"{float(row['cam_y_min_m']):.6f}\t{float(row['cam_y_max_m']):.6f}"
        )


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    sessions = session_dirs(root)
    ref_name = args.reference_session
    if ref_name.endswith("/"):
        ref_name = ref_name.rstrip("/")
    if "/" in ref_name:
        ref_name = Path(ref_name).name
    ref_dir = root / ref_name if not root.name.startswith("session_") else root
    if not (ref_dir / "annotation.hdf5").is_file():
        raise SystemExit(f"missing reference session {ref_dir}")
    target, _ = early_camera_y(ref_dir, args.early_frames)
    rows: list[dict[str, object]] = []
    for session in sessions:
        row = measure(session, args.early_frames)
        shift = target - float(row["early_cam_y_m"])
        row["reference_session"] = ref_name
        row["reference_cam_y_m"] = target
        row["world_y_shift_m"] = 0.0 if session.name == ref_name else shift
        if session.name == ref_name:
            row["status"] = "reference"
        elif abs(shift) < 1e-4:
            row["status"] = "already_matched"
        else:
            row["status"] = "needs_shift"
        if args.apply and row["status"] == "needs_shift":
            existing = row["already_shifted_m"]
            if existing is not None and abs(float(existing) - shift) <= 1e-4:
                row["status"] = "already_applied"
            else:
                remaining = shift if existing is None else shift
                # If a previous correction exists, remaining is still
                # target - current, which is the residual to apply now.
                shift_source(session, remaining)
                after_y, after_all = early_camera_y(session, args.early_frames)
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "method": "match_early_camera_y_to_reference",
                    "reference_session": ref_name,
                    "early_frames": int(args.early_frames),
                    "reference_cam_y_m": target,
                    "session_cam_y_before_m": float(row["early_cam_y_m"]),
                    "world_y_shift_m": float(remaining),
                    "session_cam_y_after_m": float(after_y),
                    "applied_to": [
                        "mesh.ply",
                        "annotation.hdf5:/cam-pose/translations",
                    ],
                    "not_applied_to": [
                        "hand-pose",
                        "depth",
                        "imu",
                        "calibrations",
                        "rgb.mp4",
                    ],
                }
                write_record(session, record)
                row["early_cam_y_m"] = after_y
                row["cam_y_min_m"] = float(after_all.min())
                row["cam_y_max_m"] = float(after_all.max())
                row["world_y_shift_m"] = float(remaining)
                row["status"] = "shifted"
        rows.append(row)
    summarize(rows, target)


if __name__ == "__main__":
    main()
