#!/usr/bin/env python3
"""Lift each source session by one constant +Y so first-5 cam height matches a target.

Edits exported egocentric sessions only: mesh.ply and cam-pose translations.
Hands, depth, IMU, calibrations, and RGB are unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import trimesh

TARGET_FIRST5_CAM_Y_M = 1.28
SCHEMA_VERSION = 1
METHOD_SUFFIX = "height_match_shift"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--target-first5-cam-y-m",
        type=float,
        default=TARGET_FIRST5_CAM_Y_M,
        help="Target mean of the first 5 camera Y values (meters)",
    )
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def first5_cam_y(h5_path: Path) -> tuple[float, np.ndarray]:
    with h5py.File(h5_path, "r") as archive:
        y = np.asarray(archive["cam-pose/translations"][:, 1], dtype=np.float64)
    n = min(5, int(y.size))
    if n == 0:
        raise RuntimeError(f"empty cam-pose in {h5_path}")
    return float(y[:n].mean()), y


def mesh_y_extents(mesh_path: Path) -> list[float]:
    mesh = trimesh.load(mesh_path, process=False)
    y = np.asarray(mesh.vertices, dtype=np.float64)[:, 1]
    return [float(y.min()), float(y.max())]


def already_applied(payload: dict, target: float) -> bool:
    extra = payload.get("extra_cam_height_shift")
    record = payload.get("height_match_shift")
    if record is None and extra is None:
        return False
    target_existing = None
    if isinstance(record, dict):
        target_existing = record.get("target_first5_cam_y_m")
    if target_existing is None:
        target_existing = payload.get("target_first5_cam_y_m")
    if target_existing is None:
        return False
    return abs(float(target_existing) - target) <= 1e-9


def shift_session(session: Path, shift_y: float, target: float) -> None:
    mesh_path = session / "mesh.ply"
    h5_path = session / "annotation.hdf5"
    mesh = trimesh.load(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    vertices[:, 1] += shift_y
    mesh.vertices = vertices
    mesh.export(mesh_path)
    with h5py.File(h5_path, "a") as archive:
        translations = archive["cam-pose/translations"]
        values = translations[:]
        values[:, 1] += np.asarray(shift_y, dtype=values.dtype)
        translations[:] = values
        translations.parent.attrs["height_match_target_first5_m"] = float(target)
        translations.parent.attrs["height_match_last_shift_m"] = float(shift_y)
        if "metadata" in archive:
            archive["metadata"].attrs["height_match_last_shift_m"] = float(shift_y)


def update_json(
    path: Path,
    *,
    shift_y: float,
    target: float,
    first5_before: float,
    first5_after: float,
    cam_y_after: list[float],
    mesh_y_after: list[float],
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    method = str(payload.get("method") or "")
    if method and METHOD_SUFFIX not in method:
        payload["method"] = f"{method}+{METHOD_SUFFIX}"
    payload["extra_cam_height_shift"] = float(shift_y)
    payload["extra_cam_height_shift_reason"] = (
        "constant +Y so first-5 cam height matches 1.25-1.31 m standing band"
    )
    payload["target_first5_cam_y_m"] = float(target)
    payload["cam_y_first5_mean_before"] = float(first5_before)
    payload["cam_pose_y_after"] = cam_y_after
    payload["mesh_y_after"] = mesh_y_after
    payload["height_match_shift"] = {
        "schema_version": SCHEMA_VERSION,
        "method": "first5_cam_height_match",
        "target_first5_cam_y_m": float(target),
        "cam_y_first5_mean_before": float(first5_before),
        "cam_y_first5_mean_after": float(first5_after),
        "world_y_shift_m": float(shift_y),
        "applied_to": ["mesh.ply", "annotation.hdf5:/cam-pose/translations"],
        "not_applied_to": ["hand-pose", "depth", "imu", "calibrations", "rgb.mp4"],
        "camera_relative_motion_unchanged": True,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    sessions = (
        [root]
        if root.name.startswith("session_")
        else sorted(path for path in root.glob("session_*") if path.is_dir())
    )
    if not sessions:
        raise SystemExit(f"no sessions under {root}")
    target = float(args.target_first5_cam_y_m)
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for session in sessions:
        try:
            h5_path = session / "annotation.hdf5"
            mesh_path = session / "mesh.ply"
            if not h5_path.is_file() or not mesh_path.is_file():
                raise FileNotFoundError("missing annotation.hdf5 or mesh.ply")
            first5, y = first5_cam_y(h5_path)
            shift_y = target - first5
            ground_path = session / "ground_align.json"
            ground = (
                json.loads(ground_path.read_text(encoding="utf-8"))
                if ground_path.is_file()
                else {}
            )
            row = {
                "session": session.name,
                "first5_before": first5,
                "shift_y": shift_y,
                "n_frames": int(y.size),
            }
            if already_applied(ground, target):
                row["status"] = "already_applied"
                row["first5_after"] = first5
            elif args.apply:
                shift_session(session, shift_y, target)
                first5_after, y_after = first5_cam_y(h5_path)
                mesh_after = mesh_y_extents(mesh_path)
                cam_after = [float(y_after.min()), float(y_after.max())]
                record_kwargs = dict(
                    shift_y=shift_y,
                    target=target,
                    first5_before=first5,
                    first5_after=first5_after,
                    cam_y_after=cam_after,
                    mesh_y_after=mesh_after,
                )
                update_json(ground_path, **record_kwargs)
                update_json(session / "export_manifest.json", **record_kwargs)
                row["status"] = "shifted"
                row["first5_after"] = first5_after
            else:
                row["status"] = "dry_run"
                row["first5_after"] = first5 + shift_y
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{session.name}: {exc}")

    print(
        f"sessions={len(rows)} target_first5_cam_y_m={target:.3f} apply={args.apply}"
    )
    if rows:
        before = np.asarray([float(r["first5_before"]) for r in rows])
        after = np.asarray([float(r["first5_after"]) for r in rows])
        shifts = np.asarray([float(r["shift_y"]) for r in rows])
        print(
            "first5_before min/median/mean/max="
            f"{before.min():.4f}/{np.median(before):.4f}/{before.mean():.4f}/{before.max():.4f}"
        )
        print(
            "first5_after  min/median/mean/max="
            f"{after.min():.4f}/{np.median(after):.4f}/{after.mean():.4f}/{after.max():.4f}"
        )
        print(
            "shift_y       min/median/mean/max="
            f"{shifts.min():.4f}/{np.median(shifts):.4f}/{shifts.mean():.4f}/{shifts.max():.4f}"
        )
        print("session\tstatus\tfirst5_before\tshift_y\tfirst5_after")
        for row in rows:
            print(
                f"{row['session']}\t{row['status']}\t{float(row['first5_before']):.6f}\t"
                f"{float(row['shift_y']):.6f}\t{float(row['first5_after']):.6f}"
            )
    if failures:
        print(f"failures={len(failures)}")
        for line in failures:
            print(line)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
