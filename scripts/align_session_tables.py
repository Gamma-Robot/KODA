#!/usr/bin/env python3
"""Align exported session meshes and source camera poses to one table height."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import trimesh


SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--target-height-m", type=float, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def detect_table_height(mesh_path: Path) -> float:
    mesh = trimesh.load(mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"expected one triangle mesh: {mesh_path}")
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    horizontal = np.abs(normals[:, 1]) >= 0.94
    heights = centers[horizontal, 1]
    weights = areas[horizontal]
    if heights.size == 0:
        raise RuntimeError(f"no horizontal faces: {mesh_path}")
    order = np.argsort(heights)
    sorted_heights = heights[order]
    cumulative_area = np.cumsum(weights[order])
    horizontal_area = float(cumulative_area[-1])
    if horizontal_area <= 0.0:
        raise RuntimeError(f"zero horizontal surface area in {mesh_path}")
    index = int(np.searchsorted(cumulative_area, horizontal_area * 0.5))
    return float(sorted_heights[index])


def shift_mesh(mesh_path: Path, shift: float) -> None:
    mesh = trimesh.load(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    vertices[:, 1] += shift
    mesh.vertices = vertices
    mesh.export(mesh_path)


def shift_camera(h5_path: Path, shift: float, target: float) -> None:
    with h5py.File(h5_path, "a") as archive:
        translations = archive["cam-pose/translations"]
        values = translations[:]
        values[:, 1] += np.asarray(shift, dtype=values.dtype)
        translations[:] = values
        translations.parent.attrs["table_height_target_m"] = target
        translations.parent.attrs["table_height_last_shift_m"] = shift


def update_json(path: Path, record: dict[str, object]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    payload["table_height_alignment"] = record
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    sessions = (
        [root]
        if root.is_dir() and root.name.startswith("session_")
        else sorted(path for path in root.glob("session_*") if path.is_dir())
    )
    if not sessions:
        raise RuntimeError(f"no sessions under {root}")
    results = []
    for session in sessions:
        mesh_path = session / "mesh.ply"
        h5_path = session / "annotation.hdf5"
        if not mesh_path.is_file() or not h5_path.is_file():
            raise FileNotFoundError(f"incomplete source geometry: {session}")
        existing_path = session / "ground_align.json"
        existing = json.loads(existing_path.read_text()) if existing_path.is_file() else {}
        prior = existing.get("table_height_alignment")
        prior_total_shift = float(prior.get("world_y_shift_m", 0.0)) if prior else 0.0
        if prior and abs(float(prior["target_height_m"]) - args.target_height_m) < 1e-9:
            measured = detect_table_height(mesh_path)
            if abs(measured - args.target_height_m) <= 0.0001:
                results.append((session.name, measured, 0.0, "already_aligned"))
                continue
        measured = detect_table_height(mesh_path)
        shift = float(args.target_height_m - measured)
        results.append((session.name, measured, shift, "planned" if args.dry_run else "aligned"))
        if args.dry_run:
            continue
        total_shift = prior_total_shift + shift
        shift_mesh(mesh_path, shift)
        shift_camera(h5_path, shift, args.target_height_m)
        after = detect_table_height(mesh_path)
        for _ in range(4):
            residual = float(args.target_height_m - after)
            if abs(residual) <= 0.00005:
                break
            shift_mesh(mesh_path, residual)
            shift_camera(h5_path, residual, args.target_height_m)
            total_shift += residual
            after = detect_table_height(mesh_path)
        record = {
            "schema_version": SCHEMA_VERSION,
            "method": "area_weighted_median_horizontal_mesh_faces",
            "axis": "source_world_y_up",
            "target_height_m": float(args.target_height_m),
            "detected_height_before_m": measured,
            "world_y_shift_m": total_shift,
            "detected_height_after_m": after,
            "applied_to": ["mesh.ply", "annotation.hdf5:/cam-pose/translations"],
            "not_applied_to": [
                "kinematics.npz",
                "camera_kinematics_30hz.npz",
                "rgb_final_50hz.mp4",
                "rgb.mp4",
                "depth",
                "hand-pose",
                "imu",
                "calibrations",
            ],
            "derived_outputs_require_regeneration": True,
        }
        update_json(existing_path, record)
        update_json(session / "export_manifest.json", record)
    shifts = np.asarray([item[2] for item in results], dtype=np.float64)
    print(f"sessions={len(results)} dry_run={args.dry_run}")
    print(
        "shift_m min/median/mean/max="
        f"{shifts.min():.6f}/{np.median(shifts):.6f}/{shifts.mean():.6f}/{shifts.max():.6f}"
    )
    for item in results:
        print(f"{item[0]} table={item[1]:.6f} shift={item[2]:+.6f} {item[3]}")


if __name__ == "__main__":
    main()
