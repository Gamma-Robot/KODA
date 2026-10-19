#!/usr/bin/env python3
"""Apply one audited world translation to an exported source session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--table-height-before-m", type=float, required=True)
    parser.add_argument("--target-table-height-m", type=float, default=0.75005)
    parser.add_argument("--recenter-first-camera-xz", action="store_true")
    return parser.parse_args()


def update_json(path: Path, record: dict[str, object]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    payload["world_origin_repair"] = record
    if "table_height_alignment" in payload:
        table = payload["table_height_alignment"]
        table["method"] = "semantic_upper_table_plane_manual_audit"
        table["target_height_m"] = record["target_table_height_m"]
        table["detected_height_before_m"] = record["table_height_before_m"]
        table["world_y_shift_m"] = record["world_translation_xyz_m"][1]
        table["detected_height_after_m"] = record["target_table_height_m"]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    session = args.session.expanduser().resolve()
    mesh_path = session / "mesh.ply"
    h5_path = session / "annotation.hdf5"
    if not mesh_path.is_file() or not h5_path.is_file():
        raise FileNotFoundError(f"incomplete source session: {session}")

    ground_path = session / "ground_align.json"
    ground_payload = (
        json.loads(ground_path.read_text(encoding="utf-8"))
        if ground_path.is_file()
        else {}
    )
    prior = ground_payload.get("world_origin_repair")
    if prior is not None:
        same_target = abs(
            float(prior["target_table_height_m"]) - args.target_table_height_m
        ) <= 1.0e-9
        same_source = abs(
            float(prior["table_height_before_m"]) - args.table_height_before_m
        ) <= 1.0e-6
        if same_target and same_source:
            print(json.dumps({"status": "already_applied", **prior}, indent=2))
            return
        raise RuntimeError(f"different world-origin repair already exists: {prior}")

    with h5py.File(h5_path, "r") as archive:
        translations = np.asarray(archive["cam-pose/translations"])
    shift = np.zeros(3, dtype=np.float64)
    if args.recenter_first_camera_xz:
        shift[[0, 2]] = -translations[0, [0, 2]]
    shift[1] = args.target_table_height_m - args.table_height_before_m

    mesh = trimesh.load(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    vertices += shift
    mesh.vertices = vertices
    mesh.export(mesh_path)

    with h5py.File(h5_path, "a") as archive:
        dataset = archive["cam-pose/translations"]
        values = dataset[:]
        values += shift.astype(values.dtype)
        dataset[:] = values
        dataset.parent.attrs["world_origin_repair_translation_xyz_m"] = shift

    record = {
        "schema_version": 1,
        "method": "semantic_table_height_and_first_camera_horizontal_origin",
        "table_height_before_m": float(args.table_height_before_m),
        "target_table_height_m": float(args.target_table_height_m),
        "world_translation_xyz_m": shift.tolist(),
        "first_camera_xyz_before_m": translations[0].astype(float).tolist(),
        "first_camera_xyz_after_m": (translations[0] + shift).astype(float).tolist(),
        "applied_to": ["mesh.ply", "annotation.hdf5:/cam-pose/translations"],
        "camera_relative_motion_unchanged": True,
    }
    update_json(ground_path, record)
    update_json(session / "export_manifest.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
