#!/usr/bin/env python3
"""Measure/align table height from the first seconds of source RGB-D.

This edits exported egocentric source sessions only (mesh.ply and
cam-pose translations). It does not change the tracking/view-change pipeline.

The conversion RANSAC takes the lowest horizontal plane on the full mesh.
The table is only visible in the first seconds, so this detector uses those
frames and keeps the dominant (most-inlier) near-horizontal plane instead.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import trimesh


TARGET_TABLE_Y_M = 0.758
EARLY_SECONDS = 3.0
SCHEMA_VERSION = 1
TABLE_AS_GROUND_MAX_M = 0.20
FLOOR_AS_GROUND_TOL_M = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--target-height-m", type=float, default=TARGET_TABLE_Y_M)
    parser.add_argument("--early-seconds", type=float, default=EARLY_SECONDS)
    parser.add_argument("--apply", action="store_true", help="Shift source mesh and cam-pose")
    return parser.parse_args()


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def load_calibration(session: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    calib = session / "calibrations"
    depth_k_path = _first_existing([calib / "depth_K.npy", calib / "K_depth.npy"])
    depth_d_path = _first_existing([calib / "depth_D.npy", calib / "D_depth.npy"])
    o2l_path = calib / "R_optical_to_link.npy"
    if depth_k_path is None:
        raise FileNotFoundError(f"missing depth_K.npy under {calib}")
    if not o2l_path.is_file():
        raise FileNotFoundError(o2l_path)
    depth_k = np.load(depth_k_path).astype(np.float64)
    depth_d = (
        np.load(depth_d_path).astype(np.float64).reshape(-1)
        if depth_d_path is not None
        else np.zeros(5, dtype=np.float64)
    )
    if depth_d.size < 5:
        depth_d = np.pad(depth_d, (0, 5 - depth_d.size))
    r_o2l = np.load(o2l_path).astype(np.float64)
    return depth_k, depth_d[:5], r_o2l


def unproject_depth_optical(
    depth_m: np.ndarray,
    valid: np.ndarray,
    k: np.ndarray,
    d: np.ndarray,
    stride: int = 4,
) -> np.ndarray:
    h, w = depth_m.shape
    if np.asarray(valid).ndim == 0:
        if not bool(valid):
            return np.zeros((0, 3), dtype=np.float64)
        pixel_ok = np.ones((h, w), dtype=bool)
    else:
        pixel_ok = np.asarray(valid, dtype=bool)
        if pixel_ok.shape != (h, w):
            raise ValueError(f"valid shape {pixel_ok.shape} != depth {(h, w)}")
    ys = np.arange(0, h, stride, dtype=np.float64)
    xs = np.arange(0, w, stride, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    z = depth_m[::stride, ::stride]
    ok = pixel_ok[::stride, ::stride] & np.isfinite(z) & (z > 0.25) & (z < 2.5)
    if not np.any(ok):
        return np.zeros((0, 3), dtype=np.float64)
    pixels = np.stack([grid_x[ok], grid_y[ok]], axis=1).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pixels, k, d).reshape(-1, 2)
    zz = z[ok]
    return np.column_stack([und[:, 0] * zz, und[:, 1] * zz, zz])


def to_world_yup(
    optical: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    r_o2l: np.ndarray,
) -> np.ndarray:
    return (rotation @ r_o2l @ optical.T).T + translation.reshape(1, 3)


def fit_dominant_horizontal_plane(
    xyz: np.ndarray,
    *,
    thresh: float = 0.015,
    min_up: float = 0.9,
    rng: np.random.Generator,
    max_iters: int = 600,
) -> dict[str, float | int]:
    if xyz.shape[0] < 200:
        raise RuntimeError(f"too few early points: {xyz.shape[0]}")
    fit = xyz
    if fit.shape[0] > 25000:
        fit = xyz[rng.choice(xyz.shape[0], 25000, replace=False)]
    best_count = 0
    best_normal = None
    best_d = None
    for _ in range(max_iters):
        idx = rng.choice(fit.shape[0], 3, replace=False)
        p0, p1, p2 = fit[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal = normal / norm
        if normal[1] < 0:
            normal = -normal
        if abs(normal[1]) < min_up:
            continue
        plane_d = float(-normal @ p0)
        dist = np.abs(fit @ normal + plane_d)
        count = int(np.count_nonzero(dist <= thresh))
        if count > best_count:
            best_count = count
            best_normal = normal
            best_d = plane_d
    if best_normal is None or best_count < 200:
        raise RuntimeError(f"no dominant horizontal plane (best_inliers={best_count})")
    dist = np.abs(xyz @ best_normal + best_d)
    inliers = xyz[dist <= thresh]
    return {
        "table_y_m": float(np.mean(inliers[:, 1])),
        "n_inliers": int(inliers.shape[0]),
        "n_candidates": int(xyz.shape[0]),
        "up_alignment": float(abs(best_normal[1])),
    }


def classify(table_y: float, target: float) -> str:
    if abs(table_y) <= TABLE_AS_GROUND_MAX_M:
        return "table_as_ground"
    if abs(table_y - target) <= FLOOR_AS_GROUND_TOL_M:
        return "floor_as_ground"
    return "other"


def measure_early_table(session: Path, early_seconds: float) -> dict[str, object]:
    h5_path = session / "annotation.hdf5"
    ground_path = session / "ground_align.json"
    depth_k, depth_d, r_o2l = load_calibration(session)
    with h5py.File(h5_path, "r") as archive:
        timestamps = np.asarray(archive["cam-pose/timestamps"], dtype=np.float64)
        translations = np.asarray(archive["cam-pose/translations"], dtype=np.float64)
        rotations = np.asarray(archive["cam-pose/rotations"], dtype=np.float64)
        depth_mm = np.asarray(archive["depth/frames"])
        depth_valid = np.asarray(archive["depth/valid"], dtype=bool)
    t0 = float(timestamps[0])
    cam_y_first = float(translations[0, 1])
    early = timestamps <= (t0 + early_seconds)
    frame_ids = np.flatnonzero(early)
    if frame_ids.size == 0:
        raise RuntimeError("no early frames")
    frame_ids = frame_ids[::2]
    chunks = []
    for index in frame_ids:
        optical = unproject_depth_optical(
            depth_mm[index].astype(np.float64) / 1000.0,
            depth_valid[index],
            depth_k,
            depth_d,
        )
        if optical.shape[0] == 0:
            continue
        world = to_world_yup(optical, rotations[index], translations[index], r_o2l)
        below = world[:, 1] < (cam_y_first - 0.10)
        world = world[below]
        if world.shape[0]:
            chunks.append(world)
    if not chunks:
        raise RuntimeError("no early world points below the first camera")
    xyz = np.concatenate(chunks, axis=0)
    plane = fit_dominant_horizontal_plane(xyz, rng=np.random.default_rng(0))
    table_y = float(plane["table_y_m"])
    if table_y >= cam_y_first - 0.05:
        raise RuntimeError(
            f"dominant plane y={table_y:.3f} is not below camera y={cam_y_first:.3f}"
        )
    ground = json.loads(ground_path.read_text()) if ground_path.is_file() else {}
    offset = ground.get("ground_y_offset")
    return {
        "session": session.name,
        "table_y_m": table_y,
        "cam_y_first_m": cam_y_first,
        "ground_y_offset": None if offset is None else float(offset),
        "table_minus_target_m": table_y - TARGET_TABLE_Y_M,
        "n_inliers": int(plane["n_inliers"]),
        "n_candidates": int(plane["n_candidates"]),
        "up_alignment": float(plane["up_alignment"]),
        "label": classify(table_y, TARGET_TABLE_Y_M),
        "n_early_frames": int(frame_ids.size),
    }


def shift_source(session: Path, shift_y: float, record: dict[str, object]) -> None:
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
        translations.parent.attrs["early_table_target_m"] = float(record["target_height_m"])
        translations.parent.attrs["early_table_last_shift_m"] = float(shift_y)
    payload_record = {
        "schema_version": SCHEMA_VERSION,
        "method": "early_rgb_d_dominant_horizontal_plane",
        "early_seconds": EARLY_SECONDS,
        **record,
        "applied_to": ["mesh.ply", "annotation.hdf5:/cam-pose/translations"],
        "not_applied_to": ["hand-pose", "depth", "imu", "calibrations", "rgb.mp4"],
    }
    for name in ("ground_align.json", "export_manifest.json"):
        path = session / name
        payload = json.loads(path.read_text()) if path.is_file() else {}
        payload["early_table_alignment"] = payload_record
        path.write_text(json.dumps(payload, indent=2) + "\n")


def summarize(rows: list[dict[str, object]], target: float) -> None:
    labels: dict[str, int] = {}
    heights = np.asarray([float(row["table_y_m"]) for row in rows], dtype=np.float64)
    print(f"sessions={len(rows)} target_table_y_m={target:.3f}")
    print(
        "table_y_m min/median/mean/max="
        f"{heights.min():.4f}/{np.median(heights):.4f}/{heights.mean():.4f}/{heights.max():.4f}"
    )
    for row in rows:
        labels[str(row["label"])] = labels.get(str(row["label"]), 0) + 1
    print("counts " + " ".join(f"{key}={value}" for key, value in sorted(labels.items())))
    print(
        "session\tlabel\ttable_y_m\tcam_y_first_m\tground_y_offset\ttable_minus_0.758_m\tn_inliers"
    )
    for row in rows:
        offset = row["ground_y_offset"]
        offset_s = "nan" if offset is None else f"{float(offset):.6f}"
        print(
            f"{row['session']}\t{row['label']}\t{float(row['table_y_m']):.6f}\t"
            f"{float(row['cam_y_first_m']):.6f}\t{offset_s}\t"
            f"{float(row['table_minus_target_m']):.6f}\t{row['n_inliers']}"
        )


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
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for session in sessions:
        try:
            row = measure_early_table(session, args.early_seconds)
            row["table_minus_target_m"] = float(row["table_y_m"]) - args.target_height_m
            row["label"] = classify(float(row["table_y_m"]), args.target_height_m)
            if args.apply:
                prior_path = session / "ground_align.json"
                prior = json.loads(prior_path.read_text()) if prior_path.is_file() else {}
                existing = prior.get("early_table_alignment")
                already = (
                    existing
                    and abs(float(existing.get("target_height_m", 0.0)) - args.target_height_m)
                    <= 1e-9
                    and abs(
                        float(existing.get("detected_height_after_m", 1e9))
                        - args.target_height_m
                    )
                    <= 0.005
                )
                if already:
                    row["status"] = "already_aligned"
                else:
                    shift = args.target_height_m - float(row["table_y_m"])
                    apply_record = {
                        "target_height_m": args.target_height_m,
                        "detected_height_before_m": float(row["table_y_m"]),
                        "world_y_shift_m": shift,
                        "label_before": row["label"],
                    }
                    shift_source(session, shift, apply_record)
                    after = measure_early_table(session, args.early_seconds)
                    apply_record["detected_height_after_m"] = float(after["table_y_m"])
                    for name in ("ground_align.json", "export_manifest.json"):
                        path = session / name
                        if not path.is_file():
                            continue
                        payload = json.loads(path.read_text())
                        payload["early_table_alignment"] = {
                            "schema_version": SCHEMA_VERSION,
                            "method": "early_rgb_d_dominant_horizontal_plane",
                            "early_seconds": args.early_seconds,
                            **apply_record,
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
                        path.write_text(json.dumps(payload, indent=2) + "\n")
                    row["table_y_m"] = float(after["table_y_m"])
                    row["cam_y_first_m"] = float(after["cam_y_first_m"])
                    row["table_minus_target_m"] = (
                        float(after["table_y_m"]) - args.target_height_m
                    )
                    row["label"] = classify(
                        float(after["table_y_m"]), args.target_height_m
                    )
                    row["status"] = "aligned"
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{session.name}: {exc}")
    if rows:
        summarize(rows, args.target_height_m)
    if failures:
        print(f"failures={len(failures)}")
        for line in failures:
            print(line)
        raise SystemExit(1)
    if not rows:
        raise SystemExit("no sessions measured")


if __name__ == "__main__":
    main()
