#!/usr/bin/env python3
"""Quality and provenance helpers for the offline RGB-D novel-view pipeline.

The helpers in this module intentionally depend only on NumPy and the standard
library.  They are used by the warp, compositor, and inpainting stages so that
``valid`` retains its geometric meaning while generated RGB is tracked
separately through ``synthetic`` and ``provenance`` maps.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


# Stable on-disk provenance codes.  Keep these values compatible when adding
# stages; downstream VLA loaders may persist them as uint8 tensors.
PROVENANCE_INVALID = np.uint8(0)
PROVENANCE_OBSERVED = np.uint8(1)
PROVENANCE_FUSED = np.uint8(2)
PROVENANCE_ROBOT = np.uint8(3)
PROVENANCE_SYNTHETIC = np.uint8(4)

PROVENANCE_NAMES = {
    int(PROVENANCE_INVALID): "invalid",
    int(PROVENANCE_OBSERVED): "observed",
    int(PROVENANCE_FUSED): "fused",
    int(PROVENANCE_ROBOT): "robot",
    int(PROVENANCE_SYNTHETIC): "synthetic",
}


def provenance_from_geometry(
    valid: np.ndarray,
    source_count: np.ndarray | None = None,
    *,
    current_frame: int | None = None,
    last_frame: np.ndarray | None = None,
) -> np.ndarray:
    """Build a provenance map for a rendered geometry frame.

    ``valid`` is the geometric validity mask.  If source frame indices are
    available, points last observed in the current source frame are labelled
    ``observed`` and points supplied by history are labelled ``fused``.
    Without indices, a positive source count is conservatively labelled
    ``fused``; callers can overwrite known current-frame pixels afterwards.
    """
    v = np.asarray(valid, dtype=bool)
    out = np.full(v.shape, PROVENANCE_INVALID, dtype=np.uint8)
    if source_count is None:
        out[v] = PROVENANCE_FUSED
        return out
    count = np.asarray(source_count)
    out[v & (count <= 1)] = PROVENANCE_OBSERVED
    out[v & (count > 1)] = PROVENANCE_FUSED
    if current_frame is not None and last_frame is not None:
        lf = np.asarray(last_frame)
        if lf.shape == v.shape:
            out[v & (lf == int(current_frame))] = PROVENANCE_OBSERVED
    return out


def mark_robot(provenance: np.ndarray, robot_mask: np.ndarray) -> np.ndarray:
    """Return a copy with robot pixels marked, without changing geometry masks."""
    out = np.asarray(provenance, dtype=np.uint8).copy()
    m = np.asarray(robot_mask, dtype=bool)
    if m.shape != out.shape:
        raise ValueError(f"robot mask shape {m.shape} != provenance {out.shape}")
    out[m] = PROVENANCE_ROBOT
    return out


def mark_synthetic(provenance: np.ndarray, synthetic_mask: np.ndarray) -> np.ndarray:
    """Return a copy with generated RGB pixels marked as synthetic."""
    out = np.asarray(provenance, dtype=np.uint8).copy()
    m = np.asarray(synthetic_mask, dtype=bool)
    if m.shape != out.shape:
        raise ValueError(f"synthetic mask shape {m.shape} != provenance {out.shape}")
    out[m] = PROVENANCE_SYNTHETIC
    return out


def provenance_fractions(provenance: np.ndarray) -> dict[str, float]:
    """Return fractions for every known provenance class."""
    p = np.asarray(provenance, dtype=np.uint8)
    denom = max(int(p.size), 1)
    return {
        name: float(np.count_nonzero(p == code) / denom)
        for code, name in sorted(PROVENANCE_NAMES.items())
    }


def frame_metrics(
    *,
    frame: int,
    valid: np.ndarray,
    hole: np.ndarray,
    confidence: np.ndarray | None = None,
    provenance: np.ndarray | None = None,
    source_count: np.ndarray | None = None,
    front_ratio: float | None = None,
    in_bounds_ratio: float | None = None,
    valid_source_ratio: float | None = None,
    reprojection_error_px: float | None = None,
    timing_offset_ms: float | None = None,
    depth_relative_error: float | None = None,
    robot_boundary_iou: float | None = None,
) -> dict[str, Any]:
    """Create a JSON-ready per-frame quality row."""
    v = np.asarray(valid, dtype=bool)
    h = np.asarray(hole, dtype=bool)
    if v.shape != h.shape:
        raise ValueError("valid and hole shapes differ")
    row: dict[str, Any] = {
        "frame": int(frame),
        "valid_ratio": float(v.mean()) if v.size else 0.0,
        "hole_ratio": float(h.mean()) if h.size else 1.0,
        "front_ratio": None if front_ratio is None else float(front_ratio),
        "in_bounds_ratio": None if in_bounds_ratio is None else float(in_bounds_ratio),
        "valid_source_ratio": None if valid_source_ratio is None else float(valid_source_ratio),
        "reprojection_error_px": None if reprojection_error_px is None else float(reprojection_error_px),
        "timing_offset_ms": None if timing_offset_ms is None else float(timing_offset_ms),
        "depth_relative_error": None if depth_relative_error is None else float(depth_relative_error),
        "robot_boundary_iou": None if robot_boundary_iou is None else float(robot_boundary_iou),
    }
    if confidence is not None:
        c = np.asarray(confidence, dtype=np.float32)
        row["confidence_mean_valid"] = (
            float(np.mean(c[v])) if c.shape == v.shape and np.any(v) else None
        )
    if source_count is not None:
        sc = np.asarray(source_count)
        row["source_count_mean_valid"] = (
            float(np.mean(sc[v])) if sc.shape == v.shape and np.any(v) else None
        )
    if provenance is not None:
        row["provenance"] = provenance_fractions(provenance)
    return row


def temporal_flicker(mask_sequence: Iterable[np.ndarray]) -> float | None:
    """Measure valid-mask transitions per pixel between adjacent frames."""
    masks = [np.asarray(m, dtype=bool) for m in mask_sequence]
    if len(masks) < 2:
        return None
    pairs = []
    for a, b in zip(masks[:-1], masks[1:]):
        if a.shape != b.shape:
            raise ValueError("mask sequence contains different shapes")
        pairs.append(float(np.mean(a != b)))
    return float(np.mean(pairs)) if pairs else None


def _safe_mean(rows: list[Mapping[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    return float(np.mean(vals)) if vals else None


def aggregate_episode(
    rows: Iterable[Mapping[str, Any]],
    *,
    calibration_status: str = "unknown",
    geometry_status: str = "unknown",
    flicker: float | None = None,
    synthetic_limit: float = 0.25,
    min_score: float = 80.0,
) -> dict[str, Any]:
    """Aggregate frame rows into an auditable episode report.

    Hard geometry/calibration failures are kept separate from the soft score.
    This lets callers retain rejected artifacts for review without accidentally
    treating them as accepted VLA observations.
    """
    items = [dict(r) for r in rows]
    n = len(items)
    bad = [bool(r.get("bad", False)) for r in items]
    bad_ratio = float(np.mean(bad)) if bad else 0.0
    synthetic = [
        float(r["provenance"].get("synthetic", 0.0))
        for r in items
        if isinstance(r.get("provenance"), Mapping)
    ]
    synthetic_ratio = float(np.mean(synthetic)) if synthetic else 0.0

    # A transparent score, not a replacement for hard gates.  Missing soft
    # metrics contribute a neutral value and are reported as unavailable.
    coverage = _safe_mean(items, "valid_ratio")
    depth = 1.0 - (_safe_mean(items, "depth_relative_error") or 0.0)
    depth = float(np.clip(depth, 0.0, 1.0))
    reproj = _safe_mean(items, "reprojection_error_px")
    reproj_score = 1.0 if reproj is None else float(np.clip(1.0 - reproj / 8.0, 0.0, 1.0))
    temporal_score = 1.0 if flicker is None else float(np.clip(1.0 - flicker / 0.08, 0.0, 1.0))
    robot = _safe_mean(items, "robot_boundary_iou")
    robot_score = 1.0 if robot is None else float(np.clip(robot, 0.0, 1.0))
    coverage_score = 0.0 if coverage is None else float(np.clip(coverage / 0.7, 0.0, 1.0))
    score = 100.0 * (
        0.30 * coverage_score
        + 0.20 * depth
        + 0.20 * reproj_score
        + 0.15 * temporal_score
        + 0.15 * robot_score
    )

    hard_failures = []
    if calibration_status not in ("pass", "known", "complete"):
        hard_failures.append("calibration_status")
    # bypassed/off: warp skipped the pose-convention preflight.
    # fallback: gate ran but did not hard-fail.
    if geometry_status not in ("pass", "known", "bypassed", "off", "fallback"):
        hard_failures.append("geometry_status")
    if bad_ratio > 0.05:
        hard_failures.append("bad_frame_ratio")
    if synthetic_ratio > float(synthetic_limit):
        hard_failures.append("synthetic_ratio")
    accepted = not hard_failures and score >= float(min_score)
    return {
        "frame_count": n,
        "bad_frame_ratio": bad_ratio,
        "synthetic_ratio": synthetic_ratio,
        "flicker": None if flicker is None else float(flicker),
        "means": {
            "valid_ratio": coverage,
            "depth_relative_error": _safe_mean(items, "depth_relative_error"),
            "reprojection_error_px": reproj,
            "robot_boundary_iou": robot,
        },
        "score": float(score),
        "score_threshold": float(min_score),
        "hard_failures": hard_failures,
        "status": "accepted" if accepted else "rejected",
    }


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _indexed_images(path: Path | None) -> list[Path]:
    if path is None or not path.is_dir():
        return []
    files = []
    for p in path.glob("*.png"):
        try:
            int(p.stem)
        except ValueError:
            continue
        files.append(p)
    return sorted(files, key=lambda p: int(p.stem))


def _mask_sequence(path: Path | None) -> list[np.ndarray]:
    if path is None:
        return []
    try:
        from frame_volume import as_mask, is_packed_stream, open_volume
    except ImportError:  # pragma: no cover
        from .frame_volume import as_mask, is_packed_stream, open_volume
    if is_packed_stream(path):
        volume = open_volume(path)
        return [as_mask(volume[i]) for i in range(len(volume))]
    files = _indexed_images(path)
    if not files:
        return []
    import cv2

    return [cv2.imread(str(item), cv2.IMREAD_GRAYSCALE) > 127 for item in files]


def _provenance_sequence(path: Path | None) -> list[np.ndarray]:
    if path is None:
        return []
    try:
        from frame_volume import is_packed_stream, open_volume
    except ImportError:  # pragma: no cover
        from .frame_volume import is_packed_stream, open_volume
    if is_packed_stream(path):
        volume = open_volume(path)
        out = []
        for i in range(len(volume)):
            prov = np.asarray(volume[i])
            if prov.ndim == 3:
                prov = prov[..., 0]
            out.append(prov)
        return out
    files = _indexed_images(path)
    if not files:
        return []
    import cv2

    out = []
    for item in files:
        prov = cv2.imread(str(item), cv2.IMREAD_UNCHANGED)
        if prov is None:
            continue
        if prov.ndim == 3:
            prov = prov[..., 0]
        out.append(prov)
    return out






def _cli() -> None:
    p = argparse.ArgumentParser(description="Aggregate offline RGB-D quality metrics")
    p.add_argument("--frame-metrics", type=Path, required=True)
    p.add_argument("--geometry-gate", type=Path, default=None)
    p.add_argument("--valid-dir", type=Path, default=None)
    p.add_argument("--provenance-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--calibration-status", default=None)
    p.add_argument("--synthetic-limit", type=float, default=0.25)
    p.add_argument("--min-score", type=float, default=80.0)
    args = p.parse_args()

    rows = _read_jsonl(args.frame_metrics)
    valid_seq = _mask_sequence(args.valid_dir)
    flicker = temporal_flicker(valid_seq) if valid_seq else None
    for i, prov in enumerate(_provenance_sequence(args.provenance_dir)):
        if i >= len(rows):
            break
        rows[i]["provenance"] = provenance_fractions(prov)
    gate = {}
    if args.geometry_gate and args.geometry_gate.is_file():
        gate = json.loads(args.geometry_gate.read_text())
    geometry_status = str(gate.get("status", "unknown"))
    calibration_status = args.calibration_status
    if calibration_status is None:
        calibration_status = str(gate.get("calibration_status", "unknown"))
    report = aggregate_episode(
        rows,
        calibration_status=calibration_status,
        geometry_status=geometry_status,
        flicker=flicker,
        synthetic_limit=args.synthetic_limit,
        min_score=args.min_score,
    )
    report["geometry_gate"] = gate
    report["synthetic_limit"] = float(args.synthetic_limit)
    report["schema_version"] = 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] != "accepted":
        review = args.out.parent / "rejected" / "review.json"
        review.parent.mkdir(parents=True, exist_ok=True)
        review.write_text(json.dumps({"status": "review", "episode_quality": str(args.out), "reasons": report["hard_failures"]}, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "score": report["score"], "out": str(args.out)}))


if __name__ == "__main__":
    _cli()
