#!/usr/bin/env python3
"""Geometry quality checks and depth confidence helpers for offline RGB-D data.

The module is deliberately dependency-light (NumPy only) so it can be imported by
the adapter and by the viewport renderer.  A gate result is JSON serialisable and
contains both the measured values and the threshold decisions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import json

import numpy as np


@dataclass(frozen=True)
class GeometryGateThresholds:
    valid_ratio: float = 0.30
    target_front_ratio: float = 0.85
    in_bounds_ratio: float = 0.30
    reprojection_p50_px: float = 3.0
    reprojection_p95_px: float = 8.0
    timing_p95_ms: float = 8.0
    timing_max_ms: float = 20.0
    depth_relative_p95: float = 0.15
    bad_frame_ratio: float = 0.05
    max_bad_frame_run: int = 5


DEFAULT_THRESHOLDS = GeometryGateThresholds()


def _finite_positive(depth: np.ndarray) -> np.ndarray:
    d = np.asarray(depth)
    return np.isfinite(d) & (d > 1e-6)


def compute_depth_confidence(
    depth: np.ndarray,
    rgb: np.ndarray | None = None,
    sensor_confidence: np.ndarray | None = None,
    previous_depth: np.ndarray | None = None,
    *,
    min_depth: float = 0.05,
    max_depth: float = 20.0,
    edge_gradient: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(confidence, valid, edge_invalid)`` for a depth image.

    Sensor confidence is preferred when supplied. Otherwise confidence is derived
    from validity, range, local depth gradients, and (when available) RGB edges.
    Values are in [0, 1]; edge-invalid pixels are excluded from ``valid``.
    """
    d = np.asarray(depth, dtype=np.float32)
    valid = _finite_positive(d) & (d >= min_depth) & (d <= max_depth)
    conf = valid.astype(np.float32)
    if sensor_confidence is not None:
        sc = np.asarray(sensor_confidence, dtype=np.float32)
        if sc.shape != d.shape:
            raise ValueError(f"sensor confidence shape {sc.shape} != depth {d.shape}")
        if sc.max(initial=0.0) > 1.0:
            sc = sc / 255.0
        conf = np.clip(np.nan_to_num(sc, nan=0.0), 0.0, 1.0) * valid
    else:
        # A local gradient detects depth discontinuities where splatting creates
        # foreground fly-pixels.  Invalid neighbours are treated as boundaries.
        gx = np.zeros_like(d)
        gy = np.zeros_like(d)
        if d.ndim == 2 and min(d.shape) > 1:
            gx[:, 1:] = np.abs(d[:, 1:] - d[:, :-1])
            gy[1:, :] = np.abs(d[1:, :] - d[:-1, :])
        grad = np.maximum(gx, gy)
        scale = np.maximum(d, 1e-3)
        rel_grad = grad / scale
        threshold = 0.08 if edge_gradient is None else float(edge_gradient)
        edge = valid & (rel_grad > threshold)
        # RGB edges are a secondary cue, only reducing confidence (never making
        # an invalid point valid).
        if rgb is not None:
            im = np.asarray(rgb)
            if im.ndim == 3 and im.shape[:2] == d.shape:
                gray = im.astype(np.float32).mean(axis=2)
                rg = np.zeros_like(gray)
                rg[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
                rg[1:, :] = np.maximum(rg[1:, :], np.abs(gray[1:, :] - gray[:-1, :]))
                edge |= valid & (rg > 32.0) & (rel_grad > threshold * 0.5)
        conf[edge] *= 0.15
        edge_invalid = edge
        if previous_depth is not None:
            prev = np.asarray(previous_depth, dtype=np.float32)
            if prev.shape == d.shape:
                temporal_valid = _finite_positive(prev) & valid
                rel_delta = np.abs(d - prev) / np.maximum(np.maximum(d, prev), 1e-3)
                # This is intentionally only a confidence reduction: raw
                # camera motion can change per-pixel depth, so temporal
                # disagreement must not silently invalidate geometry.
                conf[temporal_valid & (rel_delta > 0.25)] *= 0.5
    if sensor_confidence is not None:
        edge_invalid = valid & (conf < 0.2)
    valid = valid & ~edge_invalid & (conf >= 0.2)
    return conf.astype(np.float32), valid.astype(bool), edge_invalid.astype(bool)


def compute_depth_confidence_batch(
    depth: np.ndarray,
    rgb: np.ndarray | None = None,
    sensor_confidence: np.ndarray | None = None,
    *,
    rgb_present: np.ndarray | None = None,
    use_temporal: bool = False,
    min_depth: float = 0.05,
    max_depth: float = 20.0,
    edge_gradient: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized ``compute_depth_confidence`` over a ``(N, H, W)`` stack.

    Temporal reduction uses ``depth[i-1]`` as the previous frame, matching the
    sequential adapter loop. RGB edges are applied only where ``rgb_present`` is
    true; a missing mask treats every frame as present.
    """
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim != 3:
        raise ValueError(f"batched depth must be (N, H, W); got {d.shape}")
    n, height, width = d.shape
    valid = _finite_positive(d) & (d >= min_depth) & (d <= max_depth)
    conf = valid.astype(np.float32)
    if sensor_confidence is not None:
        sc = np.asarray(sensor_confidence, dtype=np.float32)
        if sc.shape != d.shape:
            raise ValueError(f"sensor confidence shape {sc.shape} != depth {d.shape}")
        frame_max = sc.reshape(n, -1).max(axis=1)
        scale = np.where(frame_max > 1.0, 255.0, 1.0).astype(np.float32)[:, None, None]
        sc = np.clip(np.nan_to_num(sc / scale, nan=0.0), 0.0, 1.0)
        conf = sc * valid
        edge_invalid = valid & (conf < 0.2)
    else:
        gx = np.zeros_like(d)
        gy = np.zeros_like(d)
        if min(height, width) > 1:
            gx[:, :, 1:] = np.abs(d[:, :, 1:] - d[:, :, :-1])
            gy[:, 1:, :] = np.abs(d[:, 1:, :] - d[:, :-1, :])
        grad = np.maximum(gx, gy)
        scale = np.maximum(d, 1e-3)
        rel_grad = grad / scale
        threshold = 0.08 if edge_gradient is None else float(edge_gradient)
        edge = valid & (rel_grad > threshold)
        if rgb is not None:
            im = np.asarray(rgb)
            if im.shape[:3] != d.shape:
                raise ValueError(f"rgb shape {im.shape} incompatible with depth {d.shape}")
            gray = im.astype(np.float32).mean(axis=-1)
            rg = np.zeros_like(gray)
            if min(height, width) > 1:
                rg[:, :, 1:] = np.abs(gray[:, :, 1:] - gray[:, :, :-1])
                rg[:, 1:, :] = np.maximum(
                    rg[:, 1:, :], np.abs(gray[:, 1:, :] - gray[:, :-1, :])
                )
            rgb_edge = valid & (rg > 32.0) & (rel_grad > threshold * 0.5)
            if rgb_present is not None:
                present = np.asarray(rgb_present, dtype=bool).reshape(n)
                rgb_edge &= present[:, None, None]
            edge |= rgb_edge
        conf = conf.copy()
        conf[edge] *= 0.15
        edge_invalid = edge
        if use_temporal and n > 1:
            prev = np.empty_like(d)
            prev[0] = np.nan
            prev[1:] = d[:-1]
            temporal_valid = _finite_positive(prev) & valid
            rel_delta = np.abs(d - prev) / np.maximum(np.maximum(d, prev), 1e-3)
            conf[temporal_valid & (rel_delta > 0.25)] *= 0.5
    valid = valid & ~edge_invalid & (conf >= 0.2)
    return conf.astype(np.float32), valid.astype(bool), edge_invalid.astype(bool)


def _percentile(values: Iterable[float] | None, q: float) -> float | None:
    if values is None:
        return None
    a = np.asarray(list(values), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size else None


def _max_bad_run(flags: list[bool]) -> int:
    best = cur = 0
    for flag in flags:
        cur = cur + 1 if flag else 0
        best = max(best, cur)
    return best


def evaluate_geometry_gate(
    frame_metrics: Iterable[Mapping[str, Any]],
    *,
    thresholds: GeometryGateThresholds = DEFAULT_THRESHOLDS,
    calibration_complete: bool = True,
    timing_available: bool = True,
    mode: str = "strict",
    rz_alternate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate per-frame geometry metrics and return a JSON-ready gate report.

    Metric keys are intentionally plain: ``valid_ratio``, ``front_ratio``,
    ``in_bounds_ratio``, ``reprojection_error_px``, ``timing_offset_ms``, and
    ``depth_relative_error``. Missing optional metrics are reported as unavailable.
    """
    rows = [dict(m) for m in frame_metrics]
    def vals(key: str) -> list[float]:
        out = []
        for r in rows:
            v = r.get(key)
            if v is not None and np.isfinite(v):
                out.append(float(v))
        return out

    valid = vals("valid_ratio")
    front = vals("front_ratio")
    bounds = vals("in_bounds_ratio")
    reproj_p50 = vals("reprojection_error_p50_px") or vals("reprojection_error_px")
    reproj_p95 = vals("reprojection_error_p95_px") or vals("reprojection_error_px")
    timing = vals("timing_offset_ms")
    depth_err = vals("depth_relative_error")
    bad_flags = []
    resolution_flags = []
    for r in rows:
        bad_flags.append(bool(r.get("bad", False)))
        resolution_flags.append(bool(r.get("depth_resolution_mismatch", False)))
    checks: dict[str, dict[str, Any]] = {
        "valid_ratio": {"value": float(np.mean(valid)) if valid else None, "threshold": thresholds.valid_ratio,
                         "pass": bool(valid) and float(np.mean(valid)) >= thresholds.valid_ratio},
        "target_front_ratio": {"value": float(np.mean(front)) if front else None, "threshold": thresholds.target_front_ratio,
                                "pass": bool(front) and float(np.mean(front)) >= thresholds.target_front_ratio},
        "in_bounds_ratio": {"value": float(np.mean(bounds)) if bounds else None, "threshold": thresholds.in_bounds_ratio,
                             "pass": bool(bounds) and float(np.mean(bounds)) >= thresholds.in_bounds_ratio},
        "reprojection_p50_px": {"value": _percentile(reproj_p50, 50), "threshold": thresholds.reprojection_p50_px,
                                 "pass": (not reproj_p50) or _percentile(reproj_p50, 50) <= thresholds.reprojection_p50_px},
        "reprojection_p95_px": {"value": _percentile(reproj_p95, 95), "threshold": thresholds.reprojection_p95_px,
                                 "pass": (not reproj_p95) or _percentile(reproj_p95, 95) <= thresholds.reprojection_p95_px},
        "timing_p95_ms": {"value": _percentile(np.abs(timing), 95), "threshold": thresholds.timing_p95_ms,
                           "pass": (not timing) or _percentile(np.abs(timing), 95) <= thresholds.timing_p95_ms},
        "timing_max_ms": {"value": float(np.max(np.abs(timing))) if timing else None, "threshold": thresholds.timing_max_ms,
                          "pass": (not timing) or float(np.max(np.abs(timing))) <= thresholds.timing_max_ms},
        "depth_relative_p95": {"value": _percentile(depth_err, 95), "threshold": thresholds.depth_relative_p95,
                                "pass": (not depth_err) or _percentile(depth_err, 95) <= thresholds.depth_relative_p95},
        "bad_frame_ratio": {"value": float(np.mean(bad_flags)) if bad_flags else None, "threshold": thresholds.bad_frame_ratio,
                             "pass": (not bad_flags) or float(np.mean(bad_flags)) <= thresholds.bad_frame_ratio},
        "max_bad_frame_run": {"value": _max_bad_run(bad_flags) if bad_flags else None, "threshold": thresholds.max_bad_frame_run,
                               "pass": (not bad_flags) or _max_bad_run(bad_flags) <= thresholds.max_bad_frame_run},
        "depth_resolution": {
            "value": float(np.mean(resolution_flags)) if resolution_flags else None,
            "threshold": 0.0,
            "pass": not any(resolution_flags),
        },
    }
    unavailable_required = [k for k in ("valid_ratio", "target_front_ratio", "in_bounds_ratio") if checks[k]["value"] is None]
    failed = [k for k, c in checks.items() if not c["pass"]]
    if not calibration_complete:
        failed.append("calibration_complete")
    # Missing timing is a diagnostic in legacy mode; strict callers can pass
    # timing_available=False and receive a clear reason without a silent pass.
    if not timing_available:
        checks["timing_available"] = {"value": False, "threshold": True, "pass": mode != "strict"}
        if mode == "strict":
            failed.append("timing_available")
    status = "pass" if not failed else ("legacy_debug" if mode != "strict" else "fail")
    return {
        "status": status,
        "mode": mode,
        "thresholds": asdict(thresholds),
        "frame_count": len(rows),
        "checks": checks,
        "failed_checks": sorted(set(failed)),
        "unavailable_required": unavailable_required,
        "rz_alternate": dict(rz_alternate) if rz_alternate is not None else None,
    }


def write_gate_report(path: str | Path, report: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


# Stable aliases for callers that prefer an imperative API name.
run_geometry_gate = evaluate_geometry_gate
check_geometry_gate = evaluate_geometry_gate
