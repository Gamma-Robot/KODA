#!/usr/bin/env python3
"""Offline wrist/palm and Dex3 finger-retargeting ablation.

This is an experiment-only sidecar.  It reads the same source annotation and
body kinematics as the view-change pipeline, but never edits either input or
the production pipeline.  Wrist candidates are expressed as desired
``wrist_yaw_link`` poses.  Finger candidates are evaluated with fixed-base
Dex3 forward kinematics on the native human-hand timeline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, Slerp


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
EXPERIMENT_VECTOR_CONFIG = SCRIPT_DIR / "configs" / "unitree_dex3_vector_experiment.yml"

from koda.retargeting.evaluation import (  # noqa: E402
    _build_fixed_base_fk,
    _evaluate_method,
    _human_wrist_local_tips,
)
from koda.retargeting.dex3 import (  # noqa: E402
    HAND_JOINT_NAMES,
    _load_session_keypoints,
    _interpolate_rotation_samples,
    _retarget_points,
)


YUP_TO_ZUP = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
WRIST_LINK_LOCAL_RX_MINUS_90 = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)

WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 9
MIDDLE_MCP = 10
MIDDLE_TIP = 14
RING_MCP = 15
LITTLE_MCP = 20
LITTLE_TIP = 24
MCP_INDICES = (INDEX_MCP, MIDDLE_MCP, RING_MCP, LITTLE_MCP)

ROBOT_PALM_SITE = np.asarray([0.08, 0.0, 0.0], dtype=np.float64)


@dataclass(frozen=True)
class RobotPalmLandmarks:
    wrist: np.ndarray
    palm: np.ndarray
    index_root: np.ndarray
    middle_root: np.ndarray

    @property
    def root_mid(self) -> np.ndarray:
        return 0.5 * (self.index_root + self.middle_root)


def robot_landmarks(side: str) -> RobotPalmLandmarks:
    if side not in ("left", "right"):
        raise ValueError(side)
    thickness = 0.0046 if side == "left" else -0.0046
    return RobotPalmLandmarks(
        wrist=np.zeros(3, dtype=np.float64),
        palm=ROBOT_PALM_SITE.copy(),
        index_root=np.asarray([0.1192, thickness, 0.0285], dtype=np.float64),
        middle_root=np.asarray([0.1192, thickness, -0.0285], dtype=np.float64),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session-dir", type=Path, required=True)
    p.add_argument("--kinematics-npz", type=Path, required=True)
    p.add_argument(
        "--target-timestamps-npz",
        type=Path,
        default=None,
        help=(
            "Optional NPZ providing the 50 Hz timestamps when the body "
            "reference stores poses and timeline targets separately."
        ),
    )
    p.add_argument(
        "--template-palm-from-current-kinematics",
        action="store_true",
        help=(
            "Use the supplied body reference as the authoritative "
            "template_palm_kabsch wrist candidate instead of fitting a second "
            "independent copy."
        ),
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-native-frames", type=int, default=None)
    p.add_argument("--flexion-scale", type=float, default=0.7)
    p.add_argument("--scaling-factor", type=float, default=1.0)
    p.add_argument("--low-pass-alpha", type=float, default=0.2)
    p.add_argument("--limit-margin", type=float, default=0.01)
    p.add_argument(
        "--finger-methods",
        default="dexpilot,vector,position,dexpilot_close",
        help="Comma-separated: dexpilot, vector, position, dexpilot_close",
    )
    p.add_argument(
        "--wrist-methods",
        default=(
            "historical_tip_wrist,mcp_frame_wrist,mcp_mid_palm_site,"
            "mcp_arc_palm_site,palm_polygon_site,template_palm_kabsch,"
            "projected_hull_centroid,mano_palm_surface_centroid,"
            "calibrated_mcp_center,root_mid_index_ring,root_mid_index_little,"
            "kabsch_index_ring,kabsch_index_little"
        ),
        help="Comma-separated wrist/palm candidates",
    )
    p.add_argument(
        "--mano-asset",
        type=Path,
        default=Path(os.environ.get("KODA_MANO_RIGHT", str(PROJECT_ROOT / "assets/models/mano/MANO_RIGHT.pkl"))),
        help="MANO topology/regressor pickle; topology is shared by both handed outputs",
    )
    return p.parse_args()


def stats(values: np.ndarray) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if not a.size:
        return {k: float("nan") for k in ("mean", "median", "p95", "max")}
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
    }


def normalize(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm < 1e-10:
        raise ValueError("degenerate direction")
    return v / norm


def historical_frame(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    x = normalize(p[MIDDLE_TIP] - p[WRIST])
    across = normalize(p[INDEX_TIP] - p[LITTLE_TIP])
    z = -normalize(np.cross(across, x))
    y = normalize(np.cross(z, x))
    x = normalize(np.cross(y, z))
    return np.stack([x, y, z], axis=1) @ WRIST_LINK_LOCAL_RX_MINUS_90


def landmark_frame(points: np.ndarray, outer_a: int, outer_b: int) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    distal = 0.25 * sum((p[index] for index in MCP_INDICES), np.zeros(3))
    if outer_a != INDEX_MCP or outer_b != LITTLE_MCP:
        distal = 0.5 * (p[outer_a] + p[outer_b])
    x = normalize(distal - p[WRIST])
    across = p[outer_a] - p[outer_b]
    across = normalize(across - x * float(np.dot(across, x)))
    z = -normalize(np.cross(across, x))
    y = normalize(np.cross(z, x))
    x = normalize(np.cross(y, z))
    return np.stack([x, y, z], axis=1) @ WRIST_LINK_LOCAL_RX_MINUS_90


def palm_polygon_centroid(points: np.ndarray) -> np.ndarray:
    """Return the area centroid of the wrist-to-MCP palm polygon.

    The polygon is triangulated as wrist-(index,middle), wrist-(middle,ring),
    and wrist-(ring,little). This is an explicit geometric palm-site proxy and
    does not assign empirical landmark weights.
    """
    p = np.asarray(points, dtype=np.float64)
    triangles = (
        (WRIST, INDEX_MCP, MIDDLE_MCP),
        (WRIST, MIDDLE_MCP, RING_MCP),
        (WRIST, RING_MCP, LITTLE_MCP),
    )
    weighted = np.zeros(3, dtype=np.float64)
    total_area = 0.0
    for indices in triangles:
        triangle = p[list(indices)]
        area = 0.5 * float(
            np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
        )
        if np.isfinite(area) and area > 1e-12:
            weighted += area * np.mean(triangle, axis=0)
            total_area += area
    if total_area <= 1e-12:
        raise ValueError("degenerate wrist-to-MCP palm polygon")
    return weighted / total_area


def _palm_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a stable 3-D palm plane origin and two in-plane axes."""
    anchors = np.asarray(points, dtype=np.float64)[[WRIST, *MCP_INDICES]]
    if not np.all(np.isfinite(anchors)):
        raise ValueError("non-finite palm anchors")
    origin = np.mean(anchors, axis=0)
    _, _, vt = np.linalg.svd(anchors - origin, full_matrices=False)
    if vt.shape != (3, 3):
        raise ValueError("palm anchors are rank deficient")
    axis_u = normalize(vt[0])
    axis_v = normalize(vt[1] - axis_u * float(np.dot(vt[1], axis_u)))
    normal = normalize(np.cross(axis_u, axis_v))
    return origin, axis_u, normal


def projected_convex_hull_centroid(points: np.ndarray) -> np.ndarray:
    """Compute the centroid of the wrist/MCP convex hull in its palm plane."""
    p = np.asarray(points, dtype=np.float64)
    anchors = p[[WRIST, *MCP_INDICES]]
    origin, axis_u, normal = _palm_plane(p)
    axis_v = normalize(np.cross(normal, axis_u))
    uv = np.column_stack(((anchors - origin) @ axis_u, (anchors - origin) @ axis_v))
    try:
        hull = ConvexHull(uv)
    except Exception as error:
        raise ValueError("degenerate palm convex hull") from error
    polygon = uv[hull.vertices]
    rolled = np.roll(polygon, -1, axis=0)
    cross = polygon[:, 0] * rolled[:, 1] - rolled[:, 1] * rolled[:, 0]
    area2 = float(np.sum(cross))
    if not np.isfinite(area2) or abs(area2) <= 1e-12:
        raise ValueError("zero palm convex-hull area")
    centroid_uv = np.sum((polygon + rolled) * cross[:, None], axis=0) / (3.0 * area2)
    return origin + centroid_uv[0] * axis_u + centroid_uv[1] * axis_v


def mano_palm_vertex_centroid(
    points: np.ndarray, vertices: np.ndarray, *, plane_tolerance_m: float = 0.018
) -> np.ndarray:
    """Estimate a palm surface center from MANO vertices inside the palm hull.

    The HDF5 recording contains MANO vertices but not the mesh face table.  We
    therefore use the projected wrist/MCP convex hull and a bounded distance to
    the fitted palm plane.  This is a vertex-area proxy, explicitly recorded as
    such in the manifest rather than being called a true mesh surface centroid.
    """
    p = np.asarray(points, dtype=np.float64)
    vtx = np.asarray(vertices, dtype=np.float64)
    if vtx.ndim != 2 or vtx.shape[1] != 3 or not len(vtx):
        raise ValueError("MANO vertices must have shape (N, 3)")
    origin, axis_u, normal = _palm_plane(p)
    axis_v = normalize(np.cross(normal, axis_u))
    anchors = p[[WRIST, *MCP_INDICES]]
    uv = np.column_stack(((anchors - origin) @ axis_u, (anchors - origin) @ axis_v))
    hull = ConvexHull(uv)
    polygon = uv[hull.vertices]
    projected = np.column_stack(((vtx - origin) @ axis_u, (vtx - origin) @ axis_v))
    # A winding-independent inside test for the convex polygon.
    edge = np.roll(polygon, -1, axis=0) - polygon
    rel = projected[:, None, :] - polygon[None, :, :]
    cross = edge[None, :, 0] * rel[..., 1] - edge[None, :, 1] * rel[..., 0]
    inside = np.all(cross >= -1e-8, axis=1) | np.all(cross <= 1e-8, axis=1)
    distance = np.abs((vtx - origin) @ normal)
    selected = inside & (distance <= float(plane_tolerance_m))
    if np.count_nonzero(selected) < 8:
        raise ValueError("too few MANO palm vertices")
    return np.mean(vtx[selected], axis=0)


def _load_mano_asset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load MANO joint regressor and faces without requiring chumpy at runtime.

    The WiLoR MANO pickle stores ``shapedirs`` as a legacy chumpy object, even
    though this experiment only needs the static joint regressor and topology.
    A minimal unpickle stub is therefore used when chumpy is not installed.
    """
    try:
        with path.open("rb") as handle:
            asset = pickle.load(handle, encoding="latin1")
    except ModuleNotFoundError as error:
        if error.name != "chumpy":
            raise

        class _ChumpyStub:
            def __setstate__(self, state: object) -> None:
                self.__dict__["state"] = state

        chumpy = types.ModuleType("chumpy")
        chumpy.__path__ = []  # type: ignore[attr-defined]
        reordering = types.ModuleType("chumpy.reordering")
        reordering.Select = _ChumpyStub  # type: ignore[attr-defined]
        chumpy_ch = types.ModuleType("chumpy.ch")
        chumpy_ch.Ch = _ChumpyStub  # type: ignore[attr-defined]
        sys.modules.update(
            {"chumpy": chumpy, "chumpy.reordering": reordering, "chumpy.ch": chumpy_ch}
        )
        with path.open("rb") as handle:
            asset = pickle.load(handle, encoding="latin1")

    regressor = asset["J_regressor"]
    if hasattr(regressor, "toarray"):
        regressor = regressor.toarray()
    regressor = np.asarray(regressor, dtype=np.float64)
    faces = np.asarray(asset["f"], dtype=np.int64)
    if regressor.shape != (16, 778) or faces.shape != (1538, 3):
        raise ValueError(
            f"unexpected MANO asset shapes regressor={regressor.shape}, faces={faces.shape}"
        )
    return regressor, faces


def align_mano_vertices_to_points(
    vertices: np.ndarray,
    points: np.ndarray,
    *,
    asset_path: Path,
    max_fit_rmse_m: float = 0.02,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rigidly place local MANO vertices into the depth-anchored hand frame.

    MANO joint order is ``wrist, index-MCP, middle-MCP, middle-PIP/...`` with
    ring and little MCP at indices 10 and 7 respectively.  The five anchors
    are matched to the 21-landmark wrist/MCP points, so no camera pose is
    applied to the already depth-anchored hand landmarks.
    """
    vtx = np.asarray(vertices, dtype=np.float64)
    p = np.asarray(points, dtype=np.float64)
    if vtx.ndim != 3 or vtx.shape[1:] != (778, 3) or p.ndim != 3 or p.shape[1:] != (25, 3):
        raise ValueError(f"invalid MANO alignment shapes vertices={vtx.shape}, points={p.shape}")
    if len(vtx) != len(p):
        raise ValueError("MANO and hand landmark timelines differ")
    regressor, _ = _load_mano_asset(asset_path)
    mano_joints = np.einsum("jv,tvc->tjc", regressor, vtx)
    mano_indices = (0, 1, 4, 10, 7)
    point_indices = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, LITTLE_MCP)
    aligned = np.full_like(vtx, np.nan)
    fit_rmse = np.full(len(p), np.nan, dtype=np.float64)
    valid = np.zeros(len(p), dtype=bool)
    rotations = np.full((len(p), 3, 3), np.nan, dtype=np.float64)
    translations = np.full((len(p), 3), np.nan, dtype=np.float64)
    for frame in range(len(p)):
        source = mano_joints[frame, list(mano_indices)]
        target = p[frame, list(point_indices)]
        if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
            continue
        try:
            rotation, translation = robust_rigid_fit(source, target)
        except (ValueError, np.linalg.LinAlgError):
            continue
        residual = (rotation @ source.T).T + translation - target
        rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        fit_rmse[frame] = rmse
        if not np.isfinite(rmse) or rmse > float(max_fit_rmse_m):
            continue
        aligned[frame] = (rotation @ vtx[frame].T).T + translation
        rotations[frame] = rotation
        translations[frame] = translation
        valid[frame] = True
    finite = fit_rmse[np.isfinite(fit_rmse)]
    report = {
        "asset_path": str(asset_path),
        "anchor_mapping": {
            "mano": list(mano_indices),
            "landmark_25": list(point_indices),
        },
        "max_fit_rmse_m": float(max_fit_rmse_m),
        "valid_frames": int(np.count_nonzero(valid)),
        "invalid_frames": int(np.count_nonzero(~valid)),
        "fit_rmse_m": stats(finite),
        "valid": valid,
        "rotations": rotations,
        "translations": translations,
    }
    return aligned, report


def mano_palm_surface_centroid(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    plane_tolerance_m: float = 0.018,
) -> np.ndarray:
    """Area-weighted centroid of MANO triangles belonging to the palm.

    A face is selected when its three vertices lie inside the projected
    wrist/four-MCP convex hull and its centroid is within the palm thickness
    tolerance.  This is a mesh surface centroid, unlike a uniform vertex mean.
    """
    p = np.asarray(points, dtype=np.float64)
    vtx = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    if p.shape != (25, 3) or vtx.ndim != 2 or vtx.shape[1:] != (3,) or tri.ndim != 2 or tri.shape[1:] != (3,):
        raise ValueError(f"invalid MANO surface shapes points={p.shape}, vertices={vtx.shape}, faces={tri.shape}")
    if not len(tri) or np.any(tri < 0) or np.any(tri >= len(vtx)):
        raise ValueError("MANO faces are empty or out of bounds")
    if not np.all(np.isfinite(vtx)):
        raise ValueError("MANO aligned vertices contain non-finite values")
    origin, axis_u, normal = _palm_plane(p)
    axis_v = normalize(np.cross(normal, axis_u))
    anchors = p[[WRIST, *MCP_INDICES]]
    uv = np.column_stack(((anchors - origin) @ axis_u, (anchors - origin) @ axis_v))
    hull = ConvexHull(uv)
    polygon = uv[hull.vertices]
    projected = np.column_stack(((vtx - origin) @ axis_u, (vtx - origin) @ axis_v))
    edge = np.roll(polygon, -1, axis=0) - polygon
    rel = projected[:, None, :] - polygon[None, :, :]
    cross = edge[None, :, 0] * rel[..., 1] - edge[None, :, 1] * rel[..., 0]
    inside = np.all(cross >= -1e-8, axis=1) | np.all(cross <= 1e-8, axis=1)
    triangles = vtx[tri]
    centroids = np.mean(triangles, axis=1)
    distance = np.abs((centroids - origin) @ normal)
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    selected = np.all(inside[tri], axis=1) & (distance <= float(plane_tolerance_m))
    selected &= np.isfinite(areas) & (areas > 1e-12)
    if np.count_nonzero(selected) < 8:
        raise ValueError("too few MANO palm faces")
    return np.sum(areas[selected, None] * centroids[selected], axis=0) / np.sum(areas[selected])


def calibrated_mcp_center(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit the wrist-to-MCP interpolation fraction to the polygon center."""
    p = np.asarray(points, dtype=np.float64)
    if p.ndim == 2:
        p = p[None]
    vectors = np.mean(p[:, list(MCP_INDICES)], axis=1) - p[:, WRIST]
    targets = np.stack([palm_polygon_centroid(hand) for hand in p])
    denom = np.sum(vectors * vectors, axis=1)
    alpha_each = np.sum((targets - p[:, WRIST]) * vectors, axis=1) / np.maximum(denom, 1e-12)
    alpha = float(np.clip(np.median(alpha_each[np.isfinite(alpha_each)]), 0.0, 1.0))
    centers = p[:, WRIST] + alpha * vectors
    return centers if len(centers) > 1 else centers[0], alpha


def load_mano_vertices_world(
    annotation_path: Path, session_dir: Path, side: str
) -> np.ndarray:
    """Load MANO vertices and apply the same optical-to-world contract as joints."""
    import h5py

    with h5py.File(annotation_path, "r") as h5:
        vertex_key = f"hand-pose/{side}_mano_vertices"
        required = (
            vertex_key,
            "hand-pose/timestamps",
            "cam-pose/timestamps",
            "cam-pose/rotations",
            "cam-pose/translations",
        )
        missing = [key for key in required if key not in h5]
        if missing:
            raise KeyError(f"missing MANO/world datasets: {missing}")
        vertices = np.asarray(h5[vertex_key], dtype=np.float64)
        hand_absolute = np.asarray(h5["hand-pose/timestamps"], dtype=np.float64).reshape(-1)
        camera_absolute = np.asarray(h5["cam-pose/timestamps"], dtype=np.float64).reshape(-1)
        camera_rotations = np.asarray(h5["cam-pose/rotations"], dtype=np.float64)
        camera_translations = np.asarray(h5["cam-pose/translations"], dtype=np.float64)
    if vertices.ndim != 3 or vertices.shape[2] != 3 or len(vertices) != len(hand_absolute):
        raise ValueError(f"unexpected MANO vertex shape {vertices.shape}")
    hand_time = hand_absolute - hand_absolute[0]
    camera_time = camera_absolute - hand_absolute[0]
    camera_rotations = _interpolate_rotation_samples(camera_rotations, camera_time, hand_time)
    camera_translations = np.column_stack(
        [np.interp(hand_time, camera_time, camera_translations[:, axis]) for axis in range(3)]
    )
    calibration = session_dir / "calibrations" / "R_optical_to_link.npy"
    optical_to_link = np.asarray(np.load(calibration), dtype=np.float64)
    world_from_optical = camera_rotations @ optical_to_link
    world = np.einsum("tij,tkj->tki", world_from_optical, vertices)
    return np.einsum("ij,tkj->tki", YUP_TO_ZUP, world)


def load_mano_vertices_local(
    annotation_path: Path, side: str
) -> np.ndarray:
    """Load raw local MANO vertices without applying camera/world transforms."""
    import h5py

    key = f"hand-pose/{side}_mano_vertices"
    with h5py.File(annotation_path, "r") as h5:
        if key not in h5:
            raise KeyError(f"missing MANO vertices: {key}")
        vertices = np.asarray(h5[key], dtype=np.float64)
    if vertices.ndim != 3 or vertices.shape[1:] != (778, 3):
        raise ValueError(f"unexpected MANO vertex shape {vertices.shape}")
    return vertices


def build_palm_template(points: np.ndarray) -> np.ndarray:
    """Build a sequence-specific rigid palm template in wrist-link axes."""
    p = np.asarray(points, dtype=np.float64)
    anchor_indices = (WRIST, *MCP_INDICES)
    local_anchors = []
    for hand in p:
        center = palm_polygon_centroid(hand)
        rotation = landmark_frame(hand, INDEX_MCP, LITTLE_MCP)
        local_anchors.append((hand[list(anchor_indices)] - center) @ rotation)
    template = np.median(np.asarray(local_anchors), axis=0)
    if not np.all(np.isfinite(template)):
        raise ValueError("non-finite palm template")
    return template


def robust_rigid_fit(
    source: np.ndarray,
    target: np.ndarray,
    *,
    huber_delta_m: float = 0.008,
    iterations: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """IRLS Kabsch fit that limits the influence of a noisy palm landmark."""
    weights = np.ones(len(source), dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    for _ in range(max(1, int(iterations))):
        rotation, translation = rigid_fit(source, target, weights=weights)
        residual = np.linalg.norm(
            (rotation @ np.asarray(source).T).T + translation - target,
            axis=1,
        )
        weights = np.minimum(1.0, float(huber_delta_m) / np.maximum(residual, 1e-12))
    return rotation, translation


def batch_frames(
    points: np.ndarray,
    method: str,
    side: str,
    *,
    mano_vertices: np.ndarray | None = None,
    mano_faces: np.ndarray | None = None,
    calibrated_alpha: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(points, dtype=np.float64)
    positions = np.empty((len(p), 3), dtype=np.float64)
    rotations = np.empty((len(p), 3, 3), dtype=np.float64)
    robot = robot_landmarks(side)
    previous = np.eye(3, dtype=np.float64)
    palm_template = (
        build_palm_template(p) if method == "template_palm_kabsch" else None
    )
    anchor_indices = (WRIST, *MCP_INDICES)

    for frame, hand in enumerate(p):
        mcp_mean = np.mean(hand[list(MCP_INDICES)], axis=0)
        mid_palm = 0.5 * (hand[WRIST] + mcp_mean)
        try:
            if method == "historical_tip_wrist":
                rotation = historical_frame(hand)
                position = hand[WRIST]
            elif method == "mcp_frame_wrist":
                rotation = landmark_frame(hand, INDEX_MCP, LITTLE_MCP)
                position = hand[WRIST]
            elif method == "mcp_mid_palm_site":
                rotation = landmark_frame(hand, INDEX_MCP, LITTLE_MCP)
                position = mid_palm - rotation @ robot.palm
            elif method == "mcp_arc_palm_site":
                rotation = landmark_frame(hand, INDEX_MCP, LITTLE_MCP)
                position = mcp_mean - rotation @ robot.palm
            elif method == "palm_polygon_site":
                rotation = landmark_frame(hand, INDEX_MCP, LITTLE_MCP)
                position = palm_polygon_centroid(hand) - rotation @ robot.palm
            elif method == "projected_hull_centroid":
                rotation = landmark_frame(hand, INDEX_MCP, LITTLE_MCP)
                position = projected_convex_hull_centroid(hand) - rotation @ robot.palm
            elif method == "mano_palm_vertex_centroid":
                if mano_vertices is None:
                    raise ValueError("MANO vertices are required for this method")
                rotation = landmark_frame(hand, INDEX_MCP, LITTLE_MCP)
                position = mano_palm_vertex_centroid(
                    hand, mano_vertices[frame]
                ) - rotation @ robot.palm
            elif method == "mano_palm_surface_centroid":
                if mano_vertices is None or mano_faces is None:
                    raise ValueError("aligned MANO vertices/faces are required for this method")
                rotation = landmark_frame(hand, INDEX_MCP, LITTLE_MCP)
                position = mano_palm_surface_centroid(
                    hand,
                    mano_vertices[frame],
                    mano_faces,
                ) - rotation @ robot.palm
            elif method == "calibrated_mcp_center":
                if calibrated_alpha is None:
                    raise ValueError("calibrated alpha is required for this method")
                rotation = landmark_frame(hand, INDEX_MCP, LITTLE_MCP)
                position = (
                    hand[WRIST]
                    + float(calibrated_alpha) * (mcp_mean - hand[WRIST])
                    - rotation @ robot.palm
                )
            elif method == "template_palm_kabsch":
                assert palm_template is not None
                rotation, palm_position = robust_rigid_fit(
                    palm_template, hand[list(anchor_indices)]
                )
                position = palm_position - rotation @ robot.palm
            elif method in ("root_mid_index_ring", "root_mid_index_little"):
                outer_b = RING_MCP if method.endswith("ring") else LITTLE_MCP
                rotation = landmark_frame(hand, INDEX_MCP, outer_b)
                human_mid = 0.5 * (hand[INDEX_MCP] + hand[outer_b])
                position = human_mid - rotation @ robot.root_mid
            elif method in ("kabsch_index_ring", "kabsch_index_little"):
                outer_b = RING_MCP if method.endswith("ring") else LITTLE_MCP
                source = np.stack(
                    [robot.wrist, robot.index_root, robot.middle_root], axis=0
                )
                target = np.stack(
                    [hand[WRIST], hand[INDEX_MCP], hand[outer_b]], axis=0
                )
                rotation, position = rigid_fit(source, target)
            else:
                raise ValueError(f"unknown wrist method {method}")
        except (ValueError, np.linalg.LinAlgError):
            rotation = previous
            position = hand[WRIST]
        rotations[frame] = rotation
        positions[frame] = position
        previous = rotation
    return positions, rotations


def rigid_fit(
    source: np.ndarray,
    target: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return R,t minimizing ||R source + t - target|| with R in SO(3)."""
    q = np.asarray(source, dtype=np.float64)
    p = np.asarray(target, dtype=np.float64)
    if q.shape != p.shape or q.ndim != 2 or q.shape[1] != 3:
        raise ValueError(f"invalid rigid fit shapes source={q.shape} target={p.shape}")
    if weights is None:
        w = np.ones(len(q), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape != (len(q),) or np.any(~np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError(f"invalid rigid-fit weights {w}")
    weight_sum = float(np.sum(w))
    if weight_sum <= 1e-12:
        raise ValueError("rigid-fit weights sum to zero")
    w = w / weight_sum
    q_mean = np.sum(w[:, None] * q, axis=0)
    p_mean = np.sum(w[:, None] * p, axis=0)
    covariance = (w[:, None] * (q - q_mean)).T @ (p - p_mean)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = p_mean - rotation @ q_mean
    return rotation, translation


def transform_points(
    positions: np.ndarray, rotations: np.ndarray, local: np.ndarray
) -> np.ndarray:
    return np.einsum("tij,pj->tpi", rotations, local) + positions[:, None]


def resample_pose(
    positions: np.ndarray,
    rotations: np.ndarray,
    source_time: np.ndarray,
    target_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_time = np.asarray(source_time, dtype=np.float64)
    target_time = np.asarray(target_time, dtype=np.float64)
    pos = np.column_stack(
        [np.interp(target_time, source_time, positions[:, axis]) for axis in range(3)]
    )
    if len(source_time) == 1:
        rot = np.repeat(rotations[:1], len(target_time), axis=0)
    else:
        clipped = np.clip(target_time, source_time[0], source_time[-1])
        rot = Slerp(source_time, Rotation.from_matrix(rotations))(clipped).as_matrix()
    return pos, rot


def rotation_error_rad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (Rotation.from_matrix(a).inv() * Rotation.from_matrix(b)).magnitude()


def temporal_metrics(
    positions: np.ndarray, rotations: np.ndarray, timestamps: np.ndarray
) -> dict[str, Any]:
    if len(timestamps) < 2:
        return {"position_speed_m_s": stats([]), "angular_speed_deg_s": stats([])}
    dt = np.diff(timestamps)
    linear = np.linalg.norm(np.diff(positions, axis=0), axis=1) / dt
    angular = np.degrees(rotation_error_rad(rotations[:-1], rotations[1:])) / dt
    return {
        "position_speed_m_s": stats(linear),
        "angular_speed_deg_s": stats(angular),
    }


def wrist_landmark_metrics(
    method: str,
    side: str,
    points: np.ndarray,
    positions: np.ndarray,
    rotations: np.ndarray,
    timestamps: np.ndarray,
) -> dict[str, Any]:
    robot = robot_landmarks(side)
    local = np.stack(
        [robot.wrist, robot.palm, robot.index_root, robot.middle_root], axis=0
    )
    transformed = transform_points(positions, rotations, local)
    wrist, palm, index_root, middle_root = np.moveaxis(transformed, 1, 0)
    mcp_mean = np.mean(points[:, list(MCP_INDICES)], axis=1)
    mid_palm = 0.5 * (points[:, WRIST] + mcp_mean)
    polygon_center = np.stack([palm_polygon_centroid(hand) for hand in points])
    human_ir = np.stack([points[:, INDEX_MCP], points[:, RING_MCP]], axis=1)
    human_il = np.stack([points[:, INDEX_MCP], points[:, LITTLE_MCP]], axis=1)
    robot_roots = np.stack([index_root, middle_root], axis=1)
    ir_rmse = np.sqrt(np.mean(np.sum((robot_roots - human_ir) ** 2, axis=2), axis=1))
    il_rmse = np.sqrt(np.mean(np.sum((robot_roots - human_il) ** 2, axis=2), axis=1))
    return {
        "method": method,
        "side": side,
        "frames": int(len(points)),
        "wrist_to_human_wrist_m": stats(np.linalg.norm(wrist - points[:, WRIST], axis=1)),
        "palm_site_to_mid_palm_m": stats(np.linalg.norm(palm - mid_palm, axis=1)),
        "palm_site_to_mcp_arc_m": stats(np.linalg.norm(palm - mcp_mean, axis=1)),
        "palm_site_to_polygon_centroid_m": stats(
            np.linalg.norm(palm - polygon_center, axis=1)
        ),
        "root_pair_to_index_ring_rmse_m": stats(ir_rmse),
        "root_pair_to_index_little_rmse_m": stats(il_rmse),
        "temporal": temporal_metrics(positions, rotations, timestamps),
    }


def load_kinematics(
    path: Path, target_timestamps_path: Path | None = None
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = ("body_names", "body_pos_w", "body_quat_w")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"missing kinematics fields {missing}: {path}")
        result = {key: np.asarray(data[key]) for key in required}
        if "timestamps" in data.files:
            result["timestamps"] = np.asarray(data["timestamps"])
    if "timestamps" not in result:
        if target_timestamps_path is None:
            raise KeyError(
                f"missing kinematics timestamps in {path}; provide "
                "--target-timestamps-npz"
            )
        with np.load(target_timestamps_path, allow_pickle=False) as timeline:
            if "timestamps" not in timeline.files:
                raise KeyError(f"missing timestamps: {target_timestamps_path}")
            result["timestamps"] = np.asarray(timeline["timestamps"])
    frames = int(np.asarray(result["body_pos_w"]).shape[0])
    if np.asarray(result["timestamps"]).shape != (frames,):
        raise ValueError(
            f"body/timeline mismatch frames={frames} "
            f"timestamps={np.asarray(result['timestamps']).shape}"
        )
    return result


def current_wrist_pose(
    kinematics: dict[str, np.ndarray], side: str
) -> tuple[np.ndarray, np.ndarray]:
    names = [str(name) for name in kinematics["body_names"]]
    name = f"{side}_wrist_yaw_link"
    if name not in names:
        raise KeyError(f"{name} missing from body_names")
    index = names.index(name)
    position = np.asarray(kinematics["body_pos_w"][:, index], dtype=np.float64)
    wxyz = np.asarray(kinematics["body_quat_w"][:, index], dtype=np.float64)
    xyzw = wxyz[:, [1, 2, 3, 0]]
    rotation = Rotation.from_quat(xyzw).as_matrix()
    return position, rotation


def candidate_vs_current(
    candidate_pos: np.ndarray,
    candidate_rot: np.ndarray,
    current_pos: np.ndarray,
    current_rot: np.ndarray,
) -> dict[str, Any]:
    return {
        "position_correction_m": stats(
            np.linalg.norm(candidate_pos - current_pos, axis=1)
        ),
        "orientation_correction_deg": stats(
            np.degrees(rotation_error_rad(current_rot, candidate_rot))
        ),
    }


def baseline_local_offset(
    candidate_pos: np.ndarray,
    candidate_rot: np.ndarray,
    baseline_pos: np.ndarray,
    baseline_rot: np.ndarray,
) -> dict[str, Any]:
    delta = np.einsum(
        "tji,tj->ti", baseline_rot, candidate_pos - baseline_pos
    )
    angle = np.degrees(rotation_error_rad(baseline_rot, candidate_rot))
    return {
        "translation_xyz_m": {
            "mean": np.mean(delta, axis=0).tolist(),
            "median": np.median(delta, axis=0).tolist(),
            "p95_abs": np.percentile(np.abs(delta), 95, axis=0).tolist(),
        },
        "orientation_delta_deg": stats(angle),
    }


def slim_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in evaluation.items() if key != "arrays"}


def finger_interaction_metrics(
    evaluation: dict[str, Any],
    left_human: np.ndarray,
    right_human: np.ndarray,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for side, human in (("left", left_human), ("right", right_human)):
        robot = evaluation["arrays"][f"{side}_robot_tips"]
        human_dist = np.stack(
            [
                np.linalg.norm(human[:, 0] - human[:, 1], axis=1),
                np.linalg.norm(human[:, 0] - human[:, 2], axis=1),
                np.linalg.norm(human[:, 1] - human[:, 2], axis=1),
            ],
            axis=1,
        )
        robot_dist = np.stack(
            [
                np.linalg.norm(robot[:, 0] - robot[:, 1], axis=1),
                np.linalg.norm(robot[:, 0] - robot[:, 2], axis=1),
                np.linalg.norm(robot[:, 1] - robot[:, 2], axis=1),
            ],
            axis=1,
        )
        scale = float(evaluation["tip_procrustes_rmse_m"][f"median_scale_{side}"])
        expected_robot = human_dist / max(scale, 1e-8)
        grasp_gap = 0.5 * (human_dist[:, 0] + human_dist[:, 1])
        threshold = float(np.percentile(grasp_gap, 20))
        contact_like = grasp_gap <= threshold
        output[side] = {
            "human_to_robot_shape_scale": scale,
            "normalized_pair_distance_error_m": stats(
                np.abs(robot_dist - expected_robot)
            ),
            "contact_like_frames": int(np.count_nonzero(contact_like)),
            "contact_like_robot_thumb_gap_m": stats(
                0.5 * (robot_dist[contact_like, 0] + robot_dist[contact_like, 1])
            ),
            "contact_like_pair_error_m": stats(
                np.abs(robot_dist[contact_like] - expected_robot[contact_like])
            ),
        }
    return output


def plot_wrist_corrections(
    output: Path,
    timeline: np.ndarray,
    metrics_arrays: dict[str, dict[str, np.ndarray]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for side_index, side in enumerate(("left", "right")):
        for method, arrays in metrics_arrays[side].items():
            axes[0, side_index].plot(
                timeline,
                np.linalg.norm(arrays["candidate_pos"] - arrays["current_pos"], axis=1),
                label=method,
                linewidth=1.0,
            )
            axes[1, side_index].plot(
                timeline,
                np.degrees(
                    rotation_error_rad(arrays["current_rot"], arrays["candidate_rot"])
                ),
                label=method,
                linewidth=1.0,
            )
        axes[0, side_index].set_title(f"{side} wrist position correction")
        axes[0, side_index].set_ylabel("m")
        axes[1, side_index].set_title(f"{side} wrist orientation correction")
        axes[1, side_index].set_ylabel("deg")
        axes[1, side_index].set_xlabel("time (s)")
        for row in range(2):
            axes[row, side_index].grid(True, alpha=0.25)
    axes[0, 1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_wrist_keyframes(
    output: Path,
    left_points: np.ndarray,
    right_points: np.ndarray,
    candidates: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
) -> dict[str, int]:
    selected: dict[str, int] = {}
    methods = (
        "historical_tip_wrist",
        "mcp_frame_wrist",
        "mcp_arc_palm_site",
        "palm_polygon_site",
        "template_palm_kabsch",
        "projected_hull_centroid",
        "mano_palm_surface_centroid",
        "calibrated_mcp_center",
        "root_mid_index_ring",
        "kabsch_index_ring",
    )
    fig = plt.figure(figsize=(14, 6))
    for column, (side, points) in enumerate(
        (("left", left_points), ("right", right_points)), start=1
    ):
        tips = points[:, [THUMB_TIP, INDEX_TIP, MIDDLE_TIP]]
        gap = 0.5 * (
            np.linalg.norm(tips[:, 0] - tips[:, 1], axis=1)
            + np.linalg.norm(tips[:, 0] - tips[:, 2], axis=1)
        )
        frame = int(np.argmin(gap))
        selected[side] = frame
        ax = fig.add_subplot(1, 2, column, projection="3d")
        human = points[frame, [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, LITTLE_MCP]]
        ax.scatter(human[:, 0], human[:, 1], human[:, 2], c="black", s=35, label="human")
        ax.plot(human[:, 0], human[:, 1], human[:, 2], c="black", linewidth=1.0)
        robot = robot_landmarks(side)
        local = np.stack(
            [robot.wrist, robot.palm, robot.index_root, robot.middle_root], axis=0
        )
        for method in methods:
            if method not in candidates[side]:
                continue
            position, rotation = candidates[side][method]
            transformed = (rotation[frame] @ local.T).T + position[frame]
            ax.scatter(
                transformed[:, 0], transformed[:, 1], transformed[:, 2], s=22, label=method
            )
            ax.plot(
                transformed[[0, 2, 3], 0],
                transformed[[0, 2, 3], 1],
                transformed[[0, 2, 3], 2],
                linewidth=0.8,
            )
        ax.set_title(f"{side} contact-like frame {frame}")
        ax.set_xlabel("world X")
        ax.set_ylabel("world Y")
        ax.set_zlabel("world Z")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return selected


def plot_finger_comparison(
    output: Path,
    timestamps: np.ndarray,
    evaluations: dict[str, dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    t = timestamps - timestamps[0]
    for column, side in enumerate(("left", "right")):
        for method, evaluation in evaluations.items():
            axes[0, column].plot(
                t,
                evaluation["arrays"][f"{side}_rmse"],
                label=method,
                linewidth=1.0,
            )
            robot = evaluation["arrays"][f"{side}_robot_tips"]
            thumb_gap = 0.5 * (
                np.linalg.norm(robot[:, 0] - robot[:, 1], axis=1)
                + np.linalg.norm(robot[:, 0] - robot[:, 2], axis=1)
            )
            axes[1, column].plot(t, thumb_gap, label=method, linewidth=1.0)
        axes[0, column].set_title(f"{side} Procrustes tip RMSE")
        axes[0, column].set_ylabel("m")
        axes[1, column].set_title(f"{side} robot thumb-to-opposition gap")
        axes[1, column].set_ylabel("m")
        axes[1, column].set_xlabel("time (s)")
        for row in range(2):
            axes[row, column].grid(True, alpha=0.25)
    axes[0, 1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def write_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **values)


def main() -> None:
    args = parse_args()
    session = args.session_dir.expanduser().resolve()
    annotation = session / "annotation.hdf5"
    kinematics_path = args.kinematics_npz.expanduser().resolve()
    out = args.out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not annotation.is_file():
        raise FileNotFoundError(annotation)
    if not kinematics_path.is_file():
        raise FileNotFoundError(kinematics_path)

    left_world, right_world, native_time, _, source_meta = _load_session_keypoints(
        annotation, session
    )
    requested_wrist_methods = [
        item.strip() for item in args.wrist_methods.split(",") if item.strip()
    ]
    supported_wrist_methods = {
        "historical_tip_wrist",
        "mcp_frame_wrist",
        "mcp_mid_palm_site",
        "mcp_arc_palm_site",
        "palm_polygon_site",
        "template_palm_kabsch",
        "projected_hull_centroid",
        "mano_palm_surface_centroid",
        "calibrated_mcp_center",
        "root_mid_index_ring",
        "root_mid_index_little",
        "kabsch_index_ring",
        "kabsch_index_little",
    }
    unknown_wrist = set(requested_wrist_methods) - supported_wrist_methods
    if unknown_wrist:
        raise ValueError(f"unknown wrist methods: {sorted(unknown_wrist)}")
    if not requested_wrist_methods:
        raise ValueError("at least one wrist method is required")
    mano_vertices_by_side: dict[str, np.ndarray | None] = {"left": None, "right": None}
    mano_surface_reports: dict[str, dict[str, Any]] = {}
    mano_faces: np.ndarray | None = None
    if "mano_palm_surface_centroid" in requested_wrist_methods:
        if not args.mano_asset.is_file():
            raise FileNotFoundError(f"MANO asset is required: {args.mano_asset}")
        _, mano_faces = _load_mano_asset(args.mano_asset.expanduser().resolve())
        mano_vertices_by_side = {
            side: load_mano_vertices_local(annotation, side)
            for side in ("left", "right")
        }
    if args.max_native_frames is not None:
        count = max(1, min(int(args.max_native_frames), len(native_time)))
        left_world = left_world[:count]
        right_world = right_world[:count]
        native_time = native_time[:count]
    left = np.einsum("ij,tkj->tki", YUP_TO_ZUP, left_world)
    right = np.einsum("ij,tkj->tki", YUP_TO_ZUP, right_world)

    if "mano_palm_surface_centroid" in requested_wrist_methods:
        for side, points in (("left", left), ("right", right)):
            local_vertices = mano_vertices_by_side[side]
            assert local_vertices is not None
            local_vertices = local_vertices[: len(points)]
            aligned, report = align_mano_vertices_to_points(
                local_vertices,
                points,
                asset_path=args.mano_asset.expanduser().resolve(),
            )
            mano_vertices_by_side[side] = aligned
            # Arrays are kept out of JSON but retained for later candidate use.
            mano_surface_reports[side] = report

    timeline_path = (
        args.target_timestamps_npz.expanduser().resolve()
        if args.target_timestamps_npz is not None
        else None
    )
    if timeline_path is not None and not timeline_path.is_file():
        raise FileNotFoundError(timeline_path)
    kinematics = load_kinematics(kinematics_path, timeline_path)
    target_time_all = np.asarray(kinematics["timestamps"], dtype=np.float64)
    target_time_all = target_time_all - target_time_all[0]
    target_mask = target_time_all <= native_time[-1] + 1e-9
    target_time = target_time_all[target_mask]
    if not len(target_time):
        raise ValueError("source and target timelines do not overlap")

    wrist_methods = tuple(requested_wrist_methods)
    correction_baseline_method = (
        "historical_tip_wrist"
        if "historical_tip_wrist" in wrist_methods
        else wrist_methods[0]
    )
    wrist_native: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
        "left": {},
        "right": {},
    }
    wrist_target: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
        "left": {},
        "right": {},
    }
    wrist_summary: dict[str, Any] = {"methods": {}, "timeline_frames": int(len(target_time))}
    target_archive: dict[str, np.ndarray] = {"timestamps": target_time}
    native_archive: dict[str, np.ndarray] = {"timestamps": native_time}
    correction_arrays: dict[str, dict[str, dict[str, np.ndarray]]] = {
        "left": {},
        "right": {},
    }

    for side, points in (("left", left), ("right", right)):
        current_pos_all, current_rot_all = current_wrist_pose(kinematics, side)
        current_pos = current_pos_all[target_mask]
        current_rot = current_rot_all[target_mask]
        calibrated_alpha = None
        if "calibrated_mcp_center" in wrist_methods:
            _, calibrated_alpha = calibrated_mcp_center(points)
        for method in wrist_methods:
            if args.template_palm_from_current_kinematics and method == "template_palm_kabsch":
                native_pos, native_rot = resample_pose(
                    current_pos_all,
                    current_rot_all,
                    target_time_all,
                    native_time,
                )
                target_pos = current_pos.copy()
                target_rot = current_rot.copy()
            else:
                native_pos, native_rot = batch_frames(
                    points,
                    "mano_palm_surface_centroid" if method == "mano_palm_surface_centroid" else method,
                    side,
                    mano_vertices=mano_vertices_by_side[side],
                    mano_faces=mano_faces,
                    calibrated_alpha=calibrated_alpha,
                )
                target_pos, target_rot = resample_pose(
                    native_pos, native_rot, native_time, target_time
                )
            wrist_native[side][method] = (native_pos, native_rot)
            wrist_target[side][method] = (target_pos, target_rot)
            metric = wrist_landmark_metrics(
                method, side, points, native_pos, native_rot, native_time
            )
            metric["correction_from_current_kinematics"] = candidate_vs_current(
                target_pos, target_rot, current_pos, current_rot
            )
            wrist_summary["methods"].setdefault(method, {})[side] = metric
            if method == "calibrated_mcp_center":
                metric["calibrated_alpha"] = float(calibrated_alpha)
            if method == "mano_palm_surface_centroid":
                metric["mano_alignment"] = {
                    key: value
                    for key, value in mano_surface_reports[side].items()
                    if key not in {"valid", "rotations", "translations"}
                }
            native_archive[f"{method}__{side}_position"] = native_pos
            native_archive[f"{method}__{side}_rotation"] = native_rot
            target_archive[f"{method}__{side}_position"] = target_pos
            target_archive[f"{method}__{side}_rotation"] = target_rot
            correction_arrays[side][method] = {
                "candidate_pos": target_pos,
                "candidate_rot": target_rot,
                "current_pos": current_pos,
                "current_rot": current_rot,
            }

        baseline_pos, baseline_rot = wrist_native[side][correction_baseline_method]
        for method in wrist_methods:
            pos, rot = wrist_native[side][method]
            wrist_summary["methods"][method][side]["delta_from_historical_local"] = (
                baseline_local_offset(pos, rot, baseline_pos, baseline_rot)
            )

    keyframes = plot_wrist_keyframes(
        out / "wrist_candidate_keyframes.png", left, right, wrist_native
    )
    plot_wrist_corrections(
        out / "wrist_correction_from_current.png", target_time, correction_arrays
    )
    wrist_summary["contact_like_keyframes"] = keyframes
    write_npz(out / "wrist_candidates_native.npz", native_archive)
    write_npz(out / "wrist_candidates_50hz.npz", target_archive)
    (out / "wrist_metrics.json").write_text(
        json.dumps(wrist_summary, indent=2) + "\n"
    )

    finger_methods = [
        item.strip() for item in args.finger_methods.split(",") if item.strip()
    ]
    unknown = set(finger_methods) - {
        "dexpilot",
        "vector",
        "position",
        "dexpilot_close",
    }
    if unknown:
        raise ValueError(f"unknown finger methods: {sorted(unknown)}")
    left_fk = _build_fixed_base_fk("left")
    right_fk = _build_fixed_base_fk("right")
    evaluations: dict[str, dict[str, Any]] = {}
    finger_summary: dict[str, Any] = {"methods": {}}
    finger_archive: dict[str, np.ndarray] = {"timestamps": native_time}
    left_human = _human_wrist_local_tips(left_world[: len(native_time)])
    right_human = _human_wrist_local_tips(right_world[: len(native_time)])

    for method in finger_methods:
        retarget_type = "dexpilot" if method == "dexpilot_close" else method
        human_close_gamma = 0.5 if method == "dexpilot_close" else None
        config_yml = EXPERIMENT_VECTOR_CONFIG if method == "vector" else None
        qpos, parameters = _retarget_points(
            left_world[: len(native_time)],
            right_world[: len(native_time)],
            retarget_type=retarget_type,
            config_yml=config_yml,
            scaling_factor=float(args.scaling_factor),
            project_dist=None,
            escape_dist=None,
            low_pass_alpha=float(args.low_pass_alpha),
            flexion_scale=float(args.flexion_scale),
            limit_margin=float(args.limit_margin),
            human_close_gamma=human_close_gamma,
        )
        evaluation = _evaluate_method(
            method,
            qpos,
            left_world[: len(native_time)],
            right_world[: len(native_time)],
            left_fk,
            right_fk,
        )
        evaluations[method] = evaluation
        finger_summary["methods"][method] = {
            "parameters": parameters,
            "geometry": slim_evaluation(evaluation),
            "interaction_proxy": finger_interaction_metrics(
                evaluation, left_human, right_human
            ),
        }
        finger_archive[f"{method}__qpos"] = qpos
        finger_archive[f"{method}__left_tips"] = evaluation["arrays"][
            "left_robot_tips"
        ]
        finger_archive[f"{method}__right_tips"] = evaluation["arrays"][
            "right_robot_tips"
        ]

    plot_finger_comparison(
        out / "finger_method_comparison.png", native_time, evaluations
    )
    write_npz(out / "finger_candidates_native.npz", finger_archive)
    (out / "finger_metrics.json").write_text(
        json.dumps(finger_summary, indent=2) + "\n"
    )

    manifest = {
        "schema_version": 1,
        "experiment_only": True,
        "production_pipeline_modified": False,
        "session_dir": str(session),
        "annotation_hdf5": str(annotation),
        "kinematics_npz": str(kinematics_path),
        "source_metadata": source_meta,
        "native_frames": int(len(native_time)),
        "target_frames_evaluated": int(len(target_time)),
        "wrist_methods": list(wrist_methods),
        "finger_methods": finger_methods,
        "correction_baseline_method": correction_baseline_method,
        "robot_landmarks": {
            side: {
                "wrist": robot_landmarks(side).wrist.tolist(),
                "palm": robot_landmarks(side).palm.tolist(),
                "index_root": robot_landmarks(side).index_root.tolist(),
                "middle_root": robot_landmarks(side).middle_root.tolist(),
            }
            for side in ("left", "right")
        },
        "palm_method_contract": {
            "palm_polygon_site": (
                "area centroid of the wrist-index-middle-ring-little MCP polygon"
            ),
            "template_palm_kabsch": (
                "sequence median wrist+four-MCP template in wrist-link axes, "
                "fit per frame with Huber IRLS Kabsch; template origin is the "
                "palm polygon area centroid"
            ),
            "projected_hull_centroid": (
                "area centroid of the convex hull of wrist+four MCP landmarks "
                "after projection to a fitted palm plane"
            ),
            "mano_palm_surface_centroid": (
                "area-weighted centroid of MANO faces inside the wrist/MCP convex "
                "hull after per-frame 5-anchor rigid alignment; invalid when the "
                "alignment RMSE exceeds the configured threshold"
            ),
            "calibrated_mcp_center": (
                "wrist + alpha*(mean MCP-wrist), with alpha fitted per sequence "
                "to the polygon area centroid"
            ),
        },
        "object_interaction_status": (
            "finger contact-like proxy only; explicit cup segmentation and surface "
            "penetration scoring are a separate experiment stage"
        ),
        "mano_alignment": {
            side: {
                key: value
                for key, value in report.items()
                if key not in {"valid", "rotations", "translations"}
            }
            for side, report in mano_surface_reports.items()
        },
        "outputs": [
            "wrist_metrics.json",
            "wrist_candidates_native.npz",
            "wrist_candidates_50hz.npz",
            "wrist_candidate_keyframes.png",
            "wrist_correction_from_current.png",
            "finger_metrics.json",
            "finger_candidates_native.npz",
            "finger_method_comparison.png",
        ],
    }
    (out / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
