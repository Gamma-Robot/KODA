#!/usr/bin/env python3
"""Refine BrainCo wrists and 6-DoF hands with Holosoma sequential SQP.

The loop matches ``InteractionMeshRetargeter`` in amazon-far/holosoma: each
contact frame builds a Delaunay interaction mesh, freezes that frame's object
samples O inside the SQP inner loop, linearizes Laplacian matching plus
anchors/smoothness/limits, and solves a small trust-region step for ``dx``.
The next frame is warm-started from the previous solution.

Anti-penetration is not part of the SQP cost.  After SQP, lerp, and
smoothing, the full stored neighborhood object cloud is tested against the
dense BrainCo visual mesh.  Laplacian SQP may use a prefix of that cloud so
the Delaunay graph stays small; pull-out always uses every stored sample.
The wrist translates toward the recorded human hand (16 MANO landmarks) so
interior points leave along that direction instead of sliding along palm
geom z.  The 16 landmarks are the Laplacian graph and the pull-out target;
the collision surface is the dense mesh.

We retarget the 16-landmark hand instead of the whole body.  Holosoma samples
object points once on a watertight mesh; here O is a per-frame voxel-sparse
unprojection around the hand (visibility and the moving object change the set),
but those samples stay fixed while that frame's SQP iterates.

Non-contact frames do not enter the objective; their wrist/joint offsets lerp
between the neighbouring contact endpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import h5py
import mujoco
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from brainco_hand_retarget import (  # noqa: E402
    BRAINCO_ACTION_JOINT_NAMES,
    BRAINCO_HAND_JOINT_NAMES,
    BRAINCO_LEFT_HAND_JOINT_NAMES,
    BRAINCO_LEFT_HAND_TIP_BODIES,
    BRAINCO_MIMIC_RULES,
    BRAINCO_PALM_GEOM_MESH_NAMES,
    BRAINCO_RIGHT_HAND_JOINT_NAMES,
    BRAINCO_RIGHT_HAND_TIP_BODIES,
    G1_STANDING_HEIGHT,
    MANO_HAND_KEYPOINT_INDICES,
    _load_model,
    _visual_palm_geom,
    brainco_qpos_writer,
)
from contact_pointcloud_analysis import (  # noqa: E402
    FINGER_NAMES,
    to_world,
)
from interaction_mesh import (  # noqa: E402
    build_interaction_adjacency,
    calculate_laplacian_coordinates,
    calculate_laplacian_matrix,
    deepest_inside_world_push,
    retract_axis_away_from_object,
    retract_axis_toward_target,
)
from retarget_template_palm_reference import (  # noqa: E402
    WRIST_BODIES,
    _angular_velocity_world,
    _atomic_json,
    _atomic_npz,
    _continuous_wxyz,
    _finite_difference,
    _load_npz,
)

HAND_CONTRACT = "dual_wrist_yaw_link_SE3_plus_g1_brainco_6DoF_per_hand"
SAMPLE_GROUPS = ("palm",) + FINGER_NAMES
DELTA_DIM = 12
QPOS_SLICE = {"left": slice(0, 6), "right": slice(6, 12)}
METHOD_NAME = "g1_brainco_palm_refine+neighborhood_laplacian"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--human-mask-dir", type=Path, default=None)
    parser.add_argument("--contact-anchors", type=Path, required=True)
    parser.add_argument("--input-reference", type=Path, required=True)
    parser.add_argument("--input-targets", type=Path, required=True)
    parser.add_argument("--input-hand", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--output-reference", type=Path, required=True)
    parser.add_argument("--output-targets", type=Path, required=True)
    parser.add_argument("--output-hand", type=Path, required=True)
    parser.add_argument("--meta-json", type=Path, required=True)
    parser.add_argument("--output-wrist-candidate-left", type=Path, default=None)
    parser.add_argument("--output-wrist-candidate-right", type=Path, default=None)
    parser.add_argument("--mesh-samples-per-geom", type=int, default=200)
    parser.add_argument("--wrist-translation-bound-m", type=float, default=0.015)
    parser.add_argument("--wrist-rotation-bound-deg", type=float, default=12.0)
    parser.add_argument("--joint-delta-bound-rad", type=float, default=0.28)
    parser.add_argument("--wrist-anchor-weight", type=float, default=0.10)
    parser.add_argument("--joint-anchor-weight", type=float, default=0.018)
    parser.add_argument("--temporal-wrist-weight", type=float, default=12.0)
    parser.add_argument("--temporal-joint-weight", type=float, default=2.0)
    parser.add_argument("--limit-weight", type=float, default=40.0)
    parser.add_argument(
        "--laplacian-weight",
        type=float,
        default=10.0,
        help="Holosoma per-vertex weight on ||Δ(P_r ∪ O) - Δ(P_h ∪ O)||",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.2,
        help="Holosoma SQP trust region ||dx|| per inner iteration",
    )
    parser.add_argument(
        "--sqp-iters-init",
        type=int,
        default=20,
        help="SQP iterations on the first contact frame (Holosoma uses 50)",
    )
    parser.add_argument(
        "--uniform-weight",
        dest="uniform_weight",
        action="store_true",
        default=True,
        help="Holosoma uniform Laplacian weights (default)",
    )
    parser.add_argument(
        "--distance-weight",
        dest="uniform_weight",
        action="store_false",
        help="use inverse-distance Laplacian weights",
    )
    parser.add_argument(
        "--penetration-weight",
        type=float,
        default=30.0,
        help="unused; anti-penetration is a post-retarget mesh pull-out",
    )
    parser.add_argument("--penetration-thickness-m", type=float, default=0.018)
    parser.add_argument("--penetration-eps-m", type=float, default=0.001)
    parser.add_argument(
        "--pullout-iters",
        type=int,
        default=8,
        help="post-retarget iterations that pull the wrist out of interior "
        "neighborhood points; 0 skips pull-out and keeps the SQP pose",
    )
    parser.add_argument("--min-object-cloud-points", type=int, default=0)
    parser.add_argument(
        "--laplacian-object-cloud-max-points",
        type=int,
        default=512,
        help="cap on neighborhood O vertices in the SQP Delaunay/Laplacian graph; "
        "pull-out keeps every stored sample from contact_anchors.npz",
    )
    parser.add_argument(
        "--adaptive-wrist-d0-m",
        type=float,
        default=0.025,
        help="widen the wrist translation box when init palm-to-object >= this",
    )
    parser.add_argument(
        "--adaptive-wrist-bound-max-m",
        type=float,
        default=0.040,
        help="cap on the per-frame wrist translation box after adaptive widening",
    )
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument(
        "--max-nfev",
        type=int,
        default=10,
        help="SQP iterations per later contact frame (Holosoma uses 10)",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rgb-shape", default="1080,1920")
    return parser.parse_args()


def adaptive_wrist_translation_bound(
    d0_m: float,
    *,
    default_m: float = 0.015,
    trigger_m: float = 0.025,
    max_m: float = 0.040,
) -> float:
    """Widen the wrist translation box only when the init palm is far from O."""
    default_m = float(default_m)
    if not np.isfinite(d0_m) or float(d0_m) < float(trigger_m):
        return default_m
    return float(np.clip(float(d0_m), default_m, float(max_m)))


def packed_object_points(
    points: np.ndarray | None,
    count: int | None,
    min_points: int = 0,
) -> np.ndarray:
    if points is None:
        return np.zeros((0, 3), dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = int(count) if count is not None else len(pts)
    n = max(0, min(n, len(pts)))
    pts = pts[:n]
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if len(pts) < int(min_points):
        return np.zeros((0, 3), dtype=np.float64)
    return pts


def cap_object_points(points: np.ndarray, max_points: int) -> np.ndarray:
    """Keep a stable prefix of packed neighborhood samples for the Laplacian graph."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return pts
    cap = max(1, int(max_points))
    if len(pts) <= cap:
        return pts
    return pts[:cap]


def estimate_outward_normals(
    points: np.ndarray, center: np.ndarray, k: int = 8
) -> np.ndarray:
    """PCA normals on the hand sample cloud, oriented away from ``center``."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    origin = np.asarray(center, dtype=np.float64).reshape(3)
    n = len(pts)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    away = pts - origin
    fallback = away / np.maximum(np.linalg.norm(away, axis=1, keepdims=True), 1e-8)
    if n == 1:
        return fallback
    kk = min(max(3, int(k)), n)
    _, idx = cKDTree(pts).query(pts, k=kk, workers=1)
    idx = np.atleast_2d(np.asarray(idx, dtype=np.int64))
    normals = np.zeros((n, 3), dtype=np.float64)
    for i, ids in enumerate(idx):
        local = pts[np.atleast_1d(ids)]
        if len(local) < 3:
            normals[i] = fallback[i]
            continue
        centered = local - local.mean(axis=0)
        cov = centered.T @ centered
        _evals, vecs = np.linalg.eigh(cov)
        normal = vecs[:, 0]
        if float(np.dot(normal, away[i])) < 0.0:
            normal = -normal
        norm = float(np.linalg.norm(normal))
        normals[i] = normal / norm if norm > 1e-8 else fallback[i]
    return normals


def nearest_indices(source_time: np.ndarray, query_time: np.ndarray) -> np.ndarray:
    source = np.asarray(source_time, dtype=np.float64)
    query = np.asarray(query_time, dtype=np.float64)
    right = np.searchsorted(source, query, side="left")
    right = np.clip(right, 0, max(len(source) - 1, 0))
    left = np.clip(right - 1, 0, max(len(source) - 1, 0))
    if len(source) == 0:
        return np.zeros(len(query), dtype=np.int64)
    use_right = np.abs(source[right] - query) < np.abs(query - source[left])
    return np.where(use_right, right, left).astype(np.int64)


def resample_binary(
    flags: np.ndarray, source_time: np.ndarray, query_time: np.ndarray
) -> np.ndarray:
    if len(flags) == 0:
        return np.zeros(len(query_time), dtype=np.int8)
    return np.asarray(flags, dtype=np.int8)[nearest_indices(source_time, query_time)]


def consecutive_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(flags, dtype=np.int8)
    runs: list[tuple[int, int]] = []
    start = None
    for index, flag in enumerate(values):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def smooth_delta_series(deltas: np.ndarray, window: int) -> np.ndarray:
    """Low-pass wrist/joint offsets after the per-frame solve."""
    values = np.asarray(deltas, dtype=np.float64)
    size = int(window)
    if size < 3 or len(values) < size:
        return values.copy()
    if size % 2 == 0:
        size += 1
    poly = 2 if size > 3 else 1
    out = values.copy()
    out[:, :3] = savgol_filter(values[:, :3], size, poly, axis=0, mode="interp")
    out[:, 3:6] = savgol_filter(values[:, 3:6], size, poly, axis=0, mode="interp")
    out[:, 6:] = savgol_filter(values[:, 6:], size, poly, axis=0, mode="interp")
    return out


def interpolate_deltas(contact: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """Lerp non-contact offsets between neighbouring contact endpoints."""
    flags = np.asarray(contact, dtype=np.int8).reshape(-1)
    values = np.asarray(deltas, dtype=np.float64)
    filled = np.zeros_like(values)
    runs = consecutive_runs(flags)
    if not runs:
        return filled
    for start, end in runs:
        filled[start : end + 1] = values[start : end + 1]
    zero = np.zeros(values.shape[1], dtype=np.float64)
    first_start = runs[0][0]
    if first_start > 0:
        head = values[first_start]
        for frame in range(first_start):
            alpha = frame / float(first_start)
            filled[frame] = _blend_delta(zero, head, alpha)
    last_end = runs[-1][1]
    last_index = len(flags) - 1
    if last_end < last_index:
        tail = values[last_end]
        span = last_index - last_end
        for frame in range(last_end + 1, len(flags)):
            alpha = (frame - last_end) / float(span)
            filled[frame] = _blend_delta(tail, zero, alpha)
    for (_prev_start, prev_end), (next_start, _next_end) in zip(runs, runs[1:]):
        if next_start <= prev_end + 1:
            continue
        start_delta = values[prev_end]
        end_delta = values[next_start]
        span = next_start - prev_end
        for frame in range(prev_end + 1, next_start):
            alpha = (frame - prev_end) / float(span)
            filled[frame] = _blend_delta(start_delta, end_delta, alpha)
    return filled


def _blend_delta(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    out = np.empty_like(start)
    out[:3] = (1.0 - alpha) * start[:3] + alpha * end[:3]
    out[3:6] = _lerp_rotvec(start[3:6], end[3:6], alpha)
    out[6:] = (1.0 - alpha) * start[6:] + alpha * end[6:]
    return out


def _lerp_rotvec(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    start_rot = Rotation.from_rotvec(np.asarray(start, dtype=np.float64))
    end_rot = Rotation.from_rotvec(np.asarray(end, dtype=np.float64))
    delta = (start_rot.inv() * end_rot).as_rotvec()
    return (start_rot * Rotation.from_rotvec(float(alpha) * delta)).as_rotvec()


def apply_wrist_delta(
    position: np.ndarray, rotation: np.ndarray, delta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(delta, dtype=np.float64)
    return (
        np.asarray(position, dtype=np.float64) + rotation @ value[:3],
        np.asarray(rotation, dtype=np.float64)
        @ Rotation.from_rotvec(value[3:6]).as_matrix(),
    )


def _interp_camera(
    cam_t: np.ndarray, cam_r: np.ndarray, cam_x: np.ndarray, query: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    cam_time = np.asarray(cam_t, dtype=np.float64)
    query_time = np.asarray(query, dtype=np.float64)
    rotation = Slerp(cam_time, Rotation.from_matrix(cam_r))(
        np.clip(query_time, cam_time[0], cam_time[-1])
    ).as_matrix()
    translation = np.column_stack(
        [
            np.interp(query_time, cam_time, np.asarray(cam_x, dtype=np.float64)[:, axis])
            for axis in range(3)
        ]
    )
    return rotation, translation


def mano_landmarks_world(
    joints_optical: np.ndarray,
    camera_rotation: np.ndarray,
    camera_translation: np.ndarray,
    r_optical_to_link: np.ndarray,
) -> np.ndarray:
    optical = np.asarray(joints_optical, dtype=np.float64).reshape(-1, 3)
    world = to_world(optical, camera_rotation, camera_translation, r_optical_to_link)
    return world[np.asarray(MANO_HAND_KEYPOINT_INDICES, dtype=np.int64)]


class BraincoMeshSampler:
    """Sample BrainCo visual meshes and 16 FK landmarks in the wrist frame."""

    def __init__(self, model: mujoco.MjModel, side: str, samples_per_geom: int, seed: int) -> None:
        if side not in ("left", "right"):
            raise ValueError(side)
        self.side = side
        self.model = model
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = 0.0
        self.data.qpos[2] = G1_STANDING_HEIGHT
        self.data.qpos[3] = 1.0
        self.qpos_writer = brainco_qpos_writer(model)
        self.wrist_body = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, WRIST_BODIES[side])
        )
        if self.wrist_body < 0:
            raise KeyError(WRIST_BODIES[side])
        self.hand_names = (
            BRAINCO_LEFT_HAND_JOINT_NAMES
            if side == "left"
            else BRAINCO_RIGHT_HAND_JOINT_NAMES
        )
        self.hand_ids = np.asarray(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.hand_names
            ],
            dtype=int,
        )
        if np.any(self.hand_ids < 0):
            raise KeyError(self.hand_names)
        prefix = "left_" if side == "left" else "right_"
        dependent_names = (
            f"{prefix}thumb_distal_joint",
            f"{prefix}index_distal_joint",
            f"{prefix}middle_distal_joint",
            f"{prefix}ring_distal_joint",
            f"{prefix}pinky_distal_joint",
        )
        self.dependent_ids = np.asarray(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in dependent_names
            ],
            dtype=int,
        )
        if np.any(self.dependent_ids < 0):
            raise KeyError(dependent_names)
        tip_names = (
            BRAINCO_LEFT_HAND_TIP_BODIES
            if side == "left"
            else BRAINCO_RIGHT_HAND_TIP_BODIES
        )
        self.tip_body_ids = np.asarray(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in tip_names
            ],
            dtype=int,
        )
        if np.any(self.tip_body_ids < 0):
            raise KeyError(tip_names)
        self.lower = model.jnt_range[self.hand_ids, 0].copy()
        self.upper = model.jnt_range[self.hand_ids, 1].copy()
        rng = np.random.default_rng(seed)
        self.geom_ids: list[tuple[int, str]] = []
        self.geom_vertices: list[tuple[int, str, np.ndarray]] = []
        palm_mesh = BRAINCO_PALM_GEOM_MESH_NAMES[0 if side == "left" else 1]
        self.palm_geom_id = _visual_palm_geom(model, palm_mesh)
        for geom_id in range(model.ngeom):
            if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue
            if int(model.geom_group[geom_id]) != 2:
                continue
            body_id = int(model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name is None or not str(body_name).startswith(prefix):
                continue
            rest = str(body_name)[len(prefix) :].lower()
            mesh_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[geom_id])
            )
            if geom_id == self.palm_geom_id or (
                mesh_name is not None and "palm" in str(mesh_name).lower()
            ):
                group = "palm"
            else:
                group = next(
                    (name for name in FINGER_NAMES if rest.startswith(name)),
                    None,
                )
            if group is None:
                continue
            self.geom_ids.append((geom_id, group))
            # Dense visual-mesh vertices are the post-retarget collision surface.
            # SQP still uses only 16 landmarks; pull-out uses these samples.
            mesh_id = int(model.geom_dataid[geom_id])
            start = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            vertices = np.asarray(
                model.mesh_vert[start : start + count], dtype=np.float64
            )
            keep = min(max(16, int(samples_per_geom)), len(vertices))
            if keep < len(vertices):
                indices = np.sort(rng.choice(len(vertices), size=keep, replace=False))
                vertices = vertices[indices]
            self.geom_vertices.append((geom_id, group, vertices))
        groups = {group for _, group in self.geom_ids}
        if "palm" not in groups:
            raise RuntimeError(f"no {side} BrainCo visual palm mesh found")
        if not any(group in FINGER_NAMES for group in groups):
            raise RuntimeError(f"no {side} BrainCo visual finger meshes found")

    def _forward(self, q6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.clip(np.asarray(q6, dtype=np.float64).reshape(-1), self.lower, self.upper)
        action = np.zeros(len(BRAINCO_ACTION_JOINT_NAMES), dtype=np.float64)
        offset = 14 if self.side == "left" else 20
        action[offset : offset + 6] = q
        self.data.qpos[:] = 0.0
        self.data.qpos[2] = G1_STANDING_HEIGHT
        self.data.qpos[3] = 1.0
        self.qpos_writer(self.data.qpos, action)
        mujoco.mj_forward(self.model, self.data)
        wrist_position = np.asarray(self.data.xpos[self.wrist_body], dtype=np.float64)
        wrist_rotation = np.asarray(
            self.data.xmat[self.wrist_body], dtype=np.float64
        ).reshape(3, 3)
        return wrist_position, wrist_rotation

    def _collect_landmarks_world(self) -> np.ndarray:
        active = np.asarray(self.data.xanchor[self.hand_ids], dtype=np.float64)
        dependent = np.asarray(self.data.xanchor[self.dependent_ids], dtype=np.float64)
        tips = np.asarray(self.data.xpos[self.tip_body_ids], dtype=np.float64)
        points = np.empty((16, 3), dtype=np.float64)
        points[0] = active[0]
        points[1] = active[1]
        points[2] = dependent[0]
        points[3] = tips[0]
        for finger in range(4):
            out = 4 + 3 * finger
            points[out] = active[2 + finger]
            points[out + 1] = dependent[1 + finger]
            points[out + 2] = tips[1 + finger]
        return points

    def forward_local(
        self, q6: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """16 landmarks, geom-origin proxies, radial normals, palm (wrist frame)."""
        wrist_position, wrist_rotation = self._forward(q6)
        landmarks = (self._collect_landmarks_world() - wrist_position) @ wrist_rotation
        proxies = np.empty((len(self.geom_ids), 3), dtype=np.float64)
        for index, (geom_id, _group) in enumerate(self.geom_ids):
            geom_position = np.asarray(self.data.geom_xpos[geom_id], dtype=np.float64)
            proxies[index] = (geom_position - wrist_position) @ wrist_rotation
        palm_world = np.asarray(self.data.geom_xpos[self.palm_geom_id], dtype=np.float64)
        palm_center = (palm_world - wrist_position) @ wrist_rotation
        cloud = (
            np.concatenate([landmarks, proxies], axis=0)
            if len(proxies)
            else landmarks
        )
        away = cloud - palm_center.reshape(1, 3)
        normals = away / np.maximum(np.linalg.norm(away, axis=1, keepdims=True), 1e-8)
        return landmarks, cloud, normals, palm_center

    def local_state(
        self, q6: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Finger/palm samples, palm center, and palm z-axis, all in the wrist frame."""
        wrist_position, wrist_rotation = self._forward(q6)
        grouped: dict[str, list[np.ndarray]] = {name: [] for name in SAMPLE_GROUPS}
        for geom_id, group, vertices in self.geom_vertices:
            geom_position = np.asarray(self.data.geom_xpos[geom_id], dtype=np.float64)
            geom_rotation = np.asarray(
                self.data.geom_xmat[geom_id], dtype=np.float64
            ).reshape(3, 3)
            world = vertices @ geom_rotation.T + geom_position
            local = (world - wrist_position) @ wrist_rotation
            grouped[group].append(local)
        samples = {
            group: np.concatenate(parts, axis=0)
            for group, parts in grouped.items()
            if parts
        }
        palm_world = np.asarray(self.data.geom_xpos[self.palm_geom_id], dtype=np.float64)
        palm_rot = np.asarray(
            self.data.geom_xmat[self.palm_geom_id], dtype=np.float64
        ).reshape(3, 3)
        palm_center_local = (palm_world - wrist_position) @ wrist_rotation
        palm_normal_local = wrist_rotation.T @ palm_rot[:, 2]
        return samples, palm_center_local, palm_normal_local

    def local_samples(self, q6: np.ndarray) -> dict[str, np.ndarray]:
        samples, _, _ = self.local_state(q6)
        return samples

    def local_kinematics(
        self, q6: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Dense samples, outward normals, 16 landmarks, palm center, palm z."""
        samples, palm_center, palm_normal = self.local_state(q6)
        wrist_position = np.asarray(self.data.xpos[self.wrist_body], dtype=np.float64)
        wrist_rotation = np.asarray(
            self.data.xmat[self.wrist_body], dtype=np.float64
        ).reshape(3, 3)
        landmarks = (self._collect_landmarks_world() - wrist_position) @ wrist_rotation
        parts = [samples[name] for name in SAMPLE_GROUPS if name in samples]
        cloud = (
            np.concatenate(parts, axis=0)
            if parts
            else np.zeros((0, 3), dtype=np.float64)
        )
        # Palm geom origin sits on the palmar face.  Orienting PCA away from
        # that point flips palmar-face normals inward, so -n pushes the wrist
        # through the object and parks the cloud on the dorsum.  The sample
        # centroid is inside the solid.
        origin = cloud.mean(axis=0) if len(cloud) else palm_center
        normals = estimate_outward_normals(cloud, origin)
        return cloud, normals, landmarks, palm_center, palm_normal


def interaction_vertices(
    hand_landmarks: np.ndarray, object_points: np.ndarray
) -> np.ndarray:
    hand = np.asarray(hand_landmarks, dtype=np.float64).reshape(-1, 3)
    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    if len(obj) == 0:
        return hand
    return np.concatenate([hand, obj], axis=0)


JOINT_FD_EPS = 1e-4
WRIST_FD_EPS = 1e-4


def _transform_local(
    points_local: np.ndarray, position: np.ndarray, rotation: np.ndarray
) -> np.ndarray:
    pts = np.asarray(points_local, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return pts @ np.asarray(rotation, dtype=np.float64).T + np.asarray(
        position, dtype=np.float64
    )


def trust_region_step(
    jacobian: np.ndarray,
    residual: np.ndarray,
    *,
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    step_size: float,
    damping: float = 1e-4,
) -> np.ndarray:
    """Holosoma-style linearized step: min ||J dx + r||^2 with box + ||dx|| trust."""
    n = int(len(x))
    hess = jacobian.T @ jacobian + float(damping) * np.eye(n)
    grad = jacobian.T @ residual
    try:
        delta = np.linalg.solve(hess, -grad)
    except np.linalg.LinAlgError:
        delta = np.linalg.lstsq(hess, -grad, rcond=None)[0]
    delta = np.clip(delta, lower - x, upper - x)
    norm = float(np.linalg.norm(delta))
    if norm > float(step_size) and norm > 0.0:
        delta *= float(step_size) / norm
    return np.asarray(delta, dtype=np.float64)


def assemble_sqp_system(
    x: np.ndarray,
    *,
    q_base: np.ndarray,
    wrist_position: np.ndarray,
    wrist_rotation: np.ndarray,
    sampler: BraincoMeshSampler,
    object_points: np.ndarray,
    lap_matrix: np.ndarray,
    target_laplacian: np.ndarray,
    previous: np.ndarray,
    translation_bound: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearize Laplacian matching + anchors/smoothness at ``x``.

    Matches Holosoma ``solve_single_iteration``: ``L`` and object vertices are
    frozen, robot keypoints get a 12-DoF Jacobian, object rows of ``J_V`` are 0.
    Neighborhood anti-penetration is applied after this solve, not here.
    """
    q = np.clip(q_base + x[6:], sampler.lower, sampler.upper)
    landmarks_local, _cloud_local, _normals_local, _palm = sampler.forward_local(q)
    position, rotation = apply_wrist_delta(wrist_position, wrist_rotation, x[:6])
    robot_landmarks = _transform_local(landmarks_local, position, rotation)
    vertices = interaction_vertices(robot_landmarks, object_points)
    n_hand = 16
    n_vertex = len(vertices)
    if vertices.shape != target_laplacian.shape:
        raise ValueError(
            f"Laplacian vertex count mismatch: {vertices.shape} vs {target_laplacian.shape}"
        )

    # J_V: d vec(vertices)/dx.  Object samples are frozen this frame.
    jacobian_vertices = np.zeros((3 * n_vertex, DELTA_DIM), dtype=np.float64)
    base_hand = robot_landmarks.reshape(-1)
    for index in range(n_hand):
        jacobian_vertices[3 * index : 3 * index + 3, :3] = wrist_rotation
    for axis in range(3):
        x_pert = x.copy()
        x_pert[3 + axis] += WRIST_FD_EPS
        pos_p, rot_p = apply_wrist_delta(wrist_position, wrist_rotation, x_pert[:6])
        hand_p = _transform_local(landmarks_local, pos_p, rot_p).reshape(-1)
        jacobian_vertices[: 3 * n_hand, 3 + axis] = (hand_p - base_hand) / WRIST_FD_EPS
    for joint in range(6):
        q_pert = q.copy()
        q_pert[joint] = np.clip(
            q_pert[joint] + JOINT_FD_EPS, sampler.lower[joint], sampler.upper[joint]
        )
        landmarks_pert, _c, _n, _p = sampler.forward_local(q_pert)
        hand_p = _transform_local(landmarks_pert, position, rotation).reshape(-1)
        jacobian_vertices[: 3 * n_hand, 6 + joint] = (hand_p - base_hand) / JOINT_FD_EPS

    kron = np.kron(lap_matrix, np.eye(3, dtype=np.float64))
    jacobian_lap = kron @ jacobian_vertices
    lap_vec = (lap_matrix @ vertices).reshape(-1)
    target_vec = np.asarray(target_laplacian, dtype=np.float64).reshape(-1)
    sqrt_w = math.sqrt(float(args.laplacian_weight)) * np.ones(3 * n_vertex)
    residual_lap = sqrt_w * (lap_vec - target_vec)
    jacobian_lap = sqrt_w[:, None] * jacobian_lap

    rotation_bound = np.radians(float(args.wrist_rotation_bound_deg))
    residual_blocks = [
        residual_lap,
        args.wrist_anchor_weight * x[:3],
        args.wrist_anchor_weight * x[3:6],
        args.joint_anchor_weight * x[6:],
        args.temporal_wrist_weight * (x[:6] - previous[:6]),
        args.temporal_joint_weight * (x[6:] - previous[6:]),
        np.asarray(
            [
                float(args.limit_weight)
                * max(float(np.linalg.norm(x[:3])) - float(translation_bound), 0.0),
                float(args.limit_weight)
                * max(float(np.linalg.norm(x[3:6])) - rotation_bound, 0.0),
            ],
            dtype=np.float64,
        ),
    ]
    jacobian_blocks = [
        jacobian_lap,
        args.wrist_anchor_weight * np.eye(3, DELTA_DIM, 0),
        args.wrist_anchor_weight * np.eye(3, DELTA_DIM, 3),
        args.joint_anchor_weight * np.eye(6, DELTA_DIM, 6),
        args.temporal_wrist_weight * np.eye(6, DELTA_DIM, 0),
        args.temporal_joint_weight * np.eye(6, DELTA_DIM, 6),
        np.zeros((2, DELTA_DIM), dtype=np.float64),
    ]
    trans_norm = float(np.linalg.norm(x[:3]))
    rot_norm = float(np.linalg.norm(x[3:6]))
    if trans_norm > float(translation_bound) + 1e-12:
        jacobian_blocks[-1][0, :3] = (
            float(args.limit_weight) * x[:3] / trans_norm
        )
    if rot_norm > rotation_bound + 1e-12:
        jacobian_blocks[-1][1, 3:6] = float(args.limit_weight) * x[3:6] / rot_norm

    residual = np.concatenate(
        [np.asarray(block, dtype=np.float64).reshape(-1) for block in residual_blocks]
    )
    jacobian = np.concatenate(jacobian_blocks, axis=0)
    return residual, jacobian


def solve_sqp_frame(
    x0: np.ndarray,
    *,
    n_iter: int,
    q_base: np.ndarray,
    wrist_position: np.ndarray,
    wrist_rotation: np.ndarray,
    sampler: BraincoMeshSampler,
    object_points: np.ndarray,
    lap_matrix: np.ndarray,
    target_laplacian: np.ndarray,
    previous: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    translation_bound: float,
    args: argparse.Namespace,
) -> np.ndarray:
    """Holosoma ``iterate``: up to ``n_iter`` linearized trust-region steps."""
    x = np.clip(np.asarray(x0, dtype=np.float64), lower, upper)
    last_cost = np.inf
    for _ in range(max(1, int(n_iter))):
        residual, jacobian = assemble_sqp_system(
            x,
            q_base=q_base,
            wrist_position=wrist_position,
            wrist_rotation=wrist_rotation,
            sampler=sampler,
            object_points=object_points,
            lap_matrix=lap_matrix,
            target_laplacian=target_laplacian,
            previous=previous,
            translation_bound=translation_bound,
            args=args,
        )
        cost = float(residual @ residual)
        if np.isclose(cost, last_cost):
            break
        last_cost = cost
        delta = trust_region_step(
            jacobian,
            residual,
            x=x,
            lower=lower,
            upper=upper,
            step_size=float(args.step_size),
        )
        if float(np.linalg.norm(delta)) < 2e-4:
            x = np.clip(x + delta, lower, upper)
            break
        x = np.clip(x + delta, lower, upper)
    return x


def apply_wrist_local_translation(
    x: np.ndarray,
    world_push: np.ndarray,
    wrist_rotation: np.ndarray,
    translation_bound: float,
) -> np.ndarray:
    """Add a world-frame wrist translation, stored in the original wrist frame."""
    updated = np.asarray(x, dtype=np.float64).copy()
    dx_local = np.asarray(wrist_rotation, dtype=np.float64).T @ np.asarray(
        world_push, dtype=np.float64
    ).reshape(3)
    bound = float(translation_bound)
    updated[:3] = np.clip(updated[:3] + dx_local, -bound, bound)
    return updated


def pull_hand_out_of_neighborhood(
    x: np.ndarray,
    *,
    q_base: np.ndarray,
    wrist_position: np.ndarray,
    wrist_rotation: np.ndarray,
    sampler: BraincoMeshSampler,
    object_points: np.ndarray,
    translation_bound: float,
    args: argparse.Namespace,
    human_landmarks: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Translate the wrist so interior neighborhood points leave the dense mesh.

    Collision uses the visual-mesh samples, not the 16 Laplacian landmarks.
    Joint offsets stay as retargeted; only ``x[:3]`` moves, clipped to the same
    adaptive wrist box used by SQP.  The translation aims at the recorded
    human-hand landmark centroid.  If that target is missing, fall back to
    backing off the object centroid.
    """
    updated = np.asarray(x, dtype=np.float64).copy()
    stats = {
        "inside_before": 0,
        "inside_after": 0,
        "max_depth_before_m": 0.0,
        "push_m": 0.0,
        "iters": 0,
        "clipped": False,
    }
    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    if len(obj) == 0:
        return updated, stats
    human = (
        None
        if human_landmarks is None
        else np.asarray(human_landmarks, dtype=np.float64).reshape(-1, 3)
    )

    def _query(state: np.ndarray) -> tuple[np.ndarray, int, float]:
        q = np.clip(q_base + state[6:], sampler.lower, sampler.upper)
        cloud_local, normals_local, landmarks_local, palm_local, palm_n_local = (
            sampler.local_kinematics(q)
        )
        position, rotation = apply_wrist_delta(
            wrist_position, wrist_rotation, state[:6]
        )
        cloud_world = _transform_local(cloud_local, position, rotation)
        normals_world = normals_local @ rotation.T
        robot_landmarks = _transform_local(landmarks_local, position, rotation)
        axis = None
        if human is not None:
            finite = np.isfinite(human).all(axis=1)
            if np.any(finite) and len(robot_landmarks) > 0:
                axis = retract_axis_toward_target(
                    robot_landmarks.mean(axis=0),
                    human[finite].mean(axis=0),
                )
        if axis is None:
            palm_world = np.asarray(rotation, dtype=np.float64) @ np.asarray(
                palm_local, dtype=np.float64
            ).reshape(3) + np.asarray(position, dtype=np.float64)
            palm_n_world = np.asarray(palm_n_local, dtype=np.float64).reshape(3) @ (
                np.asarray(rotation, dtype=np.float64).T
            )
            axis = retract_axis_away_from_object(palm_n_world, palm_world, obj)
        return deepest_inside_world_push(
            obj,
            cloud_world,
            normals_world,
            thickness_m=float(args.penetration_thickness_m),
            clearance_m=float(args.penetration_eps_m),
            retract_axis=axis,
        )

    push, n_inside, depth_before = _query(updated)
    stats["inside_before"] = int(n_inside)
    stats["max_depth_before_m"] = float(depth_before)
    if n_inside == 0:
        stats["inside_after"] = 0
        return updated, stats

    moved = 0.0
    clipped = False
    used = 0
    for _ in range(max(1, int(args.pullout_iters))):
        if n_inside == 0:
            break
        used += 1
        candidate = apply_wrist_local_translation(
            updated, push, wrist_rotation, translation_bound
        )
        if np.allclose(candidate[:3], updated[:3], atol=1e-9):
            clipped = True
            break
        moved += float(np.linalg.norm(candidate[:3] - updated[:3]))
        updated = candidate
        push, n_inside, _depth = _query(updated)
        if float(np.linalg.norm(push)) < 1e-9:
            break

    stats["inside_after"] = int(n_inside)
    stats["push_m"] = float(moved)
    stats["iters"] = int(used)
    stats["clipped"] = bool(clipped)
    return updated, stats


def optimize_side(
    *,
    side: str,
    sampler: BraincoMeshSampler,
    qpos: np.ndarray,
    wrist_position: np.ndarray,
    wrist_rotation: np.ndarray,
    contact_50: np.ndarray,
    native_index: np.ndarray,
    human_landmarks_native: np.ndarray,
    object_points: np.ndarray | None,
    object_count: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    frames = len(qpos)
    deltas = np.zeros((frames, DELTA_DIM), dtype=np.float64)
    q_slice = QPOS_SLICE[side]
    bounds_r = np.radians(float(args.wrist_rotation_bound_deg))
    bounds_q = float(args.joint_delta_bound_rad)
    previous = np.zeros(DELTA_DIM, dtype=np.float64)
    solved = 0
    widened = 0
    d0_values: list[float] = []
    first_contact = True
    laplacian_counts: list[int] = []
    pullout_counts: list[int] = []
    translation_bounds = np.full(
        frames, float(args.wrist_translation_bound_m), dtype=np.float64
    )
    for frame in range(frames):
        if not bool(contact_50[frame]):
            continue
        native = int(native_index[frame])
        cloud_pts = packed_object_points(
            None if object_points is None else object_points[native],
            None if object_count is None else int(object_count[native]),
            int(args.min_object_cloud_points),
        )
        lap_pts = cap_object_points(
            cloud_pts, int(args.laplacian_object_cloud_max_points)
        )
        P_h = np.asarray(human_landmarks_native[native], dtype=np.float64).reshape(16, 3)
        source = interaction_vertices(P_h, lap_pts)
        adj_list = build_interaction_adjacency(source)
        lap_matrix = calculate_laplacian_matrix(
            source, adj_list, uniform_weight=bool(args.uniform_weight)
        )
        target_laplacian = calculate_laplacian_coordinates(
            source, adj_list, uniform_weight=bool(args.uniform_weight)
        )
        _, _, _, palm_local = sampler.forward_local(qpos[frame, q_slice])
        palm_init = palm_local @ wrist_rotation[frame].T + wrist_position[frame]
        if len(lap_pts) > 0:
            d0 = float(np.linalg.norm(palm_init - lap_pts.mean(axis=0)))
        else:
            d0 = 0.0
        d0_values.append(d0)
        bound_t = adaptive_wrist_translation_bound(
            d0,
            default_m=float(args.wrist_translation_bound_m),
            trigger_m=float(args.adaptive_wrist_d0_m),
            max_m=float(args.adaptive_wrist_bound_max_m),
        )
        if bound_t > float(args.wrist_translation_bound_m) + 1e-9:
            widened += 1
        translation_bounds[frame] = bound_t
        lower = np.concatenate(
            [
                np.full(3, -bound_t),
                np.full(3, -bounds_r),
                np.full(6, -bounds_q),
            ]
        )
        upper = -lower
        x0 = previous.copy()
        x0[:3] = np.clip(x0[:3], -bound_t, bound_t)
        n_iter = (
            int(args.sqp_iters_init) if first_contact else int(args.max_nfev)
        )
        previous = solve_sqp_frame(
            x0,
            n_iter=n_iter,
            q_base=qpos[frame, q_slice],
            wrist_position=wrist_position[frame],
            wrist_rotation=wrist_rotation[frame],
            sampler=sampler,
            object_points=lap_pts,
            lap_matrix=lap_matrix,
            target_laplacian=target_laplacian,
            previous=previous,
            lower=lower,
            upper=upper,
            translation_bound=bound_t,
            args=args,
        )
        first_contact = False
        deltas[frame] = previous
        solved += 1
        laplacian_counts.append(len(lap_pts))
        pullout_counts.append(len(cloud_pts))
    filled = interpolate_deltas(contact_50, deltas)
    filled = smooth_delta_series(filled, int(args.smooth_window))
    pullout_inside_before = 0
    pullout_inside_after = 0
    pullout_frames = 0
    pullout_clipped = 0
    pullout_push: list[float] = []
    do_pullout = int(args.pullout_iters) > 0
    if do_pullout:
        for frame in range(frames):
            if not bool(contact_50[frame]):
                continue
            native = int(native_index[frame])
            cloud_pts = packed_object_points(
                None if object_points is None else object_points[native],
                None if object_count is None else int(object_count[native]),
                int(args.min_object_cloud_points),
            )
            if len(cloud_pts) == 0:
                continue
            filled[frame], pull_stats = pull_hand_out_of_neighborhood(
                filled[frame],
                q_base=qpos[frame, q_slice],
                wrist_position=wrist_position[frame],
                wrist_rotation=wrist_rotation[frame],
                sampler=sampler,
                object_points=cloud_pts,
                translation_bound=float(translation_bounds[frame]),
                args=args,
                human_landmarks=np.asarray(
                    human_landmarks_native[native], dtype=np.float64
                ).reshape(16, 3),
            )
            pullout_frames += 1
            pullout_inside_before += int(pull_stats["inside_before"])
            pullout_inside_after += int(pull_stats["inside_after"])
            pullout_push.append(float(pull_stats["push_m"]))
            if pull_stats["clipped"]:
                pullout_clipped += 1
    report = {
        "side": side,
        "contact_frames": int(np.count_nonzero(contact_50)),
        "solved_frames": solved,
        "adaptive_wrist_widened_frames": widened,
        "init_palm_object_mean_m": float(np.mean(d0_values)) if d0_values else 0.0,
        "init_palm_object_max_m": float(np.max(d0_values)) if d0_values else 0.0,
        "mean_abs_translation_m": float(np.mean(np.abs(filled[:, :3]))) if frames else 0.0,
        "mean_abs_joint_rad": float(np.mean(np.abs(filled[:, 6:]))) if frames else 0.0,
        "method": METHOD_NAME,
        "solver": "holosoma_sequential_sqp",
        "postprocess": "neighborhood_mesh_pullout" if do_pullout else "none",
        "pullout_frames": pullout_frames,
        "pullout_inside_points_before": pullout_inside_before,
        "pullout_inside_points_after": pullout_inside_after,
        "pullout_clipped_frames": pullout_clipped,
        "pullout_mean_push_m": float(np.mean(pullout_push)) if pullout_push else 0.0,
        "laplacian_object_cloud_max_points": int(args.laplacian_object_cloud_max_points),
        "laplacian_object_points_mean": (
            float(np.mean(laplacian_counts)) if laplacian_counts else 0.0
        ),
        "pullout_object_points_mean": (
            float(np.mean(pullout_counts)) if pullout_counts else 0.0
        ),
    }
    return filled, report


def contact_frame_indices(contact: np.ndarray) -> np.ndarray:
    flags = np.asarray(contact, dtype=np.int8).reshape(-1)
    hits = np.flatnonzero(flags)
    if len(hits) == 0:
        return np.asarray([0], dtype=np.int64)
    return np.arange(int(hits[0]), int(hits[-1]) + 1, dtype=np.int64)


def write_wrist_candidate(
    path: Path,
    *,
    timestamps: np.ndarray,
    contact: np.ndarray,
    base_position: np.ndarray,
    base_rotation: np.ndarray,
    causal_position: np.ndarray,
    causal_rotation: np.ndarray,
    side: str,
) -> None:
    flags = np.asarray(contact, dtype=np.int8).reshape(-1)
    has_contact = bool(np.any(flags))
    payload = {
        "schema_version": np.asarray(1, dtype=np.int32),
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "frame_indices": contact_frame_indices(flags),
        "base_position": np.asarray(base_position, dtype=np.float64),
        "base_rotation": np.asarray(base_rotation, dtype=np.float64),
        "causal_position": np.asarray(causal_position, dtype=np.float64),
        "causal_rotation": np.asarray(causal_rotation, dtype=np.float64),
        "status": np.asarray(
            "contact_pointcloud_2finger" if has_contact else "no_contact_detected"
        ),
        "reason": np.asarray(METHOD_NAME if has_contact else "no_contact_detected"),
        "optimization_space": np.asarray(f"{side}_wrist_yaw_link_SE3"),
        "not_joint_yaw_offset": np.asarray(True),
        "method": np.asarray(METHOD_NAME),
    }
    _atomic_npz(path, payload)


def main() -> None:
    args = parse_args()
    session = args.session_dir.expanduser().resolve()
    cal = session / "calibrations"
    r_optical_to_link = np.asarray(np.load(cal / "R_optical_to_link.npy"), dtype=np.float64)
    reference = _load_npz(args.input_reference)
    targets = _load_npz(args.input_targets)
    hand = _load_npz(args.input_hand)
    with np.load(args.contact_anchors, allow_pickle=False) as archive:
        anchors_npz = {name: np.asarray(archive[name]) for name in archive.files}
    body_names = [str(value) for value in reference["body_names"].reshape(-1).tolist()]
    fps = float(np.asarray(reference["fps"]).reshape(()))
    target_time = np.asarray(targets["timestamps"], dtype=np.float64)
    frames = len(target_time)
    qpos_key = "qpos" if "qpos" in hand else "hand_qpos"
    qpos = np.asarray(hand[qpos_key], dtype=np.float64).copy()
    if qpos.shape != (frames, 12):
        raise ValueError(f"expected hand qpos {(frames, 12)}, got {qpos.shape}")

    native_time = np.asarray(anchors_npz["timestamps"], dtype=np.float64)
    native_index = nearest_indices(native_time, target_time)
    with h5py.File(session / "annotation.hdf5", "r") as h5:
        cam_t = np.asarray(h5["cam-pose/timestamps"], dtype=np.float64)
        cam_r = np.asarray(h5["cam-pose/rotations"], dtype=np.float64)
        cam_x = np.asarray(h5["cam-pose/translations"], dtype=np.float64)
        hand_abs = np.asarray(h5["hand-pose/timestamps"], dtype=np.float64)
        joints = {
            side: np.asarray(h5[f"hand-pose/{side}_joints"], dtype=np.float64)
            for side in ("left", "right")
        }
    native_abs = hand_abs[0] + native_time
    camera_rotation_native, camera_translation_native = _interp_camera(
        cam_t, cam_r, cam_x, native_abs
    )
    human_landmarks = {}
    for side in ("left", "right"):
        n_native = min(len(native_time), len(joints[side]))
        landmarks = np.full((len(native_time), 16, 3), np.nan, dtype=np.float64)
        for index in range(n_native):
            landmarks[index] = mano_landmarks_world(
                joints[side][index],
                camera_rotation_native[index],
                camera_translation_native[index],
                r_optical_to_link,
            )
        human_landmarks[side] = landmarks

    model = _load_model(args.mjcf)
    output_reference = {name: value.copy() for name, value in reference.items()}
    output_targets = {name: value.copy() for name, value in targets.items()}
    qpos_out = qpos.copy()
    side_reports: dict[str, Any] = {}
    contact_50 = {}
    candidate_paths = {
        "left": args.output_wrist_candidate_left,
        "right": args.output_wrist_candidate_right,
    }
    for side in ("left", "right"):
        native_contact = np.asarray(anchors_npz[f"{side}_contact"], dtype=np.int8)
        contact_50[side] = resample_binary(native_contact, native_time, target_time)
        body_index = body_names.index(WRIST_BODIES[side])
        wrist_position = np.asarray(
            reference["body_pos_w"][:, body_index], dtype=np.float64
        )
        wrist_rotation = Rotation.from_quat(
            np.asarray(reference["body_quat_w"][:, body_index], dtype=np.float64),
            scalar_first=True,
        ).as_matrix()
        new_position = wrist_position.copy()
        new_rotation = wrist_rotation.copy()
        if int(np.count_nonzero(contact_50[side])) == 0:
            side_reports[side] = {
                "side": side,
                "contact_frames": 0,
                "solved_frames": 0,
                "skipped": True,
                "method": METHOD_NAME,
            }
        else:
            sampler = BraincoMeshSampler(
                model, side, int(args.mesh_samples_per_geom), int(args.seed)
            )
            filled, report = optimize_side(
                side=side,
                sampler=sampler,
                qpos=qpos,
                wrist_position=wrist_position,
                wrist_rotation=wrist_rotation,
                contact_50=contact_50[side],
                native_index=native_index,
                human_landmarks_native=human_landmarks[side],
                object_points=(
                    np.asarray(anchors_npz[f"{side}_object_points"])
                    if f"{side}_object_points" in anchors_npz
                    else None
                ),
                object_count=(
                    np.asarray(anchors_npz[f"{side}_object_count"])
                    if f"{side}_object_count" in anchors_npz
                    else None
                ),
                args=args,
            )
            q_slice = QPOS_SLICE[side]
            qpos_out[:, q_slice] = np.clip(
                qpos[:, q_slice] + filled[:, 6:], sampler.lower, sampler.upper
            )
            for frame in range(frames):
                new_position[frame], new_rotation[frame] = apply_wrist_delta(
                    wrist_position[frame], wrist_rotation[frame], filled[frame, :6]
                )
            quaternion = _continuous_wxyz(new_rotation)
            output_reference["body_pos_w"][:, body_index] = new_position.astype(
                output_reference["body_pos_w"].dtype
            )
            output_reference["body_quat_w"][:, body_index] = quaternion.astype(
                output_reference["body_quat_w"].dtype
            )
            output_reference["body_lin_vel_w"][:, body_index] = _finite_difference(
                new_position, fps
            ).astype(output_reference["body_lin_vel_w"].dtype)
            output_reference["body_ang_vel_w"][:, body_index] = _angular_velocity_world(
                new_rotation, fps
            ).astype(output_reference["body_ang_vel_w"].dtype)
            output_targets[f"{side}_wrist"] = new_position.astype(np.float32)
            output_targets[f"{side}_wrist_quat_wxyz"] = quaternion.astype(np.float32)
            side_reports[side] = report
        candidate_path = candidate_paths[side]
        if candidate_path is not None:
            write_wrist_candidate(
                candidate_path.expanduser().resolve(),
                timestamps=target_time,
                contact=contact_50[side],
                base_position=wrist_position,
                base_rotation=wrist_rotation,
                causal_position=new_position,
                causal_rotation=new_rotation,
                side=side,
            )

    wrist_indices = [body_names.index(name) for name in WRIST_BODIES.values()]
    non_wrist = np.ones(len(body_names), dtype=bool)
    non_wrist[wrist_indices] = False
    for field in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        if not np.array_equal(reference[field][:, non_wrist], output_reference[field][:, non_wrist]):
            raise RuntimeError(f"non-wrist values changed in {field}")

    qvel = np.gradient(qpos_out, target_time, axis=0, edge_order=1)
    left_contact = contact_50["left"].astype(np.int8)
    right_contact = contact_50["right"].astype(np.int8)
    native_left = np.asarray(anchors_npz["left_contact"], dtype=np.int8)
    native_right = np.asarray(anchors_npz["right_contact"], dtype=np.int8)
    hand_out = dict(hand)
    hand_out["qpos"] = qpos_out.astype(np.float32)
    hand_out["qvel"] = qvel.astype(np.float32)
    hand_out["left_qpos"] = qpos_out[:, :6].astype(np.float32)
    hand_out["right_qpos"] = qpos_out[:, 6:].astype(np.float32)
    hand_out["left_qvel"] = qvel[:, :6].astype(np.float32)
    hand_out["right_qvel"] = qvel[:, 6:].astype(np.float32)
    hand_out["hand_joint_names"] = np.asarray(BRAINCO_HAND_JOINT_NAMES)
    hand_out["left_contact"] = left_contact
    hand_out["right_contact"] = right_contact
    hand_out["contact_50hz"] = np.maximum(left_contact, right_contact).astype(np.int8)
    hand_out["left_contact_30hz"] = native_left
    hand_out["right_contact_30hz"] = native_right
    hand_out["contact_30hz"] = np.maximum(native_left, native_right).astype(np.int8)
    hand_out["native_timestamps"] = native_time
    hand_out["contact_detector"] = np.asarray("source_rgbd_pointcloud_anchors")
    hand_out["hand_contract"] = np.asarray(HAND_CONTRACT)
    hand_out["mimic_rules_json"] = np.asarray(json.dumps(BRAINCO_MIMIC_RULES))
    output_reference["wrist_reference_method"] = np.asarray(METHOD_NAME)
    output_targets["wrist_reference_method"] = output_reference["wrist_reference_method"]

    report = {
        "schema_version": 1,
        "hand_contract": HAND_CONTRACT,
        "method": METHOD_NAME,
        "object_source": str(np.asarray(anchors_npz.get("object_source", "")).reshape(())),
        "sides": side_reports,
        "inputs": {
            "anchors": str(args.contact_anchors.expanduser().resolve()),
            "reference": str(args.input_reference.expanduser().resolve()),
            "hand": str(args.input_hand.expanduser().resolve()),
        },
    }
    _atomic_npz(args.output_reference, output_reference)
    _atomic_npz(args.output_targets, output_targets)
    _atomic_npz(args.output_hand, hand_out)
    _atomic_json(args.meta_json, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
