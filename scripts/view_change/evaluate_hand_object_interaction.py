#!/usr/bin/env python3
"""Evaluate Dex3 wrist/finger candidates against an observed cup surface.

This experiment-only sidecar consumes the candidates produced by
``hand_retarget_experiment.py``. It does not alter the production retargeting
or view-change pipeline. Cup observations come from RGB masks plus registered
metric depth, while hand geometry comes from the complete G1+Dex3 MuJoCo
model so the evaluated frame is the actual ``wrist_yaw_link`` frame.

RGB-D unprojection uses the same ``unproject_masked_optical`` helper as
contact-pointcloud analysis.  ``--device cuda`` keeps that projection on
Torch CUDA; ``--device cpu`` is the NumPy fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.optimize import least_squares
from sklearn.cluster import DBSCAN


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parents[1]
# SparseTrack is vendored through the project resources link.  The old
# sibling-path assumption only worked in the original checkout layout and
# made contact preprocessing fail for every newly discovered session.
SPARSE_SCRIPTS = WORKSPACE / "resources" / "SparseTrack" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SPARSE_SCRIPTS))

from cup_eval_fast import (  # noqa: E402
    evaluate_packed_configuration,
    geoms_to_wrist_local,
    pack_local_samples,
)
from dex3_kinematics_utils import (  # noqa: E402
    DEFAULT_G1_MESHDIR,
    evaluate_mimic_rule,
    joint_mimic_rules,
    load_dex3_model,
)
from egodex_to_dex3_hand_qpos import HAND_JOINT_NAMES  # noqa: E402
from frame_volume import as_mask, open_volume  # noqa: E402


def _contact_projection():
    """Load the shared RGB-D unprojection helpers without importing Torch at module load."""
    from contact_pointcloud_analysis import (  # noqa: WPS433
        TorchDepthProjector,
        resolve_device,
        unproject_masked_optical,
    )

    return TorchDepthProjector, resolve_device, unproject_masked_optical


DEFAULT_BRAINCO_XML = (
    WORKSPACE / "resources/brainco/robots/g1_brainco/g1_mocap_29dof_with_brainco_hands.xml"
)
DEFAULT_BRAINCO_MESHDIR = WORKSPACE / "resources/brainco/robots/g1_brainco"
BRAINCO_HAND_JOINT_NAMES = (
    "left_thumb_metacarpal_joint",
    "left_thumb_proximal_joint",
    "left_index_proximal_joint",
    "left_middle_proximal_joint",
    "left_ring_proximal_joint",
    "left_pinky_proximal_joint",
    "right_thumb_metacarpal_joint",
    "right_thumb_proximal_joint",
    "right_index_proximal_joint",
    "right_middle_proximal_joint",
    "right_ring_proximal_joint",
    "right_pinky_proximal_joint",
)
HAND_MODE_DEX3 = "dex3"
HAND_MODE_BRAINCO = "g1_brainco"


YUP_TO_ZUP = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
GROUP_ORDER = ("palm", "thumb", "index", "middle")
GROUP_COLORS = {
    "palm": "#666666",
    "thumb": "#d62728",
    "index": "#2ca02c",
    "middle": "#ff7f0e",
}


@dataclass(frozen=True)
class CupModel:
    frame: int
    center_xy: tuple[float, float]
    z_reference: float
    radius_reference: float
    radius_slope: float
    z_min: float
    z_max: float
    points: int
    inliers: int
    inlier_fraction: float
    residual_p50_m: float
    residual_p95_m: float
    depth_frame: int
    depth_time_offset_ms: float
    camera_time_offset_ms: float

    def radius_at(self, z: np.ndarray | float) -> np.ndarray:
        return self.radius_reference + self.radius_slope * (
            np.asarray(z, dtype=np.float64) - self.z_reference
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=300)
    parser.add_argument("--end-frame", type=int, default=360)
    parser.add_argument("--keyframe", type=int, default=331)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument(
        "--hand-mode",
        choices=(HAND_MODE_DEX3, HAND_MODE_BRAINCO),
        default=HAND_MODE_DEX3,
    )
    parser.add_argument("--wrist-methods", default="")
    parser.add_argument("--finger-methods", default="")
    parser.add_argument("--contact-threshold-m", type=float, default=0.008)
    parser.add_argument("--penetration-tolerance-m", type=float, default=0.005)
    parser.add_argument("--penetration-ratio-tolerance", type=float, default=0.02)
    parser.add_argument("--dbscan-eps-m", type=float, default=0.012)
    parser.add_argument("--dbscan-min-samples", type=int, default=8)
    parser.add_argument("--cylinder-inlier-m", type=float, default=0.010)
    parser.add_argument("--ransac-iterations", type=int, default=1000)
    parser.add_argument("--mesh-samples-per-geom", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--allow-object-fit-fallback", action="store_true")
    parser.add_argument("--max-object-fallback-run", type=int, default=5)
    parser.add_argument("--object-center-step-m", type=float, default=0.05)
    parser.add_argument("--object-radius-step-m", type=float, default=0.012)
    parser.add_argument("--object-height-step-m", type=float, default=0.04)
    parser.add_argument("--object-temporal-gap-cap", type=int, default=2)
    parser.add_argument(
        "--device",
        default="auto",
        help="RGB-D unprojection device: auto, cuda[:N], or cpu",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip heatmap/keyframe PNGs; JSONL and summary are still written.",
    )
    return parser.parse_args()


def finite_stats(values: np.ndarray | list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not array.size:
        return {key: float("nan") for key in ("mean", "median", "p95", "max")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def comma_list(value: str, available: list[str]) -> list[str]:
    selected = [item.strip() for item in value.split(",") if item.strip()]
    if not selected:
        return available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"unknown methods {unknown}; available={available}")
    return selected


def nearest_index(timestamps: np.ndarray, query: float) -> int:
    """Return nearest timestamp index, choosing the earlier item on a tie."""
    time = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if not time.size:
        raise ValueError("empty timestamp array")
    right = int(np.searchsorted(time, query, side="left"))
    if right <= 0:
        return 0
    if right >= len(time):
        return len(time) - 1
    left = right - 1
    if query - time[left] <= time[right] - query:
        return left
    return right


def circle_from_three(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    p = np.asarray(points, dtype=np.float64)
    if p.shape != (3, 2):
        raise ValueError(p.shape)
    a = 2.0 * np.asarray(
        [p[1] - p[0], p[2] - p[0]], dtype=np.float64
    )
    b = np.asarray(
        [np.dot(p[1], p[1]) - np.dot(p[0], p[0]),
         np.dot(p[2], p[2]) - np.dot(p[0], p[0])],
        dtype=np.float64,
    )
    if abs(float(np.linalg.det(a))) < 1e-10:
        return None
    center = np.linalg.solve(a, b)
    radius = float(np.linalg.norm(p[0] - center))
    if not np.isfinite(radius):
        return None
    return center, radius


def fit_upright_tapered_cylinder(
    points_world: np.ndarray,
    *,
    frame: int,
    depth_frame: int = -1,
    depth_time_offset_ms: float = 0.0,
    camera_time_offset_ms: float = 0.0,
    inlier_threshold_m: float = 0.010,
    ransac_iterations: int = 1000,
    seed: int = 0,
) -> tuple[CupModel, np.ndarray]:
    """Robustly fit an upright tapered cup body, rejecting its handle."""
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 100:
        raise ValueError(f"insufficient cup points: {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("cup points contain non-finite values")

    rng = np.random.default_rng(seed)
    sample_count = min(len(points), 4000)
    sample_ids = rng.choice(len(points), size=sample_count, replace=False)
    xy = points[sample_ids, :2]
    best: tuple[int, float, np.ndarray, float] | None = None
    relaxed = max(inlier_threshold_m, 0.012)
    for _ in range(max(50, int(ransac_iterations))):
        ids = rng.choice(sample_count, size=3, replace=False)
        candidate = circle_from_three(xy[ids])
        if candidate is None:
            continue
        center, radius = candidate
        if not 0.025 <= radius <= 0.10:
            continue
        residual = np.abs(np.linalg.norm(xy - center, axis=1) - radius)
        inliers = int(np.count_nonzero(residual <= relaxed))
        median = float(np.median(np.minimum(residual, relaxed * 2.0)))
        key = (inliers, -median)
        if best is None or key > (best[0], -best[1]):
            best = (inliers, median, center, radius)
    if best is None or best[0] < 50:
        raise ValueError("cup circle RANSAC found no plausible body")

    initial_center = best[2]
    initial_radius = best[3]
    z_reference = float(np.median(points[:, 2]))
    radial_initial = np.linalg.norm(points[:, :2] - initial_center, axis=1)
    prefit = np.abs(radial_initial - initial_radius) <= relaxed * 1.5
    fit_points = points[prefit]
    if len(fit_points) < 100:
        raise ValueError("too few cup body points after handle rejection")

    def residual_function(parameters: np.ndarray, xyz: np.ndarray) -> np.ndarray:
        cx, cy, radius, slope = parameters
        radial = np.linalg.norm(xyz[:, :2] - np.asarray([cx, cy]), axis=1)
        expected = radius + slope * (xyz[:, 2] - z_reference)
        return radial - expected

    bounds_low = np.asarray(
        [points[:, 0].min() - 0.05, points[:, 1].min() - 0.05, 0.020, -0.35]
    )
    bounds_high = np.asarray(
        [points[:, 0].max() + 0.05, points[:, 1].max() + 0.05, 0.110, 0.35]
    )
    result = least_squares(
        residual_function,
        np.asarray([*initial_center, initial_radius, 0.0]),
        args=(fit_points,),
        bounds=(bounds_low, bounds_high),
        loss="soft_l1",
        f_scale=0.004,
        max_nfev=300,
    )
    residual_all = np.abs(residual_function(result.x, points))
    inliers = residual_all <= inlier_threshold_m
    if np.count_nonzero(inliers) < 100:
        raise ValueError("tapered cylinder has too few final inliers")

    final = least_squares(
        residual_function,
        result.x,
        args=(points[inliers],),
        bounds=(bounds_low, bounds_high),
        loss="soft_l1",
        f_scale=0.003,
        max_nfev=300,
    )
    residual_all = np.abs(residual_function(final.x, points))
    inliers = residual_all <= inlier_threshold_m
    radial_body_ids = np.flatnonzero(inliers)
    sorted_ids = radial_body_ids[np.argsort(points[radial_body_ids, 2])]
    sorted_z = points[sorted_ids, 2]
    split_locations = np.flatnonzero(np.diff(sorted_z) > 0.025) + 1
    components = np.split(sorted_ids, split_locations)
    body_ids = max(components, key=len)
    if len(body_ids) < 100:
        raise ValueError("cup axial component has too few points")
    axial_low = float(np.min(points[body_ids, 2])) - 0.003
    axial_high = float(np.max(points[body_ids, 2])) + 0.003
    axial_window = (points[:, 2] >= axial_low) & (points[:, 2] <= axial_high)
    final = least_squares(
        residual_function,
        final.x,
        args=(points[body_ids],),
        bounds=(bounds_low, bounds_high),
        loss="soft_l1",
        f_scale=0.003,
        max_nfev=300,
    )
    residual_all = np.abs(residual_function(final.x, points))
    inliers = (residual_all <= inlier_threshold_m) & axial_window
    body = points[inliers]
    z_min, z_max = np.percentile(body[:, 2], [1.0, 99.0])
    cx, cy, radius, slope = (float(value) for value in final.x)
    end_radii = radius + slope * (np.asarray([z_min, z_max]) - z_reference)
    if not 0.055 <= z_max - z_min <= 0.18 or np.any(end_radii < 0.020):
        raise ValueError(
            f"implausible cup dimensions height={z_max-z_min:.4f}, "
            f"end_radii={end_radii.tolist()}"
        )
    model = CupModel(
        frame=int(frame),
        center_xy=(cx, cy),
        z_reference=z_reference,
        radius_reference=radius,
        radius_slope=slope,
        z_min=float(z_min),
        z_max=float(z_max),
        points=int(len(points)),
        inliers=int(np.count_nonzero(inliers)),
        inlier_fraction=float(np.mean(inliers)),
        residual_p50_m=float(np.percentile(residual_all[inliers], 50)),
        residual_p95_m=float(np.percentile(residual_all[inliers], 95)),
        depth_frame=int(depth_frame),
        depth_time_offset_ms=float(depth_time_offset_ms),
        camera_time_offset_ms=float(camera_time_offset_ms),
    )
    return model, inliers


class CupObservationLoader:
    def __init__(
        self,
        session_dir: Path,
        mask_dir: Path,
        *,
        dbscan_eps_m: float,
        dbscan_min_samples: int,
        device=None,
    ) -> None:
        self.session_dir = session_dir
        self.mask_dir = mask_dir
        self.h5 = h5py.File(session_dir / "annotation.hdf5", "r")
        calibration = session_dir / "calibrations"
        self.rgb_k = np.asarray(np.load(calibration / "rgb_K.npy"), dtype=np.float64)
        self.depth_k = np.asarray(
            np.load(calibration / "depth_K.npy"), dtype=np.float64
        )
        self.optical_to_link = np.asarray(
            np.load(calibration / "R_optical_to_link.npy"), dtype=np.float64
        )
        self.hand_time = np.asarray(
            self.h5["hand-pose/timestamps"], dtype=np.float64
        )
        self.depth_time = np.asarray(self.h5["depth/timestamps"], dtype=np.float64)
        self.camera_time = np.asarray(
            self.h5["cam-pose/timestamps"], dtype=np.float64
        )
        self.depth_frames = np.asarray(self.h5["depth/frames"][:], dtype=np.float32)
        self.camera_rotations = np.asarray(
            self.h5["cam-pose/rotations"][:], dtype=np.float64
        )
        self.camera_translations = np.asarray(
            self.h5["cam-pose/translations"][:], dtype=np.float64
        )
        self.depth_valid = (
            np.asarray(self.h5["depth/valid"][:], dtype=bool)
            if "depth/valid" in self.h5
            else None
        )
        height, width = self.depth_frames.shape[1:]
        self.dbscan_eps_m = float(dbscan_eps_m)
        self.dbscan_min_samples = int(dbscan_min_samples)
        self.device = device
        projector_cls, _resolve_device, unproject = _contact_projection()
        self._unproject_masked_optical = unproject
        self.projector = None
        if device is not None and getattr(device, "type", None) == "cuda":
            self.projector = projector_cls(
                self.depth_k,
                self.rgb_k,
                (int(height), int(width)),
                device,
            )
        self._mask_volume = open_volume(mask_dir)
        self._mask_bool = (
            np.asarray(
                as_mask(np.ascontiguousarray(self._mask_volume.array)),
                dtype=bool,
            )
            if self._mask_volume.packed
            else None
        )

    def close(self) -> None:
        self.h5.close()

    def __enter__(self) -> "CupObservationLoader":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def points_for_frame(
        self, frame: int
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        if self._mask_bool is not None:
            if frame < 0 or frame >= len(self._mask_bool):
                raise FileNotFoundError(f"{self.mask_dir} has no packed frame {frame}")
            mask = np.asarray(self._mask_bool[frame], dtype=bool)
        else:
            mask = as_mask(self._mask_volume[frame])
        metadata = self.metadata_for_frame(frame)
        depth_frame = int(metadata["depth_frame"])
        camera_frame = int(metadata["camera_frame"])
        if self.depth_valid is not None and not bool(self.depth_valid[depth_frame]):
            raise ValueError(f"depth frame {depth_frame} is invalid")
        depth_mm = np.asarray(self.depth_frames[depth_frame], dtype=np.float64)
        optical = self._unproject_masked_optical(
            depth_mm,
            self.depth_k,
            self.rgb_k,
            mask,
            projector=self.projector,
        )
        if len(optical) < 100:
            raise ValueError(f"frame {frame} has only {len(optical)} masked depth points")

        labels = DBSCAN(
            eps=self.dbscan_eps_m,
            min_samples=self.dbscan_min_samples,
            n_jobs=1,
        ).fit_predict(optical)
        cluster_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        if not len(cluster_labels):
            raise ValueError(f"frame {frame} cup mask has no depth cluster")
        largest = int(cluster_labels[int(np.argmax(counts))])
        optical = optical[labels == largest]

        camera_rotation = np.asarray(
            self.camera_rotations[camera_frame], dtype=np.float64
        )
        camera_translation = np.asarray(
            self.camera_translations[camera_frame], dtype=np.float64
        )
        world_yup = (
            (camera_rotation @ self.optical_to_link @ optical.T).T
            + camera_translation
        )
        world_zup = (YUP_TO_ZUP @ world_yup.T).T
        metadata["cluster_points"] = int(len(world_zup))
        return world_zup, metadata

    def metadata_for_frame(self, frame: int) -> dict[str, float | int]:
        """Return alignment metadata even when the object mask has no points."""
        query_time = float(self.hand_time[frame])
        depth_frame = nearest_index(self.depth_time, query_time)
        camera_frame = nearest_index(
            self.camera_time, float(self.depth_time[depth_frame])
        )
        return {
            "depth_frame": int(depth_frame),
            "camera_frame": int(camera_frame),
            "depth_time_offset_ms": 1000.0
            * float(self.depth_time[depth_frame] - query_time),
            "camera_time_offset_ms": 1000.0
            * float(self.camera_time[camera_frame] - self.depth_time[depth_frame]),
            "cluster_points": 0,
        }


class Dex3MeshSampler:
    """Sample visual hand meshes in the actual G1 wrist_yaw_link frame."""

    def __init__(
        self,
        side: str,
        samples_per_geom: int,
        seed: int,
        hand_mode: str = HAND_MODE_DEX3,
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"unsupported side {side}")
        if hand_mode not in (HAND_MODE_DEX3, HAND_MODE_BRAINCO):
            raise ValueError(f"unsupported hand mode {hand_mode}")
        self.side = side
        self.hand_mode = hand_mode
        if hand_mode == HAND_MODE_BRAINCO:
            self.model = load_dex3_model(
                DEFAULT_BRAINCO_XML,
                meshdir=DEFAULT_BRAINCO_MESHDIR,
                with_d435=False,
                hand_mode=HAND_MODE_BRAINCO,
            )
            self.hand_joint_names = BRAINCO_HAND_JOINT_NAMES
            self.mimic_rules = joint_mimic_rules(self.model)
            self.group_names = GROUP_ORDER + ("ring", "pinky")
        else:
            self.model = load_dex3_model(
                meshdir=DEFAULT_G1_MESHDIR, with_d435=False
            )
            self.hand_joint_names = HAND_JOINT_NAMES
            self.mimic_rules = []
            self.group_names = GROUP_ORDER
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = 0.0
        self.data.qpos[3] = 1.0
        self.wrist_body = self._require_id(
            mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link"
        )
        self.joint_qpos_address = {
            name: int(
                self.model.jnt_qposadr[
                    self._require_id(mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
            )
            for name in self.hand_joint_names
        }
        for rule in self.mimic_rules:
            for name in (str(rule["dependent"]), str(rule["reference"])):
                if name in self.joint_qpos_address:
                    continue
                self.joint_qpos_address[name] = int(
                    self.model.jnt_qposadr[
                        self._require_id(mujoco.mjtObj.mjOBJ_JOINT, name)
                    ]
                )
        self.side_joint_indices = [
            index
            for index, name in enumerate(self.hand_joint_names)
            if name.startswith(f"{side}_")
        ]
        self.geom_vertices: list[tuple[int, str, np.ndarray]] = []
        rng = np.random.default_rng(seed)
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            if body_name is None:
                continue
            group = self._group_for_body(str(body_name))
            if group is None:
                continue
            if int(self.model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue
            if hand_mode == HAND_MODE_BRAINCO and int(self.model.geom_group[geom_id]) != 2:
                continue
            mesh_id = int(self.model.geom_dataid[geom_id])
            start = int(self.model.mesh_vertadr[mesh_id])
            count = int(self.model.mesh_vertnum[mesh_id])
            vertices = np.asarray(
                self.model.mesh_vert[start : start + count], dtype=np.float64
            )
            keep = min(max(32, samples_per_geom), len(vertices))
            if keep < len(vertices):
                indices = np.sort(rng.choice(len(vertices), size=keep, replace=False))
                vertices = vertices[indices]
            self.geom_vertices.append((geom_id, group, np.ascontiguousarray(vertices)))
        if not self.geom_vertices:
            raise RuntimeError(f"no {side} {hand_mode} visual meshes found")
        self._rebuild_packed_layout()

    def _require_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = int(mujoco.mj_name2id(self.model, object_type, name))
        if object_id < 0:
            raise KeyError(name)
        return object_id

    def _group_for_body(self, body_name: str) -> str | None:
        prefix = f"{self.side}_"
        if self.hand_mode == HAND_MODE_BRAINCO:
            if body_name in (
                f"{self.side}_wrist_yaw_link",
                f"{self.side}_base_link",
                f"{self.side}_base2_link",
            ):
                return "palm"
            if not body_name.startswith(prefix):
                return None
            rest = body_name[len(prefix) :].lower()
            for group in ("thumb", "index", "middle", "ring", "pinky"):
                if rest.startswith(f"{group}_"):
                    return group
            return None
        if body_name == f"{self.side}_wrist_yaw_link":
            return "palm"
        if not body_name.startswith(prefix + "hand_"):
            return None
        for group in ("thumb", "index", "middle"):
            if f"hand_{group}_" in body_name:
                return group
        return None

    def side_joint_limits(self) -> np.ndarray:
        limits = np.empty((len(self.side_joint_indices), 2), dtype=np.float64)
        for row, index in enumerate(self.side_joint_indices):
            name = self.hand_joint_names[index]
            joint_id = self._require_id(mujoco.mjtObj.mjOBJ_JOINT, name)
            limits[row] = np.asarray(self.model.jnt_range[joint_id], dtype=np.float64)
        return limits

    def _rebuild_packed_layout(self) -> None:
        grouped: dict[str, list[tuple[int, np.ndarray]]] = {
            key: [] for key in self.group_names
        }
        for geom_id, group, vertices in self.geom_vertices:
            grouped[group].append((geom_id, vertices))
        names: list[str] = []
        offsets = [0]
        geoms: list[tuple[int, np.ndarray, int, int]] = []
        cursor = 0
        for group in self.group_names:
            parts = grouped[group]
            if not parts:
                continue
            names.append(group)
            for geom_id, vertices in parts:
                count = int(vertices.shape[0])
                geoms.append((geom_id, vertices, cursor, cursor + count))
                cursor += count
            offsets.append(cursor)
        self.packed_names = tuple(names)
        self.packed_offsets = np.asarray(offsets, dtype=np.int64)
        self.packed_count = cursor
        self._packed_geoms = geoms
        packed_vertices = np.empty((cursor, 3), dtype=np.float64)
        packed_geom_ids = np.empty(cursor, dtype=np.int64)
        for geom_id, vertices, start, end in geoms:
            packed_vertices[start:end] = vertices
            packed_geom_ids[start:end] = geom_id
        self._packed_vertices = packed_vertices
        self._packed_geom_ids = packed_geom_ids

    def _set_hand_qpos(self, qpos14: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos14, dtype=np.float64).reshape(-1)
        expected = len(self.hand_joint_names)
        if qpos.shape != (expected,):
            raise ValueError(f"expected {expected} hand joints, got {qpos.shape}")
        for index in self.side_joint_indices:
            name = self.hand_joint_names[index]
            self.data.qpos[self.joint_qpos_address[name]] = qpos[index]
        for rule in self.mimic_rules:
            reference_name = str(rule["reference"])
            dependent_name = str(rule["dependent"])
            if reference_name not in self.joint_qpos_address:
                continue
            if dependent_name not in self.joint_qpos_address:
                continue
            reference_q = self.data.qpos[self.joint_qpos_address[reference_name]]
            dependent_q, _ = evaluate_mimic_rule(
                rule, np.asarray(reference_q), np.asarray(0.0)
            )
            self.data.qpos[self.joint_qpos_address[dependent_name]] = float(
                np.asarray(dependent_q).reshape(())
            )
        return qpos

    def fill_packed_local_samples(
        self, qpos14: np.ndarray, out: np.ndarray
    ) -> np.ndarray:
        packed = np.asarray(out, dtype=np.float64)
        if packed.shape != (self.packed_count, 3):
            raise ValueError(
                f"packed samples must be ({self.packed_count}, 3), got {packed.shape}"
            )
        self._set_hand_qpos(qpos14)
        # Hand visual poses depend only on qpos kinematics. mj_forward also
        # runs collisions/actuators and dominated the joint-Powell inner loop.
        mujoco.mj_kinematics(self.model, self.data)
        wrist_position = np.ascontiguousarray(
            self.data.xpos[self.wrist_body], dtype=np.float64
        )
        wrist_rotation = np.ascontiguousarray(
            np.asarray(self.data.xmat[self.wrist_body], dtype=np.float64).reshape(3, 3)
        )
        geoms_to_wrist_local(
            self._packed_vertices,
            self._packed_geom_ids,
            np.ascontiguousarray(self.data.geom_xpos, dtype=np.float64),
            np.ascontiguousarray(self.data.geom_xmat, dtype=np.float64),
            wrist_position,
            wrist_rotation,
            packed,
        )
        return packed

    def unpack_packed_samples(self, packed: np.ndarray) -> dict[str, np.ndarray]:
        return {
            name: np.ascontiguousarray(
                packed[int(self.packed_offsets[index]) : int(self.packed_offsets[index + 1])]
            )
            for index, name in enumerate(self.packed_names)
        }

    def local_samples(self, qpos14: np.ndarray) -> dict[str, np.ndarray]:
        packed = np.empty((self.packed_count, 3), dtype=np.float64)
        self.fill_packed_local_samples(qpos14, packed)
        return self.unpack_packed_samples(packed)


def point_group_metrics(
    points: np.ndarray, model: CupModel, contact_threshold_m: float
) -> tuple[dict[str, float | int | bool], np.ndarray]:
    xyz = np.asarray(points, dtype=np.float64)
    center = np.asarray(model.center_xy, dtype=np.float64)
    delta_xy = xyz[:, :2] - center
    rho = np.linalg.norm(delta_xy, axis=1)
    expected_radius = model.radius_at(xyz[:, 2])
    lateral = rho - expected_radius
    within_height = (xyz[:, 2] >= model.z_min) & (xyz[:, 2] <= model.z_max)
    axial_outside = np.maximum(
        np.maximum(model.z_min - xyz[:, 2], xyz[:, 2] - model.z_max), 0.0
    )
    surface_distance = np.sqrt(lateral * lateral + axial_outside * axial_outside)
    eligible = np.flatnonzero(within_height)
    if eligible.size:
        best_index = int(eligible[np.argmin(np.abs(lateral[eligible]))])
        minimum = float(np.abs(lateral[best_index]))
        contact_count = int(
            np.count_nonzero(within_height & (np.abs(lateral) <= contact_threshold_m))
        )
    else:
        best_index = int(np.argmin(surface_distance))
        minimum = float(surface_distance[best_index])
        contact_count = 0
    radial = delta_xy[best_index]
    radial_norm = float(np.linalg.norm(radial))
    normal = radial / radial_norm if radial_norm > 1e-10 else np.zeros(2)
    penetration = np.where(within_height, np.maximum(-lateral, 0.0), 0.0)
    positive = penetration[penetration > 0.0]
    metrics: dict[str, float | int | bool] = {
        "surface_distance_min_m": minimum,
        "contact_points": contact_count,
        "contact": bool(contact_count >= 3),
        "penetrating_points": int(np.count_nonzero(penetration > 0.0)),
        "penetration_ratio": float(np.mean(penetration > 0.0)),
        "penetration_mean_positive_m": (
            float(np.mean(positive)) if positive.size else 0.0
        ),
        "penetration_p95_m": (
            float(np.percentile(positive, 95)) if positive.size else 0.0
        ),
        "penetration_max_m": float(np.max(penetration)),
        "closest_point_x": float(xyz[best_index, 0]),
        "closest_point_y": float(xyz[best_index, 1]),
        "closest_point_z": float(xyz[best_index, 2]),
        "closest_radial_normal_x": float(normal[0]),
        "closest_radial_normal_y": float(normal[1]),
    }
    return metrics, penetration


def evaluate_configuration(
    local_samples: dict[str, np.ndarray],
    wrist_position: np.ndarray,
    wrist_rotation: np.ndarray,
    cup: CupModel,
    contact_threshold_m: float,
    penetration_tolerance_m: float,
    penetration_ratio_tolerance: float,
    return_world_samples: bool = True,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    packed, offsets, names = pack_local_samples(local_samples)
    return evaluate_packed_configuration(
        packed,
        offsets,
        names,
        wrist_position,
        wrist_rotation,
        cup,
        contact_threshold_m,
        penetration_tolerance_m,
        penetration_ratio_tolerance,
        return_world_samples=return_world_samples,
    )


def max_true_run(values: list[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def interpolate_cup_model(
    frame: int,
    left: CupModel,
    right: CupModel,
    observation: dict[str, float | int],
) -> CupModel:
    """Interpolate a failed cup fit between two independently valid fits."""
    if not left.frame < frame < right.frame:
        raise ValueError(
            f"cup fallback frame {frame} is not bracketed by {left.frame}, {right.frame}"
        )
    alpha = float(frame - left.frame) / float(right.frame - left.frame)

    def blend(name: str) -> float:
        return (1.0 - alpha) * float(getattr(left, name)) + alpha * float(
            getattr(right, name)
        )

    return CupModel(
        frame=int(frame),
        center_xy=(
            (1.0 - alpha) * float(left.center_xy[0]) + alpha * float(right.center_xy[0]),
            (1.0 - alpha) * float(left.center_xy[1]) + alpha * float(right.center_xy[1]),
        ),
        z_reference=blend("z_reference"),
        radius_reference=blend("radius_reference"),
        radius_slope=blend("radius_slope"),
        z_min=blend("z_min"),
        z_max=blend("z_max"),
        points=int(round(blend("points"))),
        inliers=int(round(blend("inliers"))),
        inlier_fraction=blend("inlier_fraction"),
        residual_p50_m=blend("residual_p50_m"),
        residual_p95_m=blend("residual_p95_m"),
        depth_frame=int(observation["depth_frame"]),
        depth_time_offset_ms=float(observation["depth_time_offset_ms"]),
        camera_time_offset_ms=float(observation["camera_time_offset_ms"]),
    )


def copy_cup_model(
    frame: int,
    source: CupModel,
    observation: dict[str, float | int],
) -> CupModel:
    """Hold/backfill a nearby valid model while preserving frame metadata."""
    values = asdict(source)
    values.update(
        frame=int(frame),
        depth_frame=int(observation["depth_frame"]),
        depth_time_offset_ms=float(observation["depth_time_offset_ms"]),
        camera_time_offset_ms=float(observation["camera_time_offset_ms"]),
    )
    values["center_xy"] = tuple(values["center_xy"])
    return CupModel(**values)


def cup_temporal_error(
    previous: CupModel,
    current: CupModel,
    *,
    center_step_m: float,
    radius_step_m: float,
    height_step_m: float,
    temporal_gap_cap: int = 2,
) -> str | None:
    delta_frames = max(1, int(current.frame - previous.frame))
    effective_delta = min(delta_frames, max(1, int(temporal_gap_cap)))
    center_previous = np.asarray(
        [previous.center_xy[0], previous.center_xy[1], previous.z_reference]
    )
    center_current = np.asarray(
        [current.center_xy[0], current.center_xy[1], current.z_reference]
    )
    center_delta = float(np.linalg.norm(center_current - center_previous))
    radius_delta = abs(float(current.radius_reference - previous.radius_reference))
    previous_height = float(previous.z_max - previous.z_min)
    current_height = float(current.z_max - current.z_min)
    height_delta = abs(current_height - previous_height)
    violations = []
    if center_delta > float(center_step_m) * effective_delta:
        violations.append(f"center_step={center_delta:.4f}m")
    if radius_delta > float(radius_step_m) * effective_delta:
        violations.append(f"radius_step={radius_delta:.4f}m")
    if height_delta > float(height_step_m) * effective_delta:
        violations.append(f"height_step={height_delta:.4f}m")
    if not violations:
        return None
    return "cup temporal continuity gate: " + ", ".join(violations)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    bilateral = [bool(record["bilateral_contact"]) for record in records]
    valid_grasp = [bool(record["valid_grasp"]) for record in records]
    transitions = sum(a != b for a, b in zip(bilateral[:-1], bilateral[1:]))
    return {
        "frames": len(records),
        "bilateral_contact_rate": float(np.mean(bilateral)),
        "bilateral_contact_frames": int(np.count_nonzero(bilateral)),
        "bilateral_contact_longest_run": max_true_run(bilateral),
        "bilateral_contact_transitions": int(transitions),
        "valid_grasp_rate": float(np.mean(valid_grasp)),
        "valid_grasp_frames": int(np.count_nonzero(valid_grasp)),
        "valid_grasp_longest_run": max_true_run(valid_grasp),
        "thumb_surface_distance_m": finite_stats(
            [record["thumb_surface_distance_m"] for record in records]
        ),
        "opposition_surface_distance_m": finite_stats(
            [record["opposition_surface_distance_m"] for record in records]
        ),
        "opposition_normal_dot": finite_stats(
            [record["opposition_normal_dot"] for record in records]
        ),
        "hand_penetration_ratio": finite_stats(
            [record["hand_penetration_ratio"] for record in records]
        ),
        "hand_penetration_p95_m": finite_stats(
            [record["hand_penetration_p95_m"] for record in records]
        ),
        "hand_penetration_max_m": finite_stats(
            [record["hand_penetration_max_m"] for record in records]
        ),
        "ranking_objective_m": finite_stats(
            [record["ranking_objective_m"] for record in records]
        ),
    }


def plot_heatmaps(
    output: Path,
    wrist_methods: list[str],
    finger_methods: list[str],
    summaries: dict[str, dict[str, Any]],
) -> None:
    fields = (
        ("valid_grasp_rate", "Contact without excessive penetration", True),
        ("ranking_objective_m.median", "Median interaction objective (m)", False),
        ("hand_penetration_p95_m.p95", "Penetration p95-of-p95 (m)", False),
        ("thumb_surface_distance_m.median", "Median thumb distance (m)", False),
    )

    def value_for(summary: dict[str, Any], path: str) -> float:
        value: Any = summary
        for key in path.split("."):
            value = value[key]
        return float(value)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for axis, (field, title, higher_better) in zip(axes.flat, fields):
        matrix = np.asarray(
            [
                [
                    value_for(summaries[f"{wrist}__{finger}"], field)
                    for finger in finger_methods
                ]
                for wrist in wrist_methods
            ]
        )
        image = axis.imshow(matrix, aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(finger_methods)), finger_methods, rotation=25)
        axis.set_yticks(range(len(wrist_methods)), wrist_methods)
        axis.set_title(title + (" (higher better)" if higher_better else " (lower better)"))
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.3f}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] < np.nanmedian(matrix) else "black",
                    fontsize=7,
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def set_equal_3d(axis: Any, points: np.ndarray) -> None:
    low = np.min(points, axis=0)
    high = np.max(points, axis=0)
    center = 0.5 * (low + high)
    radius = 0.55 * float(np.max(high - low))
    for setter, value in zip(
        (axis.set_xlim, axis.set_ylim, axis.set_zlim), center
    ):
        setter(value - radius, value + radius)


def plot_keyframe(
    output: Path,
    frame: int,
    cup: CupModel,
    cup_points: np.ndarray,
    entries: list[tuple[str, str, dict[str, Any], dict[str, np.ndarray]]],
    *,
    point_cloud_only: bool = False,
    dpi: int = 160,
    point_size: float = 2.0,
    point_alpha: float = 0.45,
) -> None:
    columns = 3
    rows = int(math.ceil(len(entries) / columns))
    fig = plt.figure(figsize=(5.2 * columns, 4.8 * rows))
    rng = np.random.default_rng(frame)
    if len(cup_points) > 1200:
        cup_points = cup_points[
            rng.choice(len(cup_points), size=1200, replace=False)
        ]
    angles = np.linspace(0.0, 2.0 * np.pi, 80)
    heights = np.linspace(cup.z_min, cup.z_max, 8)
    cylinder_lines = []
    for z in heights:
        radius = float(cup.radius_at(z))
        cylinder_lines.append(
            np.column_stack(
                [
                    cup.center_xy[0] + radius * np.cos(angles),
                    cup.center_xy[1] + radius * np.sin(angles),
                    np.full_like(angles, z),
                ]
            )
        )
    for index, (wrist, finger, metric, samples) in enumerate(entries, start=1):
        axis = fig.add_subplot(rows, columns, index, projection="3d")
        axis.scatter(
            cup_points[:, 0], cup_points[:, 1], cup_points[:, 2],
            c="#5dade2", s=2, alpha=0.25,
        )
        if not point_cloud_only:
            for line in cylinder_lines:
                axis.plot(line[:, 0], line[:, 1], line[:, 2], c="#2471a3", lw=0.5)
        all_points = [cup_points]
        for group in GROUP_ORDER:
            points = samples[group]
            all_points.append(points)
            axis.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=GROUP_COLORS[group], s=point_size, alpha=point_alpha,
                label=group,
            )
        axis.set_title(
            f"{wrist}\n{finger} | valid={int(metric['valid_grasp'])} "
            f"pen={100*metric['hand_penetration_ratio']:.1f}%"
        )
        if point_cloud_only:
            axis.grid(False)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_zticks([])
            for dimension in (axis.xaxis, axis.yaxis, axis.zaxis):
                dimension.pane.fill = False
                dimension.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
                dimension.line.set_color((1.0, 1.0, 1.0, 0.0))
        else:
            axis.set_xlabel("world X")
            axis.set_ylabel("world Y")
            axis.set_zlabel("world Z")
        set_equal_3d(axis, np.concatenate(all_points, axis=0))
        if index == 1:
            axis.legend(fontsize=7)
    fig.suptitle(f"Observed cup and Dex3 visual mesh, source frame {frame}")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    session = args.session_dir.expanduser().resolve()
    experiment = args.experiment_dir.expanduser().resolve()
    mask_dir = args.mask_dir.expanduser().resolve()
    output = args.out_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    wrist_path = experiment / "wrist_candidates_native.npz"
    finger_path = experiment / "finger_candidates_native.npz"
    for required in (
        session / "annotation.hdf5",
        session / "calibrations" / "rgb_K.npy",
        session / "calibrations" / "depth_K.npy",
        session / "calibrations" / "R_optical_to_link.npy",
        wrist_path,
        finger_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not mask_dir.is_dir():
        raise FileNotFoundError(mask_dir)
    _projector_cls, resolve_device, _unproject = _contact_projection()
    device = resolve_device(args.device)
    use_cuda = device is not None and getattr(device, "type", None) == "cuda"
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    print(
        f"[interaction] device={device or 'cpu'} "
        f"projection_backend={'torch_cuda' if use_cuda else 'numpy_cpu'}",
        flush=True,
    )

    with np.load(wrist_path, allow_pickle=False) as archive:
        wrist_data = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(finger_path, allow_pickle=False) as archive:
        finger_data = {key: np.asarray(archive[key]) for key in archive.files}
    available_wrist = sorted(
        {
            key.split("__", 1)[0]
            for key in wrist_data
            if key.endswith(f"__{args.side}_position")
        }
    )
    available_finger = sorted(
        {key.split("__", 1)[0] for key in finger_data if key.endswith("__qpos")}
    )
    wrist_methods = comma_list(args.wrist_methods, available_wrist)
    finger_methods = comma_list(args.finger_methods, available_finger)
    frame_count = min(
        len(wrist_data["timestamps"]), len(finger_data["timestamps"])
    )
    start = max(0, int(args.start_frame))
    end = min(int(args.end_frame), frame_count - 1)
    if end < start:
        raise ValueError(f"empty frame range {start}:{end}")
    frames = list(range(start, end + 1))

    cup_models: dict[int, CupModel] = {}
    cup_points_by_frame: dict[int, np.ndarray] = {}
    observations_by_frame: dict[int, dict[str, float | int]] = {}
    object_fit_errors: dict[int, str] = {}
    fit_inlier_counts: dict[int, int] = {}
    with CupObservationLoader(
        session,
        mask_dir,
        dbscan_eps_m=args.dbscan_eps_m,
        dbscan_min_samples=args.dbscan_min_samples,
        device=device,
    ) as loader:
        for frame in frames:
            observation = loader.metadata_for_frame(frame)
            try:
                points, observation = loader.points_for_frame(frame)
            except (FileNotFoundError, ValueError) as error:
                if not args.allow_object_fit_fallback:
                    raise
                cup_points_by_frame[frame] = np.empty((0, 3), dtype=np.float64)
                observations_by_frame[frame] = observation
                object_fit_errors[frame] = str(error)
                continue
            cup_points_by_frame[frame] = points
            observations_by_frame[frame] = observation
            try:
                model, inliers = fit_upright_tapered_cylinder(
                    points,
                    frame=frame,
                    depth_frame=int(observation["depth_frame"]),
                    depth_time_offset_ms=float(observation["depth_time_offset_ms"]),
                    camera_time_offset_ms=float(observation["camera_time_offset_ms"]),
                    inlier_threshold_m=float(args.cylinder_inlier_m),
                    ransac_iterations=int(args.ransac_iterations),
                    seed=int(args.seed) + frame,
                )
                cup_models[frame] = model
                fit_inlier_counts[frame] = int(np.count_nonzero(inliers))
            except ValueError as error:
                if not args.allow_object_fit_fallback:
                    raise
                object_fit_errors[frame] = str(error)

    accepted_frames: list[int] = []
    for frame in frames:
        if frame not in cup_models:
            continue
        if accepted_frames:
            temporal_error = cup_temporal_error(
                cup_models[accepted_frames[-1]],
                cup_models[frame],
                center_step_m=float(args.object_center_step_m),
                radius_step_m=float(args.object_radius_step_m),
                height_step_m=float(args.object_height_step_m),
                temporal_gap_cap=int(args.object_temporal_gap_cap),
            )
            if temporal_error is not None:
                if not args.allow_object_fit_fallback:
                    raise ValueError(f"frame {frame}: {temporal_error}")
                object_fit_errors[frame] = temporal_error
                del cup_models[frame]
                fit_inlier_counts.pop(frame, None)
                continue
        accepted_frames.append(frame)

    if object_fit_errors:
        failed = [frame in object_fit_errors for frame in frames]
        longest = max_true_run(failed)
        if longest > int(args.max_object_fallback_run):
            raise ValueError(
                f"object fit fallback run {longest} exceeds "
                f"--max-object-fallback-run={args.max_object_fallback_run}"
            )
        valid_frames = sorted(cup_models)
        for frame in sorted(object_fit_errors):
            before = [value for value in valid_frames if value < frame]
            after = [value for value in valid_frames if value > frame]
            if before and after:
                left = before[-1]
                right = after[0]
                cup_models[frame] = interpolate_cup_model(
                    frame,
                    cup_models[left],
                    cup_models[right],
                    observations_by_frame[frame],
                )
            elif before:
                cup_models[frame] = copy_cup_model(
                    frame, cup_models[before[-1]], observations_by_frame[frame]
                )
            elif after:
                cup_models[frame] = copy_cup_model(
                    frame, cup_models[after[0]], observations_by_frame[frame]
                )
            else:
                raise ValueError("all cup fits failed")

    object_records: list[dict[str, Any]] = []
    for frame in frames:
        record = asdict(cup_models[frame])
        record["cluster_points"] = int(
            observations_by_frame[frame]["cluster_points"]
        )
        record["fit_inlier_mask_count"] = fit_inlier_counts.get(frame)
        if frame in object_fit_errors:
            record["object_fit_error"] = object_fit_errors[frame]
            valid_frames = sorted(value for value in cup_models if value not in object_fit_errors)
            before = [value for value in valid_frames if value < frame]
            after = [value for value in valid_frames if value > frame]
            if before and after:
                record["object_fit_status"] = "temporal_interpolation"
                record["fallback_sources"] = [before[-1], after[0]]
            elif before:
                record["object_fit_status"] = "trailing_hold"
                record["fallback_sources"] = [before[-1]]
            else:
                record["object_fit_status"] = "leading_backfill"
                record["fallback_sources"] = [after[0]]
        else:
            record["object_fit_status"] = "direct"
            record["object_fit_error"] = None
            record["fallback_sources"] = None
        object_records.append(record)

    sampler = Dex3MeshSampler(
        args.side,
        int(args.mesh_samples_per_geom),
        int(args.seed),
        hand_mode=str(args.hand_mode),
    )
    local_cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    records_by_combination: dict[str, list[dict[str, Any]]] = {
        f"{wrist}__{finger}": []
        for wrist in wrist_methods
        for finger in finger_methods
    }
    keyframe_cache: dict[
        tuple[str, str], tuple[dict[str, Any], dict[str, np.ndarray]]
    ] = {}

    for frame in frames:
        cup = cup_models[frame]
        for finger in finger_methods:
            cache_key = (finger, frame)
            if cache_key not in local_cache:
                local_cache[cache_key] = sampler.local_samples(
                    finger_data[f"{finger}__qpos"][frame]
                )
            local = local_cache[cache_key]
            for wrist in wrist_methods:
                position = wrist_data[f"{wrist}__{args.side}_position"][frame]
                rotation = wrist_data[f"{wrist}__{args.side}_rotation"][frame]
                metric, samples = evaluate_configuration(
                    local,
                    position,
                    rotation,
                    cup,
                    float(args.contact_threshold_m),
                    float(args.penetration_tolerance_m),
                    float(args.penetration_ratio_tolerance),
                )
                metric.update(
                    {
                        "frame": frame,
                        "side": args.side,
                        "wrist_method": wrist,
                        "finger_method": finger,
                    }
                )
                records_by_combination[f"{wrist}__{finger}"].append(metric)
                if frame == int(args.keyframe):
                    keyframe_cache[(wrist, finger)] = (metric, samples)

    summaries = {
        name: summarize_records(records)
        for name, records in records_by_combination.items()
    }
    for name, summary in summaries.items():
        wrist, finger = name.split("__", 1)
        summary["wrist_method"] = wrist
        summary["finger_method"] = finger
    ranking = sorted(
        summaries,
        key=lambda name: (
            -summaries[name]["valid_grasp_rate"],
            summaries[name]["ranking_objective_m"]["median"],
            summaries[name]["hand_penetration_p95_m"]["p95"],
            -summaries[name]["bilateral_contact_rate"],
        ),
    )

    with (output / "object_models.jsonl").open("w") as stream:
        for record in object_records:
            stream.write(json.dumps(record) + "\n")
    with (output / "interaction_frame_metrics.jsonl").open("w") as stream:
        for name in sorted(records_by_combination):
            for record in records_by_combination[name]:
                stream.write(json.dumps(record) + "\n")
    summary_document = {
        "schema_version": 1,
        "side": args.side,
        "frame_range_inclusive": [start, end],
        "evaluated_frames": len(frames),
        "wrist_methods": wrist_methods,
        "finger_methods": finger_methods,
        "contact_threshold_m": float(args.contact_threshold_m),
        "penetration_tolerance_m": float(args.penetration_tolerance_m),
        "penetration_ratio_tolerance": float(args.penetration_ratio_tolerance),
        "ranking_note": (
            "ranking_objective is a heuristic ordering aid; physical contact, "
            "opposition and penetration terms remain separately reported"
        ),
        "ranking": ranking,
        "combinations": summaries,
    }
    (output / "interaction_summary.json").write_text(
        json.dumps(summary_document, indent=2) + "\n"
    )
    if not args.skip_plots:
        plot_heatmaps(
            output / "interaction_heatmaps.png",
            wrist_methods,
            finger_methods,
            summaries,
        )

    keyframe = int(args.keyframe)
    if keyframe not in cup_models:
        keyframe = frames[len(frames) // 2]
    top_entries = []
    for name in ranking[:6]:
        wrist, finger = name.split("__", 1)
        metric, samples = keyframe_cache.get((wrist, finger), (None, None))
        if metric is None:
            position = wrist_data[f"{wrist}__{args.side}_position"][keyframe]
            rotation = wrist_data[f"{wrist}__{args.side}_rotation"][keyframe]
            local = sampler.local_samples(finger_data[f"{finger}__qpos"][keyframe])
            metric, samples = evaluate_configuration(
                local,
                position,
                rotation,
                cup_models[keyframe],
                float(args.contact_threshold_m),
                float(args.penetration_tolerance_m),
                float(args.penetration_ratio_tolerance),
            )
        top_entries.append((wrist, finger, metric, samples))
    if not args.skip_plots:
        plot_keyframe(
            output / f"interaction_keyframe_{keyframe:06d}.png",
            keyframe,
            cup_models[keyframe],
            cup_points_by_frame[keyframe],
            top_entries,
        )

    manifest = {
        "schema_version": 1,
        "experiment_only": True,
        "production_pipeline_modified": False,
        "session_dir": str(session),
        "source_experiment_dir": str(experiment),
        "mask_dir": str(mask_dir),
        "output_dir": str(output),
        "coordinate_contract": (
            "depth optical -> source Y-up world -> Z-up retarget world; Dex3 "
            "visual meshes sampled in complete G1 MuJoCo wrist_yaw_link frame"
        ),
        "object_model": (
            "upright robust tapered cylinder; circle RANSAC rejects the cup handle"
        ),
        "object_fit_fallback": {
            "enabled": bool(args.allow_object_fit_fallback),
            "frames": sorted(object_fit_errors),
            "max_consecutive_allowed": int(args.max_object_fallback_run),
            "temporal_gate_per_frame": {
                "center_step_m": float(args.object_center_step_m),
                "radius_step_m": float(args.object_radius_step_m),
                "height_step_m": float(args.object_height_step_m),
                "gap_cap_frames": int(args.object_temporal_gap_cap),
            },
        },
        "limitations": [
            "the cup is assumed upright over the evaluated interval",
            "single-view RGB-D does not observe the occluded cup surface",
            "solid-body penetration proxy is conservative for the open cup interior",
            "metrics rank candidates but do not prove dynamic grasp stability",
        ],
        "top_combination": ranking[0],
        "outputs": [
            "object_models.jsonl",
            "interaction_frame_metrics.jsonl",
            "interaction_summary.json",
            "interaction_heatmaps.png",
            f"interaction_keyframe_{keyframe:06d}.png",
        ],
    }
    (output / "interaction_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
