#!/usr/bin/env python3
"""Apply a final, offline Mink dual-arm IK pass to three-point MPC output.

The JAX MPC trajectory remains the source of truth for the floating base,
torso, and lower body.  This script changes only the 14 arm joints, using the
previous refined frame as the next frame's seed.  It then rebuilds a complete,
self-consistent 50 Hz kinematics log with the exact MJX model used by MPC.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import mink
import mujoco
import numpy as np


PROJECT_ROOT = Path(os.environ.get("KODA_HOME", str(Path(__file__).resolve().parents[7])))
SPARSETRACK_SCRIPTS = PROJECT_ROOT / "resources" / "SparseTrack" / "scripts"
if str(SPARSETRACK_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SPARSETRACK_SCRIPTS))

from st_mink_wrist_ik import (  # noqa: E402
    ARMS_ONLY,
    MinkDualWristIK,
    build_model_qpos,
    extract_joint_qpos,
)


LEFT_WRIST = "left_wrist_yaw_link"
RIGHT_WRIST = "right_wrist_yaw_link"
ROOT_BODY = "pelvis"
DEFAULT_MODEL = PROJECT_ROOT / "src/koda/vendor/scalebfm/scalebridge/data/robot/g1_29dof/g1_29dof_mjx.xml"
COLLISION_GEOM_PAIRS = (
    (
        ("left_hand_coll", "left_forearm_coll"),
        ("torso_coll_l", "torso_coll_r", "pelvis_coll", "head_coll"),
    ),
    (
        ("right_hand_coll", "right_forearm_coll"),
        ("torso_coll_l", "torso_coll_r", "pelvis_coll", "head_coll"),
    ),
    (
        ("left_hand_coll", "left_forearm_coll"),
        ("right_hand_coll", "right_forearm_coll"),
    ),
)
# ``mj_geomDistance(left_shoulder_coll, torso_coll_*)`` measures the fixed
# shoulder attachment, not the visual clearance of the upper arm.  The q0
# regularizer therefore uses the elbow-side part of each shoulder capsule and
# samples its centre line after the proximal attachment.  These names are kept
# explicit so a model change fails loudly instead of silently changing the
# geometry contract.
Q0_SHOULDER_PROBES = (
    ("left_shoulder_coll", "left_elbow_link"),
    ("right_shoulder_coll", "right_elbow_link"),
)
Q0_SHOULDER_OBSTACLE_GEOMS = (
    "torso_coll_l",
    "torso_coll_r",
    "pelvis_coll",
)
DEFAULT_Q0_SHOULDER_SAMPLE_START = 0.40
DEFAULT_Q0_SHOULDER_SAMPLE_COUNT = 6
Q0_SHOULDER_CLEARANCE_TARGET_TOLERANCE_M = 1.0e-4


def _strings(values: Iterable[Any]) -> list[str]:
    result = []
    for value in np.asarray(values).reshape(-1).tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        result.append(str(value))
    return result


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _normalise_quaternions(values: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.all(np.isfinite(values)) or np.any(norms < 1.0e-8):
        raise ValueError(f"{label} contains an invalid quaternion")
    return values / norms


def _quaternion_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = _normalise_quaternions(a, "first quaternion array")
    b = _normalise_quaternions(b, "second quaternion array")
    dots = np.abs(np.sum(a * b, axis=-1))
    return 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _minimum_geom_clearance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    joint_names: list[str],
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    joint_qpos: np.ndarray,
    pairs: list[tuple[list[str], list[str]]],
) -> float:
    """Evaluate configured collision-pair clearance for one candidate qpos."""
    del qpos
    data.qpos[:] = build_model_qpos(
        model,
        root_pos=root_pos,
        root_quat_wxyz=root_quat_wxyz,
        joint_qpos=joint_qpos,
        joint_names=joint_names,
    )
    mujoco.mj_forward(model, data)
    distances = []
    for left_names, right_names in pairs:
        for left_name in left_names:
            left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, left_name)
            for right_name in right_names:
                right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, right_name)
                distances.append(mujoco.mj_geomDistance(model, data, left_id, right_id, 1.0, None))
    return float(min(distances)) if distances else float("inf")


def _expand_geom_pairs(
    model: mujoco.MjModel,
    grouped_pairs: Iterable[tuple[Iterable[str], Iterable[str]]],
) -> list[tuple[int, int]]:
    """Expand named geom groups into concrete MuJoCo geom-id pairs."""
    expanded: list[tuple[int, int]] = []
    for left_names, right_names in grouped_pairs:
        for left_name in left_names:
            left_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, left_name
            )
            for right_name in right_names:
                right_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_GEOM, right_name
                )
                if left_id < 0 or right_id < 0:
                    raise ValueError(
                        "clearance cost references a missing geom: "
                        f"{left_name!r}, {right_name!r}"
                    )
                expanded.append((int(left_id), int(right_id)))
    return expanded


def _build_shoulder_probe_ids(
    model: mujoco.MjModel,
    probes=Q0_SHOULDER_PROBES,
    obstacles=Q0_SHOULDER_OBSTACLE_GEOMS,
) -> list[tuple[int, int, list[int]]]:
    """Resolve shoulder geom, elbow body, and torso obstacle ids.

    The returned tuples are ``(shoulder_geom_id, elbow_body_id,
    obstacle_geom_ids)``.  All referenced objects are validated once before
    the solver starts.
    """
    obstacle_ids: list[int] = []
    for name in obstacles:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise ValueError(f"shoulder clearance obstacle is missing: {name!r}")
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            raise ValueError(f"shoulder clearance obstacle must be a capsule: {name!r}")
        obstacle_ids.append(int(geom_id))

    resolved_probes: list[tuple[int, int, list[int]]] = []
    for shoulder_name, elbow_name in probes:
        shoulder_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, shoulder_name
        )
        elbow_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, elbow_name)
        if shoulder_id < 0 or elbow_id < 0:
            raise ValueError(
                "shoulder clearance probe is missing: "
                f"geom={shoulder_name!r}, body={elbow_name!r}"
            )
        if int(model.geom_type[shoulder_id]) != int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            raise ValueError(f"shoulder clearance probe must be a capsule: {shoulder_name!r}")
        resolved_probes.append((int(shoulder_id), int(elbow_id), list(obstacle_ids)))
    return resolved_probes


def _collision_layout(model: mujoco.MjModel, layout: str):
    """Resolve the explicitly selected geometry without changing the MPC model."""
    mapping = {}
    if layout == "hydrax-derived":
        specs = {
            "pelvis_coll": ("hydrax_collision_2", "pelvis", 3),
            "torso_coll_l": ("hydrax_collision_34", "torso_link", 3),
            "torso_coll_r": ("hydrax_collision_35", "torso_link", 3),
            "head_coll": ("hydrax_collision_36", "torso_link", 2),
            "left_shoulder_coll": ("hydrax_collision_39", "left_shoulder_roll_link", 3),
            "left_forearm_coll": ("hydrax_collision_42", "left_wrist_roll_link", 3),
            "left_hand_coll": ("hydrax_collision_47", "left_wrist_yaw_link", 3),
            "right_shoulder_coll": ("hydrax_collision_50", "right_shoulder_roll_link", 3),
            "right_forearm_coll": ("hydrax_collision_53", "right_wrist_roll_link", 3),
            "right_hand_coll": ("hydrax_collision_58", "right_wrist_yaw_link", 3),
        }
        for canonical, (actual, body, geom_type) in specs.items():
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, actual)
            if geom_id < 0 or model.body(int(model.geom_bodyid[geom_id])).name != body:
                raise ValueError(f"invalid derived collision geometry: {actual} on {body}")
            if int(model.geom_type[geom_id]) != geom_type:
                raise ValueError(f"unexpected collision primitive type: {actual}")
            mapping[canonical] = actual
    rename = lambda name: mapping.get(name, name)
    pairs = tuple((tuple(map(rename, left)), tuple(map(rename, right))) for left, right in COLLISION_GEOM_PAIRS)
    probes = tuple((rename(geom), body) for geom, body in Q0_SHOULDER_PROBES)
    obstacles = tuple(map(rename, Q0_SHOULDER_OBSTACLE_GEOMS))
    return pairs, probes, obstacles


def _point_to_capsule_surface_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    point: np.ndarray,
    point_radius: float,
    obstacle_geom_id: int,
) -> float:
    """Distance between a spherical upper-arm sample and a capsule surface."""
    center = np.asarray(data.geom_xpos[obstacle_geom_id], dtype=np.float64)
    rotation = np.asarray(data.geom_xmat[obstacle_geom_id], dtype=np.float64).reshape(3, 3)
    half_length = float(model.geom_size[obstacle_geom_id, 1])
    axis = rotation[:, 2]
    segment_start = center - half_length * axis
    segment_end = center + half_length * axis
    segment = segment_end - segment_start
    denominator = float(segment @ segment)
    if denominator <= 1.0e-12:
        closest = segment_start
    else:
        fraction = float(
            np.clip((np.asarray(point, dtype=np.float64) - segment_start) @ segment / denominator, 0.0, 1.0)
        )
        closest = segment_start + fraction * segment
    return float(
        np.linalg.norm(np.asarray(point, dtype=np.float64) - closest)
        - float(model.geom_size[obstacle_geom_id, 0])
        - float(point_radius)
    )


def _shoulder_probe_clearances(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    probes: list[tuple[int, int, list[int]]],
    sample_start: float,
    sample_count: int,
) -> np.ndarray:
    """Return one visual upper-arm clearance value per side in metres.

    MuJoCo capsule endpoints are selected by proximity to the elbow body, so
    the sampled segment always runs from the shoulder attachment toward the
    elbow even when the arm is rotated.  The first part of the capsule is
    intentionally skipped because it is physically attached to the torso.
    Both capsule radii are subtracted, making the result a surface-to-surface
    distance rather than a centre-line distance.
    """
    fractions = np.linspace(
        float(sample_start), 1.0, int(sample_count), dtype=np.float64
    )
    clearances: list[float] = []
    for shoulder_id, elbow_body_id, obstacle_ids in probes:
        center = np.asarray(data.geom_xpos[shoulder_id], dtype=np.float64)
        rotation = np.asarray(data.geom_xmat[shoulder_id], dtype=np.float64).reshape(3, 3)
        half_length = float(model.geom_size[shoulder_id, 1])
        shoulder_radius = float(model.geom_size[shoulder_id, 0])
        endpoints = (center - half_length * rotation[:, 2], center + half_length * rotation[:, 2])
        elbow_position = np.asarray(data.xpos[elbow_body_id], dtype=np.float64)
        distal_index = int(np.argmin([np.linalg.norm(endpoint - elbow_position) for endpoint in endpoints]))
        distal = endpoints[distal_index]
        proximal = endpoints[1 - distal_index]
        side_clearance = float("inf")
        for fraction in fractions:
            point = proximal + fraction * (distal - proximal)
            for obstacle_id in obstacle_ids:
                side_clearance = min(
                    side_clearance,
                    _point_to_capsule_surface_distance(
                        model, data, point, shoulder_radius, obstacle_id
                    ),
                )
        clearances.append(side_clearance)
    return np.asarray(clearances, dtype=np.float64)


def _minimum_shoulder_probe_clearance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: list[str],
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    joint_qpos: np.ndarray,
    probes: list[tuple[int, int, list[int]]],
    sample_start: float,
    sample_count: int,
) -> float:
    """Evaluate the same probe metric used by the q0 soft task."""
    data.qpos[:] = build_model_qpos(
        model,
        root_pos=root_pos,
        root_quat_wxyz=root_quat_wxyz,
        joint_qpos=joint_qpos,
        joint_names=joint_names,
    )
    mujoco.mj_forward(model, data)
    return float(
        np.min(
            _shoulder_probe_clearances(
                model, data, probes, sample_start, sample_count
            )
        )
    )


class _ShoulderClearanceTask:
    """Soft hinge cost for visual upper-arm clearance from the torso.

    The residual uses the elbow-side shoulder-capsule samples from
    :func:`_shoulder_probe_clearances`, rather than the capsule's minimum
    ``mj_geomDistance``.  This avoids penalising the unavoidable shoulder
    attachment and makes a 6 cm threshold physically interpretable.  The
    finite-difference Jacobian is used only for frame 0.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        probes: list[tuple[int, int, list[int]]],
        arm_dof_indices: Iterable[int],
        threshold_m: float,
        cost: float,
        finite_difference_eps: float,
        sample_start: float,
        sample_count: int,
    ) -> None:
        self.model = model
        self.probes = list(probes)
        self.arm_dof_indices = tuple(int(index) for index in arm_dof_indices)
        self.threshold_m = float(threshold_m)
        self.cost = float(cost)
        self.finite_difference_eps = float(finite_difference_eps)
        self.sample_start = float(sample_start)
        self.sample_count = int(sample_count)
        self._data = mujoco.MjData(model)
        self._q_plus = np.empty(model.nq, dtype=np.float64)
        self._q_minus = np.empty(model.nq, dtype=np.float64)
        self._velocity = np.zeros(model.nv, dtype=np.float64)

    def _clearances(self, qpos: np.ndarray) -> np.ndarray:
        self._data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self._data)
        return _shoulder_probe_clearances(
            self.model,
            self._data,
            self.probes,
            self.sample_start,
            self.sample_count,
        )

    def compute_qp_residual(self, configuration: mink.Configuration):
        if self.cost <= 0.0:
            return (
                np.zeros((0, self.model.nv), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
                0.0,
            )
        qpos = np.asarray(configuration.q, dtype=np.float64)
        rows: list[np.ndarray] = []
        errors: list[float] = []
        distances = _shoulder_probe_clearances(
            self.model,
            configuration.data,
            self.probes,
            self.sample_start,
            self.sample_count,
        )
        for probe_index, distance in enumerate(distances):
            distance = float(distance)
            if not np.isfinite(distance) or distance >= self.threshold_m:
                continue
            error = self.threshold_m - distance
            jacobian = np.zeros(self.model.nv, dtype=np.float64)
            for dof_index in self.arm_dof_indices:
                self._velocity.fill(0.0)
                self._velocity[dof_index] = 1.0
                # Integrate from the current qpos for the two-sided derivative.
                self._q_plus[:] = qpos
                self._q_minus[:] = qpos
                mujoco.mj_integratePos(
                    self.model,
                    self._q_plus,
                    self._velocity,
                    self.finite_difference_eps,
                )
                mujoco.mj_integratePos(
                    self.model,
                    self._q_minus,
                    self._velocity,
                    -self.finite_difference_eps,
                )
                plus = self._clearances(self._q_plus)[probe_index]
                minus = self._clearances(self._q_minus)[probe_index]
                jacobian[dof_index] = (plus - minus) / (
                    2.0 * self.finite_difference_eps
                )
            # The task error is threshold - distance, so its Jacobian is
            # minus the distance Jacobian and the desired residual is -error.
            rows.append(-jacobian)
            errors.append(error)
        self._data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self._data)
        if not rows:
            return (
                np.zeros((0, self.model.nv), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
                0.0,
            )
        weighted_jacobian = self.cost * np.asarray(rows, dtype=np.float64)
        weighted_error = self.cost * (-np.asarray(errors, dtype=np.float64))
        return weighted_jacobian, weighted_error, 0.0


class _PostureWithClearanceTask:
    """Delegate the normal posture task and append one extra residual task."""

    def __init__(self, posture_task: Any, clearance_task: _ShoulderClearanceTask):
        self.posture_task = posture_task
        self.clearance_task = clearance_task

    def set_target_from_configuration(self, configuration: mink.Configuration) -> None:
        self.posture_task.set_target_from_configuration(configuration)

    def compute_qp_residual(self, configuration: mink.Configuration):
        posture = self.posture_task.compute_qp_residual(configuration)
        clearance = self.clearance_task.compute_qp_residual(configuration)
        return (
            np.vstack((posture[0], clearance[0])),
            np.concatenate((posture[1], clearance[1])),
            float(posture[2]) + float(clearance[2]),
        )

    def compute_qp_objective(self, configuration: mink.Configuration):
        """Combine the clearance residual with Mink 0.0.x posture QP."""
        posture = self.posture_task.compute_qp_objective(configuration)
        jacobian, error, damping = self.clearance_task.compute_qp_residual(
            configuration
        )
        if jacobian.shape[0] == 0:
            return posture
        clearance_h = jacobian.T @ jacobian
        clearance_h += float(damping) * np.eye(configuration.model.nv)
        clearance_c = -error.T @ jacobian
        return posture._replace(
            H=posture.H + clearance_h,
            c=posture.c + clearance_c,
        )


def _model_indices(
    model: mujoco.MjModel,
    joint_names: list[str],
    body_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    joint_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names],
        dtype=np.int32,
    )
    body_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names],
        dtype=np.int32,
    )
    missing_joints = [name for name, index in zip(joint_names, joint_ids) if index < 0]
    missing_bodies = [name for name, index in zip(body_names, body_ids) if index < 0]
    if missing_joints or missing_bodies:
        raise ValueError(
            f"model name mismatch: missing_joints={missing_joints}, "
            f"missing_bodies={missing_bodies}"
        )
    if any(
        int(model.jnt_type[index]) != int(mujoco.mjtJoint.mjJNT_HINGE)
        for index in joint_ids
    ):
        raise ValueError("all logged joints must be one-DoF hinge joints")
    return (
        joint_ids,
        np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int32),
        np.asarray(model.jnt_dofadr[joint_ids], dtype=np.int32),
        body_ids,
    )


def _finite_difference_arm_positions(
    joint_pos: np.ndarray,
    arm_indices: np.ndarray,
    fps: float,
) -> np.ndarray:
    result = np.zeros_like(joint_pos, dtype=np.float64)
    if joint_pos.shape[0] <= 1:
        return result
    arm = np.unwrap(np.asarray(joint_pos[:, arm_indices], dtype=np.float64), axis=0)
    result[0, arm_indices] = (arm[1] - arm[0]) * fps
    result[-1, arm_indices] = (arm[-1] - arm[-2]) * fps
    if arm.shape[0] > 2:
        result[1:-1, arm_indices] = 0.5 * (arm[2:] - arm[:-2]) * fps
    return result


def _reconstruct_base_qvel(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    root_body_id: int,
    target_linear_w: np.ndarray,
    target_angular_w: np.ndarray,
) -> np.ndarray:
    jacobian_pos = np.zeros((3, model.nv), dtype=np.float64)
    jacobian_rot = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacBodyCom(
        model, data, jacobian_pos, jacobian_rot, int(root_body_id)
    )
    jacobian = np.concatenate((jacobian_pos, jacobian_rot), axis=0)
    target = np.concatenate((target_linear_w, target_angular_w), axis=0)
    residual = target - jacobian @ data.qvel
    try:
        base_qvel = np.linalg.solve(jacobian[:, :6], residual)
    except np.linalg.LinAlgError as error:
        raise RuntimeError("pelvis free-joint Jacobian is singular") from error
    if not np.all(np.isfinite(base_qvel)):
        raise RuntimeError("reconstructed free-joint velocity is non-finite")
    return base_qvel


def refine(args: argparse.Namespace) -> dict[str, Any]:
    raw_path = args.raw_kinematics.expanduser().resolve()
    raw_meta_path = args.raw_metadata.expanduser().resolve()
    reference_path = args.reference.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = _load_npz(raw_path)
    reference = _load_npz(reference_path)
    raw_meta = _load_json(raw_meta_path)
    joint_names = _strings(raw_meta.get("joint_names", []))
    body_names = _strings(raw_meta.get("body_names", []))
    reference_body_names = _strings(reference.get("body_names", []))
    if len(joint_names) != 29 or len(body_names) != 30:
        raise ValueError(
            f"expected 29 joints and 30 bodies, got {len(joint_names)}/{len(body_names)}"
        )
    if len(set(joint_names)) != 29 or len(set(body_names)) != 30:
        raise ValueError("joint/body metadata contains duplicate names")
    if body_names[0] != ROOT_BODY:
        raise ValueError(f"body order must begin with {ROOT_BODY!r}")
    missing_arms = sorted(set(ARMS_ONLY) - set(joint_names))
    if missing_arms:
        raise ValueError(f"raw trajectory is missing arm joints: {missing_arms}")
    for wrist in (LEFT_WRIST, RIGHT_WRIST):
        if wrist not in body_names or wrist not in reference_body_names:
            raise ValueError(f"missing wrist body {wrist!r}")

    required_raw = (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "fps",
    )
    missing = [name for name in required_raw if name not in raw]
    if missing:
        raise ValueError(f"raw kinematics is missing fields: {missing}")
    fps = float(np.asarray(raw["fps"]).reshape(()))
    if not np.isclose(fps, 50.0):
        raise ValueError(f"expected 50 Hz kinematics, got {fps:g} Hz")
    frames = int(np.asarray(raw["joint_pos"]).shape[0])
    if args.max_frames is not None:
        frames = min(frames, int(args.max_frames))
    if frames <= 0:
        raise ValueError("no frames selected")

    raw_joint_pos = np.asarray(raw["joint_pos"][:frames], dtype=np.float64)
    raw_joint_vel = np.asarray(raw["joint_vel"][:frames], dtype=np.float64)
    raw_body_pos = np.asarray(raw["body_pos_w"][:frames], dtype=np.float64)
    raw_body_quat = _normalise_quaternions(
        raw["body_quat_w"][:frames], "raw body quaternion"
    )
    raw_body_lin = np.asarray(raw["body_lin_vel_w"][:frames], dtype=np.float64)
    raw_body_ang = np.asarray(raw["body_ang_vel_w"][:frames], dtype=np.float64)
    expected_shapes = {
        "joint_pos": (frames, 29),
        "joint_vel": (frames, 29),
        "body_pos_w": (frames, 30, 3),
        "body_quat_w": (frames, 30, 4),
        "body_lin_vel_w": (frames, 30, 3),
        "body_ang_vel_w": (frames, 30, 3),
    }
    actual = {
        "joint_pos": raw_joint_pos.shape,
        "joint_vel": raw_joint_vel.shape,
        "body_pos_w": raw_body_pos.shape,
        "body_quat_w": raw_body_quat.shape,
        "body_lin_vel_w": raw_body_lin.shape,
        "body_ang_vel_w": raw_body_ang.shape,
    }
    mismatched = {
        name: (actual[name], shape)
        for name, shape in expected_shapes.items()
        if actual[name] != shape
    }
    if mismatched:
        raise ValueError(f"raw field shape mismatch: {mismatched}")
    for name, values in (
        ("joint_pos", raw_joint_pos),
        ("joint_vel", raw_joint_vel),
        ("body_pos_w", raw_body_pos),
        ("body_lin_vel_w", raw_body_lin),
        ("body_ang_vel_w", raw_body_ang),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"raw field {name} contains non-finite values")

    ref_pos_all = np.asarray(reference["body_pos_w"][:frames], dtype=np.float64)
    ref_quat_all = _normalise_quaternions(
        reference["body_quat_w"][:frames], "reference body quaternion"
    )
    if ref_pos_all.shape[0] != frames or ref_quat_all.shape[0] != frames:
        raise ValueError("reference has fewer frames than raw kinematics")
    raw_wrist_indices = np.asarray(
        [body_names.index(LEFT_WRIST), body_names.index(RIGHT_WRIST)], dtype=np.int32
    )
    ref_wrist_indices = np.asarray(
        [reference_body_names.index(LEFT_WRIST), reference_body_names.index(RIGHT_WRIST)],
        dtype=np.int32,
    )
    target_pos = ref_pos_all[:, ref_wrist_indices]
    target_quat = ref_quat_all[:, ref_wrist_indices]

    model = mujoco.MjModel.from_xml_path(str(model_path))
    if (model.nq, model.nv) != (36, 35):
        raise ValueError(
            f"expected the body-only MPC model nq=36,nv=35, got {model.nq},{model.nv}"
        )
    joint_ids, qpos_addresses, qvel_addresses, body_ids = _model_indices(
        model, joint_names, body_names
    )
    del joint_ids
    geom_pairs, shoulder_probes, shoulder_obstacles = _collision_layout(model, args.collision_layout)
    shoulder_probe_ids = _build_shoulder_probe_ids(model, shoulder_probes, shoulder_obstacles)
    root_body_id = int(body_ids[body_names.index(ROOT_BODY)])
    arm_indices = np.asarray([joint_names.index(name) for name in ARMS_ONLY], dtype=np.int32)
    non_arm_indices = np.asarray(
        [index for index in range(len(joint_names)) if index not in set(arm_indices)],
        dtype=np.int32,
    )
    arm_body_mask = np.asarray(
        [
            name.startswith(("left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"))
            for name in body_names
        ],
        dtype=bool,
    )
    non_arm_body_mask = ~arm_body_mask

    ik_dt = 1.0 / fps if args.ik_dt is None else float(args.ik_dt)
    max_arm_delta = (
        float(args.arm_velocity_limit) / fps
        if args.max_arm_delta is None
        else float(args.max_arm_delta)
    )
    solver = MinkDualWristIK(
        model,
        left_frame=LEFT_WRIST,
        right_frame=RIGHT_WRIST,
        orientation_cost=float(args.orientation_cost),
        posture_cost=float(args.posture_cost),
        dt=ik_dt,
        max_iters=int(args.ik_iters),
        pos_tol_m=float(args.position_tolerance),
        ori_tol_rad=float(args.orientation_tolerance),
        use_collision=False,
        arm_vel_limit=float(args.arm_velocity_limit),
        max_arm_delta_rad=max_arm_delta,
        joint_names=list(ARMS_ONLY),
    )
    collision_pairs = [
        (list(left), list(right)) for left, right in geom_pairs
    ]
    q0_collision_pairs = list(collision_pairs)
    # Do not add shoulder-vs-torso capsule pairs to CollisionAvoidanceLimit:
    # MuJoCo/Mink filters parent-child pairs, and the proximal shoulder contact
    # is an intentional attachment.  The dedicated probe task below is the
    # actual q0 clearance contract.
    for names in collision_pairs:
        for group in names:
            missing_geoms = [
                name
                for name in group
                if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) < 0
            ]
            if missing_geoms:
                raise ValueError(f"collision geometry missing from exact MPC model: {missing_geoms}")
    if not args.no_collision:
        solver.limits.append(
            mink.CollisionAvoidanceLimit(
                model,
                collision_pairs,
                gain=0.85,
                minimum_distance_from_collisions=float(args.collision_margin),
                collision_detection_distance=max(0.08, float(args.collision_margin)),
            )
        )

    # The artificial all-zero arm posture is only a seed for frame 0.  It is
    # not a video frame and must not impose a temporal velocity/delta penalty
    # on the q0 solve.  Keep a separate solver instance so the normal solver,
    # used from frame 1 onward, remains byte-for-byte configured as before.
    q0_solver = None
    q0_collision_margin = None
    q0_clearance_cost_enabled = False
    if args.q0_zero_arm_no_smoothness:
        q0_solver = MinkDualWristIK(
            model,
            left_frame=LEFT_WRIST,
            right_frame=RIGHT_WRIST,
            orientation_cost=float(args.q0_orientation_cost),
            posture_cost=float(args.q0_posture_cost),
            dt=ik_dt,
            max_iters=int(args.ik_iters),
            pos_tol_m=float(args.position_tolerance),
            ori_tol_rad=float(args.orientation_tolerance),
            use_collision=False,
            # A very high velocity ceiling removes the temporal velocity
            # constraint for this one-shot initialization only.  The explicit
            # q0 delta/amplitude bound and MuJoCo joint ranges remain active.
            arm_vel_limit=float(args.q0_arm_velocity_limit),
            max_arm_delta_rad=float(args.q0_max_arm_delta),
            joint_names=list(ARMS_ONLY),
        )
        if not args.no_collision:
            q0_collision_margin = (
                float(args.q0_collision_margin)
                if args.q0_collision_margin is not None
                else float(args.collision_margin)
            )
            q0_solver.limits.append(
                mink.CollisionAvoidanceLimit(
                    model,
                    q0_collision_pairs,
                    gain=0.85,
                    minimum_distance_from_collisions=q0_collision_margin,
                    collision_detection_distance=max(0.08, q0_collision_margin),
                )
            )
        else:
            q0_collision_margin = None

        clearance_enabled = bool(
            args.q0_shoulder_clearance or args.q0_shoulder_clearance_cost > 0.0
        )
        if clearance_enabled and args.q0_shoulder_clearance_cost > 0.0:
            clearance_task = _ShoulderClearanceTask(
                model=model,
                probes=shoulder_probe_ids,
                arm_dof_indices=sorted(q0_solver._arm_dof_indices),
                threshold_m=float(args.q0_shoulder_clearance_threshold),
                cost=float(args.q0_shoulder_clearance_cost),
                finite_difference_eps=float(args.q0_shoulder_clearance_fd_eps),
                sample_start=float(args.q0_shoulder_clearance_sample_start),
                sample_count=int(args.q0_shoulder_clearance_sample_count),
            )
            q0_solver.posture_task = _PostureWithClearanceTask(
                q0_solver.posture_task, clearance_task
            )
            q0_clearance_cost_enabled = True

    refined_joint_pos = raw_joint_pos.copy()
    iterations = np.zeros(frames, dtype=np.int32)
    solver_success = np.zeros(frames, dtype=np.int8)
    solver_max_delta = np.zeros(frames, dtype=np.float32)
    solver_pos = np.zeros((frames, 2), dtype=np.float32)
    solver_ori = np.zeros((frames, 2), dtype=np.float32)
    previous_arms: np.ndarray | None = None
    q0_branch_source = "disabled"
    q0_branch_fallback_used = False
    q0_candidate_data = mujoco.MjData(model)
    for frame in range(frames):
        seed = raw_joint_pos[frame].copy()
        active_solver = solver
        if frame == 0 and q0_solver is not None:
            # Kimodo/JAX supplies the torso and lower body; only arm joints
            # are reset to zero for this diagnostic initialization solve.
            seed[arm_indices] = 0.0
            active_solver = q0_solver
        elif previous_arms is not None:
            seed[arm_indices] = previous_arms
        seed_qpos = build_model_qpos(
            model,
            root_pos=raw_body_pos[frame, 0],
            root_quat_wxyz=raw_body_quat[frame, 0],
            joint_qpos=seed,
            joint_names=joint_names,
        )
        result = active_solver.solve(
            seed_qpos=seed_qpos,
            left_pos=target_pos[frame, 0],
            left_quat_wxyz=target_quat[frame, 0],
            right_pos=target_pos[frame, 1],
            right_quat_wxyz=target_quat[frame, 1],
            joint_names=list(ARMS_ONLY),
        )
        if frame == 0 and q0_solver is not None:
            q0_branch_source = "zero_arm_seed"
            if args.q0_open_branch_fallback:
                solved_probe = extract_joint_qpos(model, result.qpos, joint_names)
                zero_clearance = _minimum_shoulder_probe_clearance(
                    model,
                    q0_candidate_data,
                    joint_names,
                    raw_body_pos[frame, 0],
                    raw_body_quat[frame, 0],
                    solved_probe,
                    shoulder_probe_ids,
                    float(args.q0_shoulder_clearance_sample_start),
                    int(args.q0_shoulder_clearance_sample_count),
                )
                if zero_clearance < float(args.q0_min_shoulder_clearance):
                    # This is a static branch retry, not a temporal warm start:
                    # the raw frame-0 arm is the Kimodo/JAX physical branch.
                    alternate_seed = raw_joint_pos[frame].copy()
                    alternate_qpos = build_model_qpos(
                        model,
                        root_pos=raw_body_pos[frame, 0],
                        root_quat_wxyz=raw_body_quat[frame, 0],
                        joint_qpos=alternate_seed,
                        joint_names=joint_names,
                    )
                    alternate_result = q0_solver.solve(
                        seed_qpos=alternate_qpos,
                        left_pos=target_pos[frame, 0],
                        left_quat_wxyz=target_quat[frame, 0],
                        right_pos=target_pos[frame, 1],
                        right_quat_wxyz=target_quat[frame, 1],
                        joint_names=list(ARMS_ONLY),
                    )
                    alternate_solution = extract_joint_qpos(
                        model, alternate_result.qpos, joint_names
                    )
                    alternate_clearance = _minimum_shoulder_probe_clearance(
                        model,
                        q0_candidate_data,
                        joint_names,
                        raw_body_pos[frame, 0],
                        raw_body_quat[frame, 0],
                        alternate_solution,
                        shoulder_probe_ids,
                        float(args.q0_shoulder_clearance_sample_start),
                        int(args.q0_shoulder_clearance_sample_count),
                    )
                    alternate_ok = (
                        max(
                            alternate_result.left_pos_err_m,
                            alternate_result.right_pos_err_m,
                        )
                        <= max(0.05, float(args.position_tolerance) * 5.0)
                        and max(
                            alternate_result.left_ori_err_rad,
                            alternate_result.right_ori_err_rad,
                        )
                        <= max(0.5, float(args.orientation_tolerance) * 2.5)
                    )
                    if alternate_ok and alternate_clearance > zero_clearance:
                        result = alternate_result
                        q0_branch_source = "raw_mpc_static_branch_fallback"
                        q0_branch_fallback_used = True
        solved = extract_joint_qpos(model, result.qpos, joint_names)
        solved[non_arm_indices] = raw_joint_pos[frame, non_arm_indices]
        if not np.all(np.isfinite(solved)):
            raise RuntimeError(f"Mink returned a non-finite solution at frame {frame}")
        refined_joint_pos[frame] = solved
        previous_arms = solved[arm_indices].copy()
        iterations[frame] = int(result.n_iters)
        solver_success[frame] = np.int8(bool(result.success))
        solver_max_delta[frame] = float(result.max_arm_delta_rad)
        solver_pos[frame] = (result.left_pos_err_m, result.right_pos_err_m)
        solver_ori[frame] = (result.left_ori_err_rad, result.right_ori_err_rad)
        if frame == 0 or (frame + 1) % 100 == 0 or frame + 1 == frames:
            print(
                f"[mink-final] frame={frame + 1}/{frames} "
                f"pos_mm={1e3 * float(np.mean(solver_pos[frame])):.2f} "
                f"success={bool(solver_success[frame])}",
                flush=True,
            )

    arm_fd = _finite_difference_arm_positions(refined_joint_pos, arm_indices, fps)
    refined_joint_vel = raw_joint_vel.copy()
    refined_joint_vel[:, arm_indices] = arm_fd[:, arm_indices]
    body_pos = np.empty_like(raw_body_pos, dtype=np.float32)
    body_quat = np.empty_like(raw_body_quat, dtype=np.float32)
    body_lin = np.empty_like(raw_body_lin, dtype=np.float32)
    body_ang = np.empty_like(raw_body_ang, dtype=np.float32)
    post_pos = np.empty((frames, 2), dtype=np.float32)
    post_ori = np.empty((frames, 2), dtype=np.float32)
    collision_distance = np.empty(frames, dtype=np.float32)
    shoulder_clearance_distance = np.empty(frames, dtype=np.float32)
    shoulder_clearance_per_side = np.empty((frames, 2), dtype=np.float32)
    base_velocity_error = 0.0
    data = mujoco.MjData(model)
    object_velocity = np.empty(6, dtype=np.float64)
    expanded_pairs = _expand_geom_pairs(model, geom_pairs)

    for frame in range(frames):
        data.qpos[:] = model.qpos0
        data.qvel[:] = 0.0
        data.qpos[:3] = raw_body_pos[frame, 0]
        data.qpos[3:7] = raw_body_quat[frame, 0]
        data.qpos[qpos_addresses] = refined_joint_pos[frame]
        data.qvel[qvel_addresses] = refined_joint_vel[frame]
        mujoco.mj_forward(model, data)
        data.qvel[:6] = _reconstruct_base_qvel(
            model,
            data,
            root_body_id,
            raw_body_lin[frame, 0],
            raw_body_ang[frame, 0],
        )
        mujoco.mj_forward(model, data)
        body_pos[frame] = data.xpos[body_ids]
        body_quat[frame] = data.xquat[body_ids]
        for body_index, body_id in enumerate(body_ids):
            mujoco.mj_objectVelocity(
                model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                int(body_id),
                object_velocity,
                0,
            )
            body_ang[frame, body_index] = object_velocity[:3]
            body_lin[frame, body_index] = object_velocity[3:]
        reconstructed_root = np.concatenate((body_lin[frame, 0], body_ang[frame, 0]))
        requested_root = np.concatenate((raw_body_lin[frame, 0], raw_body_ang[frame, 0]))
        base_velocity_error = max(
            base_velocity_error,
            float(np.max(np.abs(reconstructed_root - requested_root))),
        )
        post_pos[frame] = np.linalg.norm(
            body_pos[frame, raw_wrist_indices] - target_pos[frame], axis=-1
        )
        post_ori[frame] = _quaternion_error(
            body_quat[frame, raw_wrist_indices], target_quat[frame]
        )
        collision_distance[frame] = min(
            mujoco.mj_geomDistance(model, data, left, right, 1.0, None)
            for left, right in expanded_pairs
        )
        shoulder_clearance_per_side[frame] = _shoulder_probe_clearances(
            model,
            data,
            shoulder_probe_ids,
            float(args.q0_shoulder_clearance_sample_start),
            int(args.q0_shoulder_clearance_sample_count),
        )
        shoulder_clearance_distance[frame] = float(
            np.min(shoulder_clearance_per_side[frame])
        )

    pre_pos = np.linalg.norm(
        raw_body_pos[:, raw_wrist_indices] - target_pos, axis=-1
    ).astype(np.float32)
    pre_ori = _quaternion_error(
        raw_body_quat[:, raw_wrist_indices], target_quat
    ).astype(np.float32)
    non_arm_joint_position_error = float(
        np.max(
            np.abs(
                refined_joint_pos[:, non_arm_indices]
                - raw_joint_pos[:, non_arm_indices]
            )
        )
    )
    non_arm_joint_velocity_error = float(
        np.max(
            np.abs(
                refined_joint_vel[:, non_arm_indices]
                - raw_joint_vel[:, non_arm_indices]
            )
        )
    )
    non_arm_body_position_error = float(
        np.max(
            np.abs(
                body_pos[:, non_arm_body_mask]
                - raw_body_pos[:, non_arm_body_mask]
            )
        )
    )
    non_arm_body_orientation_error = float(
        np.max(
            _quaternion_error(
                body_quat[:, non_arm_body_mask], raw_body_quat[:, non_arm_body_mask]
            )
        )
    )
    if frames > 1:
        interframe_delta = np.abs(
            np.diff(np.unwrap(refined_joint_pos[:, arm_indices], axis=0), axis=0)
        )
        max_interframe_delta = float(np.max(interframe_delta))
        max_arm_velocity = float(np.max(np.abs(refined_joint_vel[:, arm_indices])))
    else:
        max_interframe_delta = 0.0
        max_arm_velocity = 0.0
    arm_joint_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARMS_ONLY],
        dtype=np.int32,
    )
    ranges = np.asarray(model.jnt_range[arm_joint_ids], dtype=np.float64)
    limited = np.asarray(model.jnt_limited[arm_joint_ids], dtype=bool)
    arm_values = refined_joint_pos[:, arm_indices]
    violations = np.maximum(
        ranges[:, 0][None] - arm_values,
        arm_values - ranges[:, 1][None],
    )
    max_joint_limit_violation = float(
        max(0.0, np.max(np.where(limited[None], violations, -np.inf)))
    )
    wrist_pre_mean = float(np.mean(pre_pos))
    wrist_post_mean = float(np.mean(post_pos))
    minimum_collision_distance = float(np.min(collision_distance))
    minimum_shoulder_clearance = float(np.min(shoulder_clearance_distance))
    gates = {
        "all_output_values_finite": bool(
            all(
                np.all(np.isfinite(values))
                for values in (
                    refined_joint_pos,
                    refined_joint_vel,
                    body_pos,
                    body_quat,
                    body_lin,
                    body_ang,
                    post_pos,
                    post_ori,
                    collision_distance,
                    shoulder_clearance_distance,
                )
            )
        ),
        "non_arm_joint_positions_exact": non_arm_joint_position_error == 0.0,
        "non_arm_joint_velocities_exact": non_arm_joint_velocity_error == 0.0,
        "non_arm_body_pose_preserved": (
            non_arm_body_position_error <= 2.0e-6
            and non_arm_body_orientation_error <= 2.0e-6
        ),
        "base_velocity_reconstructed": base_velocity_error <= 2.0e-6,
        "arm_joint_limits_satisfied": max_joint_limit_violation <= 1.0e-6,
        "arm_interframe_delta_bounded": max_interframe_delta <= max_arm_delta + 2.0e-5,
        "arm_velocity_bounded": max_arm_velocity <= float(args.arm_velocity_limit) + 2.0e-4,
        "collision_clearance_at_margin": (
            bool(args.no_collision)
            or minimum_collision_distance >= float(args.collision_margin) - 1.0e-6
        ),
        "wrist_position_mean_under_5cm": wrist_post_mean < 0.05,
        "wrist_position_max_under_5cm": float(np.max(post_pos)) < 0.05,
        "wrist_position_improved": wrist_post_mean < wrist_pre_mean,
        "wrist_orientation_mean_improved": float(np.mean(post_ori))
        < float(np.mean(pre_ori)),
    }
    accepted = all(gates.values())

    timestamps = np.arange(frames, dtype=np.float64) / fps
    output_arrays = {
        "joint_pos": refined_joint_pos.astype(np.float32),
        "joint_vel": refined_joint_vel.astype(np.float32),
        "root_pos_w": body_pos[:, 0].copy(),
        "root_quat_wxyz": body_quat[:, 0].copy(),
        "root_lin_vel_w": body_lin[:, 0].copy(),
        "root_ang_vel_w": body_ang[:, 0].copy(),
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": body_lin,
        "body_ang_vel_w": body_ang,
        "joint_names": np.asarray(joint_names),
        "body_names": np.asarray(body_names),
        "timestamps": timestamps,
        "fps": np.asarray(int(round(fps)), dtype=np.int64),
        "ik_pre_wrist_position_m": pre_pos,
        "ik_post_wrist_position_m": post_pos,
        "ik_pre_wrist_orientation_rad": pre_ori,
        "ik_post_wrist_orientation_rad": post_ori,
        "ik_solver_position_m": solver_pos,
        "ik_solver_orientation_rad": solver_ori,
        "ik_solver_success": solver_success,
        "ik_iterations": iterations,
        "ik_solver_max_delta_rad": solver_max_delta,
        "ik_min_collision_distance_m": collision_distance,
        "ik_shoulder_clearance_distance_m": shoulder_clearance_distance,
        "ik_left_shoulder_clearance_distance_m": shoulder_clearance_per_side[:, 0],
        "ik_right_shoulder_clearance_distance_m": shoulder_clearance_per_side[:, 1],
        "offline_mink_arms_ik": np.asarray(1, dtype=np.int32),
        "source_raw_kinematics": np.asarray(str(raw_path)),
        "source_reference": np.asarray(str(reference_path)),
    }
    output_path = output_dir / "kinematics.npz"
    summary_path = output_dir / "mink_ik_summary.json"
    metadata_path = output_dir / "meta.json"
    errors_path = output_dir / "mink_ik_errors.csv"
    _atomic_npz(output_path, output_arrays)

    per_link = {}
    for wrist_index, wrist_name in enumerate((LEFT_WRIST, RIGHT_WRIST)):
        per_link[wrist_name] = {
            "position_m": {
                "pre": _stats(pre_pos[:, wrist_index]),
                "post": _stats(post_pos[:, wrist_index]),
            },
            "orientation_rad": {
                "pre": _stats(pre_ori[:, wrist_index]),
                "post": _stats(post_ori[:, wrist_index]),
            },
        }
    diagnostics = {
        "solver_success_rate": float(np.mean(solver_success)),
        "solver_success_note": (
            "Diagnostic convergence flag uses the solver's strict per-wrist "
            "position/orientation tolerances; output acceptance uses measured FK errors."
        ),
        "mean_iterations": float(np.mean(iterations)),
        "max_solver_delta_rad": float(np.max(solver_max_delta)),
        "max_interframe_arm_delta_rad": max_interframe_delta,
        "max_arm_velocity_rad_per_s": max_arm_velocity,
        "max_joint_limit_violation_rad": max_joint_limit_violation,
        "minimum_configured_collision_distance_m": minimum_collision_distance,
        "minimum_shoulder_clearance_m": minimum_shoulder_clearance,
        "minimum_left_shoulder_clearance_m": float(
            np.min(shoulder_clearance_per_side[:, 0])
        ),
        "minimum_right_shoulder_clearance_m": float(
            np.min(shoulder_clearance_per_side[:, 1])
        ),
        "q0_shoulder_clearance_m": float(shoulder_clearance_distance[0]),
        "q0_left_shoulder_clearance_m": float(shoulder_clearance_per_side[0, 0]),
        "q0_right_shoulder_clearance_m": float(shoulder_clearance_per_side[0, 1]),
        "q0_shoulder_clearance_target_met": (
            bool(
                shoulder_clearance_distance[0]
                >= float(args.q0_shoulder_clearance_threshold)
                - Q0_SHOULDER_CLEARANCE_TARGET_TOLERANCE_M
            )
            if args.q0_zero_arm_no_smoothness
            else None
        ),
        "q0_shoulder_clearance_target_tolerance_m": (
            Q0_SHOULDER_CLEARANCE_TARGET_TOLERANCE_M
            if args.q0_zero_arm_no_smoothness
            else None
        ),
        "max_non_arm_joint_position_change_rad": non_arm_joint_position_error,
        "max_non_arm_joint_velocity_change_rad_per_s": non_arm_joint_velocity_error,
        "max_non_arm_body_position_change_m": non_arm_body_position_error,
        "max_non_arm_body_orientation_change_rad": non_arm_body_orientation_error,
        "max_base_velocity_reconstruction_error": base_velocity_error,
    }
    summary = {
        "status": "completed_accepted" if accepted else "completed_failed_acceptance",
        "accepted": accepted,
        "stage": "final offline dual-arm Mink IK after three-point JAX MPC",
        "sources": {
            "raw_kinematics": {"path": str(raw_path), "sha256": _sha256(raw_path)},
            "raw_metadata": {"path": str(raw_meta_path), "sha256": _sha256(raw_meta_path)},
            "three_point_reference": {
                "path": str(reference_path),
                "sha256": _sha256(reference_path),
            },
            "mjx_model": {"path": str(model_path), "sha256": _sha256(model_path)},
        },
        "output": {
            "kinematics": str(output_path),
            "metadata": str(metadata_path),
            "per_frame_errors": str(errors_path),
            "frames": frames,
            "fps": fps,
        },
        "invariants": {
            "modified_joints": list(ARMS_ONLY),
            "fixed_state": "floating base, torso/waist, and both legs from raw MPC",
            "target": "left/right wrist 6D poses from torso_umi3_reference.npz",
            "warm_start": "previous refined arm state",
            "q0_zero_arm_no_smoothness": bool(args.q0_zero_arm_no_smoothness),
            "q0_branch_source": q0_branch_source,
            "q0_branch_fallback_used": q0_branch_fallback_used,
        },
        "solver": {
            "backend": "MinkDualWristIK/daqp",
            "ik_iters": int(args.ik_iters),
            "ik_dt_s": ik_dt,
            "position_cost": 1.0,
            "orientation_cost": float(args.orientation_cost),
            "posture_cost": float(args.posture_cost),
            "position_tolerance_m": float(args.position_tolerance),
            "orientation_tolerance_rad": float(args.orientation_tolerance),
            "arm_velocity_limit_rad_per_s": float(args.arm_velocity_limit),
            "max_arm_delta_rad_per_frame": max_arm_delta,
            "configuration_limit": True,
            "collision_avoidance": not args.no_collision,
            "collision_layout": args.collision_layout,
            "collision_margin_m": float(args.collision_margin),
            "collision_geom_pairs": [
                [list(left), list(right)] for left, right in geom_pairs
            ],
            "q0_initialization": {
                "enabled": bool(args.q0_zero_arm_no_smoothness),
                "arm_seed": "all_zero" if args.q0_zero_arm_no_smoothness else "raw_mpc",
                "temporal_smoothness": (
                    "disabled_for_frame_0_only"
                    if args.q0_zero_arm_no_smoothness
                    else "normal"
                ),
                "posture_cost": (
                    float(args.q0_posture_cost)
                    if args.q0_zero_arm_no_smoothness
                    else None
                ),
                "max_arm_delta_rad": (
                    float(args.q0_max_arm_delta)
                    if args.q0_zero_arm_no_smoothness
                    else None
                ),
                "arm_velocity_limit_rad_per_s": (
                    float(args.q0_arm_velocity_limit)
                    if args.q0_zero_arm_no_smoothness
                    else None
                ),
                "shoulder_clearance": bool(
                    args.q0_zero_arm_no_smoothness
                    and (
                        args.q0_shoulder_clearance
                        or args.q0_shoulder_clearance_cost > 0.0
                    )
                ),
                "collision_margin_m": q0_collision_margin,
                "shoulder_clearance_cost": (
                    float(args.q0_shoulder_clearance_cost)
                    if args.q0_zero_arm_no_smoothness
                    else 0.0
                ),
                "shoulder_clearance_cost_enabled": q0_clearance_cost_enabled,
                "shoulder_clearance_threshold_m": (
                    float(args.q0_shoulder_clearance_threshold)
                    if args.q0_zero_arm_no_smoothness
                    else None
                ),
                "shoulder_clearance_target_tolerance_m": (
                    Q0_SHOULDER_CLEARANCE_TARGET_TOLERANCE_M
                    if args.q0_zero_arm_no_smoothness
                    else None
                ),
                "shoulder_clearance_jacobian": (
                    "finite_difference_elbow_side_capsule_probe"
                    if q0_clearance_cost_enabled
                    else None
                ),
                "shoulder_clearance_geometry": {
                    "metric": (
                        "minimum_surface_distance_of_elbow_side_upper_arm_samples"
                    ),
                    "shoulder_probes": [
                        {"geom": geom, "elbow_body": body}
                        for geom, body in shoulder_probes
                    ],
                    "obstacles": list(shoulder_obstacles),
                    "sample_start_fraction": float(
                        args.q0_shoulder_clearance_sample_start
                    ),
                    "sample_count": int(args.q0_shoulder_clearance_sample_count),
                    "radii_subtracted": True,
                },
                "branch_fallback": {
                    "enabled": bool(args.q0_open_branch_fallback),
                    "minimum_shoulder_clearance_m": float(args.q0_min_shoulder_clearance),
                    "selected_source": q0_branch_source,
                },
            },
        },
        "wrist_tracking": {
            "combined_position_m": {
                "pre": _stats(pre_pos),
                "post": _stats(post_pos),
                "improvement_percent_mean": 100.0
                * (wrist_pre_mean - wrist_post_mean)
                / wrist_pre_mean,
            },
            "combined_orientation_rad": {
                "pre": _stats(pre_ori),
                "post": _stats(post_ori),
            },
            "per_link": per_link,
        },
        "diagnostics": diagnostics,
        "gates": gates,
    }
    metadata = dict(raw_meta)
    metadata.update(
        {
            "format_version": 2,
            "source": "final offline Mink-refined three-point JAX MPC kinematics",
            "joint_names": joint_names,
            "body_names": body_names,
            "dimensions": {"T": frames, "J": len(joint_names), "B": len(body_names)},
            "offline_refinement": summary,
            "coordinate_conventions": {
                **raw_meta.get("coordinate_conventions", {}),
                "body_velocity_point": "MuJoCo body center of mass",
                "body_velocity_source": "mj_objectVelocity(local=0), world axes",
            },
        }
    )
    rows = []
    for frame in range(frames):
        rows.append(
            {
                "frame": frame,
                "time_s": timestamps[frame],
                "pre_left_position_m": pre_pos[frame, 0],
                "pre_right_position_m": pre_pos[frame, 1],
                "post_left_position_m": post_pos[frame, 0],
                "post_right_position_m": post_pos[frame, 1],
                "pre_left_orientation_rad": pre_ori[frame, 0],
                "pre_right_orientation_rad": pre_ori[frame, 1],
                "post_left_orientation_rad": post_ori[frame, 0],
                "post_right_orientation_rad": post_ori[frame, 1],
                "solver_success": int(solver_success[frame]),
                "solver_iterations": int(iterations[frame]),
                "solver_max_delta_rad": solver_max_delta[frame],
                "min_collision_distance_m": collision_distance[frame],
                "shoulder_clearance_distance_m": shoulder_clearance_distance[frame],
                "left_shoulder_clearance_distance_m": shoulder_clearance_per_side[
                    frame, 0
                ],
                "right_shoulder_clearance_distance_m": shoulder_clearance_per_side[
                    frame, 1
                ],
            }
        )
    _atomic_json(metadata_path, metadata)
    _atomic_json(summary_path, summary)
    _atomic_csv(errors_path, rows)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-kinematics", type=Path, required=True)
    parser.add_argument("--raw-metadata", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--collision-layout", choices=("production", "hydrax-derived"), default=os.environ.get("MINK_COLLISION_LAYOUT", "hydrax-derived"))
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--ik-iters", type=int, default=40)
    parser.add_argument("--ik-dt", type=float)
    parser.add_argument("--orientation-cost", type=float, default=0.08)
    parser.add_argument("--posture-cost", type=float, default=0.12)
    parser.add_argument("--position-tolerance", type=float, default=0.01)
    parser.add_argument("--orientation-tolerance", type=float, default=0.2)
    parser.add_argument("--arm-velocity-limit", type=float, default=2.5)
    parser.add_argument("--max-arm-delta", type=float)
    parser.add_argument(
        "--q0-zero-arm-no-smoothness",
        action="store_true",
        help="For frame 0 only, seed all arm joints at zero and solve without the "
        "normal temporal velocity/delta constraint. Frames >=1 keep the default solver.",
    )
    parser.add_argument(
        "--q0-max-arm-delta",
        type=float,
        default=3.0,
        help="Maximum q0 arm displacement allowed by the one-shot solver (rad).",
    )
    parser.add_argument(
        "--q0-arm-velocity-limit",
        type=float,
        default=1.0e4,
        help="Numerical velocity ceiling for q0 only; does not affect later frames.",
    )
    parser.add_argument(
        "--q0-posture-cost",
        type=float,
        default=0.0,
        help="Optional q0 branch prior. Default 0 removes the zero-posture bias.",
    )
    parser.add_argument(
        "--q0-orientation-cost",
        type=float,
        default=0.08,
        help="Wrist orientation cost for the q0 one-shot solve.",
    )
    parser.add_argument(
        "--q0-shoulder-clearance",
        action="store_true",
        help="Enable the elbow-side upper-arm clearance geometry for q0. "
        "The soft cost is set with --q0-shoulder-clearance-cost.",
    )
    parser.add_argument(
        "--q0-collision-margin",
        type=float,
        help="Optional collision margin for q0 only; later frames keep --collision-margin.",
    )
    parser.add_argument(
        "--q0-shoulder-clearance-cost",
        type=float,
        default=0.0,
        help="Soft q0 hinge-cost weight for visual upper-arm clearance. "
        "The metric samples the elbow-side shoulder capsule and subtracts "
        "both capsule radii; zero disables the cost.",
    )
    parser.add_argument(
        "--q0-shoulder-clearance-threshold",
        type=float,
        default=0.06,
        help="q0 visual upper-arm surface-clearance threshold in meters.",
    )
    parser.add_argument(
        "--q0-shoulder-clearance-fd-eps",
        type=float,
        default=1.0e-4,
        help="Finite-difference step (radians) for the q0 clearance Jacobian.",
    )
    parser.add_argument(
        "--q0-shoulder-clearance-sample-start",
        type=float,
        default=DEFAULT_Q0_SHOULDER_SAMPLE_START,
        help="Fraction of the shoulder capsule centre line at which clearance "
        "sampling starts; the proximal attachment is skipped.",
    )
    parser.add_argument(
        "--q0-shoulder-clearance-sample-count",
        type=int,
        default=DEFAULT_Q0_SHOULDER_SAMPLE_COUNT,
        help="Number of elbow-side upper-arm clearance samples.",
    )
    parser.add_argument(
        "--q0-open-branch-fallback",
        action="store_true",
        help="Retry q0 from the Kimodo/JAX arm branch when the zero-seed solution "
        "has insufficient shoulder clearance.",
    )
    parser.add_argument(
        "--q0-min-shoulder-clearance",
        type=float,
        default=0.01,
        help="Clearance threshold that triggers the static q0 branch retry (m).",
    )
    parser.add_argument("--collision-margin", type=float, default=0.015)
    parser.add_argument("--no-collision", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ik_iters <= 0:
        raise ValueError("--ik-iters must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.arm_velocity_limit <= 0.0:
        raise ValueError("--arm-velocity-limit must be positive")
    if args.q0_max_arm_delta <= 0.0:
        raise ValueError("--q0-max-arm-delta must be positive")
    if args.q0_arm_velocity_limit <= 0.0:
        raise ValueError("--q0-arm-velocity-limit must be positive")
    if args.q0_posture_cost < 0.0 or args.q0_orientation_cost < 0.0:
        raise ValueError("q0 costs must be non-negative")
    if args.q0_collision_margin is not None and args.q0_collision_margin < 0.0:
        raise ValueError("--q0-collision-margin must be non-negative")
    if args.q0_shoulder_clearance_cost < 0.0:
        raise ValueError("--q0-shoulder-clearance-cost must be non-negative")
    if args.q0_shoulder_clearance_threshold <= 0.0:
        raise ValueError("--q0-shoulder-clearance-threshold must be positive")
    if args.q0_shoulder_clearance_fd_eps <= 0.0:
        raise ValueError("--q0-shoulder-clearance-fd-eps must be positive")
    if not 0.0 <= args.q0_shoulder_clearance_sample_start < 1.0:
        raise ValueError(
            "--q0-shoulder-clearance-sample-start must be in [0, 1)"
        )
    if args.q0_shoulder_clearance_sample_count < 2:
        raise ValueError("--q0-shoulder-clearance-sample-count must be >= 2")
    if args.q0_min_shoulder_clearance < 0.0:
        raise ValueError("--q0-min-shoulder-clearance must be non-negative")
    summary = refine(args)
    if not summary["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
