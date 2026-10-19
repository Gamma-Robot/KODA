#!/usr/bin/env python3
"""Replace a prepared dual-wrist reference with G1 + BrainCo palm-refined poses.

This is the ``HAND_MODE=g1_brainco`` counterpart of
``inject_contact_aware_wrist_reference.py``: the template-palm reference stays
the carrier and only the two ``*_wrist_yaw_link`` pose/velocity streams and the
matching Kimodo wrist targets are replaced.  The same pass emits the 12-DoF
BrainCo hand archive on the identical 50 Hz timeline, so wrist and fingers are
solved against one set of MANO landmarks.

The method is ported verbatim from Open-AoE ``phantom`` (``--robot g1_brainco``):

1.  A dex-retargeting ``position`` seed maps wrist-relative fingertips to the
    six active joints per hand.  A dummy 6-DoF free joint absorbs global SE(3),
    so the seed is frame-invariant.
2.  Palm poses are reconstructed from MANO wrist + MCP landmarks; the prior
    wrist rotation only disambiguates axis signs.
3.  ``scipy.optimize.least_squares`` refines the six active joints plus wrist
    translation against 16 MANO landmarks, with the wrist rotation frozen.

Every step is SE(3)-equivariant, so it runs directly in the ScaleTrack Z-up
world frame.  ``phantom``'s arm scaling, shoulder re-anchoring, and head-camera
alignment are deliberately not ported: they exist to reach the G1 base frame,
which this pipeline is already in.

Palm-axis signs are resolved against the MJCF local-z correction *before*
the causal continuity pass.  Continuity is restricted to frames whose
constructed axes are degenerate against that corrected reference, so a
slow interpolated ramp cannot latch a 180° branch.

Palm-axis signs are resolved against the MJCF local-z correction *before*
the causal continuity pass.  Continuity is restricted to frames whose
constructed axes are degenerate against that corrected reference, so a
slow interpolated ramp cannot latch a 180° branch.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retarget_template_palm_reference import (  # noqa: E402
    WRIST_BODIES,
    _angular_velocity_world,
    _atomic_json,
    _atomic_npz,
    _continuous_wxyz,
    _finite_difference,
    _load_npz,
    _load_world_points_zup,
    _resample_pose,
    _sha256,
    _stats,
    template_palm_kabsch,
)

logger = logging.getLogger("brainco-hand-retarget")


# ── Constants ported from phantom/constants/__init__.py ─────────────────────

CONF_THRESHOLD = 0.10
G1_STANDING_HEIGHT = 0.793

JOINT_REFINE_HAND_MAX_DELTA = 0.2

MANO_HAND_MAX_NFEV = 20
MANO_HAND_KEYPOINT_SCALE = 0.015
MANO_HAND_ORIENTATION_SCALE = 0.35
MANO_HAND_FIXED_WRIST_ORIENTATION_SCALE = 1.00
MANO_HAND_JOINT_TEMPORAL_SCALE = 0.15
MANO_HAND_WRIST_POSITION_TEMPORAL_SCALE = 0.02
MANO_HAND_WRIST_ROTATION_TEMPORAL_SCALE = 0.10
MANO_HAND_WRIST_POSITION_MAX_DELTA = 0.04
MANO_HAND_WRIST_ROTATION_MAX_DELTA = 0.20
MANO_HAND_SEED_JOINT_SCALE = 0.25
MANO_HAND_SEED_WRIST_POSITION_SCALE = 0.06
MANO_HAND_SEED_WRIST_ROTATION_SCALE = 0.30

HAND_KEYPOINT_VALIDATION_THRESHOLD_M = 0.05

LEFT_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# phantom/pipeline.py: the 16 MANO landmarks used as refinement targets.
MANO_HAND_KEYPOINT_INDICES = np.array(
    [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20],
    dtype=int,
)


# ── Constants ported from phantom/constants/g1_brainco.py ───────────────────

BRAINCO_LEFT_HAND_JOINT_NAMES = [
    "left_thumb_metacarpal_joint",
    "left_thumb_proximal_joint",
    "left_index_proximal_joint",
    "left_middle_proximal_joint",
    "left_ring_proximal_joint",
    "left_pinky_proximal_joint",
]
BRAINCO_RIGHT_HAND_JOINT_NAMES = [
    "right_thumb_metacarpal_joint",
    "right_thumb_proximal_joint",
    "right_index_proximal_joint",
    "right_middle_proximal_joint",
    "right_ring_proximal_joint",
    "right_pinky_proximal_joint",
]
BRAINCO_HAND_JOINT_NAMES = (
    BRAINCO_LEFT_HAND_JOINT_NAMES + BRAINCO_RIGHT_HAND_JOINT_NAMES
)
BRAINCO_ACTION_JOINT_NAMES = (
    LEFT_ARM_JOINT_NAMES
    + RIGHT_ARM_JOINT_NAMES
    + BRAINCO_LEFT_HAND_JOINT_NAMES
    + BRAINCO_RIGHT_HAND_JOINT_NAMES
)
BRAINCO_ACTION_DIM = len(BRAINCO_ACTION_JOINT_NAMES)

BRAINCO_MIMIC_RULES: list[tuple[str, str, float, float]] = []
for _prefix in ("left_", "right_"):
    BRAINCO_MIMIC_RULES.extend(
        [
            (f"{_prefix}thumb_distal_joint", f"{_prefix}thumb_proximal_joint", 1.0, 0.0),
            (f"{_prefix}index_distal_joint", f"{_prefix}index_proximal_joint", 1.155, 0.0),
            (f"{_prefix}middle_distal_joint", f"{_prefix}middle_proximal_joint", 1.155, 0.0),
            (f"{_prefix}ring_distal_joint", f"{_prefix}ring_proximal_joint", 1.155, 0.0),
            (f"{_prefix}pinky_distal_joint", f"{_prefix}pinky_proximal_joint", 1.155, 0.0),
        ]
    )

BRAINCO_LEFT_HAND_TIP_BODIES = (
    "left_thumb_tip_Link",
    "left_index_tip_Link",
    "left_middle_tip_Link",
    "left_ring_tip_Link",
    "left_pinky_tip_Link",
)
BRAINCO_RIGHT_HAND_TIP_BODIES = (
    "right_thumb_tip",
    "right_index_tip",
    "right_middle_tip",
    "right_ring_tip",
    "right_pinky_tip",
)

BRAINCO_WRIST_TARGET_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
BRAINCO_PALM_GEOM_MESH_NAMES = ("left_base_link", "right_base_link")

# The derived local correction must be a rotation about the palm z axis.
PALM_CORRECTION_AXIS_TOLERANCE = 0.10
PALM_CORRECTION_MIN_DEG = 45.0
PALM_CORRECTION_MAX_DEG = 135.0
# Per-frame absolute sign tests are trusted above this |dot|; below it the
# constructed axis is nearly orthogonal to the reference and continuity may
# inherit the previous sign.  0.3 is cos(72°).  A true 90° mounting mismatch
# sits near 0.0; a well-aligned BrainCo/Inspire palm sits near 1.0.
PALM_SIGN_ALIGN_MIN_ABS_DOT = 0.3
PALM_SIGN_ALIGN_MEDIAN_MIN_ABS_DOT = 0.5

HAND_CONTRACT = "dual_wrist_yaw_link_SE3_plus_g1_brainco_6DoF_per_hand"
WRIST_REFERENCE_METHOD = "g1_brainco_palm_refine"


# ── Palm-axis chain ported from phantom/aoe/transform.py ────────────────────


def _normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    out = np.broadcast_to(fallback, vector.shape).astype(np.float64).copy()
    ok = norm[..., 0] > 1e-8
    out[ok] = vector[ok] / norm[ok]
    return out


def _moving_average(data: np.ndarray, radius: int = 2) -> np.ndarray:
    if radius <= 0 or data.shape[0] <= 1:
        return data.copy()
    out = np.empty_like(data, dtype=np.float64)
    for index in range(data.shape[0]):
        low = max(0, index - radius)
        high = min(data.shape[0], index + radius + 1)
        out[index] = np.mean(data[low:high], axis=0)
    return out


def _orthonormalize_palm_axes(
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    ref_x: np.ndarray | None = None,
    ref_z: np.ndarray | None = None,
    normal_sign: float = 1.0,
) -> np.ndarray:
    z = _normalize(z_axis, np.array([1.0, 0.0, 0.0]))
    x = x_axis - z * np.sum(x_axis * z, axis=-1, keepdims=True)
    x = _normalize(x, np.array([0.0, 1.0, 0.0]))

    if ref_z is not None:
        flip = np.sum(z * ref_z, axis=-1) < 0.0
        z[flip] *= -1.0
    if ref_x is not None:
        flip = np.sum(x * ref_x, axis=-1) < 0.0
        x[flip] *= -1.0
    x *= float(normal_sign)

    y = np.cross(z, x)
    y = _normalize(y, np.array([0.0, 0.0, 1.0]))
    x = np.cross(y, z)
    x = _normalize(x, np.array([0.0, 1.0, 0.0]))

    poses = np.zeros((x.shape[0], 3, 3), dtype=np.float64)
    poses[:, :, 0] = x
    poses[:, :, 1] = y
    poses[:, :, 2] = z
    return poses


def _raw_palm_sign_confidence(
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    ref_x: np.ndarray,
    ref_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """|dot| of constructed axes against references, before sign matching."""
    z = _normalize(z_axis, np.array([1.0, 0.0, 0.0]))
    x = x_axis - z * np.sum(x_axis * z, axis=-1, keepdims=True)
    x = _normalize(x, np.array([0.0, 1.0, 0.0]))
    return (
        np.abs(np.sum(x * ref_x, axis=-1)),
        np.abs(np.sum(z * ref_z, axis=-1)),
    )


def _enforce_palm_normal_continuity(
    rotations: np.ndarray,
    degenerate: np.ndarray | None = None,
) -> np.ndarray:
    """Choose the palm-plane normal sign continuously over time.

    Unconditional causal continuity will follow a slow 180° ramp into the
    flipped branch and never return, even when each frame's absolute x/z
    sign is well determined.  When ``degenerate`` is provided, inherit the
    previous y-sign only on those frames; high-confidence frames keep the
    per-frame absolute sign from ``_orthonormalize_palm_axes``.
    """
    rotations = np.asarray(rotations, dtype=np.float64).copy()
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("rotations must have shape (T, 3, 3)")
    if degenerate is not None:
        degenerate = np.asarray(degenerate, dtype=bool)
        if degenerate.shape != (rotations.shape[0],):
            raise ValueError("degenerate must have shape (T,)")
    for index in range(1, rotations.shape[0]):
        if degenerate is not None and not bool(degenerate[index]):
            continue
        if float(np.dot(rotations[index - 1, :, 1], rotations[index, :, 1])) < 0.0:
            rotations[index, :, 0] *= -1.0
            rotations[index, :, 1] *= -1.0
    return rotations


def _compose_palm_correction(
    local_z_correction_deg: float,
    local_correction_matrix: np.ndarray | None,
) -> np.ndarray | None:
    correction = None
    if local_correction_matrix is not None:
        correction = np.asarray(local_correction_matrix, dtype=np.float64)
        if correction.shape != (3, 3):
            raise ValueError("local_correction_matrix must have shape (3, 3)")
    if local_z_correction_deg:
        angle = np.deg2rad(float(local_z_correction_deg))
        cosine, sine = np.cos(angle), np.sin(angle)
        local_rotation = np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        correction = (
            local_rotation if correction is None else correction @ local_rotation
        )
    if correction is None:
        return None
    if not np.allclose(correction.T @ correction, np.eye(3), atol=1e-6):
        raise ValueError("local correction must be orthonormal")
    if np.linalg.det(correction) <= 0.0:
        raise ValueError("local correction must be a proper rotation")
    return correction


def estimate_palm_poses_from_keypoints(
    keypoints_source: np.ndarray,
    wrist_poses: np.ndarray,
    wrist_to_palm: np.ndarray,
    smooth_radius: int = 2,
    local_z_correction_deg: float = 0.0,
    local_correction_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate target robot palm poses from MANO 21-point hand geometry.

    The palm center comes from MANO wrist plus MCP landmarks; the orientation
    is reconstructed from the distal direction and palm plane normal.  The
    prior wrist rotation is used only to resolve axis signs.

    The MJCF local-z correction (~90° for BrainCo/Inspire) is applied to the
    *reference* before those sign tests.  The constructed lateral axis is
    otherwise structurally orthogonal to the robot visual-palm x axis, so
    ``dot(x, ref_x)`` has no signal and a causal y-continuity pass can latch
    a 180° flip.  The angle itself is measured from the robot's neutral
    geometry in ``estimate_palm_local_correction``, not from any session.
    """
    keypoints_source = np.asarray(keypoints_source, dtype=np.float64)
    wrist_poses = np.asarray(wrist_poses, dtype=np.float64)
    wrist_to_palm = np.asarray(wrist_to_palm, dtype=np.float64)
    wrist_source = keypoints_source[:, 0, :]
    mcps_source = keypoints_source[:, (5, 9, 13, 17), :]
    mcp_mean_source = mcps_source.mean(axis=1)
    palm_center_source = 0.5 * (wrist_source + mcp_mean_source)

    palm_center = wrist_poses[:, :3, 3] + (palm_center_source - wrist_source)

    distal = mcp_mean_source - wrist_source
    lateral = mcps_source[:, 0] - mcps_source[:, 3]

    correction = _compose_palm_correction(
        local_z_correction_deg, local_correction_matrix
    )
    ref_palm_rot = np.einsum(
        "tij,jk->tik", wrist_poses[:, :3, :3], wrist_to_palm[:3, :3]
    )
    # ``correction`` later maps constructed axes into the robot visual palm.
    # Express the robot-palm reference in that same constructed frame so the
    # x-axis sign test is not comparing two axes that differ by a quarter turn.
    ref_for_sign = (
        np.einsum("tij,jk->tik", ref_palm_rot, correction)
        if correction is not None
        else ref_palm_rot
    )
    ref_x = ref_for_sign[:, :, 0]
    ref_z = ref_for_sign[:, :, 2]

    abs_x_dot, abs_z_dot = _raw_palm_sign_confidence(
        lateral, distal, ref_x, ref_z
    )
    if (
        correction is not None
        and float(np.median(abs_x_dot)) < PALM_SIGN_ALIGN_MEDIAN_MIN_ABS_DOT
    ):
        raise ValueError(
            "palm x-axis sign reference is still degenerate after applying "
            f"the MJCF local correction (median |dot|={float(np.median(abs_x_dot)):.3f}); "
            "the mounting convention or correction order is wrong"
        )
    degenerate = (abs_x_dot < PALM_SIGN_ALIGN_MIN_ABS_DOT) | (
        abs_z_dot < PALM_SIGN_ALIGN_MIN_ABS_DOT
    )

    palm_rot = _orthonormalize_palm_axes(
        lateral, distal, ref_x, ref_z, normal_sign=-1.0
    )
    palm_rot = _enforce_palm_normal_continuity(palm_rot, degenerate=degenerate)

    if smooth_radius > 0:
        palm_center = _moving_average(palm_center, smooth_radius)
        x = _moving_average(palm_rot[:, :, 0], smooth_radius)
        z = _moving_average(palm_rot[:, :, 2], smooth_radius)
        abs_x_dot, abs_z_dot = _raw_palm_sign_confidence(x, z, ref_x, ref_z)
        degenerate = (abs_x_dot < PALM_SIGN_ALIGN_MIN_ABS_DOT) | (
            abs_z_dot < PALM_SIGN_ALIGN_MIN_ABS_DOT
        )
        palm_rot = _orthonormalize_palm_axes(x, z, ref_x, ref_z, normal_sign=-1.0)
        palm_rot = _enforce_palm_normal_continuity(palm_rot, degenerate=degenerate)

    if correction is not None:
        palm_rot = np.einsum("tij,jk->tik", palm_rot, correction)

    palm_poses = np.zeros((keypoints_source.shape[0], 4, 4), dtype=np.float64)
    palm_poses[:, :3, :3] = palm_rot
    palm_poses[:, :3, 3] = palm_center
    palm_poses[:, 3, 3] = 1.0
    return palm_poses


def wrist_poses_from_palm_targets(
    palm_poses: np.ndarray,
    wrist_to_palm: np.ndarray,
) -> np.ndarray:
    """Convert desired robot palm poses into wrist-yaw-link target poses."""
    palm_poses = np.asarray(palm_poses, dtype=np.float64)
    wrist_to_palm = np.asarray(wrist_to_palm, dtype=np.float64)
    rotation_wp = wrist_to_palm[:3, :3]
    position_wp = wrist_to_palm[:3, 3]

    out = np.zeros_like(palm_poses, dtype=np.float64)
    out[:, 3, 3] = 1.0
    out[:, :3, :3] = np.einsum("tij,kj->tik", palm_poses[:, :3, :3], rotation_wp)
    out[:, :3, 3] = palm_poses[:, :3, 3] - np.einsum(
        "tij,j->ti", out[:, :3, :3], position_wp
    )
    return out


# ── BrainCo MuJoCo model helpers ────────────────────────────────────────────


def _object_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise KeyError(f"MuJoCo object not found: {name}")
    return int(obj_id)


def _visual_palm_geom(model: mujoco.MjModel, mesh_name: str) -> int:
    for geom_id in range(model.ngeom):
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0 or int(model.geom_group[geom_id]) != 2:
            continue
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) == mesh_name:
            return geom_id
    raise KeyError(f"Visual BrainCo palm geom not found: {mesh_name}")


def brainco_qpos_writer(model: mujoco.MjModel) -> Callable[[np.ndarray, np.ndarray], None]:
    """Write a 26-DoF action into qpos and resolve the BrainCo mimic joints."""

    def address(name: str) -> int:
        return int(model.jnt_qposadr[_object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])

    action_addresses = np.asarray(
        [address(name) for name in BRAINCO_ACTION_JOINT_NAMES], dtype=int
    )
    mimic_addresses = [
        (address(mimic), address(driver), float(multiplier), float(offset))
        for mimic, driver, multiplier, offset in BRAINCO_MIMIC_RULES
    ]

    def write(qpos_full: np.ndarray, action: np.ndarray) -> None:
        qpos_full[action_addresses] = np.asarray(action, dtype=np.float64)
        for mimic, driver, multiplier, offset in mimic_addresses:
            qpos_full[mimic] = multiplier * qpos_full[driver] + offset

    return write


def _neutral_data(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    data.qpos[2] = G1_STANDING_HEIGHT
    data.qpos[3] = 1.0
    mujoco.mj_kinematics(model, data)
    return data


def estimate_wrist_to_palm(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    """Measure each target-body to visual-palm transform in the neutral pose."""
    data = _neutral_data(model)
    out: dict[str, np.ndarray] = {}
    for index, side in enumerate(("left", "right")):
        wrist_id = _object_id(
            model, mujoco.mjtObj.mjOBJ_BODY, BRAINCO_WRIST_TARGET_BODIES[index]
        )
        palm_geom = _visual_palm_geom(model, BRAINCO_PALM_GEOM_MESH_NAMES[index])
        wrist_rot = data.xmat[wrist_id].reshape(3, 3).copy()
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = wrist_rot.T @ data.geom_xmat[palm_geom].reshape(3, 3)
        transform[:3, 3] = wrist_rot.T @ (
            data.geom_xpos[palm_geom] - data.xpos[wrist_id]
        )
        out[side] = transform
    return out


def estimate_palm_local_correction(
    model: mujoco.MjModel, wrist_to_palm: dict[str, np.ndarray]
) -> dict[str, float]:
    """Derive the local palm z correction from the robot's own neutral geometry.

    ``phantom`` hardcodes this angle per robot: it rotates the palm frame built
    from MANO landmarks into the frame of the robot's visual palm mesh.  The
    angle is fully determined by the MJCF, so measure it by running the same
    axis construction on the robot's own wrist and MCP anchors.  Applying this
    to ``g1_inspire`` reproduces phantom's hardcoded +90/-90; BrainCo mounts its
    ``*_base_link`` palm mesh the other way round and needs the opposite sign.
    """
    data = _neutral_data(model)
    out: dict[str, float] = {}
    for index, side in enumerate(("left", "right")):
        wrist_id = _object_id(
            model, mujoco.mjtObj.mjOBJ_BODY, BRAINCO_WRIST_TARGET_BODIES[index]
        )
        palm_geom = _visual_palm_geom(model, BRAINCO_PALM_GEOM_MESH_NAMES[index])
        hand_names = (
            BRAINCO_LEFT_HAND_JOINT_NAMES
            if side == "left"
            else BRAINCO_RIGHT_HAND_JOINT_NAMES
        )
        mcp_ids = np.asarray(
            [
                _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in hand_names[2:6]
            ],
            dtype=int,
        )
        wrist_pos = data.xpos[wrist_id]
        wrist_rot = data.xmat[wrist_id].reshape(3, 3)
        palm_rot = data.geom_xmat[palm_geom].reshape(3, 3)
        mcp = data.xanchor[mcp_ids]
        reference = wrist_rot @ wrist_to_palm[side][:3, :3]
        geometric = _orthonormalize_palm_axes(
            (mcp[0] - mcp[3])[None, :],
            (mcp.mean(axis=0) - wrist_pos)[None, :],
            reference[None, :, 0],
            reference[None, :, 2],
            normal_sign=-1.0,
        )[0]
        correction = geometric.T @ palm_rot
        axis_error = abs(correction[2, 0]) + abs(correction[2, 1])
        if axis_error > PALM_CORRECTION_AXIS_TOLERANCE:
            raise ValueError(
                f"{side} palm correction is not a rotation about the palm z axis "
                f"(off-axis magnitude {axis_error:.3f})"
            )
        angle = float(np.degrees(np.arctan2(correction[1, 0], correction[0, 0])))
        if not PALM_CORRECTION_MIN_DEG <= abs(angle) <= PALM_CORRECTION_MAX_DEG:
            raise ValueError(
                f"{side} palm correction {angle:.2f} deg is outside the expected "
                "quarter-turn mounting convention"
            )
        out[side] = angle
    return out


# ── Stage 1: dex-retargeting position seed ──────────────────────────────────


class BraincoHandRetargeter:
    """Wrist-relative fingertips to six active BrainCo joints, per side.

    Ported from ``phantom/retarget/hand_retarget_base.py`` and
    ``hand_retarget_brainco.py``.
    """

    N_TIPS = 5
    N_HAND_DOF = 6

    def __init__(self, side: str, urdf_dir: Path, config_dir: Path):
        if side not in ("left", "right"):
            raise ValueError(f"invalid side: {side}")
        from dex_retargeting.retargeting_config import RetargetingConfig

        self.side = side
        RetargetingConfig.set_default_urdf_dir(str(urdf_dir))
        config_path = Path(config_dir) / f"brainco_hand_{side}.yml"
        if not config_path.is_file():
            raise FileNotFoundError(f"BrainCo retarget config not found: {config_path}")
        self.config_path = config_path
        self.retargeting = RetargetingConfig.load_from_file(str(config_path)).build()

        canonical = self.canonical_joint_names(side)
        target_names = [
            name
            for name in getattr(self.retargeting.optimizer, "target_joint_names", [])
            if not name.startswith("dummy_")
        ]
        if sorted(target_names) != sorted(canonical):
            raise AssertionError(
                f"BrainCo optimizer target mismatch: expected {canonical}, "
                f"got {target_names}"
            )
        dof_names = [
            name
            for name in self.retargeting.optimizer.robot.dof_joint_names
            if not name.startswith("dummy_")
        ]
        if not set(canonical).issubset(dof_names) or len(dof_names) != 11:
            raise AssertionError(
                f"BrainCo optimizer DOF mismatch: expected 6 active + 5 mimic, "
                f"got {dof_names}"
            )
        self._reorder_index = np.asarray(
            [dof_names.index(name) for name in canonical], dtype=int
        )

    @staticmethod
    def canonical_joint_names(side: str) -> list[str]:
        return (
            BRAINCO_LEFT_HAND_JOINT_NAMES
            if side == "left"
            else BRAINCO_RIGHT_HAND_JOINT_NAMES
        )

    def retarget_frame(self, fingertips_relative: np.ndarray) -> np.ndarray:
        if fingertips_relative.shape != (self.N_TIPS, 3):
            raise ValueError(f"expected ({self.N_TIPS}, 3) fingertips")
        qpos_full = self.retargeting.retarget(fingertips_relative.astype(np.float64))
        return qpos_full[6:][self._reorder_index].astype(np.float32)

    def retarget_episode(
        self,
        fingertips: np.ndarray,
        wrist_position: np.ndarray,
        tips_confidence: np.ndarray | None = None,
    ) -> np.ndarray:
        frames = fingertips.shape[0]
        out = np.zeros((frames, self.N_HAND_DOF), dtype=np.float32)
        if tips_confidence is None:
            tips_confidence = np.ones((frames, self.N_TIPS), dtype=np.float32)

        self.retargeting.reset()
        last_good = np.zeros(self.N_HAND_DOF, dtype=np.float32)
        for index in range(frames):
            if np.any(tips_confidence[index] < CONF_THRESHOLD):
                out[index] = last_good
                continue
            try:
                solution = self.retarget_frame(
                    fingertips[index] - wrist_position[index]
                )
                out[index] = solution
                last_good = solution.copy()
            except Exception as error:  # noqa: BLE001 - mirror phantom's policy
                logger.warning(
                    "BrainCo %s seed retargeting failed at frame %d: %s",
                    self.side,
                    index,
                    error,
                )
                out[index] = last_good
        return out


# ── Stage 2: MANO 16-point refinement ───────────────────────────────────────


@dataclass(frozen=True)
class _SideModel:
    name: str
    action_indices: np.ndarray
    hand_joint_ids: np.ndarray
    dependent_joint_ids: np.ndarray
    tip_body_ids: np.ndarray
    wrist_body_id: int
    palm_geom_id: int


class BraincoJointRefiner:
    """Refine BrainCo's six active hand joints and wrist translation.

    Ported from ``phantom/retarget/joint_refine_inspire.py`` and
    ``joint_refine_brainco.py``.
    """

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)
        self.qpos_writer = brainco_qpos_writer(model)
        self._standing_qpos = np.zeros(model.nq, dtype=np.float64)
        self._standing_qpos[2] = G1_STANDING_HEIGHT
        self._standing_qpos[3] = 1.0
        self._sides = {side: self._build_side(side) for side in ("left", "right")}

    def _build_side(self, side: str) -> _SideModel:
        prefix = "left_" if side == "left" else "right_"
        hand_names = (
            BRAINCO_LEFT_HAND_JOINT_NAMES
            if side == "left"
            else BRAINCO_RIGHT_HAND_JOINT_NAMES
        )
        dependent_names = (
            f"{prefix}thumb_distal_joint",
            f"{prefix}index_distal_joint",
            f"{prefix}middle_distal_joint",
            f"{prefix}ring_distal_joint",
            f"{prefix}pinky_distal_joint",
        )
        tip_names = (
            BRAINCO_LEFT_HAND_TIP_BODIES
            if side == "left"
            else BRAINCO_RIGHT_HAND_TIP_BODIES
        )
        side_index = 0 if side == "left" else 1
        return _SideModel(
            name=side,
            action_indices=np.concatenate(
                [
                    np.arange(0, 7) if side == "left" else np.arange(7, 14),
                    np.arange(14, 20) if side == "left" else np.arange(20, 26),
                ]
            ).astype(int),
            hand_joint_ids=np.asarray(
                [
                    _object_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    for name in hand_names
                ],
                dtype=int,
            ),
            dependent_joint_ids=np.asarray(
                [
                    _object_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    for name in dependent_names
                ],
                dtype=int,
            ),
            tip_body_ids=np.asarray(
                [
                    _object_id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                    for name in tip_names
                ],
                dtype=int,
            ),
            wrist_body_id=_object_id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                BRAINCO_WRIST_TARGET_BODIES[side_index],
            ),
            palm_geom_id=_visual_palm_geom(
                self.model, BRAINCO_PALM_GEOM_MESH_NAMES[side_index]
            ),
        )

    def hand_joint_limits(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        ids = self._sides[side].hand_joint_ids
        return (
            self.model.jnt_range[ids, 0].copy(),
            self.model.jnt_range[ids, 1].copy(),
        )

    def _set_action(self, action: np.ndarray) -> None:
        self.data.qpos[:] = self._standing_qpos
        self.qpos_writer(self.data.qpos, action)
        mujoco.mj_kinematics(self.model, self.data)

    def _collect_points(self, side_model: _SideModel) -> np.ndarray:
        active = self.data.xanchor[side_model.hand_joint_ids]
        dependent = self.data.xanchor[side_model.dependent_joint_ids]
        tips = self.data.xpos[side_model.tip_body_ids]
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

    def _hand_local_keypoints(
        self, side: str, hand_q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the 16 landmarks and palm rotation in wrist coordinates.

        Arm joints stay at zero, so the result depends only on the hand pose.
        Mimic joints are resolved by the qpos writer and are not independent
        optimization variables.
        """
        side_model = self._sides[side]
        action = np.zeros(BRAINCO_ACTION_DIM, dtype=np.float64)
        action[side_model.action_indices[-6:]] = np.asarray(hand_q, dtype=np.float64)
        self._set_action(action)

        wrist_pos = self.data.xpos[side_model.wrist_body_id]
        wrist_rot = self.data.xmat[side_model.wrist_body_id].reshape(3, 3)
        points_world = self._collect_points(side_model)

        # Row-vector convention: local = (world - wrist) @ R_wrist.
        points_local = (points_world - wrist_pos[None, :]) @ wrist_rot
        palm_local_rot = wrist_rot.T @ self.data.geom_xmat[
            side_model.palm_geom_id
        ].reshape(3, 3)
        return points_local, palm_local_rot

    def evaluate_hand_keypoints(
        self, side: str, hand_q: np.ndarray, wrist_poses: np.ndarray
    ) -> np.ndarray:
        """Evaluate the rendered 16-point layout for a refined hand trajectory."""
        frames = hand_q.shape[0]
        out = np.empty((frames, 16, 3), dtype=np.float64)
        for index in range(frames):
            local, _ = self._hand_local_keypoints(side, hand_q[index])
            pose = wrist_poses[index]
            out[index] = local @ pose[:3, :3].T + pose[:3, 3]
        return out

    @staticmethod
    def _pose_from_delta(
        base_pose: np.ndarray,
        delta_position: np.ndarray,
        delta_rotation: np.ndarray,
    ) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = base_pose[:3, :3] @ Rotation.from_rotvec(
            delta_rotation
        ).as_matrix()
        pose[:3, 3] = base_pose[:3, 3] + delta_position
        return pose

    def refine_hand_side(
        self,
        side: str,
        seed_hand: np.ndarray,
        seed_wrist: np.ndarray,
        keypoint_targets: np.ndarray,
        palm_targets: np.ndarray,
        wrist_confidence: np.ndarray,
        *,
        freeze_wrist_rotation: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        palm_orientation_scale = (
            MANO_HAND_FIXED_WRIST_ORIENTATION_SCALE
            if freeze_wrist_rotation
            else MANO_HAND_ORIENTATION_SCALE
        )
        side_model = self._sides[side]
        frames = seed_hand.shape[0]
        hand_out = np.asarray(seed_hand, dtype=np.float64).copy()
        wrist_out = np.asarray(seed_wrist, dtype=np.float64).copy()
        keypoint_error_before = np.full(frames, np.nan, dtype=np.float32)
        keypoint_error_after = np.full(frames, np.nan, dtype=np.float32)
        wrist_delta = np.zeros((frames, 6), dtype=np.float32)
        accepted = np.zeros(frames, dtype=bool)
        held = np.zeros(frames, dtype=bool)
        nfev = np.zeros(frames, dtype=np.int16)

        joint_lower_model = self.model.jnt_range[side_model.hand_joint_ids, 0].copy()
        joint_upper_model = self.model.jnt_range[side_model.hand_joint_ids, 1].copy()
        previous_hand: np.ndarray | None = None
        previous_wrist: np.ndarray | None = None

        for frame_index in range(frames):
            target = np.asarray(keypoint_targets[frame_index], dtype=np.float64)
            palm_target = np.asarray(palm_targets[frame_index], dtype=np.float64)
            valid = bool(
                wrist_confidence[frame_index] >= CONF_THRESHOLD
                and target.shape == (16, 3)
                and np.isfinite(target).all()
                and palm_target.shape == (4, 4)
                and np.isfinite(palm_target).all()
            )
            if not valid:
                held[frame_index] = previous_hand is not None
                if previous_hand is not None and previous_wrist is not None:
                    hand_out[frame_index] = previous_hand
                    wrist_out[frame_index] = previous_wrist
                previous_hand = hand_out[frame_index].copy()
                previous_wrist = wrist_out[frame_index].copy()
                continue

            seed_hand_frame = np.asarray(seed_hand[frame_index], dtype=np.float64)
            seed_wrist_frame = np.asarray(seed_wrist[frame_index], dtype=np.float64)
            base_hand = (
                previous_hand.copy()
                if previous_hand is not None
                else seed_hand_frame.copy()
            )
            base_wrist = (
                previous_wrist.copy()
                if previous_wrist is not None
                else seed_wrist_frame.copy()
            )
            if freeze_wrist_rotation:
                # Preserve the historical per-frame wrist orientation while
                # allowing the optimizer to correct only the wrist position.
                base_wrist[:3, :3] = seed_wrist_frame[:3, :3]

            lower = joint_lower_model.copy()
            upper = joint_upper_model.copy()
            if previous_hand is not None:
                lower = np.maximum(lower, base_hand - JOINT_REFINE_HAND_MAX_DELTA)
                upper = np.minimum(upper, base_hand + JOINT_REFINE_HAND_MAX_DELTA)

            wrist_variable_dim = 3 if freeze_wrist_rotation else 6
            x0 = np.concatenate(
                [np.clip(base_hand, lower, upper), np.zeros(wrist_variable_dim)]
            )
            lower_x = np.concatenate(
                [
                    lower,
                    np.full(3, -MANO_HAND_WRIST_POSITION_MAX_DELTA),
                    *(
                        ()
                        if freeze_wrist_rotation
                        else (np.full(3, -MANO_HAND_WRIST_ROTATION_MAX_DELTA),)
                    ),
                ]
            )
            upper_x = np.concatenate(
                [
                    upper,
                    np.full(3, MANO_HAND_WRIST_POSITION_MAX_DELTA),
                    *(
                        ()
                        if freeze_wrist_rotation
                        else (np.full(3, MANO_HAND_WRIST_ROTATION_MAX_DELTA),)
                    ),
                ]
            )

            def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                hand_q = x[:6]
                delta_rotation = (
                    np.zeros(3, dtype=np.float64)
                    if freeze_wrist_rotation
                    else x[9:12]
                )
                pose = self._pose_from_delta(base_wrist, x[6:9], delta_rotation)
                local_points, local_palm_rot = self._hand_local_keypoints(side, hand_q)
                points = local_points @ pose[:3, :3].T + pose[:3, 3]
                palm_rot = pose[:3, :3] @ local_palm_rot
                return points, palm_rot, pose

            def residual(x: np.ndarray) -> np.ndarray:
                points, palm_rot, pose = evaluate(x)
                orientation_error = Rotation.from_matrix(
                    palm_target[:3, :3].T @ palm_rot
                ).as_rotvec()
                temporal_hand = (x[:6] - base_hand) / MANO_HAND_JOINT_TEMPORAL_SCALE
                temporal_wrist = x[6:9] / MANO_HAND_WRIST_POSITION_TEMPORAL_SCALE
                seed_wrist_error = (
                    pose[:3, 3] - seed_wrist_frame[:3, 3]
                ) / MANO_HAND_SEED_WRIST_POSITION_SCALE
                if not freeze_wrist_rotation:
                    temporal_wrist = np.concatenate(
                        [
                            temporal_wrist,
                            x[9:12] / MANO_HAND_WRIST_ROTATION_TEMPORAL_SCALE,
                        ]
                    )
                    seed_wrist_error = np.concatenate(
                        [
                            seed_wrist_error,
                            Rotation.from_matrix(
                                seed_wrist_frame[:3, :3].T @ pose[:3, :3]
                            ).as_rotvec()
                            / MANO_HAND_SEED_WRIST_ROTATION_SCALE,
                        ]
                    )
                return np.concatenate(
                    [
                        ((points - target) / MANO_HAND_KEYPOINT_SCALE).reshape(-1),
                        orientation_error / palm_orientation_scale,
                        temporal_hand,
                        temporal_wrist,
                        (x[:6] - seed_hand_frame) / MANO_HAND_SEED_JOINT_SCALE,
                        seed_wrist_error,
                    ]
                )

            seed_points, _, _ = evaluate(
                np.concatenate([seed_hand_frame, np.zeros(wrist_variable_dim)])
            )
            keypoint_error_before[frame_index] = np.linalg.norm(
                seed_points - target, axis=1
            ).mean()
            try:
                solution = least_squares(
                    residual,
                    x0,
                    bounds=(lower_x, upper_x),
                    max_nfev=MANO_HAND_MAX_NFEV,
                    ftol=1e-5,
                    xtol=1e-5,
                    gtol=1e-5,
                )
                candidate = np.clip(solution.x, lower_x, upper_x)
                nfev[frame_index] = int(solution.nfev)
                candidate_points, _, candidate_wrist = evaluate(candidate)
                if not np.isfinite(candidate_points).all():
                    raise ValueError("non-finite MANO hand refinement result")
                hand_out[frame_index] = candidate[:6]
                wrist_out[frame_index] = candidate_wrist
                keypoint_error_after[frame_index] = np.linalg.norm(
                    candidate_points - target, axis=1
                ).mean()
                accepted[frame_index] = True
            except Exception as error:  # noqa: BLE001 - mirror phantom's policy
                logger.warning(
                    "%s MANO hand refinement failed at frame %d: %s",
                    side,
                    frame_index,
                    error,
                )
                held[frame_index] = previous_hand is not None
                hand_out[frame_index] = (
                    previous_hand if previous_hand is not None else seed_hand_frame
                )
                wrist_out[frame_index] = (
                    previous_wrist if previous_wrist is not None else seed_wrist_frame
                )
                if freeze_wrist_rotation:
                    wrist_out[frame_index, :3, :3] = seed_wrist_frame[:3, :3]
                fallback_points, _, _ = evaluate(
                    np.concatenate(
                        [hand_out[frame_index], np.zeros(wrist_variable_dim)]
                    )
                )
                keypoint_error_after[frame_index] = np.linalg.norm(
                    fallback_points - target, axis=1
                ).mean()

            wrist_delta[frame_index, :3] = (
                wrist_out[frame_index, :3, 3] - base_wrist[:3, 3]
            )
            if not freeze_wrist_rotation:
                wrist_delta[frame_index, 3:] = Rotation.from_matrix(
                    base_wrist[:3, :3].T @ wrist_out[frame_index, :3, :3]
                ).as_rotvec()
            previous_hand = hand_out[frame_index].copy()
            previous_wrist = wrist_out[frame_index].copy()

        diagnostics = {
            "keypoint_error_before_m": keypoint_error_before,
            "keypoint_error_after_m": keypoint_error_after,
            "wrist_delta": wrist_delta,
            "accepted": accepted,
            "held": held,
            "nfev": nfev,
        }
        return hand_out.astype(np.float64), wrist_out.astype(np.float64), diagnostics


# ── Driver ──────────────────────────────────────────────────────────────────


def _load_model(mjcf: Path) -> mujoco.MjModel:
    import xml.etree.ElementTree as ET

    root = ET.parse(mjcf).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", str(mjcf.parent))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def _resample_columns(
    values: np.ndarray, source_time: np.ndarray, target_time: np.ndarray
) -> np.ndarray:
    return np.column_stack(
        [
            np.interp(target_time, source_time, values[:, column])
            for column in range(values.shape[1])
        ]
    )


def _diagnostics_summary(diagnostics: dict[str, np.ndarray]) -> dict[str, Any]:
    before = diagnostics["keypoint_error_before_m"]
    after = diagnostics["keypoint_error_after_m"]
    return {
        "keypoint_error_before_m": _stats(before[np.isfinite(before)]),
        "keypoint_error_after_m": _stats(after[np.isfinite(after)]),
        "wrist_position_delta_m": _stats(
            np.linalg.norm(diagnostics["wrist_delta"][:, :3], axis=1)
        ),
        "accepted_frames": int(np.count_nonzero(diagnostics["accepted"])),
        "held_frames": int(np.count_nonzero(diagnostics["held"])),
        "mean_nfev": float(np.mean(diagnostics["nfev"])),
        "frames_over_validation_threshold": int(
            np.count_nonzero(after > HAND_KEYPOINT_VALIDATION_THRESHOLD_M)
        ),
    }


def retarget_brainco(
    *,
    annotation_hdf5: Path,
    base_reference: Path,
    base_targets: Path,
    prepare_meta: Path,
    mjcf: Path,
    hand_urdf_dir: Path,
    retarget_config_dir: Path,
    output_reference: Path,
    output_targets: Path,
    output_hand: Path,
    meta_json: Path,
    huber_delta_m: float = 0.008,
    iterations: int = 4,
    output_native_experiment_dir: Path | None = None,
) -> dict[str, Any]:
    for path in (annotation_hdf5, base_reference, base_targets, prepare_meta, mjcf):
        if not path.is_file():
            raise FileNotFoundError(path)
    calibration = annotation_hdf5.parent / "calibrations/R_optical_to_link.npy"
    if not calibration.is_file():
        raise FileNotFoundError(calibration)

    reference = _load_npz(base_reference)
    targets = _load_npz(base_targets)
    preparation = json.loads(prepare_meta.read_text(encoding="utf-8"))
    body_names = [str(value) for value in reference["body_names"].reshape(-1).tolist()]
    fps = float(np.asarray(reference["fps"]).reshape(()))
    target_time = np.asarray(targets["timestamps"], dtype=np.float64)
    frames = len(target_time)
    if reference["body_pos_w"].shape[0] != frames:
        raise ValueError("reference and target frame counts differ")
    if not np.isclose(float(targets["fps_session"]), fps, atol=1.0e-8):
        raise ValueError("reference and target FPS differ")

    selection = preparation.get("selection", {})
    start = int(selection.get("selected_source_start_index", 0))
    end = int(
        selection.get(
            "selected_source_end_index",
            start + int(preparation["source_frames"]) - 1,
        )
    )
    repair_policy = selection.get("transient_wrist_outlier_repair", {})
    world_points, native_time_all, source_meta = _load_world_points_zup(
        annotation_hdf5, calibration, repair_policy
    )
    if start < 0 or end >= len(native_time_all) or end <= start:
        raise ValueError(f"invalid selected source range [{start}, {end}]")
    native_time = native_time_all[start : end + 1] - native_time_all[start]
    if (
        target_time[0] < native_time[0] - 1.0e-9
        or target_time[-1] > native_time[-1] + 1.0e-6
    ):
        raise ValueError("prepared target timeline is outside selected hand timeline")

    model = _load_model(mjcf)
    wrist_to_palm = estimate_wrist_to_palm(model)
    palm_correction_deg = estimate_palm_local_correction(model, wrist_to_palm)
    refiner = BraincoJointRefiner(model)

    output_reference_arrays = {name: value.copy() for name, value in reference.items()}
    output_target_arrays = {name: value.copy() for name, value in targets.items()}
    native_hand: dict[str, np.ndarray] = {}
    native_rate_hand: dict[str, np.ndarray] = {}
    native_rate_wrist_pos: dict[str, np.ndarray] = {}
    native_rate_wrist_rot: dict[str, np.ndarray] = {}
    side_reports: dict[str, Any] = {}
    delta_reports: dict[str, Any] = {}

    for side in ("left", "right"):
        points = world_points[side][start : end + 1]
        native_frames = len(points)
        mano_wrist = points[:, 0, :]

        # The prior wrist rotation only disambiguates palm axis signs, so it is
        # taken from the same template-palm Kabsch fit that seeds the Dex3 path.
        _, prior_rotation, _ = template_palm_kabsch(
            points, huber_delta_m=huber_delta_m, iterations=iterations
        )
        prior_pose = np.zeros((native_frames, 4, 4), dtype=np.float64)
        prior_pose[:, 3, 3] = 1.0
        prior_pose[:, :3, :3] = prior_rotation
        prior_pose[:, :3, 3] = mano_wrist

        retargeter = BraincoHandRetargeter(side, hand_urdf_dir, retarget_config_dir)
        # dex-retargeting's five human indices are the MANO fingertips.
        fingertips = points[:, (4, 8, 12, 16, 20), :]
        seed_hand = retargeter.retarget_episode(fingertips, mano_wrist)
        joint_lower, joint_upper = refiner.hand_joint_limits(side)
        seed_hand = np.clip(seed_hand, joint_lower, joint_upper)

        palm_poses = estimate_palm_poses_from_keypoints(
            points,
            prior_pose,
            wrist_to_palm[side],
            local_z_correction_deg=palm_correction_deg[side],
        )
        seed_wrist = wrist_poses_from_palm_targets(palm_poses, wrist_to_palm[side])
        keypoint_targets = points[:, MANO_HAND_KEYPOINT_INDICES]
        confidence = np.ones(native_frames, dtype=np.float64)

        refined_hand, refined_wrist, diagnostics = refiner.refine_hand_side(
            side,
            seed_hand,
            seed_wrist,
            keypoint_targets,
            palm_poses,
            confidence,
            freeze_wrist_rotation=True,
        )

        position, rotation = _resample_pose(
            refined_wrist[:, :3, 3], refined_wrist[:, :3, :3], native_time, target_time
        )
        quaternion = _continuous_wxyz(rotation)
        native_hand[side] = _resample_columns(refined_hand, native_time, target_time)
        n_all = len(native_time_all)
        full_pos = np.repeat(refined_wrist[:1, :3, 3], n_all, axis=0)
        full_rot = np.repeat(refined_wrist[:1, :3, :3], n_all, axis=0)
        full_hand = np.repeat(refined_hand[:1], n_all, axis=0)
        full_pos[start : end + 1] = refined_wrist[:, :3, 3]
        full_rot[start : end + 1] = refined_wrist[:, :3, :3]
        full_hand[start : end + 1] = refined_hand
        if end + 1 < n_all:
            full_pos[end + 1 :] = refined_wrist[-1, :3, 3]
            full_rot[end + 1 :] = refined_wrist[-1, :3, :3]
            full_hand[end + 1 :] = refined_hand[-1]
        native_rate_wrist_pos[side] = full_pos
        native_rate_wrist_rot[side] = full_rot
        native_rate_hand[side] = full_hand
        n_all = len(native_time_all)
        full_pos = np.repeat(refined_wrist[:1, :3, 3], n_all, axis=0)
        full_rot = np.repeat(refined_wrist[:1, :3, :3], n_all, axis=0)
        full_hand = np.repeat(refined_hand[:1], n_all, axis=0)
        full_pos[start : end + 1] = refined_wrist[:, :3, 3]
        full_rot[start : end + 1] = refined_wrist[:, :3, :3]
        full_hand[start : end + 1] = refined_hand
        if end + 1 < n_all:
            full_pos[end + 1 :] = refined_wrist[-1, :3, 3]
            full_rot[end + 1 :] = refined_wrist[-1, :3, :3]
            full_hand[end + 1 :] = refined_hand[-1]
        native_rate_wrist_pos[side] = full_pos
        native_rate_wrist_rot[side] = full_rot
        native_rate_hand[side] = full_hand

        body_index = body_names.index(WRIST_BODIES[side])
        old_position = np.asarray(
            reference["body_pos_w"][:, body_index], dtype=np.float64
        )
        old_rotation = Rotation.from_quat(
            np.asarray(reference["body_quat_w"][:, body_index], dtype=np.float64),
            scalar_first=True,
        )
        orientation_delta = np.degrees(
            (old_rotation.inv() * Rotation.from_matrix(rotation)).magnitude()
        )

        output_reference_arrays["body_pos_w"][:, body_index] = position.astype(
            output_reference_arrays["body_pos_w"].dtype
        )
        output_reference_arrays["body_quat_w"][:, body_index] = quaternion.astype(
            output_reference_arrays["body_quat_w"].dtype
        )
        output_reference_arrays["body_lin_vel_w"][:, body_index] = _finite_difference(
            position, fps
        ).astype(output_reference_arrays["body_lin_vel_w"].dtype)
        output_reference_arrays["body_ang_vel_w"][:, body_index] = (
            _angular_velocity_world(rotation, fps).astype(
                output_reference_arrays["body_ang_vel_w"].dtype
            )
        )
        output_target_arrays[f"{side}_wrist"] = position.astype(np.float32)
        output_target_arrays[f"{side}_wrist_quat_wxyz"] = quaternion.astype(np.float32)

        rendered = refiner.evaluate_hand_keypoints(side, refined_hand, refined_wrist)
        side_reports[side] = {
            "native_frames": int(native_frames),
            "seed_config": str(retargeter.config_path.resolve()),
            "wrist_to_palm_translation_m": wrist_to_palm[side][:3, 3].tolist(),
            "palm_local_z_correction_deg": palm_correction_deg[side],
            "refinement": _diagnostics_summary(diagnostics),
            "rendered_keypoint_error_m": _stats(
                np.linalg.norm(rendered - keypoint_targets, axis=-1).mean(axis=1)
            ),
        }
        delta_reports[side] = {
            "position_delta_m": _stats(
                np.linalg.norm(position - old_position, axis=1)
            ),
            "orientation_delta_deg": _stats(orientation_delta),
            "first_position_zup": position[0].tolist(),
            "first_quaternion_wxyz": quaternion[0].tolist(),
        }

    output_reference_arrays["wrist_reference_method"] = np.asarray(
        WRIST_REFERENCE_METHOD
    )
    output_target_arrays["wrist_reference_method"] = np.asarray(WRIST_REFERENCE_METHOD)

    # The adapter is intentionally unable to alter the carrier, joints, clock,
    # or Kimodo torso heading.
    wrist_indices = [body_names.index(name) for name in WRIST_BODIES.values()]
    non_wrist = np.ones(len(body_names), dtype=bool)
    non_wrist[wrist_indices] = False
    for field in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        if not np.array_equal(
            reference[field][:, non_wrist], output_reference_arrays[field][:, non_wrist]
        ):
            raise RuntimeError(f"non-wrist values changed in {field}")
    for field in ("joint_pos", "joint_vel", "body_names", "joint_names", "fps"):
        if not np.array_equal(reference[field], output_reference_arrays[field]):
            raise RuntimeError(f"base reference field changed: {field}")
    for field in ("timestamps", "kimodo_pelvis_heading_quat_wxyz", "fps_session"):
        if not np.array_equal(targets[field], output_target_arrays[field]):
            raise RuntimeError(f"base target field changed: {field}")

    qpos = np.concatenate([native_hand["left"], native_hand["right"]], axis=1)
    if qpos.shape != (frames, len(BRAINCO_HAND_JOINT_NAMES)):
        raise RuntimeError(f"unexpected BrainCo hand shape {qpos.shape}")
    qvel = np.gradient(qpos, target_time, axis=0, edge_order=1)
    native_frames = int(end - start + 1)
    hand_arrays = {
        "qpos": qpos.astype(np.float32),
        "qvel": qvel.astype(np.float32),
        "left_qpos": qpos[:, :6].astype(np.float32),
        "right_qpos": qpos[:, 6:].astype(np.float32),
        "left_qvel": qvel[:, :6].astype(np.float32),
        "right_qvel": qvel[:, 6:].astype(np.float32),
        "hand_joint_names": np.asarray(BRAINCO_HAND_JOINT_NAMES),
        "left_joint_names": np.asarray(BRAINCO_LEFT_HAND_JOINT_NAMES),
        "right_joint_names": np.asarray(BRAINCO_RIGHT_HAND_JOINT_NAMES),
        "timestamps": target_time,
        "fps": np.asarray(fps, dtype=np.float64),
        "method": np.asarray(WRIST_REFERENCE_METHOD),
        "hand_contract": np.asarray(HAND_CONTRACT),
        "mimic_rules_json": np.asarray(json.dumps(BRAINCO_MIMIC_RULES)),
        "source_hdf5": np.asarray(str(annotation_hdf5.resolve())),
        "source_mjcf": np.asarray(str(mjcf.resolve())),
        "schema_version": np.asarray("brainco_hand_kinematics_v1"),
        "left_contact": np.zeros(frames, dtype=np.int8),
        "right_contact": np.zeros(frames, dtype=np.int8),
        "left_contact_30hz": np.zeros(native_frames, dtype=np.int8),
        "right_contact_30hz": np.zeros(native_frames, dtype=np.int8),
        "native_timestamps": np.asarray(native_time, dtype=np.float64),
    }

    report: dict[str, Any] = {
        "schema_version": 1,
        "method": WRIST_REFERENCE_METHOD,
        "hand_contract": HAND_CONTRACT,
        "hand_dof": len(BRAINCO_HAND_JOINT_NAMES),
        "hand_joint_names": list(BRAINCO_HAND_JOINT_NAMES),
        "selection_policy": preparation.get("selection_policy"),
        "selection": selection,
        "source_frames": preparation.get("source_frames"),
        "output_frames": frames,
        "output_fps": fps,
        "inputs": {
            "annotation_hdf5": str(annotation_hdf5.resolve()),
            "annotation_sha256": _sha256(annotation_hdf5),
            "base_reference": str(base_reference.resolve()),
            "base_reference_sha256": _sha256(base_reference),
            "base_targets": str(base_targets.resolve()),
            "base_targets_sha256": _sha256(base_targets),
            "prepare_meta": str(prepare_meta.resolve()),
            "mjcf": str(mjcf.resolve()),
            "mjcf_sha256": _sha256(mjcf),
        },
        "outputs": {
            "reference": str(output_reference.resolve()),
            "targets": str(output_targets.resolve()),
            "hand": str(output_hand.resolve()),
        },
        "contract": {
            "stage_1": "dex_retargeting position seed on wrist-relative fingertips",
            "stage_2": (
                "scipy least_squares over 6 hand joints + 3 wrist translation "
                "against 16 MANO landmarks, wrist rotation frozen"
            ),
            "palm_axes": "MANO wrist + MCP landmarks, prior rotation resolves signs",
            "palm_local_correction": (
                "derived from the MJCF neutral pose; reproduces phantom's "
                "hardcoded +90/-90 on g1_inspire"
            ),
            "palm_local_correction": (
                "derived from the MJCF neutral pose; reproduces phantom's "
                "hardcoded +90/-90 on g1_inspire"
            ),
            "coordinate_frame": "ScaleTrack Z-up world",
            "refinement_rate": "native hand timeline, resampled to the target clock",
            "wrist_position_max_delta_m": MANO_HAND_WRIST_POSITION_MAX_DELTA,
            "keypoint_validation_threshold_m": HAND_KEYPOINT_VALIDATION_THRESHOLD_M,
            "source_method": "Open-AoE phantom --robot g1_brainco",
        },
        "source": source_meta,
        "sides": side_reports,
        "delta_from_historical_reference": delta_reports,
        "gates": {
            "non_wrist_reference_unchanged": True,
            "joint_and_timeline_fields_unchanged": True,
            "finite_output": bool(
                np.all(np.isfinite(output_reference_arrays["body_pos_w"]))
                and np.all(np.isfinite(output_reference_arrays["body_quat_w"]))
                and np.all(np.isfinite(qpos))
                and np.all(np.isfinite(qvel))
            ),
        },
    }
    if not report["gates"]["finite_output"]:
        raise RuntimeError("BrainCo reference contains non-finite values")

    _atomic_npz(output_reference, output_reference_arrays)
    _atomic_npz(output_targets, output_target_arrays)
    _atomic_npz(output_hand, hand_arrays)
    if output_native_experiment_dir is not None:
        output_native_experiment_dir.mkdir(parents=True, exist_ok=True)
        method = WRIST_REFERENCE_METHOD
        native_timestamps = np.asarray(native_time_all, dtype=np.float64)
        native_timestamps = native_timestamps - native_timestamps[0]
        wrist_native = {"timestamps": native_timestamps}
        for side in ("left", "right"):
            wrist_native[f"{method}__{side}_position"] = native_rate_wrist_pos[side]
            wrist_native[f"{method}__{side}_rotation"] = native_rate_wrist_rot[side]
        native_qpos = np.concatenate(
            [native_rate_hand["left"], native_rate_hand["right"]], axis=1
        ).astype(np.float32)
        finger_native = {
            "timestamps": native_timestamps,
            "position__qpos": native_qpos,
            "hand_joint_names": np.asarray(BRAINCO_HAND_JOINT_NAMES),
        }
        wrist_path = output_native_experiment_dir / "wrist_candidates_native.npz"
        finger_path = output_native_experiment_dir / "finger_candidates_native.npz"
        _atomic_npz(wrist_path, wrist_native)
        _atomic_npz(finger_path, finger_native)
        report["outputs"]["native_wrist_candidates"] = str(wrist_path.resolve())
        report["outputs"]["native_finger_candidates"] = str(finger_path.resolve())
    _atomic_json(meta_json, report)
    return report


def parse_args() -> argparse.Namespace:
    default_assets = Path(__file__).resolve().parents[2] / "resources/brainco"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-hdf5", type=Path, required=True)
    parser.add_argument("--base-reference", type=Path, required=True)
    parser.add_argument("--base-targets", type=Path, required=True)
    parser.add_argument("--base-meta", type=Path, required=True)
    parser.add_argument("--output-reference", type=Path, required=True)
    parser.add_argument("--output-targets", type=Path, required=True)
    parser.add_argument("--output-hand", type=Path, required=True)
    parser.add_argument("--meta-json", type=Path, required=True)
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=default_assets
        / "robots/g1_brainco/g1_mocap_29dof_with_brainco_hands.xml",
    )
    parser.add_argument(
        "--hand-urdf-dir", type=Path, default=default_assets / "robots/hands"
    )
    parser.add_argument(
        "--retarget-config-dir", type=Path, default=default_assets / "configs/retarget"
    )
    parser.add_argument("--huber-delta-m", type=float, default=0.008)
    parser.add_argument("--irls-iterations", type=int, default=4)
    parser.add_argument(
        "--output-native-experiment-dir",
        type=Path,
        help=(
            "Optional directory for Dex3-shaped native wrist/finger candidate "
            "archives used by CupModel contact optimization."
        ),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="[brainco-hand-retarget] %(message)s"
    )
    args = parse_args()
    report = retarget_brainco(
        annotation_hdf5=args.annotation_hdf5.expanduser().resolve(),
        base_reference=args.base_reference.expanduser().resolve(),
        base_targets=args.base_targets.expanduser().resolve(),
        prepare_meta=args.base_meta.expanduser().resolve(),
        mjcf=args.mjcf.expanduser().resolve(),
        hand_urdf_dir=args.hand_urdf_dir.expanduser().resolve(),
        retarget_config_dir=args.retarget_config_dir.expanduser().resolve(),
        output_reference=args.output_reference.expanduser().resolve(),
        output_targets=args.output_targets.expanduser().resolve(),
        output_hand=args.output_hand.expanduser().resolve(),
        meta_json=args.meta_json.expanduser().resolve(),
        huber_delta_m=float(args.huber_delta_m),
        iterations=int(args.irls_iterations),
        output_native_experiment_dir=(
            args.output_native_experiment_dir.expanduser().resolve()
            if args.output_native_experiment_dir is not None
            else None
        ),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
