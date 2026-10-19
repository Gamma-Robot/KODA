#!/usr/bin/env python3
"""Produce contact-aware, smoothed three-point humanoid kinematics."""

from __future__ import annotations

import argparse
import atexit
import fcntl
import hashlib
import json
import os
import queue
import signal
import shutil
import socket
import sys
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_ROOT = PROJECT_ROOT / "resources"
DATA_ROOT = PROJECT_ROOT / "data"
ASSET_ROOT = Path(os.environ.get("KODA_ASSETS", str(PROJECT_ROOT / "assets"))).expanduser().resolve()
DEFAULT_INPUT_ROOT = DATA_ROOT / "egocentric_data/pick_up_the_cup_smooth"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "kinematics/pick_up_the_cup_smooth"
DEFAULT_WORK_ROOT = Path(os.environ.get("KODA_CACHE", str(PROJECT_ROOT / ".cache"))) / "torso_umi3_batch"
PY_ST = Path(os.environ.get("KODA_PYTHON", sys.executable))
PY_JAX = PY_ST
PY_MINK = PY_ST
ST = RESOURCE_ROOT / "ScaleBFM/ScaleTrack"
HYDRAX = RESOURCE_ROOT / "hydrax"
PREPARE = ST / "scripts/eval_session/prepare_output_clean_raw_eval.py"
RETARGET_WRIST_REFERENCE = (
    PROJECT_ROOT / "scripts/view_change/retarget_template_palm_reference.py"
)
INJECT_CONTACT_AWARE_REFERENCE = (
    PROJECT_ROOT / "scripts/view_change/inject_contact_aware_wrist_reference.py"
)
JOINT_CONTACT_PRECOMPUTE = (
    PROJECT_ROOT / "scripts/view_change/run_contact_precompute.sh"
)
BRAINCO_HAND_RETARGET = (
    PROJECT_ROOT / "scripts/view_change/brainco_hand_retarget.py"
)
PY_DEX_RETARGET = PY_ST
DEX_RETARGETING_SRC = (
    RESOURCE_ROOT / "xr_teleoperate/teleop/robot_control/dex-retargeting/src"
)
BRAINCO_ROOT = RESOURCE_ROOT / "brainco"
BRAINCO_MJCF = (
    BRAINCO_ROOT / "robots/g1_brainco/g1_mocap_29dof_with_brainco_hands.xml"
)
BRAINCO_MESHES = BRAINCO_ROOT / "robots/g1_brainco/meshes"
BRAINCO_HAND_URDF_DIR = BRAINCO_ROOT / "robots/hands"
BRAINCO_RETARGET_CONFIG_DIR = BRAINCO_ROOT / "configs/retarget"

HAND_MODE_DEX3 = "dex3"
HAND_MODE_BRAINCO = "g1_brainco"
HAND_MODES = (HAND_MODE_DEX3, HAND_MODE_BRAINCO)
# Dex3 carries 7 actuated joints per hand; BrainCo carries 6 active joints per
# hand and resolves its 5 mimic joints per hand at render time.
HAND_DOF_BY_MODE = {HAND_MODE_DEX3: 14, HAND_MODE_BRAINCO: 12}
HAND_CONTRACT_BY_MODE = {
    HAND_MODE_DEX3: "right_wrist_yaw_link_SE3_plus_right_Dex3_7DoF",
    HAND_MODE_BRAINCO: "dual_wrist_yaw_link_SE3_plus_g1_brainco_6DoF_per_hand",
}
BUILD_REFERENCE = ST / "scripts/eval_session/build_torso_umi3_reference.py"
SMOOTH_REFERENCE = PROJECT_ROOT / "scripts/smooth_three_point_reference.py"
KIMODO_INIT = ST / "scripts/eval_session/run_clean_kimodo_init_sweep.py"
MINK_REFINE = ST / "scripts/eval_session/refine_three_point_kinematics_mink.py"
SUMMARIZE = ST / "scripts/eval_session/summarize_three_point_tracking.py"
BUILD_ARTIFACT = HYDRAX / "scripts/pi_mjx_sim2sim/build_torso_umi3_artifacts.py"
JAX_RUNNER = HYDRAX / "scripts/pi_mjx_sim2sim/diagnostics/jax_closed_loop_session_3point.py"
JAX_WORKER = HYDRAX / "scripts/pi_mjx_sim2sim/diagnostics/jax_mpc_worker.py"
DIRECT_PARITY = (
    HYDRAX
    / "scripts/pi_mjx_sim2sim/diagnostics/direct_policy_parity_torso_umi3.py"
)
ARTIFACT_DIR = HYDRAX / "scripts/pi_mjx_sim2sim/artifacts"
ARTIFACT_MANIFEST = ARTIFACT_DIR / "artifact_manifest.json"
CHECKPOINT = Path(os.environ.get("KODA_SCALEBFM_CHECKPOINT", str(ASSET_ROOT / "models/scalebfm/model_22200.pt"))).resolve()
MPC_BACKEND_JAX = "jax"
JAX_RUN_NAME = "jax_mpc_3point_n256_h36_i4_x12_seed2"
JAX_PIPELINE_CONFIG_ID = (
    "torso_umi3_model22200_n256_h36_i4_x12_seed2_"
    "full50_template_palm_joint_contact_savgol_mink_v6"
)
RUN_NAME = JAX_RUN_NAME
PIPELINE_CONFIG_ID = JAX_PIPELINE_CONFIG_ID
FINAL_FILES = ("kinematics.npz", "meta.json", "tracking_error.json")
# A batch worker is pinned to one GPU by the view-change launcher.  The old
# single lock serialized every session across the machine, defeating the
# multi-GPU batch watcher.  Keep an explicit override for standalone runs,
# otherwise scope the lock to the worker's GPU.
_LOCK_GPU = os.environ.get("GPU") or os.environ.get("CUDA_VISIBLE_DEVICES") or "default"
_LOCK_GPU = _LOCK_GPU.replace(",", "_").replace("/", "_")
GLOBAL_LOCK_PATH = Path(os.environ.get("THREE_POINT_BATCH_LOCK", str(PROJECT_ROOT / ".cache" / f"torso_umi3_batch_{_LOCK_GPU}.lock")))
SOURCE_CALIBRATION_RELATIVE = Path("calibrations/R_optical_to_link.npy")
SCALETRACK_G1_ASSETS = ST / "source/scaletrack/scaletrack/assets/robots/g1_29dof"
SCALETRACK_G1_XML = SCALETRACK_G1_ASSETS / "g1_29dof.xml"
MJX_G1_ASSETS = (
    RESOURCE_ROOT / "ScaleBFM/scalebridge/data/robot/g1_29dof"
)
MJX_G1_XML = MJX_G1_ASSETS / "g1_29dof_mjx.xml"
KIMODO_PACKAGE = RESOURCE_ROOT / "kimodo/kimodo"
KIMODO_MODEL = Path(os.environ.get("KODA_KIMODO_MODEL", str(ASSET_ROOT / "models/kimodo/Kimodo-G1-RP-v1/model.safetensors"))).resolve()
KIMODO_MODEL_CONFIG = Path(os.environ.get("KODA_KIMODO_CONFIG", str(ASSET_ROOT / "models/kimodo/Kimodo-G1-RP-v1/config.yaml"))).resolve()
KIMODO_MODEL_STATS = Path(os.environ.get("KODA_KIMODO_STATS", str(ASSET_ROOT / "models/kimodo/Kimodo-G1-RP-v1/stats/motion"))).resolve()
KIMODO_G1_XML = KIMODO_PACKAGE / "assets/skeletons/g1skel34/xml/g1.xml"
KIMODO_G1_JOINTS = KIMODO_PACKAGE / "assets/skeletons/g1skel34/joints.p"
KIMODO_G1_REST_POSE = (
    KIMODO_PACKAGE / "assets/skeletons/g1skel34/rest_pose_local_rot.p"
)
KIMODO_TEXT_ENCODERS = Path(os.environ.get("KODA_KIMODO_TEXT_ENCODERS", str(ASSET_ROOT / "models/kimodo/text_encoders"))).resolve()
MOTION_SCHEMA_TEMPLATE = RESOURCE_ROOT / "motion_schema_template.npz"
ACTIVE_PROCESSES: dict[int, tuple[subprocess.Popen, Any]] = {}
ACTIVE_PROCESSES_LOCK = threading.Lock()
STOP_REQUESTED = threading.Event()


@dataclass(frozen=True)
class GPUInfo:
    selector: str
    index: str
    uuid: str
    name: str
    memory_free_mb: int
    utilization_percent: int


@dataclass(frozen=True)
class SessionProbe:
    session_id: str
    session_dir: Path
    source_frames: int
    source_duration_s: float


@dataclass(frozen=True)
class JointContactBundle:
    """One precomputed wrist/hand/contact bundle for a source session."""

    wrist_candidate: Path
    hand_kinematics: Path
    manifest: Path
    contact_json: Path
    contact_status: str
    left_wrist_candidate: Path | None = None
    right_wrist_candidate: Path | None = None
    inject_sides: tuple[str, ...] = ("right",)

    @property
    def inject_wrist_residual(self) -> bool:
        return bool(self.inject_sides)

    @property
    def fingerprint(self) -> str:
        return joint_contact_bundle_sha256(
            self.wrist_candidate, self.hand_kinematics, self.manifest
        )


@dataclass
class JobResult:
    session_id: str
    status: str
    gpu: str | None
    elapsed_s: float
    output_dir: str | None = None
    frames: int | None = None
    accepted: bool | None = None
    error: str | None = None
    failed_stage: str | None = None
    log: str | None = None
    work_dir: str | None = None


@dataclass(frozen=True)
class Q0MinkOptions:
    """Optional frame-0 Mink initialization settings.

    The normal (frame >= 1) solver is intentionally unaffected by these
    options.  Keeping the settings as a small value object makes it harder for
    the batch runner and the single-session refinement command to drift apart.
    """

    zero_arm_no_smoothness: bool = False
    max_arm_delta: float = 3.0
    arm_velocity_limit: float = 1.0e4
    posture_cost: float = 0.0
    orientation_cost: float = 0.08
    collision_margin: float | None = None
    shoulder_clearance: bool = False
    shoulder_clearance_cost: float = 0.0
    shoulder_clearance_threshold: float = 0.06
    shoulder_clearance_sample_start: float = 0.40
    shoulder_clearance_sample_count: int = 6
    open_branch_fallback: bool = False
    min_shoulder_clearance: float = 0.01

    @property
    def enabled(self) -> bool:
        return bool(
            self.zero_arm_no_smoothness
            or self.shoulder_clearance
            or self.shoulder_clearance_cost > 0.0
        )

    def command_args(self) -> list[str]:
        """Return arguments accepted by ``refine_three_point_kinematics_mink``."""
        if not self.enabled:
            return []
        # A clearance task is only evaluated in the q0 branch.  Enabling any
        # q0 option therefore explicitly selects the zero-arm initialization.
        result = ["--q0-zero-arm-no-smoothness"]
        result.extend(("--q0-max-arm-delta", str(self.max_arm_delta)))
        result.extend(("--q0-arm-velocity-limit", str(self.arm_velocity_limit)))
        result.extend(("--q0-posture-cost", str(self.posture_cost)))
        result.extend(("--q0-orientation-cost", str(self.orientation_cost)))
        if self.collision_margin is not None:
            result.extend(("--q0-collision-margin", str(self.collision_margin)))
        if self.shoulder_clearance:
            result.append("--q0-shoulder-clearance")
        result.extend(
            (
                "--q0-shoulder-clearance-cost",
                str(self.shoulder_clearance_cost),
                "--q0-shoulder-clearance-threshold",
                str(self.shoulder_clearance_threshold),
                "--q0-shoulder-clearance-sample-start",
                str(self.shoulder_clearance_sample_start),
                "--q0-shoulder-clearance-sample-count",
                str(self.shoulder_clearance_sample_count),
            )
        )
        if self.open_branch_fallback:
            result.append("--q0-open-branch-fallback")
        result.extend(("--q0-min-shoulder-clearance", str(self.min_shoulder_clearance)))
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "zero_arm_no_smoothness": self.zero_arm_no_smoothness,
            "max_arm_delta": self.max_arm_delta,
            "arm_velocity_limit": self.arm_velocity_limit,
            "posture_cost": self.posture_cost,
            "orientation_cost": self.orientation_cost,
            "collision_margin": self.collision_margin,
            "shoulder_clearance": self.shoulder_clearance,
            "shoulder_clearance_cost": self.shoulder_clearance_cost,
            "shoulder_clearance_threshold": self.shoulder_clearance_threshold,
            "shoulder_clearance_sample_start": self.shoulder_clearance_sample_start,
            "shoulder_clearance_sample_count": self.shoulder_clearance_sample_count,
            "open_branch_fallback": self.open_branch_fallback,
            "min_shoulder_clearance": self.min_shoulder_clearance,
            "command_args": self.command_args(),
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ReferenceSmoothingOptions:
    enabled: bool = True
    window_length: int = 5
    polyorder: int = 2
    gap_factor: float = 3.0
    wrist_outlier_max_duration_s: float = 1.1
    wrist_outlier_position_floor_m: float = 0.05
    wrist_outlier_orientation_floor_deg: float = 45.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "method": "bounded wrist excursion repair then timestamp-aware local Savitzky-Golay",
            "window_length": self.window_length,
            "polyorder": self.polyorder,
            "gap_factor": self.gap_factor,
            "wrist_outlier_repair": {
                "enabled": True,
                "max_event_duration_s": self.wrist_outlier_max_duration_s,
                "position_jump_floor_m": self.wrist_outlier_position_floor_m,
                "orientation_jump_floor_deg": self.wrist_outlier_orientation_floor_deg,
            },
            "tracking_bodies": [
                "torso_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
            "stage": "post_contact_pre_tracking",
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class StageError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def joint_contact_bundle_sha256(
    wrist_candidate: Path, hand_kinematics: Path, contact_manifest: Path
) -> str:
    payload = {
        "wrist_candidate_sha256": sha256(wrist_candidate),
        "hand_kinematics_sha256": sha256(hand_kinematics),
        "contact_manifest_sha256": sha256(contact_manifest),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def resolve_contact_aware_candidate(
    root: Path | None, session_id: str
) -> Path | None:
    """Resolve one per-session feedback1 candidate without guessing silently."""
    if root is None:
        return None
    root = root.expanduser().resolve()
    if root.is_file():
        return root
    session_root = root / session_id
    if not session_root.is_dir():
        raise FileNotFoundError(
            f"contact-aware session directory is missing: {session_root}"
        )
    candidates = sorted(
        session_root.rglob("contact_aware_wrist_candidates_native.npz")
    )
    if not candidates:
        raise FileNotFoundError(
            f"no contact_aware_wrist_candidates_native.npz under {session_root}"
        )
    feedback = [path for path in candidates if "feedback1" in path.parts]
    if len(feedback) == 1:
        return feedback[0]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        "multiple contact-aware candidates found; use a root containing one "
        f"feedback1 candidate for {session_id}: {candidates}"
    )


def resolve_joint_contact_bundle(
    root: Path | None,
    session_id: str,
    *,
    hand_mode: str = HAND_MODE_DEX3,
) -> JointContactBundle | None:
    """Resolve a complete precomputed bundle without rerunning contact stages."""
    if root is None:
        return None
    root = root.expanduser().resolve()
    candidates = (
        root,
        root / "contact_aware_auto",
        root / session_id,
        root / session_id / "contact_aware_auto",
    )
    bundle_root = next(
        (
            path
            for path in candidates
            if (path / "contact_precompute_manifest.json").is_file()
        ),
        None,
    )
    if bundle_root is None:
        raise FileNotFoundError(
            f"joint-contact bundle is missing for {session_id} under {root}"
        )
    manifest = bundle_root / "contact_precompute_manifest.json"
    contact_json = bundle_root / "contact_frames.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    mode = str(payload.get("hand_mode") or hand_mode)
    if mode not in HAND_MODES:
        mode = hand_mode
    contact_status = str(payload.get("status", "unknown"))
    joint_status = str(
        payload.get("joint_contact_optimization", {}).get("status", "missing")
    )
    contact_payload: dict[str, Any] = {}
    if contact_json.is_file():
        contact_payload = json.loads(contact_json.read_text(encoding="utf-8"))
    if mode == HAND_MODE_BRAINCO:
        left_wrist = bundle_root / "contact_aware_wrist_candidates_native_left.npz"
        right_wrist = bundle_root / "contact_aware_wrist_candidates_native_right.npz"
        base_hand = bundle_root / "joint_contact/base_hand_brainco_50hz.npz"
        optimized_hand = bundle_root / "hand_brainco_50hz_joint_contact.npz"
        wrist_candidate = right_wrist if right_wrist.is_file() else left_wrist
        for path in (manifest, contact_json, base_hand, wrist_candidate):
            if not path.is_file():
                raise FileNotFoundError(path)
        left_frames = list(contact_payload.get("left_contact_frames") or [])
        right_frames = list(contact_payload.get("right_contact_frames") or [])
        inject_sides = tuple(
            side
            for side, frames, candidate in (
                ("left", left_frames, left_wrist),
                ("right", right_frames, right_wrist),
            )
            if frames and candidate.is_file()
        )
        if contact_status.startswith("contact_pointcloud"):
            if joint_status != "enabled" or not optimized_hand.is_file():
                raise ValueError(
                    f"{session_id}: contact exists but the paired optimized hand "
                    f"is unavailable (joint status={joint_status})"
                )
            hand_kinematics = optimized_hand
        else:
            hand_kinematics = base_hand
        return JointContactBundle(
            wrist_candidate=wrist_candidate,
            hand_kinematics=hand_kinematics,
            manifest=manifest,
            contact_json=contact_json,
            contact_status=contact_status,
            left_wrist_candidate=left_wrist if left_wrist.is_file() else None,
            right_wrist_candidate=right_wrist if right_wrist.is_file() else None,
            inject_sides=inject_sides,
        )

    wrist_candidate = bundle_root / "contact_aware_wrist_candidates_native.npz"
    base_hand = bundle_root / "joint_contact/base_hand_dex3_50hz.npz"
    optimized_hand = bundle_root / "hand_dex3_50hz_joint_contact.npz"
    for path in (wrist_candidate, manifest, contact_json, base_hand):
        if not path.is_file():
            raise FileNotFoundError(path)
    if contact_status.startswith("contact_pointcloud"):
        if joint_status != "enabled" or not optimized_hand.is_file():
            raise ValueError(
                f"{session_id}: contact exists but the paired optimized hand "
                f"is unavailable (joint status={joint_status})"
            )
        hand_kinematics = optimized_hand
        inject_sides = ("right",)
    else:
        hand_kinematics = base_hand
        inject_sides = ()
    return JointContactBundle(
        wrist_candidate=wrist_candidate,
        hand_kinematics=hand_kinematics,
        manifest=manifest,
        contact_json=contact_json,
        contact_status=contact_status,
        inject_sides=inject_sides,
    )


def hash_dependency_path(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return {
            "path": str(resolved),
            "kind": "file",
            "sha256": sha256(resolved),
            "size_bytes": resolved.stat().st_size,
        }
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    digest = hashlib.sha256()
    file_count = 0
    size_bytes = 0
    for child in sorted(item for item in resolved.rglob("*") if item.is_file()):
        relative = child.relative_to(resolved).as_posix()
        child_hash = sha256(child)
        child_size = child.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(child_hash.encode("ascii"))
        digest.update(b"\0")
        file_count += 1
        size_bytes += child_size
    return {
        "path": str(resolved),
        "kind": "directory",
        "sha256": digest.hexdigest(),
        "file_count": file_count,
        "size_bytes": size_bytes,
    }


def build_dependency_manifest(
    paths: dict[str, Path],
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entries = {
        label: hash_dependency_path(path)
        for label, path in sorted(paths.items())
    }
    fingerprint_payload: dict[str, Any] = {
        "pipeline_config_id": PIPELINE_CONFIG_ID,
        "dependencies": {
            label: entry["sha256"] for label, entry in entries.items()
        },
    }
    if extra_payload:
        fingerprint_payload.update(extra_payload)
    encoded = json.dumps(
        fingerprint_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "files": entries,
    }


def pipeline_dependency_manifest() -> dict[str, Any]:
    diagnostics = HYDRAX / "scripts/pi_mjx_sim2sim/diagnostics"
    scripts = HYDRAX / "scripts/pi_mjx_sim2sim"
    eval_scripts = ST / "scripts/eval_session"
    paths = {
        "artifact_manifest": ARTIFACT_MANIFEST,
        "batch_driver": Path(__file__),
        "prepare_dual_wrist_reference": PREPARE,
        "retarget_template_palm_reference": RETARGET_WRIST_REFERENCE,
        "inject_contact_aware_wrist_reference": INJECT_CONTACT_AWARE_REFERENCE,
        "joint_contact_precompute": JOINT_CONTACT_PRECOMPUTE,
        "joint_contact_hand_retarget_launcher": (
            PROJECT_ROOT / "scripts/view_change/run_hand_retarget_experiment.sh"
        ),
        "joint_contact_hand_retarget": (
            PROJECT_ROOT / "scripts/view_change/hand_retarget_experiment.py"
        ),
        "joint_contact_dex3_position_retarget": (
            PROJECT_ROOT / "src/koda/retargeting/dex3.py"
        ),
        "joint_contact_dex3_retarget_config": (
            PROJECT_ROOT / "src/koda/retargeting/configs/unitree_dex3_position.yml"
        ),
        "joint_contact_dex3_evaluation": PROJECT_ROOT / "src/koda/retargeting/evaluation.py",
        "joint_contact_dex3_kinematics": PROJECT_ROOT / "src/koda/kinematics/dex3.py",
        "joint_contact_dex3_xml": RESOURCE_ROOT / "robots/g1_dex3/g1_with_visual_dex3_hands.xml",
        "joint_contact_object_mask_selector": (
            PROJECT_ROOT / "scripts/view_change/select_contact_mask_window.py"
        ),
        "joint_contact_human_mask": (
            PROJECT_ROOT / "scripts/view_change/make_human_arm_masks.py"
        ),
        "joint_contact_sam3_backend": (
            PROJECT_ROOT / "scripts/view_change/sam3_efficientsam3_hand_masks.py"
        ),
        "joint_contact_optimizer": (
            PROJECT_ROOT
            / "scripts/view_change/optimize_contact_aware_wrist_and_fingers.py"
        ),
        "joint_contact_detector": (
            PROJECT_ROOT / "scripts/view_change/contact_pointcloud_analysis.py"
        ),
        "joint_contact_wrist_optimizer": (
            PROJECT_ROOT / "scripts/view_change/optimize_contact_aware_wrist.py"
        ),
        "joint_contact_object_interaction": (
            PROJECT_ROOT / "scripts/view_change/evaluate_hand_object_interaction.py"
        ),
        "joint_contact_identity_candidate": (
            PROJECT_ROOT / "scripts/view_change/make_identity_contact_candidate.py"
        ),
        "joint_contact_schedule": (
            PROJECT_ROOT / "scripts/view_change/expand_contact_wrist_schedule.py"
        ),
        "brainco_hand_retarget": BRAINCO_HAND_RETARGET,
        "brainco_mjcf": BRAINCO_MJCF,
        "brainco_hand_urdf_left": (
            BRAINCO_HAND_URDF_DIR / "brainco_hand/brainco_hand_left.urdf"
        ),
        "brainco_hand_urdf_right": (
            BRAINCO_HAND_URDF_DIR / "brainco_hand/brainco_hand_right.urdf"
        ),
        "brainco_retarget_config_left": (
            BRAINCO_RETARGET_CONFIG_DIR / "brainco_hand_left.yml"
        ),
        "brainco_retarget_config_right": (
            BRAINCO_RETARGET_CONFIG_DIR / "brainco_hand_right.yml"
        ),
        "motion_schema_template": MOTION_SCHEMA_TEMPLATE,
        "build_three_point_reference": BUILD_REFERENCE,
        "smooth_three_point_reference": SMOOTH_REFERENCE,
        "savgol_reference_implementation": (
            PROJECT_ROOT / "scripts/smooth_wrist_reference.py"
        ),
        "kimodo_init": KIMODO_INIT,
        "kimodo_constraint_helper": (
            eval_scripts / "package_session_kimodo_constraint_variants.py"
        ),
        "kimodo_model": KIMODO_MODEL,
        "kimodo_model_config": KIMODO_MODEL_CONFIG,
        "kimodo_joint_schema": KIMODO_G1_XML,
        "kimodo_neutral_joints": KIMODO_G1_JOINTS,
        "kimodo_rest_pose": KIMODO_G1_REST_POSE,
        "scaletrack_g1_xml": SCALETRACK_G1_XML,
        "jax_session_runner_three_point": JAX_RUNNER,
        "jax_rollout_three_point": diagnostics / "jax_closed_loop_three_point.py",
        "jax_rollout_base": diagnostics / "jax_closed_loop_push1.py",
        "jax_session_base": diagnostics / "jax_closed_loop_session.py",
        "mjx_environment_three_point": scripts / "mjx_pi_env_three_point.py",
        "mjx_environment_base": scripts / "mjx_pi_env.py",
        "cem_three_point": scripts / "cem_three_point.py",
        "cem_wrist_base": scripts / "cem_wrist.py",
        "policy_inputs": scripts / "policy_inputs.py",
        "policy_loader": scripts / "load_eager_policy.py",
        "policy_numeric_implementation": scripts / "_onboard_policy_lib.py",
        "mjx_g1_xml": MJX_G1_XML,
        "mink_refinement": MINK_REFINE,
        "mink_solver": PROJECT_ROOT / "src/koda/ik/mink_wrist.py",
        "mink_joint_helpers": PROJECT_ROOT / "src/koda/ik/wrist.py",
        "tracking_summary": SUMMARIZE,
        "tracking_metric_helpers": eval_scripts / "compare_torso_umi3_tracking.py",
    }
    kimodo_sources = [
        KIMODO_PACKAGE / "__init__.py",
        KIMODO_PACKAGE / "assets/__init__.py",
        KIMODO_PACKAGE / "constraints.py",
        KIMODO_PACKAGE / "geometry.py",
        KIMODO_PACKAGE / "postprocess.py",
        KIMODO_PACKAGE / "sanitize.py",
        KIMODO_PACKAGE / "tools.py",
        KIMODO_PACKAGE / "exports/__init__.py",
        KIMODO_PACKAGE / "exports/motion_io.py",
        KIMODO_PACKAGE / "exports/mujoco.py",
    ]
    for package_dir in ("model", "motion_rep", "skeleton"):
        kimodo_sources.extend(
            sorted((KIMODO_PACKAGE / package_dir).rglob("*.py"))
        )
    for source in sorted(set(kimodo_sources)):
        label = source.relative_to(KIMODO_PACKAGE).as_posix()
        paths[f"kimodo_source/{label}"] = source
    for stats_file in sorted(KIMODO_MODEL_STATS.rglob("*.npy")):
        label = stats_file.relative_to(KIMODO_MODEL_STATS).as_posix()
        paths[f"kimodo_stats/{label}"] = stats_file
    for encoder_file in sorted(
        item for item in KIMODO_TEXT_ENCODERS.rglob("*") if item.is_file()
    ):
        if (
            encoder_file.suffix in {".json", ".py"}
            or encoder_file.name == ".msc"
            or encoder_file.name == "adapter_model.safetensors"
        ):
            label = encoder_file.relative_to(KIMODO_TEXT_ENCODERS).as_posix()
            paths[f"kimodo_text_encoder/{label}"] = encoder_file
    # MuJoCo re-centers mesh geoms at compile time, so the palm frames that
    # seed hand refinement depend on the mesh files themselves.
    for assets_root, prefix in (
        (SCALETRACK_G1_ASSETS / "meshes", "scaletrack_mesh"),
        (MJX_G1_ASSETS / "meshes", "mjx_mesh"),
        (RESOURCE_ROOT / "robots/g1_dex3", "dex3_asset"),
        (BRAINCO_MESHES, "brainco_mesh"),
        (BRAINCO_ROOT / "robots/g1_base/meshes", "brainco_base_mesh"),
    ):
        for asset in sorted(item for item in assets_root.rglob("*") if item.is_file()):
            label = asset.relative_to(assets_root).as_posix()
            paths[f"{prefix}/{label}"] = asset
    return build_dependency_manifest(paths)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def acquire_global_lock() -> Any:
    GLOBAL_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    stream = GLOBAL_LOCK_PATH.open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        stream.seek(0)
        owner = stream.read().strip() or "unknown process"
        stream.close()
        raise RuntimeError(
            f"another three-point batch is active ({owner}); lock={GLOBAL_LOCK_PATH}"
        ) from error
    stream.seek(0)
    stream.truncate()
    stream.write(f"pid={os.getpid()} started_at={utc_now()}\n")
    stream.flush()

    def release() -> None:
        if stream.closed:
            return
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

    atexit.register(release)
    return stream


def register_process(process: subprocess.Popen, log: Any) -> None:
    with ACTIVE_PROCESSES_LOCK:
        ACTIVE_PROCESSES[process.pid] = (process, log)


def unregister_process(process: subprocess.Popen) -> None:
    with ACTIVE_PROCESSES_LOCK:
        ACTIVE_PROCESSES.pop(process.pid, None)


def terminate_active_processes() -> None:
    with ACTIVE_PROCESSES_LOCK:
        active = list(ACTIVE_PROCESSES.values())
    for process, log in active:
        try:
            terminate_process_group(process, log)
        except Exception as error:
            try:
                log.write(f"[{utc_now()}] signal cleanup failed: {error}\n")
                log.flush()
            except Exception:
                pass


def handle_termination_signal(signum: int, _frame: Any) -> None:
    STOP_REQUESTED.set()
    terminate_active_processes()
    if signum == signal.SIGINT:
        raise KeyboardInterrupt
    raise SystemExit(128 + signum)


def install_signal_handlers() -> None:
    signal.signal(signal.SIGINT, handle_termination_signal)
    signal.signal(signal.SIGTERM, handle_termination_signal)


def query_gpus() -> list[GPUInfo]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stdout.strip()}")
    inventory: list[GPUInfo] = []
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",", 4)]
        if len(parts) != 5:
            raise RuntimeError(f"unexpected nvidia-smi row: {line!r}")
        index, gpu_uuid, name, memory_free, utilization = parts
        inventory.append(
            GPUInfo(
                selector=index,
                index=index,
                uuid=gpu_uuid,
                name=name,
                memory_free_mb=int(memory_free),
                utilization_percent=int(utilization),
            )
        )
    return inventory


def inherited_visible_gpus(inventory: Sequence[GPUInfo]) -> list[GPUInfo]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return list(inventory)
    tokens = [item.strip() for item in raw.split(",") if item.strip()]
    if not tokens:
        return []
    visible: list[GPUInfo] = []
    for token in tokens:
        match = next(
            (
                gpu
                for gpu in inventory
                if token == gpu.index
                or token == gpu.uuid
                or gpu.uuid.startswith(token)
            ),
            None,
        )
        if match is None:
            raise ValueError(
                f"CUDA_VISIBLE_DEVICES contains unknown physical selector {token!r}"
            )
        if match.uuid not in {gpu.uuid for gpu in visible}:
            visible.append(match)
    return visible


def select_gpus(
    value: str,
    *,
    min_free_mb: int,
    max_utilization: int,
) -> list[GPUInfo]:
    requested = value.strip()
    if requested.lower() == "auto":
        inventory = inherited_visible_gpus(query_gpus())
        if not inventory:
            raise RuntimeError("no GPU is visible through CUDA_VISIBLE_DEVICES")
        selected = [
            gpu
            for gpu in inventory
            if gpu.memory_free_mb >= min_free_mb
            and gpu.utilization_percent <= max_utilization
        ]
        if not selected:
            raise RuntimeError(
                "no GPU satisfies the automatic selection thresholds: "
                f"free>={min_free_mb} MiB, utilization<={max_utilization}%"
            )
        return selected

    selectors = [item.strip() for item in requested.split(",") if item.strip()]
    if not selectors:
        raise ValueError("--gpus must be 'auto' or a comma-separated GPU list")

    # The batch watcher already assigns a concrete physical GPU to each
    # worker.  In restricted/containerized launches nvidia-smi can be
    # unavailable even though CUDA itself is usable.  Trust explicit numeric
    # selectors in that mode instead of rejecting every worker at preflight.
    try:
        inventory = inherited_visible_gpus(query_gpus())
    except RuntimeError:
        if all(selector.isdigit() for selector in selectors):
            return [
                GPUInfo(
                    selector=selector,
                    index=selector,
                    uuid=f"explicit-{selector}",
                    name="explicit GPU selector",
                    memory_free_mb=0,
                    utilization_percent=0,
                )
                for selector in selectors
            ]
        raise
    if not inventory:
        raise RuntimeError("no GPU is visible through CUDA_VISIBLE_DEVICES")
    selected: list[GPUInfo] = []
    for selector in selectors:
        match = next(
            (
                gpu
                for gpu in inventory
                if selector == gpu.index
                or selector == gpu.uuid
                or gpu.uuid.startswith(selector)
            ),
            None,
        )
        if match is None:
            raise ValueError(f"unknown GPU selector {selector!r}")
        selected.append(
            GPUInfo(
                selector=selector,
                index=match.index,
                uuid=match.uuid,
                name=match.name,
                memory_free_mb=match.memory_free_mb,
                utilization_percent=match.utilization_percent,
            )
        )
    if len({gpu.uuid for gpu in selected}) != len(selected):
        raise ValueError("--gpus contains duplicate devices")
    return selected


def probe_session(session_dir: Path) -> SessionProbe:
    annotation = session_dir / "annotation.hdf5"
    calibration = session_dir / SOURCE_CALIBRATION_RELATIVE
    if not annotation.is_file():
        raise FileNotFoundError(annotation)
    if not calibration.is_file():
        raise FileNotFoundError(calibration)
    calibration_array = np.asarray(np.load(calibration), dtype=np.float64)
    if calibration_array.shape != (3, 3):
        raise ValueError(
            f"calibration must have shape (3, 3), got {calibration_array.shape}"
        )
    if not np.all(np.isfinite(calibration_array)):
        raise ValueError("calibration contains non-finite values")
    with h5py.File(annotation, "r") as archive:
        required = (
            "cam-pose/timestamps",
            "cam-pose/translations",
            "cam-pose/rotations",
            "hand-pose/timestamps",
            "hand-pose/left_joints",
            "hand-pose/right_joints",
        )
        missing = [name for name in required if name not in archive]
        if missing:
            raise KeyError(f"annotation is missing datasets: {missing}")
        camera_t = np.asarray(archive["cam-pose/timestamps"], dtype=np.float64)
        hand_t = np.asarray(archive["hand-pose/timestamps"], dtype=np.float64)
        frames = len(camera_t)
        if frames < 2:
            raise ValueError("session has fewer than two camera frames")
        if len(hand_t) != frames:
            raise ValueError(
                f"camera/hand frame mismatch: {frames} vs {len(hand_t)}"
            )
        if archive["cam-pose/translations"].shape != (frames, 3):
            raise ValueError("invalid cam-pose/translations shape")
        if archive["cam-pose/rotations"].shape != (frames, 3, 3):
            raise ValueError("invalid cam-pose/rotations shape")
        for side in ("left", "right"):
            if archive[f"hand-pose/{side}_joints"].shape != (frames, 21, 3):
                raise ValueError(f"invalid {side} hand joint shape")
        if not np.all(np.isfinite(camera_t)) or np.any(np.diff(camera_t) <= 0):
            raise ValueError("camera timestamps must be finite and increasing")
        if not np.all(np.isfinite(hand_t)) or np.any(np.diff(hand_t) <= 0):
            raise ValueError("hand timestamps must be finite and increasing")
    return SessionProbe(
        session_id=session_dir.name,
        session_dir=session_dir.resolve(),
        source_frames=frames,
        source_duration_s=float(camera_t[-1] - camera_t[0]),
    )


def discover_sessions(input_root: Path, pattern: str) -> tuple[list[SessionProbe], dict[str, str]]:
    probes: list[SessionProbe] = []
    rejected: dict[str, str] = {}
    for session_dir in sorted(input_root.glob(pattern)):
        if not session_dir.is_dir():
            continue
        try:
            probes.append(probe_session(session_dir))
        except Exception as error:  # batch preflight must isolate bad inputs
            rejected[session_dir.name] = str(error)
    if not probes and not rejected:
        raise RuntimeError(f"no session directories matching {pattern!r} in {input_root}")
    return probes, rejected


def limit_discovery(
    sessions: list[SessionProbe],
    rejected: dict[str, str],
    limit: int | None,
) -> tuple[list[SessionProbe], dict[str, str]]:
    if limit is None:
        return sessions, rejected
    selected_ids = set(
        sorted(
            [session.session_id for session in sessions] + list(rejected)
        )[:limit]
    )
    return (
        [session for session in sessions if session.session_id in selected_ids],
        {
            session_id: reason
            for session_id, reason in rejected.items()
            if session_id in selected_ids
        },
    )


def read_session_list(path: Path) -> list[str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    session_ids = []
    seen = set()
    for line_number, raw_line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(), start=1
    ):
        session_id = raw_line.split("#", 1)[0].strip()
        if not session_id:
            continue
        if session_id in seen:
            raise ValueError(
                f"duplicate session ID {session_id!r} in {resolved}:{line_number}"
            )
        seen.add(session_id)
        session_ids.append(session_id)
    if not session_ids:
        raise ValueError(f"session list is empty: {resolved}")
    return session_ids


def select_discovery(
    sessions: list[SessionProbe],
    rejected: dict[str, str],
    session_ids: Sequence[str],
) -> tuple[list[SessionProbe], dict[str, str]]:
    runnable_by_id = {session.session_id: session for session in sessions}
    known = set(runnable_by_id) | set(rejected)
    missing = [session_id for session_id in session_ids if session_id not in known]
    if missing:
        raise ValueError(f"session list contains unknown IDs: {missing}")
    return (
        [runnable_by_id[session_id] for session_id in session_ids if session_id in runnable_by_id],
        {session_id: rejected[session_id] for session_id in session_ids if session_id in rejected},
    )


def output_is_complete(
    output_dir: Path,
    session_id: str,
    *,
    source_annotation: Path,
    source_calibration: Path,
    artifact_manifest_hash: str,
    pipeline_dependency_hash: str,
    orientation_offset_hash: str | None = None,
    contact_aware_hash: str | None = None,
    mink_q0_config_hash: str | None = None,
    reference_smoothing_hash: str | None = None,
    require_joint_contact_hand: bool = False,
    joint_contact_bundle_hash: str | None = None,
    hand_dof: int = HAND_DOF_BY_MODE[HAND_MODE_DEX3],
) -> bool:
    if not all((output_dir / name).is_file() for name in FINAL_FILES):
        return False
    try:
        summary = json.loads((output_dir / "tracking_error.json").read_text())
        metadata = json.loads((output_dir / "meta.json").read_text())
        inputs = summary.get("inputs", {})
        provenance = metadata.get("batch_provenance", {})
        with np.load(output_dir / "kinematics.npz", allow_pickle=False) as data:
            frames = int(np.asarray(data["joint_pos"]).shape[0])
            required = {
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "joint_names",
                "body_names",
                "fps",
                "source_annotation_sha256",
                "source_calibration_sha256",
                "artifact_manifest_sha256",
                "pipeline_dependency_sha256",
                "pipeline_config_id",
                "reference_smoothing_config_sha256",
            }
            if not required.issubset(data.files):
                return False
            if require_joint_contact_hand:
                joint_required = {
                    "hand_qpos",
                    "hand_qvel",
                    "hand_joint_names",
                    "contact_30hz",
                    "contact_50hz",
                    "left_contact",
                    "right_contact",
                    "joint_contact_bundle_sha256",
                    "joint_contact_contract",
                }
                if not joint_required.issubset(data.files):
                    return False
                if np.asarray(data["hand_qpos"]).shape != (frames, hand_dof):
                    return False
                if np.asarray(data["hand_qvel"]).shape != (frames, hand_dof):
                    return False
                if np.asarray(data["hand_joint_names"]).shape != (hand_dof,):
                    return False
                if np.asarray(data["contact_50hz"]).shape != (frames,):
                    return False
                if np.asarray(data["left_contact"]).shape != (frames,):
                    return False
                if np.asarray(data["right_contact"]).shape != (frames,):
                    return False
                if not np.all(np.isin(np.asarray(data["contact_30hz"]), [0, 1])):
                    return False
                if not np.all(np.isin(np.asarray(data["contact_50hz"]), [0, 1])):
                    return False
                if not np.all(np.isin(np.asarray(data["left_contact"]), [0, 1])):
                    return False
                if not np.all(np.isin(np.asarray(data["right_contact"]), [0, 1])):
                    return False
                if not np.array_equal(
                    np.asarray(data["contact_50hz"]),
                    np.maximum(
                        np.asarray(data["left_contact"]),
                        np.asarray(data["right_contact"]),
                    ),
                ):
                    return False
            npz_annotation_hash = str(
                np.asarray(data["source_annotation_sha256"]).reshape(()).item()
            )
            npz_calibration_hash = str(
                np.asarray(data["source_calibration_sha256"]).reshape(()).item()
            )
            npz_artifact_hash = str(
                np.asarray(data["artifact_manifest_sha256"]).reshape(()).item()
            )
            npz_dependency_hash = str(
                np.asarray(data["pipeline_dependency_sha256"]).reshape(()).item()
            )
            npz_config_id = str(
                np.asarray(data["pipeline_config_id"]).reshape(()).item()
            )
            npz_offset_hash = (
                str(
                    np.asarray(data["wrist_orientation_offset_sha256"])
                    .reshape(())
                    .item()
                )
                if "wrist_orientation_offset_sha256" in data.files
                else None
            )
            npz_contact_hash = (
                str(
                    np.asarray(data["contact_aware_reference_sha256"])
                    .reshape(())
                    .item()
                )
                if "contact_aware_reference_sha256" in data.files
                else None
            )
            npz_q0_hash = (
                str(np.asarray(data["mink_q0_config_sha256"]).reshape(()).item())
                if "mink_q0_config_sha256" in data.files
                else None
            )
            npz_smoothing_hash = str(
                np.asarray(data["reference_smoothing_config_sha256"])
                .reshape(())
                .item()
            )
            npz_joint_bundle_hash = (
                str(
                    np.asarray(data["joint_contact_bundle_sha256"])
                    .reshape(())
                    .item()
                )
                if "joint_contact_bundle_sha256" in data.files
                else None
            )
        meta_frames = int(metadata.get("dimensions", {}).get("T", -1))
        summary_frames = int(summary.get("sampling", {}).get("frames", -1))
        annotation_hash = sha256(source_annotation)
        calibration_hash = sha256(source_calibration)
        return (
            summary.get("session_id") == session_id
            and frames == meta_frames == summary_frames
            and inputs.get("pipeline_config_id") == PIPELINE_CONFIG_ID
            and inputs.get("artifact_manifest_sha256") == artifact_manifest_hash
            and inputs.get("pipeline_dependency_sha256")
            == pipeline_dependency_hash
            and inputs.get("source_annotation_sha256") == annotation_hash
            and inputs.get("source_calibration_sha256") == calibration_hash
            and provenance.get("pipeline_config_id") == PIPELINE_CONFIG_ID
            and provenance.get("artifact_manifest_sha256")
            == artifact_manifest_hash
            and provenance.get("pipeline_dependency", {}).get("sha256")
            == pipeline_dependency_hash
            and provenance.get("source_annotation_sha256") == annotation_hash
            and provenance.get("source_calibration_sha256") == calibration_hash
            and npz_config_id == PIPELINE_CONFIG_ID
            and npz_artifact_hash == artifact_manifest_hash
            and npz_dependency_hash == pipeline_dependency_hash
            and npz_annotation_hash == annotation_hash
            and npz_calibration_hash == calibration_hash
            and npz_smoothing_hash == reference_smoothing_hash
            and provenance.get("reference_smoothing_config_sha256")
            == reference_smoothing_hash
            and npz_offset_hash == orientation_offset_hash
            and provenance.get("wrist_orientation_offset_sha256")
            == orientation_offset_hash
            and (
                require_joint_contact_hand
                or (
                    npz_contact_hash == contact_aware_hash
                    and provenance.get("contact_aware_reference_sha256")
                    == contact_aware_hash
                )
            )
            and (
                not require_joint_contact_hand
                or (
                    bool(npz_joint_bundle_hash)
                    and provenance.get("joint_contact_bundle_sha256")
                    == npz_joint_bundle_hash
                    and (
                        joint_contact_bundle_hash is None
                        or npz_joint_bundle_hash == joint_contact_bundle_hash
                    )
                )
            )
            and (
                mink_q0_config_hash is None
                or (
                    npz_q0_hash == mink_q0_config_hash
                    and provenance.get("mink_q0_config_sha256")
                    == mink_q0_config_hash
                )
            )
            and inputs.get("final_kinematics_sha256")
            == sha256(output_dir / "kinematics.npz")
            and inputs.get("final_metadata_sha256")
            == sha256(output_dir / "meta.json")
        )
    except Exception:
        return False


def terminate_process_group(process: subprocess.Popen, log: Any) -> None:
    def group_alive() -> bool:
        try:
            os.killpg(process.pid, 0)
            return True
        except ProcessLookupError:
            return False

    if not group_alive():
        process.wait()
        return
    log.write(f"[{utc_now()}] terminating process group pgid={process.pid}\n")
    log.flush()
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 10.0
    while group_alive() and time.monotonic() < deadline:
        try:
            process.wait(timeout=0.1)
        except subprocess.TimeoutExpired:
            continue
        time.sleep(0.05)
    if not group_alive():
        process.wait()
        return
    log.write(f"[{utc_now()}] killing process group pgid={process.pid}\n")
    log.flush()
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def run_stage(
    stage: str,
    command: Sequence[str | Path],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    accepted_codes: tuple[int, ...] = (0,),
    timeout_s: float | None = None,
) -> int:
    if STOP_REQUESTED.is_set():
        raise StageError(stage, "batch cancellation was requested")
    rendered = [str(item) for item in command]
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] stage={stage}\n")
        log.write("command: " + " ".join(rendered) + "\n")
        log.flush()
        process = subprocess.Popen(
            rendered,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        register_process(process, log)
        try:
            try:
                returncode = process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired as error:
                terminate_process_group(process, log)
                raise StageError(stage, f"stage timed out after {timeout_s}s") from error
            except BaseException:
                terminate_process_group(process, log)
                raise
            if returncode not in accepted_codes:
                terminate_process_group(process, log)
            elapsed = time.monotonic() - started
            log.write(
                f"[{utc_now()}] returncode={returncode} elapsed_s={elapsed:.3f}\n"
            )
        finally:
            unregister_process(process)
    if returncode not in accepted_codes:
        raise StageError(
            stage,
            f"stage {stage!r} exited with code {returncode}; log={log_path}",
        )
    return returncode


def artifact_manifest_valid(manifest_path: Path) -> bool:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint = Path(manifest["checkpoint"]["path"])
        if not checkpoint.is_absolute():
            checkpoint = PROJECT_ROOT / checkpoint
        if not checkpoint.is_file() or sha256(checkpoint) != manifest["checkpoint"]["sha256"]:
            return False
        if checkpoint.resolve() != CHECKPOINT.resolve():
            return False
        if manifest.get("policy", {}).get("selected_body_slots") != [7, 10, 13]:
            return False
        if manifest.get("policy", {}).get("selected_body_names") != [
            "torso_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ]:
            return False
        for label, relative in manifest["files"].items():
            path = manifest_path.parent / relative
            if not path.is_file() or sha256(path) != manifest["artifact_sha256"][label]:
                return False
        return True
    except Exception:
        return False


def ensure_artifact(
    *,
    rebuild: bool,
    log_path: Path,
    stage_timeout_s: float | None,
) -> None:
    if not rebuild and artifact_manifest_valid(ARTIFACT_MANIFEST):
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    run_stage(
        "build_policy_artifact",
        [PY_JAX, BUILD_ARTIFACT, "--output-dir", ARTIFACT_DIR],
        cwd=HYDRAX,
        env=env,
        log_path=log_path,
        timeout_s=stage_timeout_s,
    )
    if not artifact_manifest_valid(ARTIFACT_MANIFEST):
        raise StageError(
            "build_policy_artifact",
            f"artifact validation failed after rebuild: {ARTIFACT_MANIFEST}",
        )


def run_global_parity(
    *,
    session: SessionProbe,
    gpu: str,
    work_root: Path,
    cpu_threads: int,
    log_path: Path,
    stage_timeout_s: float | None,
    orientation_offset_json: Path | None = None,
    contact_aware_candidates: Path | None = None,
    contact_transition_frames: int = 15,
    contact_edge_hold: bool = False,
    reference_smoothing: ReferenceSmoothingOptions = ReferenceSmoothingOptions(),
    max_gap_frames: int = 10,
) -> None:
    work_dir = Path(tempfile.mkdtemp(prefix="policy_parity.", dir=work_root))
    env = worker_environment(gpu, cpu_threads, work_root=work_root)
    try:
        base_dir = work_dir / "base_reference"
        raw_reference = base_dir / "raw_dual_wrist_reference.npz"
        raw_targets = base_dir / "raw_dual_wrist_targets.npz"
        raw_prepare_meta = base_dir / "prepare_meta.json"
        base_reference = base_dir / "template_palm_kabsch_reference.npz"
        targets = base_dir / "template_palm_kabsch_targets.npz"
        prepare_metadata = base_dir / "template_palm_kabsch_meta.json"
        raw_three_point_reference = (
            work_dir / "reference/torso_umi3_reference_raw.npz"
        )
        raw_reference_meta = work_dir / "reference/reference_meta_raw.json"
        reference = raw_three_point_reference
        parity_output = work_dir / "policy_parity.json"
        run_stage(
            "parity_build_dual_wrist_reference",
            [
                PY_ST,
                PREPARE,
                "--episode",
                session.session_dir,
                "--out-dir",
                base_dir,
                "--out-fps",
                "50",
                "--use-full-source",
                "--repair-transient-wrist-outliers",
                "--max-gap-frames",
                str(int(max_gap_frames)),
            ],
            cwd=ST,
            env=env,
            log_path=log_path,
            timeout_s=stage_timeout_s,
        )
        retarget_command: list[str | Path] = [
            PY_ST,
            RETARGET_WRIST_REFERENCE,
            "--annotation-hdf5",
            session.session_dir / "annotation.hdf5",
            "--base-reference",
            raw_reference,
            "--base-targets",
            raw_targets,
            "--prepare-meta",
            raw_prepare_meta,
            "--output-reference",
            base_reference,
            "--output-targets",
            targets,
            "--meta-json",
            prepare_metadata,
        ]
        if orientation_offset_json is not None:
            retarget_command.extend(
                ["--orientation-offset-json", orientation_offset_json]
            )
        run_stage(
            "parity_retarget_template_palm_reference",
            retarget_command,
            cwd=PROJECT_ROOT,
            env=env,
            log_path=log_path,
            timeout_s=stage_timeout_s,
        )
        if contact_aware_candidates is not None:
            contact_reference = base_dir / "contact_aware_reference.npz"
            contact_targets = base_dir / "contact_aware_targets.npz"
            contact_meta = base_dir / "contact_aware_meta.json"
            run_stage(
                "parity_inject_contact_aware_wrist_reference",
                [
                    PY_ST,
                    INJECT_CONTACT_AWARE_REFERENCE,
                    "--base-reference",
                    base_reference,
                    "--base-targets",
                    targets,
                    "--base-meta",
                    prepare_metadata,
                    "--contact-candidates",
                    contact_aware_candidates,
                    "--output-reference",
                    contact_reference,
                    "--output-targets",
                    contact_targets,
                    "--output-meta",
                    contact_meta,
                    "--transition-frames",
                    str(contact_transition_frames),
                ] + (["--edge-hold-policy", "edge_hold"] if contact_edge_hold else []),
                cwd=PROJECT_ROOT,
                env=env,
                log_path=log_path,
                timeout_s=stage_timeout_s,
            )
            base_reference, targets, prepare_metadata = (
                contact_reference,
                contact_targets,
                contact_meta,
            )
        frames, fps = read_reference_shape(base_reference)
        run_stage(
            "parity_build_three_point_reference",
            [
                PY_ST,
                BUILD_REFERENCE,
                "--annotation-hdf5",
                session.session_dir / "annotation.hdf5",
                "--base-reference",
                base_reference,
                "--output-npz",
                raw_three_point_reference,
                "--meta-json",
                raw_reference_meta,
                "--expected-frames",
                str(frames),
                "--expected-fps",
                str(fps),
            ],
            cwd=ST,
            env=env,
            log_path=log_path,
            timeout_s=stage_timeout_s,
        )
        if reference_smoothing.enabled:
            smoothed_reference = work_dir / "reference/torso_umi3_reference.npz"
            smoothed_targets = work_dir / "reference/smoothed_targets.npz"
            smoothing_meta = work_dir / "reference/reference_smoothing_meta.json"
            run_stage(
                "parity_smooth_three_point_reference",
                [
                    PY_ST,
                    SMOOTH_REFERENCE,
                    "--input-reference",
                    raw_three_point_reference,
                    "--input-targets",
                    targets,
                    "--output-reference",
                    smoothed_reference,
                    "--output-targets",
                    smoothed_targets,
                    "--meta-json",
                    smoothing_meta,
                    "--window-length",
                    str(reference_smoothing.window_length),
                    "--polyorder",
                    str(reference_smoothing.polyorder),
                    "--gap-factor",
                    str(reference_smoothing.gap_factor),
                    "--wrist-outlier-max-duration-s",
                    str(reference_smoothing.wrist_outlier_max_duration_s),
                    "--wrist-outlier-position-floor-m",
                    str(reference_smoothing.wrist_outlier_position_floor_m),
                    "--wrist-outlier-orientation-floor-deg",
                    str(reference_smoothing.wrist_outlier_orientation_floor_deg),
                ],
                cwd=PROJECT_ROOT,
                env=env,
                log_path=log_path,
                timeout_s=stage_timeout_s,
            )
            reference = smoothed_reference
        run_stage(
            "policy_torch_jax_parity",
            [
                PY_JAX,
                DIRECT_PARITY,
                "--motion",
                reference,
                "--artifacts",
                ARTIFACT_DIR,
                "--artifact-manifest",
                ARTIFACT_MANIFEST,
                "--torch-device",
                "cuda",
                "--output",
                parity_output,
            ],
            cwd=HYDRAX,
            env=env,
            log_path=log_path,
            timeout_s=stage_timeout_s,
        )
        if not parity_output.is_file():
            raise StageError(
                "policy_torch_jax_parity",
                f"parity output was not created: {parity_output}",
            )
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir)


def worker_environment(
    gpu: str,
    cpu_threads: int,
    *,
    work_root: Path | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
            "OMP_NUM_THREADS": str(cpu_threads),
            "MKL_NUM_THREADS": str(cpu_threads),
            "OPENBLAS_NUM_THREADS": str(cpu_threads),
            "NUMEXPR_NUM_THREADS": str(cpu_threads),
            "CHECKPOINT_DIR": str((RESOURCE_ROOT / "../assets/models/kimodo").resolve()),
            "LOCAL_CACHE": "True",
            "TEXT_ENCODER_MODE": "local",
            "TEXT_ENCODERS_DIR": str((RESOURCE_ROOT / "../assets/models/kimodo/text_encoders").resolve()),
            "TEXT_ENCODER_DEVICE": "cpu",
        }
    )
    # Resident Kimodo / JAX sockets are inherited from os.environ.copy().
    if not str(env.get("JAX_COMPILATION_CACHE_DIR", "")).strip():
        cache_root = (work_root or Path(DEFAULT_WORK_ROOT)) / "jax_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        env["JAX_COMPILATION_CACHE_DIR"] = str(cache_root)
    return env


def should_run_parity(args: argparse.Namespace | None = None) -> bool:
    if args is not None and bool(getattr(args, "run_parity", False)):
        return True
    raw = os.environ.get("THREE_POINT_RUN_PARITY", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def json_socket_request(
    sock_path: str | Path,
    payload: dict[str, Any],
    *,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    encoded = (json.dumps(payload) + "\n").encode("utf-8")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        if timeout_s is not None:
            sock.settimeout(float(timeout_s))
        sock.connect(str(sock_path))
        sock.sendall(encoded)
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError("jax mpc worker closed the connection")
            buf += chunk
        reply = json.loads(buf.split(b"\n", 1)[0].decode("utf-8"))
        if not isinstance(reply, dict):
            raise ValueError("jax mpc worker reply must be a JSON object")
        return reply
    finally:
        sock.close()


@dataclass
class JaxMpcWorkerHandle:
    gpu: str
    sock: Path
    process: subprocess.Popen[bytes]
    log_path: Path


def wait_for_jax_mpc_worker(
    handle: JaxMpcWorkerHandle,
    *,
    timeout_s: float = 900.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_report = 0.0
    while time.monotonic() < deadline:
        if handle.process.poll() is not None:
            tail = ""
            if handle.log_path.is_file():
                tail = handle.log_path.read_text(encoding="utf-8")[-4000:]
            raise RuntimeError(
                f"jax mpc worker for gpu={handle.gpu} exited "
                f"{handle.process.returncode}\n{tail}"
            )
        if handle.sock.exists():
            try:
                reply = json_socket_request(
                    handle.sock, {"cmd": "ping"}, timeout_s=5.0
                )
            except (OSError, ConnectionError, TimeoutError, socket.timeout, ValueError):
                time.sleep(0.5)
                continue
            if reply.get("ok") and reply.get("ready"):
                print(
                    f"[jax-mpc-worker] ready gpu={handle.gpu} sock={handle.sock}",
                    flush=True,
                )
                return
        now = time.monotonic()
        if now - last_report >= 15.0:
            print(
                f"[jax-mpc-worker] waiting gpu={handle.gpu} sock={handle.sock}",
                flush=True,
            )
            last_report = now
        time.sleep(0.5)
    raise TimeoutError(
        f"timed out waiting for jax mpc worker gpu={handle.gpu} sock={handle.sock}"
    )


def start_jax_mpc_workers(
    gpus: Sequence[str],
    *,
    work_root: Path,
    cpu_threads: int,
) -> list[JaxMpcWorkerHandle]:
    if not PY_JAX.is_file():
        raise FileNotFoundError(f"missing hydrax python: {PY_JAX}")
    if not JAX_WORKER.is_file():
        raise FileNotFoundError(f"missing jax mpc worker: {JAX_WORKER}")
    work_root.mkdir(parents=True, exist_ok=True)
    handles: list[JaxMpcWorkerHandle] = []
    try:
        for gpu in gpus:
            sock = work_root / f"jax_mpc_gpu_{gpu}.sock"
            log_path = work_root / f"jax_mpc_gpu_{gpu}.log"
            if sock.exists() or sock.is_socket():
                sock.unlink()
            env = worker_environment(gpu, cpu_threads, work_root=work_root)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [
                        str(PY_JAX),
                        str(JAX_WORKER),
                        "--socket",
                        str(sock),
                        "--artifact-manifest",
                        str(ARTIFACT_MANIFEST),
                    ],
                    cwd=str(HYDRAX),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            handles.append(
                JaxMpcWorkerHandle(
                    gpu=str(gpu),
                    sock=sock,
                    process=process,
                    log_path=log_path,
                )
            )
        for handle in handles:
            wait_for_jax_mpc_worker(handle)
    except Exception:
        stop_jax_mpc_workers(handles)
        raise
    return handles


def stop_jax_mpc_workers(handles: Sequence[JaxMpcWorkerHandle]) -> None:
    for handle in handles:
        if handle.process.poll() is None:
            try:
                os.killpg(handle.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
        try:
            handle.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(handle.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            handle.process.wait()
        if handle.sock.exists() or handle.sock.is_socket():
            try:
                handle.sock.unlink()
            except OSError:
                pass


def run_jax_cem_mpc(
    *,
    reference: Path,
    init_state: Path,
    run_dir: Path,
    env: dict[str, str],
    log_path: Path,
    timeout_s: float | None,
) -> None:
    command = [
        PY_JAX,
        JAX_RUNNER,
        "--motion-file",
        reference,
        "--init-state-npz",
        init_state,
        "--artifact-manifest",
        ARTIFACT_MANIFEST,
        "--output-dir",
        run_dir,
        "--seed",
        "2",
    ]
    sock = str(env.get("JAX_MPC_WORKER_SOCK", "")).strip()
    if not sock:
        run_stage(
            "jax_cem_mpc",
            command,
            cwd=HYDRAX,
            env=env,
            log_path=log_path,
            timeout_s=timeout_s,
        )
        return
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] stage=jax_cem_mpc\n")
        log.write(f"jax_mpc_worker_sock: {sock}\n")
        log.write("command: " + " ".join(str(item) for item in command) + "\n")
        log.flush()
        try:
            reply = json_socket_request(
                sock,
                {
                    "cmd": "run",
                    "motion_file": str(reference),
                    "init_state_npz": str(init_state),
                    "artifact_manifest": str(ARTIFACT_MANIFEST),
                    "output_dir": str(run_dir),
                    "seed": 2,
                    "num_samples": 256,
                    "horizon_steps": 36,
                    "execute_steps": 12,
                    "cem_iters": 4,
                },
                timeout_s=timeout_s,
            )
        except Exception as error:
            elapsed = time.monotonic() - started
            log.write(f"worker_error: {error}\n")
            log.write(f"[{utc_now()}] returncode=1 elapsed_s={elapsed:.3f}\n")
            raise StageError("jax_cem_mpc", f"worker request failed: {error}") from error
        elapsed = time.monotonic() - started
        log.write("worker_reply: " + json.dumps(reply, default=str) + "\n")
        log.write(f"[{utc_now()}] returncode=0 elapsed_s={elapsed:.3f}\n")
    if not reply.get("ok"):
        raise StageError(
            "jax_cem_mpc",
            str(reply.get("error") or "jax mpc worker returned ok=false"),
        )


def read_reference_shape(path: Path) -> tuple[int, float]:
    with np.load(path, allow_pickle=False) as data:
        frames = int(np.asarray(data["body_pos_w"]).shape[0])
        fps = float(np.asarray(data["fps"]).reshape(()))
    if frames < 2:
        raise ValueError(f"reference has only {frames} frame(s)")
    if not np.isclose(fps, 50.0, rtol=0.0, atol=1.0e-8):
        raise ValueError(f"reference fps must be 50, got {fps}")
    return frames, fps


def sanitize_transient_paths(value: Any, transient_root: Path) -> Any:
    if isinstance(value, dict):
        return {
            key: sanitize_transient_paths(item, transient_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_transient_paths(item, transient_root) for item in value]
    if isinstance(value, str):
        try:
            path = Path(value)
            relative = path.resolve().relative_to(transient_root.resolve())
        except (OSError, ValueError):
            return value
        return f"transient://{relative.as_posix()}"
    return value


def write_persistent_kinematics(
    source: Path,
    destination: Path,
    *,
    session: SessionProbe,
    artifact_manifest_hash: str,
    dependency_manifest: dict[str, Any],
    reference_preparation: dict[str, Any],
    orientation_offset_json: Path | None,
    contact_aware_candidates: Path | None,
    joint_contact_wrist_candidate: Path | None,
    joint_contact_hand: Path | None,
    contact_manifest: Path | None,
    contact_json: Path | None,
    q0_mink: Q0MinkOptions,
    reference_smoothing: ReferenceSmoothingOptions,
    reference_smoothing_meta: Path | None,
    hand_mode: str = HAND_MODE_DEX3,
    brainco_reference: Path | None = None,
    brainco_meta: Path | None = None,
) -> None:
    hand_dof = HAND_DOF_BY_MODE[hand_mode]
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    arrays["source_raw_kinematics"] = np.asarray(
        f"transient://jax_mpc/kinematics/{RUN_NAME}_kinematics.npz"
    )
    arrays["source_reference"] = np.asarray(
        "transient://three_point_reference/torso_umi3_reference.npz"
    )
    arrays["wrist_reference_method"] = np.asarray(
        reference_preparation.get("method", "template_palm_kabsch")
    )
    if joint_contact_hand is not None:
        bundle_wrist_candidate = (
            joint_contact_wrist_candidate or contact_aware_candidates
        )
        contact_bundle_published = (
            contact_manifest is not None and contact_json is not None
        )
        if contact_bundle_published and bundle_wrist_candidate is None:
            raise StageError(
                "publish",
                "joint-contact hand requires wrist candidate, manifest, and contact JSON",
            )
        if not contact_bundle_published:
            if hand_mode != HAND_MODE_BRAINCO:
                raise StageError(
                    "publish",
                    "joint-contact hand requires wrist candidate, manifest, and contact JSON",
                )
            if brainco_reference is None or brainco_meta is None:
                raise StageError(
                    "publish",
                    "BrainCo hand requires its refined reference and metadata",
                )
        with np.load(joint_contact_hand, allow_pickle=False) as hand_archive:
            hand = {name: np.asarray(hand_archive[name]) for name in hand_archive.files}
        qpos_key = "qpos" if "qpos" in hand else "hand_qpos"
        qvel_key = "qvel" if "qvel" in hand else "hand_qvel"
        names_key = (
            "hand_joint_names" if "hand_joint_names" in hand else "joint_names"
        )
        required_hand = (qpos_key, qvel_key, names_key, "timestamps")
        if any(key not in hand for key in required_hand):
            raise StageError("publish", f"invalid joint-contact hand archive: {joint_contact_hand}")
        hand_qpos = np.asarray(hand[qpos_key], dtype=np.float32)
        hand_qvel = np.asarray(hand[qvel_key], dtype=np.float32)
        hand_names = np.asarray(hand[names_key], dtype=str).reshape(-1)
        body_frames = int(np.asarray(arrays["joint_pos"]).shape[0])
        body_fps = float(np.asarray(arrays.get("fps", 50.0)).reshape(()))
        body_time = np.asarray(
            arrays.get("timestamps", np.arange(body_frames, dtype=np.float64) / body_fps),
            dtype=np.float64,
        )
        hand_time = np.asarray(hand["timestamps"], dtype=np.float64)
        body_time = body_time - body_time[0]
        hand_time = hand_time - hand_time[0]
        if hand_qpos.shape != (body_frames, hand_dof) or hand_qvel.shape != (
            body_frames,
            hand_dof,
        ):
            raise StageError(
                "publish",
                f"hand/body frame contract mismatch qpos={hand_qpos.shape} "
                f"qvel={hand_qvel.shape} body_frames={body_frames}",
            )
        if hand_names.shape != (hand_dof,):
            raise StageError(
                "publish",
                f"expected {hand_dof} {hand_mode} joints, got {hand_names.shape}",
            )
        if hand_time.shape != body_time.shape or not np.allclose(
            hand_time, body_time, rtol=0.0, atol=2.0e-6
        ):
            raise StageError("publish", "hand and body timestamps are not identical")

        if not contact_bundle_published:
            manifest_payload = {}
            if "native_timestamps" in hand:
                source_time = np.asarray(hand["native_timestamps"], dtype=np.float64)
                source_time = source_time - source_time[0]
            else:
                source_time = body_time
            left_contact_30hz = np.asarray(
                hand.get("left_contact_30hz", np.zeros(len(source_time), dtype=np.int8)),
                dtype=np.int8,
            ).reshape(-1)
            right_contact_30hz = np.asarray(
                hand.get("right_contact_30hz", np.zeros(len(source_time), dtype=np.int8)),
                dtype=np.int8,
            ).reshape(-1)
            if left_contact_30hz.shape != source_time.shape:
                left_contact_30hz = np.zeros(len(source_time), dtype=np.int8)
            if right_contact_30hz.shape != source_time.shape:
                right_contact_30hz = np.zeros(len(source_time), dtype=np.int8)
            contact_30hz = np.maximum(left_contact_30hz, right_contact_30hz)
            contact_indices = np.flatnonzero(contact_30hz).astype(np.int64)
            status = (
                "contact_pointcloud_2finger"
                if np.any(contact_30hz)
                else "no_contact_detected"
            )
            detector = str(
                np.asarray(
                    hand.get("contact_detector", "source_rgbd_pointcloud_anchors")
                ).reshape(())
            )
        else:
            manifest_payload = json.loads(
                contact_manifest.read_text(encoding="utf-8")
            )
            contact_payload = json.loads(contact_json.read_text(encoding="utf-8"))
            with np.load(
                bundle_wrist_candidate, allow_pickle=False
            ) as candidate_archive:
                source_time = np.asarray(
                    candidate_archive["timestamps"], dtype=np.float64
                )
            source_time = source_time - source_time[0]
            n_source = len(source_time)
            contact_30hz = np.zeros(n_source, dtype=np.int8)
            left_contact_30hz = np.zeros(n_source, dtype=np.int8)
            right_contact_30hz = np.zeros(n_source, dtype=np.int8)
            status = str(manifest_payload.get("status", "unknown"))
            left_frames = contact_payload.get("left_contact_frames")
            right_frames = contact_payload.get("right_contact_frames")

            def _valid_indices(raw: Any) -> np.ndarray:
                indices = np.asarray(raw or [], dtype=np.int64).reshape(-1)
                return indices[(indices >= 0) & (indices < n_source)]

            if left_frames is None and right_frames is None:
                if status.startswith("contact_pointcloud"):
                    contact_indices = _valid_indices(
                        contact_payload.get("contact_frames", [])
                    )
                    contact_30hz[contact_indices] = 1
                    right_contact_30hz = contact_30hz.copy()
                else:
                    contact_indices = np.empty(0, dtype=np.int64)
            else:
                left_contact_30hz[_valid_indices(left_frames)] = 1
                right_contact_30hz[_valid_indices(right_frames)] = 1
                contact_30hz = np.maximum(left_contact_30hz, right_contact_30hz)
                union = _valid_indices(contact_payload.get("contact_frames", []))
                contact_indices = (
                    union if len(union) else np.flatnonzero(contact_30hz).astype(np.int64)
                )
            detector = str(
                manifest_payload.get("contact_detector", "source_rgbd_pointcloud")
            )
        right = np.searchsorted(source_time, body_time, side="left")
        right = np.clip(right, 0, len(source_time) - 1)
        left = np.clip(right - 1, 0, len(source_time) - 1)
        use_right = np.abs(source_time[right] - body_time) < np.abs(
            body_time - source_time[left]
        )
        nearest = np.where(use_right, right, left)
        left_contact_50hz = left_contact_30hz[nearest]
        right_contact_50hz = right_contact_30hz[nearest]
        contact_50hz = np.maximum(left_contact_50hz, right_contact_50hz)

        arrays["hand_qpos"] = hand_qpos
        arrays["hand_qvel"] = hand_qvel
        arrays["hand_joint_names"] = hand_names
        arrays["contact_30hz"] = contact_30hz
        arrays["contact_30hz_timestamps"] = source_time
        arrays["contact_frame_indices_30hz"] = contact_indices
        arrays["contact_50hz"] = contact_50hz
        arrays["left_contact"] = left_contact_50hz
        arrays["right_contact"] = right_contact_50hz
        arrays["left_contact_30hz"] = left_contact_30hz
        arrays["right_contact_30hz"] = right_contact_30hz
        arrays["contact_detector"] = np.asarray(detector)
        arrays["contact_status"] = np.asarray(status)
        arrays["joint_contact_hand_sha256"] = np.asarray(sha256(joint_contact_hand))
        arrays["joint_contact_manifest_sha256"] = np.asarray(
            sha256(contact_manifest if contact_bundle_published else brainco_meta)
        )
        if contact_bundle_published:
            arrays["joint_contact_bundle_sha256"] = np.asarray(
                joint_contact_bundle_sha256(
                    bundle_wrist_candidate, joint_contact_hand, contact_manifest
                )
            )
        else:
            arrays["joint_contact_bundle_sha256"] = np.asarray(
                joint_contact_bundle_sha256(
                    brainco_reference, joint_contact_hand, brainco_meta
                )
            )
        arrays["joint_contact_contract"] = np.asarray(HAND_CONTRACT_BY_MODE[hand_mode])
    if orientation_offset_json is not None:
        arrays["wrist_orientation_offset_source"] = np.asarray(
            str(orientation_offset_json.resolve())
        )
        arrays["wrist_orientation_offset_sha256"] = np.asarray(
            sha256(orientation_offset_json)
        )
    # The palm refinement is the file that decided the BrainCo wrist stream, so
    # it fills the same provenance slot the contact-aware candidates fill.
    wrist_reference_source = (
        brainco_reference
        if hand_mode == HAND_MODE_BRAINCO
        else contact_aware_candidates
    )
    if wrist_reference_source is not None:
        arrays["contact_aware_reference_source"] = np.asarray(
            str(wrist_reference_source.resolve())
        )
        arrays["contact_aware_reference_sha256"] = np.asarray(
            sha256(wrist_reference_source)
        )
    arrays["source_annotation"] = np.asarray(
        str(session.session_dir / "annotation.hdf5")
    )
    arrays["source_annotation_sha256"] = np.asarray(
        sha256(session.session_dir / "annotation.hdf5")
    )
    arrays["source_calibration"] = np.asarray(
        str(session.session_dir / SOURCE_CALIBRATION_RELATIVE)
    )
    arrays["source_calibration_sha256"] = np.asarray(
        sha256(session.session_dir / SOURCE_CALIBRATION_RELATIVE)
    )
    arrays["artifact_manifest_sha256"] = np.asarray(artifact_manifest_hash)
    arrays["pipeline_dependency_sha256"] = np.asarray(
        dependency_manifest["sha256"]
    )
    arrays["pipeline_config_id"] = np.asarray(PIPELINE_CONFIG_ID)
    arrays["reference_smoothing_config_sha256"] = np.asarray(
        reference_smoothing.fingerprint
    )
    arrays["reference_smoothing_config_json"] = np.asarray(
        json.dumps(reference_smoothing.as_dict(), sort_keys=True)
    )
    if reference_smoothing_meta is not None:
        arrays["reference_smoothing_meta_sha256"] = np.asarray(
            sha256(reference_smoothing_meta)
        )
    arrays["mink_q0_config_sha256"] = np.asarray(q0_mink.fingerprint)
    arrays["mink_q0_config_json"] = np.asarray(
        json.dumps(q0_mink.as_dict(), sort_keys=True)
    )
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(destination)


def write_persistent_metadata(
    source: Path,
    destination: Path,
    *,
    transient_root: Path,
    session: SessionProbe,
    artifact_manifest_hash: str,
    dependency_manifest: dict[str, Any],
    reference_preparation: dict[str, Any],
    orientation_offset_json: Path | None,
    contact_aware_candidates: Path | None,
    joint_contact_wrist_candidate: Path | None,
    joint_contact_hand: Path | None,
    contact_manifest: Path | None,
    q0_mink: Q0MinkOptions,
    reference_smoothing: ReferenceSmoothingOptions,
    reference_smoothing_meta: Path | None,
    hand_mode: str = HAND_MODE_DEX3,
    brainco_reference: Path | None = None,
    brainco_meta: Path | None = None,
) -> None:
    is_brainco = hand_mode == HAND_MODE_BRAINCO
    wrist_reference_source = brainco_reference if is_brainco else contact_aware_candidates
    stage_manifest = (
        contact_manifest if contact_manifest is not None else brainco_meta
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload = sanitize_transient_paths(payload, transient_root)
    payload["batch_provenance"] = {
        "session_id": session.session_id,
        "source_annotation": str(session.session_dir / "annotation.hdf5"),
        "source_annotation_sha256": sha256(
            session.session_dir / "annotation.hdf5"
        ),
        "source_calibration": str(
            session.session_dir / SOURCE_CALIBRATION_RELATIVE
        ),
        "source_calibration_sha256": sha256(
            session.session_dir / SOURCE_CALIBRATION_RELATIVE
        ),
        "artifact_manifest": str(ARTIFACT_MANIFEST),
        "artifact_manifest_sha256": artifact_manifest_hash,
        "pipeline_dependency": {
            "schema_version": dependency_manifest["schema_version"],
            "sha256": dependency_manifest["sha256"],
            "entry_count": len(dependency_manifest["files"]),
        },
        "pipeline_config_id": PIPELINE_CONFIG_ID,
        "reference_smoothing": reference_smoothing.as_dict(),
        "reference_smoothing_config_sha256": reference_smoothing.fingerprint,
        "reference_smoothing_meta_sha256": (
            sha256(reference_smoothing_meta)
            if reference_smoothing_meta is not None
            else None
        ),
        "wrist_reference_method": reference_preparation.get(
            "method", "template_palm_kabsch"
        ),
        "wrist_orientation_offset": (
            str(orientation_offset_json.resolve())
            if orientation_offset_json is not None
            else None
        ),
        "wrist_orientation_offset_sha256": (
            sha256(orientation_offset_json)
            if orientation_offset_json is not None
            else None
        ),
        "contact_aware_reference_source": (
            str(wrist_reference_source.resolve())
            if wrist_reference_source is not None
            else None
        ),
        "contact_aware_reference_sha256": (
            sha256(wrist_reference_source)
            if wrist_reference_source is not None
            else None
        ),
        "joint_contact_hand_sha256": (
            sha256(joint_contact_hand) if joint_contact_hand is not None else None
        ),
        "joint_contact_manifest_sha256": (
            sha256(stage_manifest) if stage_manifest is not None else None
        ),
        "joint_contact_bundle_sha256": (
            joint_contact_bundle_sha256(
                joint_contact_wrist_candidate, joint_contact_hand, contact_manifest
            )
            if joint_contact_wrist_candidate is not None
            and joint_contact_hand is not None
            and contact_manifest is not None
            else joint_contact_bundle_sha256(
                brainco_reference, joint_contact_hand, brainco_meta
            )
            if is_brainco
            and brainco_reference is not None
            and joint_contact_hand is not None
            and brainco_meta is not None
            else None
        ),
        "joint_contact_contract": (
            HAND_CONTRACT_BY_MODE[hand_mode]
            if joint_contact_hand is not None
            else None
        ),
        "hand_mode": hand_mode,
        "mink_q0_config": q0_mink.as_dict(),
        "mink_q0_config_sha256": q0_mink.fingerprint,
        "sequence_policy": "complete source sequence resampled to 50 Hz",
    }
    payload["reference_preparation"] = reference_preparation
    atomic_json(destination, payload)


def publish_result(
    *,
    session: SessionProbe,
    output_root: Path,
    raw_final_dir: Path,
    reference: Path,
    prepare_metadata: Path,
    force: bool,
    log_path: Path,
    env: dict[str, str],
    stage_timeout_s: float | None,
    artifact_manifest_hash: str,
    dependency_manifest: dict[str, Any],
    orientation_offset_json: Path | None,
    contact_aware_candidates: Path | None,
    joint_contact_wrist_candidate: Path | None,
    joint_contact_hand: Path | None,
    contact_manifest: Path | None,
    contact_json: Path | None,
    q0_mink: Q0MinkOptions,
    reference_smoothing: ReferenceSmoothingOptions,
    reference_smoothing_meta: Path | None,
    hand_mode: str = HAND_MODE_DEX3,
    brainco_reference: Path | None = None,
    brainco_meta: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    kinematics = raw_final_dir / "kinematics.npz"
    metadata = raw_final_dir / "meta.json"
    mink_summary = raw_final_dir / "mink_ik_summary.json"
    for path in (kinematics, metadata, mink_summary):
        if not path.is_file():
            raise StageError("publish", f"missing final Mink output: {path}")
    if not prepare_metadata.is_file():
        raise StageError("publish", f"missing wrist preparation metadata: {prepare_metadata}")
    preparation_payload = json.loads(prepare_metadata.read_text(encoding="utf-8"))
    reference_preparation = {
        "method": preparation_payload.get("method"),
        "selection_policy": preparation_payload.get("selection_policy"),
        "source_frames": preparation_payload.get("source_frames"),
        "output_frames": preparation_payload.get("output_frames"),
        "output_fps": preparation_payload.get("output_fps"),
        "transient_wrist_outlier_repair": preparation_payload.get("selection", {}).get(
            "transient_wrist_outlier_repair", {}
        ),
        "landmark_outlier_repair": preparation_payload.get("source", {}).get(
            "landmark_outlier_repair", {}
        ),
        "contract": preparation_payload.get("contract", {}),
        "delta_from_historical_reference": preparation_payload.get(
            "delta_from_historical_reference", {}
        ),
        "contact_aware_injection": preparation_payload.get(
            "contact_aware_injection"
        ),
    }

    staging_root = output_root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = staging_root / f"{session.session_id}.{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        transient_root = raw_final_dir.parent
        write_persistent_kinematics(
            kinematics,
            staging / "kinematics.npz",
            session=session,
            artifact_manifest_hash=artifact_manifest_hash,
            dependency_manifest=dependency_manifest,
            reference_preparation=reference_preparation,
            orientation_offset_json=orientation_offset_json,
            contact_aware_candidates=contact_aware_candidates,
            joint_contact_wrist_candidate=joint_contact_wrist_candidate,
            joint_contact_hand=joint_contact_hand,
            contact_manifest=contact_manifest,
            contact_json=contact_json,
            q0_mink=q0_mink,
            reference_smoothing=reference_smoothing,
            reference_smoothing_meta=reference_smoothing_meta,
            hand_mode=hand_mode,
            brainco_reference=brainco_reference,
            brainco_meta=brainco_meta,
        )
        write_persistent_metadata(
            metadata,
            staging / "meta.json",
            transient_root=transient_root,
            session=session,
            artifact_manifest_hash=artifact_manifest_hash,
            dependency_manifest=dependency_manifest,
            reference_preparation=reference_preparation,
            orientation_offset_json=orientation_offset_json,
            contact_aware_candidates=contact_aware_candidates,
            joint_contact_wrist_candidate=joint_contact_wrist_candidate,
            joint_contact_hand=joint_contact_hand,
            contact_manifest=contact_manifest,
            q0_mink=q0_mink,
            reference_smoothing=reference_smoothing,
            reference_smoothing_meta=reference_smoothing_meta,
            hand_mode=hand_mode,
            brainco_reference=brainco_reference,
            brainco_meta=brainco_meta,
        )
        run_stage(
            "summarize_final_tracking",
            [
                PY_ST,
                SUMMARIZE,
                "--kinematics",
                staging / "kinematics.npz",
                "--metadata",
                staging / "meta.json",
                "--reference",
                reference,
                "--mink-summary",
                mink_summary,
                "--session-id",
                session.session_id,
                "--source-annotation",
                session.session_dir / "annotation.hdf5",
                "--source-calibration",
                session.session_dir / SOURCE_CALIBRATION_RELATIVE,
                "--artifact-manifest",
                ARTIFACT_MANIFEST,
                "--pipeline-dependency-sha256",
                dependency_manifest["sha256"],
                "--pipeline-config-id",
                PIPELINE_CONFIG_ID,
                "--output",
                staging / "tracking_error.json",
            ],
            cwd=ST,
            env=env,
            log_path=log_path,
            timeout_s=stage_timeout_s,
        )
        if {path.name for path in staging.iterdir()} != set(FINAL_FILES):
            raise StageError("publish", f"unexpected staging contents: {staging}")

        destination = output_root / session.session_id
        if destination.exists():
            if not force and output_is_complete(
                destination,
                session.session_id,
                source_annotation=session.session_dir / "annotation.hdf5",
                source_calibration=(
                    session.session_dir / SOURCE_CALIBRATION_RELATIVE
                ),
                artifact_manifest_hash=artifact_manifest_hash,
                pipeline_dependency_hash=dependency_manifest["sha256"],
                orientation_offset_hash=(
                    sha256(orientation_offset_json)
                    if orientation_offset_json is not None
                    else None
                ),
                contact_aware_hash=(
                    sha256(contact_aware_candidates)
                    if contact_aware_candidates is not None
                    else None
                ),
                mink_q0_config_hash=(
                    q0_mink.fingerprint if q0_mink.enabled else None
                ),
                reference_smoothing_hash=reference_smoothing.fingerprint,
                require_joint_contact_hand=joint_contact_hand is not None,
                hand_dof=HAND_DOF_BY_MODE[hand_mode],
            ):
                summary = json.loads(
                    (destination / "tracking_error.json").read_text()
                )
                return destination, summary
            previous_root = output_root / "_batch_previous"
            previous_root.mkdir(parents=True, exist_ok=True)
            previous = previous_root / (
                f"{session.session_id}.{int(time.time())}.{uuid.uuid4().hex[:8]}"
            )
            destination.replace(previous)
        try:
            staging.replace(destination)
        except BaseException:
            if destination.exists():
                shutil.rmtree(destination)
            if "previous" in locals() and previous.exists():
                previous.replace(destination)
            raise
        summary = json.loads((destination / "tracking_error.json").read_text())
        return destination, summary
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def run_session(
    session: SessionProbe,
    *,
    gpu_slots: queue.Queue[str],
    output_root: Path,
    work_root: Path,
    cpu_threads: int,
    keep_work: bool,
    force: bool,
    stage_timeout_s: float | None,
    artifact_manifest_hash: str,
    dependency_manifest: dict[str, Any],
    orientation_offset_json: Path | None,
    contact_aware_candidates: Path | None,
    joint_contact_precompute: bool,
    joint_contact_bundle: JointContactBundle | None,
    contact_transition_frames: int,
    contact_edge_hold: bool,
    contact_schedule_root: Path | None,
    q0_mink: Q0MinkOptions,
    reference_smoothing: ReferenceSmoothingOptions,
    wrist_reference_world_z_offset_m: float,
    hand_mode: str = HAND_MODE_DEX3,
    jax_mpc_socks: dict[str, str] | None = None,
    max_gap_frames: int = 10,
) -> JobResult:
    started = time.monotonic()
    gpu: str | None = None
    work_dir: Path | None = None
    log_path: Path | None = None
    current_stage = "preflight"
    try:
        gpu = gpu_slots.get()
        work_dir = Path(
            tempfile.mkdtemp(prefix=f"{session.session_id}.", dir=work_root)
        )
        log_path = work_dir / "pipeline.log"
        env = worker_environment(gpu, cpu_threads, work_root=work_root)
        sock = ""
        if jax_mpc_socks:
            sock = str(jax_mpc_socks.get(gpu, "")).strip()
        if not sock:
            sock = str(env.get("JAX_MPC_WORKER_SOCK", "")).strip()
        if sock:
            env["JAX_MPC_WORKER_SOCK"] = sock
        kimodo_sock = str(env.get("KIMODO_WORKER_SOCK", "")).strip()
        if kimodo_sock:
            env["KIMODO_WORKER_SOCK"] = kimodo_sock
        print(f"[start] {session.session_id} gpu={gpu}", flush=True)
        base_dir = work_dir / "base_reference"
        raw_reference = base_dir / "raw_dual_wrist_reference.npz"
        raw_targets = base_dir / "raw_dual_wrist_targets.npz"
        raw_prepare_metadata = base_dir / "prepare_meta.json"
        base_reference = base_dir / "template_palm_kabsch_reference.npz"
        targets = base_dir / "template_palm_kabsch_targets.npz"
        prepare_metadata = base_dir / "template_palm_kabsch_meta.json"
        reference_dir = work_dir / "three_point_reference"
        raw_three_point_reference = reference_dir / "torso_umi3_reference_raw.npz"
        raw_reference_meta = reference_dir / "reference_meta_raw.json"
        reference = raw_three_point_reference
        reference_smoothing_meta: Path | None = None
        init_dir = work_dir / "kimodo_init"
        init_state = init_dir / "candidate_seed_02/init_state.npz"
        init_summary = init_dir / "candidate_seed_02/ik_summary.json"
        run_dir = work_dir / "jax_mpc"
        raw_kinematics = run_dir / "kinematics" / f"{RUN_NAME}_kinematics.npz"
        raw_metadata = run_dir / "kinematics/meta.json"
        raw_summary = run_dir / f"{RUN_NAME}_summary.json"
        mink_dir = work_dir / "mink_final"

        current_stage = "build_dual_wrist_reference"
        run_stage(
            current_stage,
            [
                PY_ST,
                PREPARE,
                "--episode",
                session.session_dir,
                "--out-dir",
                base_dir,
                "--out-fps",
                "50",
                "--use-full-source",
                "--repair-transient-wrist-outliers",
                "--max-gap-frames",
                str(int(max_gap_frames)),
            ],
            cwd=ST,
            env=env,
            log_path=log_path,
            timeout_s=stage_timeout_s,
        )
        for path in (raw_reference, raw_targets, raw_prepare_metadata):
            if not path.is_file():
                raise StageError(current_stage, f"missing prepared reference output: {path}")

        current_stage = "retarget_template_palm_reference"
        retarget_command: list[str | Path] = [
            PY_ST,
            RETARGET_WRIST_REFERENCE,
            "--annotation-hdf5",
            session.session_dir / "annotation.hdf5",
            "--base-reference",
            raw_reference,
            "--base-targets",
            raw_targets,
            "--prepare-meta",
            raw_prepare_metadata,
            "--output-reference",
            base_reference,
            "--output-targets",
            targets,
            "--meta-json",
            prepare_metadata,
        ]
        if orientation_offset_json is not None:
            retarget_command.extend(
                ["--orientation-offset-json", orientation_offset_json]
            )
        if wrist_reference_world_z_offset_m != 0.0:
            retarget_command.extend(
                ["--world-z-offset-m", str(wrist_reference_world_z_offset_m)]
            )
        run_stage(
            current_stage,
            retarget_command,
            cwd=PROJECT_ROOT,
            env=env,
            log_path=log_path,
            timeout_s=stage_timeout_s,
        )
        for path in (base_reference, targets, prepare_metadata):
            if not path.is_file():
                raise StageError(current_stage, f"missing template-palm output: {path}")

        joint_contact_hand: Path | None = None
        joint_contact_wrist_candidate: Path | None = None
        contact_manifest: Path | None = None
        contact_json: Path | None = None
        brainco_reference: Path | None = None
        brainco_meta: Path | None = None
        inject_jobs: list[tuple[str, Path]] = []
        if hand_mode == HAND_MODE_BRAINCO:
            if joint_contact_precompute:
                current_stage = "brainco_contact_precompute"
                joint_root = work_dir / "joint_contact_precompute"
                experiment_root = joint_root / "experiments"
                experiment_dir = experiment_root / session.session_id
                contact_output = experiment_dir / "contact_aware_auto"
                contact_env = dict(env)
                contact_env.update(
                    {
                        "GPU": str(gpu),
                        "FORCE": "1",
                        "SESSION_ID": session.session_id,
                        "INPUT_ROOT": str(session.session_dir.parent),
                        "KINEMATICS_NPZ": str(base_reference),
                        "TARGET_TIMESTAMPS_NPZ": str(targets),
                        "TEMPLATE_PALM_FROM_KINEMATICS": "1",
                        "EXPERIMENT_ROOT": str(experiment_root),
                        "EXPERIMENT_DIR": str(experiment_dir),
                        "OUTPUT_DIR": str(contact_output),
                        "CONTACT_NO_EVIDENCE_POLICY": "identity",
                        "JOINT_CONTACT_ENABLED": "1",
                        "HAND_MODE": HAND_MODE_BRAINCO,
                        "HUMAN_MASK_BACKEND": "efficientsam3",
                    }
                )
                if "CUP_PROMPTS" in os.environ:
                    contact_env["CUP_PROMPTS"] = os.environ["CUP_PROMPTS"]
                if "CUP_PROMPTS" in os.environ:
                    contact_env["CUP_PROMPTS"] = os.environ["CUP_PROMPTS"]
                run_stage(
                    current_stage,
                    ["bash", JOINT_CONTACT_PRECOMPUTE],
                    cwd=PROJECT_ROOT,
                    env=contact_env,
                    log_path=log_path,
                    timeout_s=stage_timeout_s,
                )
                joint_contact_bundle = resolve_joint_contact_bundle(
                    contact_output,
                    session.session_id,
                    hand_mode=HAND_MODE_BRAINCO,
                )
            current_stage = "brainco_hand_retarget"
            brainco_reference = base_dir / "brainco_reference.npz"
            brainco_targets = base_dir / "brainco_targets.npz"
            brainco_meta = base_dir / "brainco_meta.json"
            brainco_hand = base_dir / "hand_brainco_50hz.npz"
            brainco_env = dict(env)
            brainco_env["PYTHONNOUSERSITE"] = "1"
            brainco_env["PYTHONPATH"] = str(DEX_RETARGETING_SRC)
            run_stage(
                current_stage,
                [
                    PY_DEX_RETARGET,
                    BRAINCO_HAND_RETARGET,
                    "--annotation-hdf5",
                    session.session_dir / "annotation.hdf5",
                    "--base-reference",
                    base_reference,
                    "--base-targets",
                    targets,
                    "--base-meta",
                    raw_prepare_metadata,
                    "--mjcf",
                    BRAINCO_MJCF,
                    "--hand-urdf-dir",
                    BRAINCO_HAND_URDF_DIR,
                    "--retarget-config-dir",
                    BRAINCO_RETARGET_CONFIG_DIR,
                    "--output-reference",
                    brainco_reference,
                    "--output-targets",
                    brainco_targets,
                    "--output-hand",
                    brainco_hand,
                    "--meta-json",
                    brainco_meta,
                ],
                cwd=PROJECT_ROOT,
                env=brainco_env,
                log_path=log_path,
                timeout_s=stage_timeout_s,
            )
            for path in (
                brainco_reference,
                brainco_targets,
                brainco_hand,
                brainco_meta,
            ):
                if not path.is_file():
                    raise StageError(current_stage, f"missing BrainCo output: {path}")
            base_reference, targets, prepare_metadata = (
                brainco_reference,
                brainco_targets,
                brainco_meta,
            )
            if joint_contact_bundle is not None:
                joint_contact_wrist_candidate = joint_contact_bundle.wrist_candidate
                joint_contact_hand = joint_contact_bundle.hand_kinematics
                contact_manifest = joint_contact_bundle.manifest
                contact_json = joint_contact_bundle.contact_json
                side_candidates = {
                    "left": joint_contact_bundle.left_wrist_candidate,
                    "right": joint_contact_bundle.right_wrist_candidate,
                }
                for side in joint_contact_bundle.inject_sides:
                    candidate = side_candidates.get(side)
                    if candidate is not None and candidate.is_file():
                        inject_jobs.append((side, candidate))
                if inject_jobs:
                    contact_aware_candidates = inject_jobs[-1][1]
                contact_edge_hold = True
            else:
                joint_contact_hand = brainco_hand
        elif joint_contact_bundle is not None:
            joint_contact_wrist_candidate = joint_contact_bundle.wrist_candidate
            joint_contact_hand = joint_contact_bundle.hand_kinematics
            contact_manifest = joint_contact_bundle.manifest
            contact_json = joint_contact_bundle.contact_json
            if joint_contact_bundle.inject_wrist_residual:
                inject_jobs.append(("right", joint_contact_bundle.wrist_candidate))
                contact_aware_candidates = joint_contact_bundle.wrist_candidate
            # A precomputed bundle obeys the same causal first/per-frame/last
            # residual schedule as the in-worker precompute path.
            contact_edge_hold = True
        elif joint_contact_precompute:
            if contact_aware_candidates is not None:
                raise StageError(
                    "joint_contact_precompute",
                    "automatic joint contact and legacy contact-aware candidates are mutually exclusive",
                )
            current_stage = "joint_contact_precompute"
            joint_root = work_dir / "joint_contact_precompute"
            experiment_root = joint_root / "experiments"
            experiment_dir = experiment_root / session.session_id
            contact_output = experiment_dir / "contact_aware_auto"
            base_hand = (
                contact_output / "joint_contact/base_hand_dex3_50hz.npz"
            )
            optimized_hand = contact_output / "hand_dex3_50hz_joint_contact.npz"
            candidate = contact_output / "contact_aware_wrist_candidates_native.npz"
            contact_manifest = contact_output / "contact_precompute_manifest.json"
            contact_json = contact_output / "contact_frames.json"
            contact_env = dict(env)
            contact_env.update(
                {
                    "GPU": str(gpu),
                    "FORCE": "1",
                    "SESSION_ID": session.session_id,
                    "INPUT_ROOT": str(session.session_dir.parent),
                    "KINEMATICS_NPZ": str(base_reference),
                    "TARGET_TIMESTAMPS_NPZ": str(targets),
                    "TEMPLATE_PALM_FROM_KINEMATICS": "1",
                    "EXPERIMENT_ROOT": str(experiment_root),
                    "EXPERIMENT_DIR": str(experiment_dir),
                    "OUTPUT_DIR": str(contact_output),
                    "BASE_HAND_50HZ": str(base_hand),
                    "CONTACT_NO_EVIDENCE_POLICY": "identity",
                    "JOINT_CONTACT_ENABLED": "1",
                    "HUMAN_MASK_BACKEND": "efficientsam3",
                }
            )
            run_stage(
                current_stage,
                ["bash", JOINT_CONTACT_PRECOMPUTE],
                cwd=PROJECT_ROOT,
                env=contact_env,
                log_path=log_path,
                timeout_s=stage_timeout_s,
            )
            for path in (candidate, base_hand, contact_manifest, contact_json):
                if not path.is_file():
                    raise StageError(current_stage, f"missing joint-contact output: {path}")
            joint_contact_wrist_candidate = candidate
            contact_payload = json.loads(contact_manifest.read_text(encoding="utf-8"))
            contact_status = str(contact_payload.get("status", "unknown"))
            joint_status = str(
                contact_payload.get("joint_contact_optimization", {}).get(
                    "status", "missing"
                )
            )
            if contact_status.startswith("contact_pointcloud"):
                if joint_status != "enabled" or not optimized_hand.is_file():
                    raise StageError(
                        current_stage,
                        "RGB-D contact was detected but paired wrist+Dex3 "
                        f"optimization is unavailable (status={joint_status})",
                    )
                joint_contact_hand = optimized_hand
            else:
                joint_contact_hand = base_hand
            # No-contact is a true identity path: retain template-palm exactly
            # and carry only the position-retargeted Dex3 hand.  The identity
            # candidate remains in the bundle solely for provenance/contact
            # timeline publication.
            contact_aware_candidates = (
                candidate if contact_status.startswith("contact_pointcloud") else None
            )
            if contact_aware_candidates is not None:
                inject_jobs.append(("right", candidate))
            # The joint optimizer defines a strict causal edge-held wrist
            # residual.  Holding the residual never freezes the absolute wrist
            # pose; it is composed with each frame's template-palm reference.
            contact_edge_hold = True

        if not inject_jobs and contact_aware_candidates is not None:
            inject_jobs.append(("right", contact_aware_candidates))

        if inject_jobs:
            current_reference = base_reference
            current_targets = targets
            current_meta = prepare_metadata
            for side, candidate in inject_jobs:
                suffix = f"_{side}" if hand_mode == HAND_MODE_BRAINCO else ""
                current_stage = (
                    f"inject_contact_aware_wrist_reference{suffix}"
                )
                contact_reference = base_dir / f"contact_aware_reference{suffix}.npz"
                contact_targets = base_dir / f"contact_aware_targets{suffix}.npz"
                contact_meta = base_dir / f"contact_aware_meta{suffix}.json"
                contact_schedule = None
                if contact_schedule_root is not None:
                    contact_schedule = (
                        contact_schedule_root.expanduser().resolve()
                        / session.session_id
                        / f"wrist_6d_offset_schedule{suffix}.npz"
                    )
                run_stage(
                    current_stage,
                    [
                        PY_ST,
                        INJECT_CONTACT_AWARE_REFERENCE,
                        "--base-reference",
                        current_reference,
                        "--base-targets",
                        current_targets,
                        "--base-meta",
                        current_meta,
                        "--contact-candidates",
                        candidate,
                        "--output-reference",
                        contact_reference,
                        "--output-targets",
                        contact_targets,
                        "--output-meta",
                        contact_meta,
                        "--side",
                        side,
                        "--transition-frames",
                        str(contact_transition_frames),
                    ]
                    + (["--edge-hold-policy", "edge_hold"] if contact_edge_hold else [])
                    + (
                        ["--schedule-output", contact_schedule]
                        if contact_schedule is not None
                        else []
                    ),
                    cwd=PROJECT_ROOT,
                    env=env,
                    log_path=log_path,
                    timeout_s=stage_timeout_s,
                )
                current_reference, current_targets, current_meta = (
                    contact_reference,
                    contact_targets,
                    contact_meta,
                )
            base_reference, targets, prepare_metadata = (
                current_reference,
                current_targets,
                current_meta,
            )
        frames, fps = read_reference_shape(base_reference)

        current_stage = "build_three_point_reference"
        run_stage(
            current_stage,
            [
                PY_ST,
                BUILD_REFERENCE,
                "--annotation-hdf5",
                session.session_dir / "annotation.hdf5",
                "--base-reference",
                base_reference,
                "--output-npz",
                raw_three_point_reference,
                "--meta-json",
                raw_reference_meta,
                "--expected-frames",
                str(frames),
                "--expected-fps",
                str(fps),
            ],
            cwd=ST,
            env=env,
            log_path=log_path,
            timeout_s=stage_timeout_s,
        )

        if reference_smoothing.enabled:
            current_stage = "smooth_three_point_reference"
            smoothed_reference = reference_dir / "torso_umi3_reference.npz"
            smoothed_targets = reference_dir / "smoothed_targets.npz"
            reference_smoothing_meta = reference_dir / "reference_smoothing_meta.json"
            run_stage(
                current_stage,
                [
                    PY_ST,
                    SMOOTH_REFERENCE,
                    "--input-reference",
                    raw_three_point_reference,
                    "--input-targets",
                    targets,
                    "--output-reference",
                    smoothed_reference,
                    "--output-targets",
                    smoothed_targets,
                    "--meta-json",
                    reference_smoothing_meta,
                    "--window-length",
                    str(reference_smoothing.window_length),
                    "--polyorder",
                    str(reference_smoothing.polyorder),
                    "--gap-factor",
                    str(reference_smoothing.gap_factor),
                    "--wrist-outlier-max-duration-s",
                    str(reference_smoothing.wrist_outlier_max_duration_s),
                    "--wrist-outlier-position-floor-m",
                    str(reference_smoothing.wrist_outlier_position_floor_m),
                    "--wrist-outlier-orientation-floor-deg",
                    str(reference_smoothing.wrist_outlier_orientation_floor_deg),
                ],
                cwd=PROJECT_ROOT,
                env=env,
                log_path=log_path,
                timeout_s=stage_timeout_s,
            )
            for path in (smoothed_reference, smoothed_targets, reference_smoothing_meta):
                if not path.is_file():
                    raise StageError(current_stage, f"missing smoothed reference output: {path}")
            reference = smoothed_reference
            targets = smoothed_targets

        current_stage = "kimodo_initial_pose"
        run_stage(
            current_stage,
            [
                PY_ST,
                KIMODO_INIT,
                "--session_npz",
                reference,
                "--targets_npz",
                targets,
                "--out_dir",
                init_dir,
                "--seeds",
                "2",
                "--duration",
                "10",
                "--diffusion_steps",
                "100",
                "--cfg_weight",
                "2",
                "5",
                "--reset_lower_body",
                "--physics_gate",
                "--skip_tracking",
                "--wrist_orientation_weight",
                "1.0",
                "--torso_init_condition",
                "--torso_ik_weight",
                "0.25",
                "--no_legacy_baselines",
            ],
            cwd=ST,
            env=env,
            log_path=log_path,
            timeout_s=stage_timeout_s,
        )
        if not init_state.is_file() or not init_summary.is_file():
            raise StageError(current_stage, "Kimodo did not produce the Seed 02 init state")
        init_payload = json.loads(init_summary.read_text(encoding="utf-8"))
        if not bool(init_payload.get("accepted", False)):
            # Some converted AR sessions have a valid, physically plausible
            # source q0 while Kimodo's generated Seed 02 is rejected by the
            # strict IK/physics gate.  Keep the gate strict for the generated
            # candidate, but allow an explicit source-state fallback so the
            # downstream ScaleTrack/Mink stages can still produce a result.
            # The fallback is recorded in the worker log and provenance by the
            # existing publish stage; it is never selected implicitly.
            allow_baseline_fallback = os.environ.get(
                "ALLOW_KIMODO_BASELINE_FALLBACK", "0"
            ) == "1"
            baseline_state = init_dir / "baseline_joint0" / "init_state.npz"
            baseline_summary = init_dir / "baseline_joint0" / "ik_summary.json"
            baseline_payload = {}
            if baseline_summary.is_file():
                baseline_payload = json.loads(
                    baseline_summary.read_text(encoding="utf-8")
                )
            baseline_audit = baseline_payload.get("physics_audit", {})
            if (
                allow_baseline_fallback
                and baseline_state.is_file()
                and bool(baseline_audit.get("physical_plausible", False))
            ):
                init_state = baseline_state
                init_summary = baseline_summary
                init_payload = {
                    **baseline_payload,
                    "accepted": True,
                    "source": "baseline_joint0",
                    "fallback_from": "candidate_seed_02",
                    "fallback_reason": "seed02_rejected_by_ik_or_physics_gate",
                }
                print(
                    "[fallback] Kimodo Seed 02 rejected; using physically "
                    "plausible baseline_joint0 init state",
                    flush=True,
                )
            else:
                raise StageError(
                    current_stage,
                    "Kimodo Seed 02 init failed its IK/physics gate",
                )

        current_stage = "jax_cem_mpc"
        run_jax_cem_mpc(
            reference=reference,
            init_state=init_state,
            run_dir=run_dir,
            env=env,
            log_path=log_path,
            timeout_s=stage_timeout_s,
        )
        for path in (raw_kinematics, raw_metadata, raw_summary):
            if not path.is_file():
                raise StageError(current_stage, f"missing JAX MPC output: {path}")

        current_stage = "mink_arm_refinement"
        mink_returncode = run_stage(
            current_stage,
            [
                PY_MINK,
                MINK_REFINE,
                "--raw-kinematics",
                raw_kinematics,
                "--raw-metadata",
                raw_metadata,
                "--reference",
                reference,
                "--output-dir",
                mink_dir,
            ] + q0_mink.command_args(),
            cwd=ST,
            env=env,
            log_path=log_path,
            accepted_codes=(0, 2),
            timeout_s=stage_timeout_s,
        )

        current_stage = "publish"
        destination, tracking = publish_result(
            session=session,
            output_root=output_root,
            raw_final_dir=mink_dir,
            reference=reference,
            prepare_metadata=prepare_metadata,
            force=force,
            log_path=log_path,
            env=env,
            stage_timeout_s=stage_timeout_s,
            artifact_manifest_hash=artifact_manifest_hash,
            dependency_manifest=dependency_manifest,
            orientation_offset_json=orientation_offset_json,
            contact_aware_candidates=contact_aware_candidates,
            joint_contact_wrist_candidate=joint_contact_wrist_candidate,
            joint_contact_hand=joint_contact_hand,
            contact_manifest=contact_manifest,
            contact_json=contact_json,
            q0_mink=q0_mink,
            reference_smoothing=reference_smoothing,
            reference_smoothing_meta=reference_smoothing_meta,
            hand_mode=hand_mode,
            brainco_reference=brainco_reference,
            brainco_meta=brainco_meta,
        )
        accepted = bool(tracking.get("accepted", False))
        status = "completed" if accepted else "completed_quality_failed"
        if mink_returncode == 2 and accepted:
            status = "completed_mink_diagnostic_failed"
        result = JobResult(
            session_id=session.session_id,
            status=status,
            gpu=gpu,
            elapsed_s=time.monotonic() - started,
            output_dir=str(destination),
            frames=int(tracking["sampling"]["frames"]),
            accepted=accepted,
        )
        print(
            f"[done] {session.session_id} gpu={gpu} status={status} "
            f"frames={result.frames} elapsed_s={result.elapsed_s:.1f}",
            flush=True,
        )
        return result
    except Exception as error:
        failed_stage = error.stage if isinstance(error, StageError) else current_stage
        logs_dir = output_root / "_batch_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        saved_log = logs_dir / f"{session.session_id}.log"
        if log_path is not None and log_path.is_file():
            shutil.copy2(log_path, saved_log)
        result = JobResult(
            session_id=session.session_id,
            status="failed",
            gpu=gpu,
            elapsed_s=time.monotonic() - started,
            error=str(error),
            failed_stage=failed_stage,
            log=str(saved_log) if saved_log.is_file() else None,
            work_dir=str(work_dir) if keep_work and work_dir is not None else None,
        )
        print(
            f"[failed] {session.session_id} gpu={gpu} stage={failed_stage}: {error}",
            flush=True,
        )
        return result
    finally:
        if gpu is not None:
            gpu_slots.put(gpu)
        if not keep_work and work_dir is not None and work_dir.exists():
            try:
                shutil.rmtree(work_dir)
            except OSError as cleanup_error:
                print(
                    f"[warn] failed to clean {work_dir}: {cleanup_error}",
                    flush=True,
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--session-pattern", default="session_*")
    parser.add_argument(
        "--session-list",
        type=Path,
        help="Text file containing the exact session IDs to process, one per line.",
    )
    parser.add_argument("--gpus", default="auto")
    parser.add_argument(
        "--mpc-backend",
        choices=(MPC_BACKEND_JAX,),
        default=MPC_BACKEND_JAX,
        help="Hydrax JAX fragment-knot CEM after Kimodo Seed 02 (only supported backend).",
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--min-free-gpu-mb", type=int, default=18000)
    parser.add_argument("--max-gpu-utilization", type=int, default=10)
    parser.add_argument("--cpu-threads-per-worker", type=int, default=4)
    parser.add_argument("--stage-timeout-s", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--run-parity",
        action="store_true",
        help=(
            "Opt in to the per-batch Torch↔JAX policy parity check. Default is "
            "skip; THREE_POINT_RUN_PARITY=1 is equivalent."
        ),
    )
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--rebuild-artifact", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--state-file",
        type=Path,
        help="Write run state here instead of OUTPUT_ROOT/batch_state.json.",
    )
    parser.add_argument(
        "--orientation-offset-root",
        type=Path,
        help=(
            "Optional root containing SESSION_ID/wrist_orientation_offset.json. "
            "Every selected session must have one."
        ),
    )
    parser.add_argument(
        "--contact-aware-root",
        type=Path,
        help=(
            "Optional per-session root or candidate file. When enabled, feedback1 "
            "causal wrist targets are injected before Kimodo/ScaleTrack/Mink."
        ),
    )
    parser.add_argument(
        "--joint-contact-precompute",
        action="store_true",
        help=(
            "Build template-palm/position hand candidates, detect RGB-D contact, "
            "and jointly optimize right wrist SE(3)+Dex3 7-DoF before "
            "Kimodo/ScaleTrack/Mink. With --hand-mode g1_brainco this instead "
            "runs cup SAM + CupModel and per-hand wrist SE(3)+6-DoF Powell. "
            "The synchronized hand trajectory is published in kinematics.npz."
        ),
    )
    parser.add_argument(
        "--joint-contact-root",
        type=Path,
        help=(
            "Root containing a precomputed per-session contact_aware_auto "
            "bundle. Dex3 publishes its synchronized base/optimized hand and "
            "injects its wrist residual. BrainCo uses the same Dex3-style "
            "bundle (12-DoF hand plus left/right wrist candidates)."
        ),
    )
    parser.add_argument(
        "--hand-mode",
        choices=HAND_MODES,
        default=HAND_MODE_DEX3,
        help=(
            "dex3 (default) keeps the contact-aware Dex3 path. g1_brainco "
            "runs the BrainCo palm refinement and, with --joint-contact-root "
            "or --joint-contact-precompute, CupModel wrist/finger Powell that "
            "publishes 12 hand DoF plus left_contact/right_contact."
        ),
    )
    parser.add_argument("--contact-transition-frames", type=int, default=15)
    parser.add_argument(
        "--wrist-reference-world-z-offset-m", type=float, default=0.0,
        help="World-Z offset applied to both wrist references before contact/tracking.",
    )
    parser.add_argument(
        "--contact-edge-hold",
        action="store_true",
        help=(
            "Use the first/last contact 6D wrist-link residual outside the "
            "contact interval. This is the required causal schedule."
        ),
    )
    parser.add_argument(
        "--contact-schedule-root",
        type=Path,
        help="Persist the expanded injected reference schedule under SESSION_ID/.",
    )
    parser.add_argument(
        "--max-gap-frames",
        type=int,
        default=10,
        help=(
            "Maximum internal missing-hand gap, in source frames, that the "
            "dual-wrist prepare stage may fill in the camera-local joint "
            "frame (Hermite wrist translation + quaternion slerp). Default "
            "10 matches the historical prepare limit."
        ),
    )
    parser.add_argument("--reference-savgol-window", type=int, default=5)
    parser.add_argument("--reference-savgol-polyorder", type=int, default=2)
    parser.add_argument("--reference-savgol-gap-factor", type=float, default=3.0)
    parser.add_argument(
        "--reference-wrist-outlier-max-duration-s", type=float, default=1.1
    )
    parser.add_argument(
        "--reference-wrist-outlier-position-floor-m", type=float, default=0.05
    )
    parser.add_argument(
        "--reference-wrist-outlier-orientation-floor-deg", type=float, default=45.0
    )
    parser.add_argument(
        "--q0-zero-arm-no-smoothness",
        action="store_true",
        help=(
            "Use the frame-0 zero-arm Mink initialization.  Frames >=1 keep "
            "the normal warm-start and temporal smoothness solver."
        ),
    )
    parser.add_argument("--q0-max-arm-delta", type=float, default=3.0)
    parser.add_argument("--q0-arm-velocity-limit", type=float, default=1.0e4)
    parser.add_argument("--q0-posture-cost", type=float, default=0.0)
    parser.add_argument("--q0-orientation-cost", type=float, default=0.08)
    parser.add_argument("--q0-collision-margin", type=float)
    parser.add_argument(
        "--q0-shoulder-clearance",
        action="store_true",
        help="Enable the q0 elbow-side upper-arm clearance geometry.",
    )
    parser.add_argument("--q0-shoulder-clearance-cost", type=float, default=0.0)
    parser.add_argument(
        "--q0-shoulder-clearance-threshold", type=float, default=0.06
    )
    parser.add_argument(
        "--q0-shoulder-clearance-sample-start", type=float, default=0.40
    )
    parser.add_argument(
        "--q0-shoulder-clearance-sample-count", type=int, default=6
    )
    parser.add_argument(
        "--q0-open-branch-fallback",
        action="store_true",
        help="Retry q0 from the raw MPC arm branch when clearance is insufficient.",
    )
    parser.add_argument("--q0-min-shoulder-clearance", type=float, default=0.01)
    return parser.parse_args()


def build_q0_mink_options(args: argparse.Namespace) -> Q0MinkOptions:
    """Validate and normalize the batch-level q0 Mink options."""
    numeric_positive = (
        ("q0_max_arm_delta", args.q0_max_arm_delta),
        ("q0_arm_velocity_limit", args.q0_arm_velocity_limit),
        ("q0_shoulder_clearance_threshold", args.q0_shoulder_clearance_threshold),
    )
    for name, value in numeric_positive:
        if float(value) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.q0_posture_cost < 0.0 or args.q0_orientation_cost < 0.0:
        raise ValueError("q0 costs must be non-negative")
    if args.q0_collision_margin is not None and args.q0_collision_margin < 0.0:
        raise ValueError("--q0-collision-margin must be non-negative")
    if args.q0_shoulder_clearance_cost < 0.0:
        raise ValueError("--q0-shoulder-clearance-cost must be non-negative")
    if not 0.0 <= args.q0_shoulder_clearance_sample_start < 1.0:
        raise ValueError("--q0-shoulder-clearance-sample-start must be in [0, 1)")
    if args.q0_shoulder_clearance_sample_count < 2:
        raise ValueError("--q0-shoulder-clearance-sample-count must be >= 2")
    if args.q0_min_shoulder_clearance < 0.0:
        raise ValueError("--q0-min-shoulder-clearance must be non-negative")
    return Q0MinkOptions(
        zero_arm_no_smoothness=bool(
            args.q0_zero_arm_no_smoothness
            or args.q0_shoulder_clearance
            or args.q0_shoulder_clearance_cost > 0.0
        ),
        max_arm_delta=float(args.q0_max_arm_delta),
        arm_velocity_limit=float(args.q0_arm_velocity_limit),
        posture_cost=float(args.q0_posture_cost),
        orientation_cost=float(args.q0_orientation_cost),
        collision_margin=(
            float(args.q0_collision_margin)
            if args.q0_collision_margin is not None
            else None
        ),
        shoulder_clearance=bool(args.q0_shoulder_clearance),
        shoulder_clearance_cost=float(args.q0_shoulder_clearance_cost),
        shoulder_clearance_threshold=float(args.q0_shoulder_clearance_threshold),
        shoulder_clearance_sample_start=float(args.q0_shoulder_clearance_sample_start),
        shoulder_clearance_sample_count=int(args.q0_shoulder_clearance_sample_count),
        open_branch_fallback=bool(args.q0_open_branch_fallback),
        min_shoulder_clearance=float(args.q0_min_shoulder_clearance),
    )


def build_reference_smoothing_options(
    args: argparse.Namespace,
) -> ReferenceSmoothingOptions:
    window = int(args.reference_savgol_window)
    polyorder = int(args.reference_savgol_polyorder)
    gap_factor = float(args.reference_savgol_gap_factor)
    outlier_duration = float(args.reference_wrist_outlier_max_duration_s)
    outlier_position = float(args.reference_wrist_outlier_position_floor_m)
    outlier_orientation = float(args.reference_wrist_outlier_orientation_floor_deg)
    if window < 3 or window % 2 != 1:
        raise ValueError("--reference-savgol-window must be an odd integer >= 3")
    if polyorder < 0 or polyorder >= window:
        raise ValueError(
            "--reference-savgol-polyorder must satisfy 0 <= polyorder < window"
        )
    if gap_factor <= 1.0:
        raise ValueError("--reference-savgol-gap-factor must be greater than 1")
    if outlier_duration <= 0.0 or outlier_position <= 0.0 or outlier_orientation <= 0.0:
        raise ValueError("reference wrist outlier repair thresholds must be positive")
    return ReferenceSmoothingOptions(
        enabled=True,
        window_length=window,
        polyorder=polyorder,
        gap_factor=gap_factor,
        wrist_outlier_max_duration_s=outlier_duration,
        wrist_outlier_position_floor_m=outlier_position,
        wrist_outlier_orientation_floor_deg=outlier_orientation,
    )


def main() -> None:
    args = parse_args()
    mpc_backend = args.mpc_backend
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    work_root = args.work_root.expanduser().resolve()
    orientation_offset_root = (
        args.orientation_offset_root.expanduser().resolve()
        if args.orientation_offset_root is not None
        else None
    )
    contact_aware_root = (
        args.contact_aware_root.expanduser().resolve()
        if args.contact_aware_root is not None
        else None
    )
    joint_contact_root = (
        args.joint_contact_root.expanduser().resolve()
        if args.joint_contact_root is not None
        else None
    )
    contact_schedule_root = (
        args.contact_schedule_root.expanduser().resolve()
        if args.contact_schedule_root is not None
        else None
    )
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    if args.workers is not None and args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.cpu_threads_per_worker <= 0:
        raise ValueError("--cpu-threads-per-worker must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.contact_transition_frames < 0:
        raise ValueError("--contact-transition-frames must be non-negative")
    if args.max_gap_frames < 1:
        raise ValueError("--max-gap-frames must be >= 1")
    q0_mink = build_q0_mink_options(args)
    reference_smoothing = build_reference_smoothing_options(args)
    hand_mode = str(args.hand_mode)
    if hand_mode == HAND_MODE_BRAINCO:
        conflicting = {
            "--contact-aware-root": contact_aware_root is not None,
            "--orientation-offset-root": orientation_offset_root is not None,
        }
        rejected_flags = sorted(name for name, used in conflicting.items() if used)
        if rejected_flags:
            raise ValueError(
                f"--hand-mode={HAND_MODE_BRAINCO} cannot combine Dex3 wrist-offset "
                f"inputs; remove {', '.join(rejected_flags)}"
            )
        required_brainco = [
            PY_DEX_RETARGET,
            BRAINCO_HAND_RETARGET,
            BRAINCO_MJCF,
            BRAINCO_HAND_URDF_DIR / "brainco_hand/brainco_hand_left.urdf",
            BRAINCO_HAND_URDF_DIR / "brainco_hand/brainco_hand_right.urdf",
            BRAINCO_RETARGET_CONFIG_DIR / "brainco_hand_left.yml",
            BRAINCO_RETARGET_CONFIG_DIR / "brainco_hand_right.yml",
        ]
        if args.joint_contact_precompute or joint_contact_root is not None:
            required_brainco.append(JOINT_CONTACT_PRECOMPUTE)
            required_brainco.append(INJECT_CONTACT_AWARE_REFERENCE)
        for path in required_brainco:
            if not path.is_file():
                raise FileNotFoundError(f"BrainCo asset is missing: {path}")
    if args.joint_contact_precompute and contact_aware_root is not None:
        raise ValueError(
            "--joint-contact-precompute replaces --contact-aware-root; do not pass both"
        )
    if args.joint_contact_precompute and joint_contact_root is not None:
        raise ValueError(
            "--joint-contact-precompute and --joint-contact-root are mutually exclusive"
        )
    if contact_aware_root is not None and joint_contact_root is not None:
        raise ValueError(
            "--contact-aware-root is the legacy wrist-only input; use only "
            "--joint-contact-root for a paired wrist/hand bundle"
        )
    if args.joint_contact_precompute and orientation_offset_root is not None:
        raise ValueError(
            "joint contact must start from plain template-palm; fixed orientation "
            "offsets are not supported in the unified tracking path"
        )
    if contact_aware_root is not None and orientation_offset_root is not None:
        raise ValueError(
            "contact-aware reference and fixed orientation offsets are mutually "
            "exclusive; contact candidates must be based on plain template-palm"
        )
    if joint_contact_root is not None and orientation_offset_root is not None:
        raise ValueError(
            "joint-contact bundles must start from plain template-palm; fixed "
            "orientation offsets are not supported"
        )

    sessions, rejected = discover_sessions(input_root, args.session_pattern)
    selected_session_ids = None
    if args.session_list is not None:
        selected_session_ids = read_session_list(args.session_list)
        sessions, rejected = select_discovery(
            sessions, rejected, selected_session_ids
        )
    sessions, rejected = limit_discovery(sessions, rejected, args.limit)
    orientation_offsets: dict[str, Path | None] = {}
    contact_candidates: dict[str, Path | None] = {}
    joint_contact_bundles: dict[str, JointContactBundle | None] = {}
    for session in sessions:
        path = (
            orientation_offset_root
            / session.session_id
            / "wrist_orientation_offset.json"
            if orientation_offset_root is not None
            else None
        )
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
        orientation_offsets[session.session_id] = path
        bundle = resolve_joint_contact_bundle(
            joint_contact_root, session.session_id, hand_mode=hand_mode
        )
        joint_contact_bundles[session.session_id] = bundle
        contact_candidates[session.session_id] = (
            bundle.wrist_candidate
            if bundle is not None and bundle.inject_wrist_residual
            else (
                None
                if hand_mode == HAND_MODE_BRAINCO
                else resolve_contact_aware_candidate(
                    contact_aware_root, session.session_id
                )
            )
        )
    selected_gpus = select_gpus(
        args.gpus,
        min_free_mb=args.min_free_gpu_mb,
        max_utilization=args.max_gpu_utilization,
    )
    workers = args.workers or len(selected_gpus)
    if workers > len(selected_gpus):
        raise ValueError(
            f"--workers={workers} exceeds selected GPU count {len(selected_gpus)}"
        )
    selected_gpus = selected_gpus[:workers]

    discovery = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "discovered_sessions": len(sessions) + len(rejected),
        "runnable_sessions": len(sessions),
        "rejected_sessions": rejected,
        "selected_gpus": [asdict(gpu) for gpu in selected_gpus],
        "workers": workers,
        "sequence_policy": "complete source sequence resampled to 50 Hz",
        "session_list": (
            str(args.session_list.expanduser().resolve())
            if args.session_list is not None
            else None
        ),
        "orientation_offset_root": (
            str(orientation_offset_root)
            if orientation_offset_root is not None
            else None
        ),
        "contact_aware_root": (
            str(contact_aware_root) if contact_aware_root is not None else None
        ),
        "hand_mode": hand_mode,
        "hand_mode": hand_mode,
        "joint_contact_precompute": bool(args.joint_contact_precompute),
        "joint_contact_root": (
            str(joint_contact_root) if joint_contact_root is not None else None
        ),
        "contact_transition_frames": int(args.contact_transition_frames),
        "wrist_reference_world_z_offset_m": float(args.wrist_reference_world_z_offset_m),
        "max_gap_frames": int(args.max_gap_frames),
        "contact_edge_hold": bool(args.contact_edge_hold),
        "contact_schedule_root": (
            str(contact_schedule_root) if contact_schedule_root is not None else None
        ),
        "reference_smoothing": reference_smoothing.as_dict(),
        "q0_mink": q0_mink.as_dict(),
        "mpc_backend": mpc_backend,
    }
    print(json.dumps(discovery, indent=2), flush=True)
    if args.dry_run:
        return

    lock_stream = acquire_global_lock()
    install_signal_handlers()
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    global_log = output_root / "_batch_artifact.log"
    stage_timeout_s = args.stage_timeout_s if args.stage_timeout_s > 0 else None
    ensure_artifact(
        rebuild=bool(args.rebuild_artifact),
        log_path=global_log,
        stage_timeout_s=stage_timeout_s,
    )
    if global_log.is_file() and artifact_manifest_valid(ARTIFACT_MANIFEST):
        global_log.unlink()
    artifact_manifest_hash = sha256(ARTIFACT_MANIFEST)
    dependency_manifest = pipeline_dependency_manifest()
    pipeline_dependency_hash = dependency_manifest["sha256"]

    started_at = utc_now()
    results: dict[str, JobResult] = {}
    for session_id, reason in rejected.items():
        results[session_id] = JobResult(
            session_id=session_id,
            status="rejected_input",
            gpu=None,
            elapsed_s=0.0,
            error=reason,
            failed_stage="input_preflight",
        )

    pending: list[SessionProbe] = []
    for session in sessions:
        destination = output_root / session.session_id
        if not args.force and output_is_complete(
            destination,
            session.session_id,
            source_annotation=session.session_dir / "annotation.hdf5",
            source_calibration=(
                session.session_dir / SOURCE_CALIBRATION_RELATIVE
            ),
            artifact_manifest_hash=artifact_manifest_hash,
            pipeline_dependency_hash=pipeline_dependency_hash,
            orientation_offset_hash=(
                sha256(orientation_offsets[session.session_id])
                if orientation_offsets[session.session_id] is not None
                else None
            ),
            contact_aware_hash=(
                sha256(contact_candidates[session.session_id])
                if contact_candidates[session.session_id] is not None
                else None
            ),
            mink_q0_config_hash=(q0_mink.fingerprint if q0_mink.enabled else None),
            reference_smoothing_hash=reference_smoothing.fingerprint,
            require_joint_contact_hand=bool(
                args.joint_contact_precompute
                or hand_mode == HAND_MODE_BRAINCO
                or joint_contact_bundles[session.session_id] is not None
            ),
            joint_contact_bundle_hash=(
                joint_contact_bundles[session.session_id].fingerprint
                if joint_contact_bundles[session.session_id] is not None
                else None
            ),
            hand_dof=HAND_DOF_BY_MODE[hand_mode],
        ):
            tracking = json.loads((destination / "tracking_error.json").read_text())
            results[session.session_id] = JobResult(
                session_id=session.session_id,
                status="skipped_complete",
                gpu=None,
                elapsed_s=0.0,
                output_dir=str(destination),
                frames=int(tracking["sampling"]["frames"]),
                accepted=bool(tracking.get("accepted", False)),
            )
        else:
            pending.append(session)

    if pending:
        if should_run_parity(args):
            print("parity=ran", flush=True)
            parity_log = output_root / "_batch_parity.log"
            run_global_parity(
                session=pending[0],
                gpu=selected_gpus[0].selector,
                work_root=work_root,
                cpu_threads=args.cpu_threads_per_worker,
                log_path=parity_log,
                stage_timeout_s=stage_timeout_s,
                orientation_offset_json=orientation_offsets[pending[0].session_id],
                contact_aware_candidates=contact_candidates[pending[0].session_id],
                contact_transition_frames=int(args.contact_transition_frames),
                contact_edge_hold=bool(args.contact_edge_hold),
                reference_smoothing=reference_smoothing,
                max_gap_frames=int(args.max_gap_frames),
            )
            if parity_log.is_file():
                parity_log.unlink()
        else:
            print("parity=skipped", flush=True)

    state_path = (
        args.state_file.expanduser().resolve()
        if args.state_file is not None
        else output_root / "batch_state.json"
    )

    def write_state() -> None:
        statuses = [result.status for result in results.values()]
        payload = {
            "format": "three-point JAX MPC batch state",
            "format_version": 1,
            "started_at": started_at,
            "updated_at": utc_now(),
            "input_root": str(input_root),
            "output_root": str(output_root),
            "artifact_manifest": str(ARTIFACT_MANIFEST),
            "artifact_manifest_sha256": artifact_manifest_hash,
            "pipeline_dependency_sha256": pipeline_dependency_hash,
            "settings": {
                "pipeline_config_id": PIPELINE_CONFIG_ID,
                "mpc_backend": mpc_backend,
                "sequence_policy": "complete source sequence resampled to 50 Hz",
                "checkpoint": str(CHECKPOINT),
                "N": 256,
                "H": 36,
                "cem_iterations": 4,
                "elite": 10,
                "execute_steps": 12,
                "seed": 2,
                "mink_arm_refinement": True,
                "hand_mode": hand_mode,
                "orientation_offset_root": (
                    str(orientation_offset_root)
                    if orientation_offset_root is not None
                    else None
                ),
                "contact_aware_root": (
                    str(contact_aware_root) if contact_aware_root is not None else None
                ),
                "joint_contact_precompute": bool(args.joint_contact_precompute),
                "joint_contact_root": (
                    str(joint_contact_root)
                    if joint_contact_root is not None
                    else None
                ),
                "contact_transition_frames": int(args.contact_transition_frames),
                "contact_edge_hold": bool(args.contact_edge_hold),
                "contact_schedule_root": (
                    str(contact_schedule_root)
                    if contact_schedule_root is not None else None
                ),
                "reference_smoothing": reference_smoothing.as_dict(),
                "q0_mink": q0_mink.as_dict(),
                "selected_gpus": [asdict(gpu) for gpu in selected_gpus],
            },
            "counts": {
                "total": len(sessions) + len(rejected),
                "reported": len(results),
                "pending": len(sessions) + len(rejected) - len(results),
                "completed": sum(status.startswith("completed") for status in statuses),
                "skipped_complete": statuses.count("skipped_complete"),
                "failed": statuses.count("failed"),
                "rejected_input": statuses.count("rejected_input"),
            },
            "sessions": {
                session_id: asdict(result)
                for session_id, result in sorted(results.items())
            },
        }
        atomic_json(state_path, payload)

    write_state()
    jax_workers: list[JaxMpcWorkerHandle] = []
    jax_mpc_socks: dict[str, str] = {}
    if pending:
        parent_sock = os.environ.get("JAX_MPC_WORKER_SOCK", "").strip()
        if parent_sock:
            print(f"jax_mpc_worker=external_socket sock={parent_sock}", flush=True)
            for gpu in selected_gpus:
                jax_mpc_socks[gpu.selector] = parent_sock
        else:
            print(
                "jax_mpc_worker=starting "
                + ",".join(gpu.selector for gpu in selected_gpus),
                flush=True,
            )
            jax_workers = start_jax_mpc_workers(
                [gpu.selector for gpu in selected_gpus],
                work_root=work_root,
                cpu_threads=args.cpu_threads_per_worker,
            )
            jax_mpc_socks = {handle.gpu: str(handle.sock) for handle in jax_workers}
            atexit.register(stop_jax_mpc_workers, jax_workers)
        gpu_slots: queue.Queue[str] = queue.Queue()
        for gpu in selected_gpus:
            gpu_slots.put(gpu.selector)
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        run_session,
                        session,
                        gpu_slots=gpu_slots,
                        output_root=output_root,
                        work_root=work_root,
                        cpu_threads=args.cpu_threads_per_worker,
                        keep_work=bool(args.keep_work),
                        force=bool(args.force),
                        stage_timeout_s=stage_timeout_s,
                        artifact_manifest_hash=artifact_manifest_hash,
                        dependency_manifest=dependency_manifest,
                        orientation_offset_json=orientation_offsets[session.session_id],
                        contact_aware_candidates=contact_candidates[session.session_id],
                        joint_contact_precompute=bool(args.joint_contact_precompute),
                        joint_contact_bundle=joint_contact_bundles[session.session_id],
                        contact_transition_frames=int(args.contact_transition_frames),
                        contact_edge_hold=bool(args.contact_edge_hold),
                        contact_schedule_root=contact_schedule_root,
                        q0_mink=q0_mink,
                        reference_smoothing=reference_smoothing,
                        wrist_reference_world_z_offset_m=float(
                            args.wrist_reference_world_z_offset_m
                        ),
                        hand_mode=hand_mode,
                        jax_mpc_socks=jax_mpc_socks,
                        max_gap_frames=int(args.max_gap_frames),
                    ): session.session_id
                    for session in pending
                }
                for future in as_completed(futures):
                    session_id = futures[future]
                    try:
                        result = future.result()
                    except Exception as error:
                        result = JobResult(
                            session_id=session_id,
                            status="failed",
                            gpu=None,
                            elapsed_s=0.0,
                            error=f"unhandled worker error: {error}",
                            failed_stage="worker_wrapper",
                        )
                    results[result.session_id] = result
                    write_state()
        finally:
            stop_jax_mpc_workers(jax_workers)

    write_state()
    staging_root = output_root / ".staging"
    if staging_root.is_dir() and not any(staging_root.iterdir()):
        staging_root.rmdir()
    failures = [
        result
        for result in results.values()
        if result.status in {"failed", "rejected_input"}
    ]
    quality_failures = [
        result
        for result in results.values()
        if result.status == "completed_quality_failed"
    ]
    print(
        f"Batch finished: total={len(results)} failures={len(failures)} "
        f"quality_failures={len(quality_failures)} state={state_path}",
        flush=True,
    )
    if failures:
        lock_stream.close()
        raise SystemExit(2)
    lock_stream.close()


if __name__ == "__main__":
    main()
