#!/usr/bin/env bash
set -Eeuo pipefail

# Build the per-session evidence consumed by the contact-aware tracking
# worker. This stage is deliberately outside whole_body_data: only the
# published kinematics/video contract is copied there by the batch driver.
# KINEMATICS_NPZ may be the temporary raw-template reference archive prepared
# by prepare_contact_template_palm_reference.sh; it must not be a prior
# tracking result.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SESSION_ID="${SESSION_ID:?SESSION_ID is required}"
INPUT_ROOT="${INPUT_ROOT:-${PROJECT_ROOT}/data/egocentric_data/pick_up_the_cup_smooth}"
KINEMATICS_ROOT="${KINEMATICS_ROOT:-${PROJECT_ROOT}/data/kinematics/pick_up_the_cup_smooth}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${PROJECT_ROOT}/outputs/hand_retarget}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${EXPERIMENT_ROOT}/${SESSION_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_DIR}/contact_aware_auto}"
GPU="${GPU:-1}"
CONTACT_DEVICE="${CONTACT_DEVICE:-cuda}"
FORCE="${FORCE:-0}"
SIDE="${SIDE:-right}"
MAX_NATIVE_FRAMES="${MAX_NATIVE_FRAMES:-}"
MIN_MASK_POINTS="${MIN_MASK_POINTS:-100}"
MAX_WINDOW_GAP="${MAX_WINDOW_GAP:-6}"
MIN_WINDOW_FRAMES="${MIN_WINDOW_FRAMES:-20}"
HAND_MODE="${HAND_MODE:-dex3}"
CUP_PROMPTS="${CUP_PROMPTS:-a blue cup|a blue mug|a cup|a mug}"
SAM_CONFIDENCE="${SAM_CONFIDENCE:-0.20}"
CONTACT_SEED="${CONTACT_SEED:-20260807}"
CONTACT_KEYFRAME="${CONTACT_KEYFRAME:-}"
CONTACT_NO_EVIDENCE_POLICY="${CONTACT_NO_EVIDENCE_POLICY:-identity}"
JOINT_CONTACT_ENABLED="${JOINT_CONTACT_ENABLED:-1}"
OBJECT_SOURCE="${OBJECT_SOURCE:-}"
CONTACT_NEIGHBORHOOD_M="${CONTACT_NEIGHBORHOOD_M:-0.04}"
CONTACT_NEIGHBORHOOD_MASK_MARGIN_PX="${CONTACT_NEIGHBORHOOD_MASK_MARGIN_PX:-0}"
CONTACT_NEIGHBORHOOD_HAND_CLEARANCE_M="${CONTACT_NEIGHBORHOOD_HAND_CLEARANCE_M:-0}"
CONTACT_DEPTH_FOOTPRINT="${CONTACT_DEPTH_FOOTPRINT:-1}"
CONTACT_OBJECT_CLOUD_MAX_POINTS="${CONTACT_OBJECT_CLOUD_MAX_POINTS:-}"
CONTACT_LAPLACIAN_OBJECT_CLOUD_MAX_POINTS="${CONTACT_LAPLACIAN_OBJECT_CLOUD_MAX_POINTS:-}"
CONTACT_OBJECT_CLOUD_RADIUS_M="${CONTACT_OBJECT_CLOUD_RADIUS_M:-}"
CONTACT_OBJECT_CLOUD_VOXEL_M="${CONTACT_OBJECT_CLOUD_VOXEL_M:-}"
CONTACT_NEIGHBORHOOD_DILATION_PX="${CONTACT_NEIGHBORHOOD_DILATION_PX:-16}"
CONTACT_NEIGHBORHOOD_Z_BAND_M="${CONTACT_NEIGHBORHOOD_Z_BAND_M:-}"
BRAINCO_MJCF="${BRAINCO_MJCF:-${PROJECT_ROOT}/resources/brainco/robots/g1_brainco/g1_mocap_29dof_with_brainco_hands.xml}"
JOINT_CONTACT_TRANSITION_FRAMES="${JOINT_CONTACT_TRANSITION_FRAMES:-15}"
DBSCAN_EPS_M="${DBSCAN_EPS_M:-0.020}"
DBSCAN_MIN_SAMPLES="${DBSCAN_MIN_SAMPLES:-3}"
MESH_SAMPLES_PER_GEOM="${MESH_SAMPLES_PER_GEOM:-400}"
RANSAC_ITERATIONS="${RANSAC_ITERATIONS:-500}"
CONTACT_DISTANCE_THRESHOLD_M="${CONTACT_DISTANCE_THRESHOLD_M:-0.020}"
CONTACT_DEPTH_GAP_THRESHOLD_M="${CONTACT_DEPTH_GAP_THRESHOLD_M:-0.020}"
CONTACT_NORMAL_DOT_THRESHOLD="${CONTACT_NORMAL_DOT_THRESHOLD:-0.10}"
CONTACT_TEMPORAL_GAP="${CONTACT_TEMPORAL_GAP:-3}"
CONTACT_MIN_RUN="${CONTACT_MIN_RUN:-3}"
CONTACT_MIN_FINGERS="${CONTACT_MIN_FINGERS:-2}"
CONTACT_OBJECT_MASK_MAX_GAP="${CONTACT_OBJECT_MASK_MAX_GAP:-4}"
HUMAN_MASK_BACKEND="${HUMAN_MASK_BACKEND:-efficientsam3}"
JOINT_CONTACT_PENETRATION_TOLERANCE_M="${JOINT_CONTACT_PENETRATION_TOLERANCE_M:-0.005}"
JOINT_CONTACT_PENETRATION_RATIO_TOLERANCE="${JOINT_CONTACT_PENETRATION_RATIO_TOLERANCE:-0.02}"
JOINT_CONTACT_ROBUST_PENETRATION_M="${JOINT_CONTACT_ROBUST_PENETRATION_M:-0.0045}"
JOINT_CONTACT_ROBUST_PENETRATION_RATIO="${JOINT_CONTACT_ROBUST_PENETRATION_RATIO:-0.018}"
JOINT_CONTACT_PENETRATION_TARGET_M="${JOINT_CONTACT_PENETRATION_TARGET_M:-0.0015}"
JOINT_CONTACT_PENETRATION_RATIO_TARGET="${JOINT_CONTACT_PENETRATION_RATIO_TARGET:-0.010}"
JOINT_CONTACT_PENETRATION_DEPTH_WEIGHT="${JOINT_CONTACT_PENETRATION_DEPTH_WEIGHT:-55.0}"
JOINT_CONTACT_PENETRATION_RATIO_WEIGHT="${JOINT_CONTACT_PENETRATION_RATIO_WEIGHT:-6.0}"
JOINT_CONTACT_FINGERTIP_PENETRATION_MAX_M="${JOINT_CONTACT_FINGERTIP_PENETRATION_MAX_M:-0.003}"
JOINT_CONTACT_FINGERTIP_PENETRATION_P95_M="${JOINT_CONTACT_FINGERTIP_PENETRATION_P95_M:-0.0015}"
JOINT_CONTACT_FINGERTIP_PENETRATION_RATIO="${JOINT_CONTACT_FINGERTIP_PENETRATION_RATIO:-0.01}"
JOINT_CONTACT_FINGERTIP_PENETRATION_WEIGHT="${JOINT_CONTACT_FINGERTIP_PENETRATION_WEIGHT:-0.0}"

HAND_PY="${HAND_PY:-${KODA_PYTHON:-${PYTHON:-python3}}}"
SAM_PY="${SAM_PY:-${KODA_PYTHON:-${PYTHON:-python3}}}"
TRACKING_PY="${TRACKING_PY:-${KODA_PYTHON:-${PYTHON:-python3}}}"
CONTACT_PY="${CONTACT_PY:-${KODA_PYTHON:-${PYTHON:-python3}}}"

SESSION_DIR="${INPUT_ROOT}/${SESSION_ID}"
KINEMATICS_NPZ="${KINEMATICS_NPZ:-${KINEMATICS_ROOT}/${SESSION_ID}/kinematics.npz}"
TARGET_TIMESTAMPS_NPZ="${TARGET_TIMESTAMPS_NPZ:-${KINEMATICS_NPZ}}"
REFERENCE_BOOTSTRAP_NPZ="${REFERENCE_BOOTSTRAP_NPZ:-}"
REFERENCE_BOOTSTRAP_SHA256="${REFERENCE_BOOTSTRAP_SHA256:-}"
WRIST_CANDIDATE="${EXPERIMENT_DIR}/wrist_candidates_native.npz"
FINGER_CANDIDATE="${EXPERIMENT_DIR}/finger_candidates_native.npz"
AUTO_ROOT="${EXPERIMENT_DIR}/contact_aware_auto"
MASK_ROOT="${EXPERIMENT_DIR}/cup_segmentation_auto"
MASK_SELECTED="${MASK_ROOT}/selected/masks"
INTERACTION_DIR="${AUTO_ROOT}/object_interaction"
INTERVAL_JSON="${AUTO_ROOT}/contact_interval.json"
POINTCLOUD_CONTACT_DIR="${AUTO_ROOT}/pointcloud_contact"
POINTCLOUD_INTERVAL_JSON="${POINTCLOUD_CONTACT_DIR}/contact_interval.json"
HUMAN_MASK_ROOT="${EXPERIMENT_DIR}/human_masks"
SOURCE_RGB_DIR="${SOURCE_RGB_DIR:-${EXPERIMENT_DIR}/source_rgb}"
CONTACT_CANDIDATE="${OUTPUT_DIR}/contact_aware_wrist_candidates_native.npz"
CONTACT_CANDIDATE_LEFT="${OUTPUT_DIR}/contact_aware_wrist_candidates_native_left.npz"
CONTACT_CANDIDATE_RIGHT="${OUTPUT_DIR}/contact_aware_wrist_candidates_native_right.npz"
WRIST_ONLY_CANDIDATE="${OUTPUT_DIR}/wrist_only/contact_aware_wrist_candidates_native.npz"
JOINT_CONTACT_DIR="${OUTPUT_DIR}/joint_contact"
BASE_HAND_50HZ="${BASE_HAND_50HZ:-}"
JOINT_HAND_OVERRIDE="${JOINT_HAND_OVERRIDE:-}"
if [[ -z "${BASE_HAND_50HZ}" ]]; then
  if [[ "${HAND_MODE}" == "g1_brainco" ]]; then
    BASE_HAND_50HZ="${JOINT_CONTACT_DIR}/base_hand_brainco_50hz.npz"
  else
    BASE_HAND_50HZ="${JOINT_CONTACT_DIR}/base_hand_dex3_50hz.npz"
  fi
fi
if [[ -z "${JOINT_HAND_OVERRIDE}" ]]; then
  if [[ "${HAND_MODE}" == "g1_brainco" ]]; then
    JOINT_HAND_OVERRIDE="${OUTPUT_DIR}/hand_brainco_50hz_joint_contact.npz"
  else
    JOINT_HAND_OVERRIDE="${OUTPUT_DIR}/hand_dex3_50hz_joint_contact.npz"
  fi
fi
MANIFEST="${OUTPUT_DIR}/contact_precompute_manifest.json"

require_file() { [[ -f "$1" ]] || { echo "[contact-precompute] missing file: $1" >&2; exit 2; }; }
require_dir() { [[ -d "$1" ]] || { echo "[contact-precompute] missing directory: $1" >&2; exit 2; }; }

# One BLAS/Torch thread per contact GPU worker.  Do not export these into
# tracking/JAX (run_three_point_batch already pins worker_environment to 4).
run_contact_py() {
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  TORCH_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES="${GPU}" \
    "${CONTACT_PY}" "$@"
}

require_dir "${SESSION_DIR}"
require_file "${SESSION_DIR}/annotation.hdf5"
require_file "${SESSION_DIR}/rgb.mp4"
require_file "${SESSION_DIR}/calibrations/rgb_K.npy"
require_file "${SESSION_DIR}/calibrations/depth_K.npy"
require_file "${SESSION_DIR}/calibrations/R_optical_to_link.npy"
require_file "${SAM_PY}"
require_file "${TRACKING_PY}"
require_file "${CONTACT_PY}"
require_file "${SCRIPT_DIR}/make_human_arm_masks.py"
require_file "${SCRIPT_DIR}/efficientsam3_client.py"
require_file "${SCRIPT_DIR}/contact_pointcloud_analysis.py"
require_file "${HAND_PY}"
if [[ "${HAND_MODE}" != "g1_brainco" ]]; then
  require_file "${PROJECT_ROOT}/src/koda/retargeting/dex3.py"
else
  require_file "${SCRIPT_DIR}/brainco_hand_retarget.py"
fi
require_file "${KINEMATICS_NPZ}"
require_file "${TARGET_TIMESTAMPS_NPZ}"
if [[ "${CONTACT_DEVICE}" == cuda* ]]; then
  # Do not silently turn a CUDA configuration error into a no-contact identity
  # candidate: the formal pipeline must either run the geometry on its worker
  # GPU or fail before writing contact evidence.
  CUDA_VISIBLE_DEVICES="${GPU}" "${TRACKING_PY}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CONTACT_DEVICE requests CUDA but no CUDA device is available")
print(f"[contact-precompute] CUDA device ready: {torch.cuda.get_device_name(0)}")
PY
  run_contact_py - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CONTACT_DEVICE requests CUDA but CONTACT_PY cannot see CUDA")
PY
fi
if [[ -n "${REFERENCE_BOOTSTRAP_NPZ}" ]]; then
  require_file "${REFERENCE_BOOTSTRAP_NPZ}"
fi
[[ "${CONTACT_NO_EVIDENCE_POLICY}" == "identity" || \
   "${CONTACT_NO_EVIDENCE_POLICY}" == "fail" ]] || {
  echo "CONTACT_NO_EVIDENCE_POLICY must be identity or fail" >&2
  exit 2
}
[[ "${JOINT_CONTACT_ENABLED}" == "0" || "${JOINT_CONTACT_ENABLED}" == "1" ]] || {
  echo "JOINT_CONTACT_ENABLED must be 0 or 1" >&2
  exit 2
}
[[ "${CONTACT_DEPTH_FOOTPRINT}" == "0" || "${CONTACT_DEPTH_FOOTPRINT}" == "1" ]] || {
  echo "CONTACT_DEPTH_FOOTPRINT must be 0 or 1" >&2
  exit 2
}
[[ "${HAND_MODE}" == "dex3" || "${HAND_MODE}" == "g1_brainco" ]] || {
  echo "HAND_MODE must be dex3 or g1_brainco" >&2
  exit 2
}
if [[ "${HAND_MODE}" == "g1_brainco" ]]; then
  SIDE="both"
  OBJECT_SOURCE="${OBJECT_SOURCE:-fingertip_neighborhood}"
  CONTACT_OBJECT_CLOUD_MAX_POINTS="${CONTACT_OBJECT_CLOUD_MAX_POINTS:-2048}"
  CONTACT_LAPLACIAN_OBJECT_CLOUD_MAX_POINTS="${CONTACT_LAPLACIAN_OBJECT_CLOUD_MAX_POINTS:-512}"
  CONTACT_OBJECT_CLOUD_RADIUS_M="${CONTACT_OBJECT_CLOUD_RADIUS_M:-0.12}"
  CONTACT_OBJECT_CLOUD_VOXEL_M="${CONTACT_OBJECT_CLOUD_VOXEL_M:-0.02}"
  CONTACT_NEIGHBORHOOD_Z_BAND_M="${CONTACT_NEIGHBORHOOD_Z_BAND_M:-0.12}"
else
  OBJECT_SOURCE="${OBJECT_SOURCE:-mask}"
  CONTACT_OBJECT_CLOUD_MAX_POINTS="${CONTACT_OBJECT_CLOUD_MAX_POINTS:-1024}"
  CONTACT_OBJECT_CLOUD_RADIUS_M="${CONTACT_OBJECT_CLOUD_RADIUS_M:-0}"
  CONTACT_OBJECT_CLOUD_VOXEL_M="${CONTACT_OBJECT_CLOUD_VOXEL_M:-0}"
  CONTACT_NEIGHBORHOOD_Z_BAND_M="${CONTACT_NEIGHBORHOOD_Z_BAND_M:-0.05}"
  if [[ "${JOINT_CONTACT_ENABLED}" == "1" && "${SIDE}" != "right" ]]; then
    echo "joint contact optimization currently supports SIDE=right only" >&2
    exit 2
  fi
fi

mkdir -p "${EXPERIMENT_DIR}" "${AUTO_ROOT}" "${OUTPUT_DIR}" "${MASK_ROOT}"
LOG="${AUTO_ROOT}/precompute.log"
exec > >(tee -a "${LOG}") 2>&1

if [[ "${FORCE}" == "1" ]]; then
  # FORCE means a fresh prompt set, not a mixture of the current prompts with
  # stale masks from an earlier experiment (for example the old blue-cup run).
  # MASK_ROOT is a per-session generated directory and is safe to rebuild.
  rm -rf -- "${MASK_ROOT}"
  mkdir -p "${MASK_ROOT}"
  echo "[contact-precompute] FORCE=1 cleared stale prompt masks under ${MASK_ROOT}"
fi

manifest_matches_pointcloud_contract() {
  "${TRACKING_PY}" - "${MANIFEST}" \
    "${CONTACT_DISTANCE_THRESHOLD_M}" "${CONTACT_DEPTH_GAP_THRESHOLD_M}" \
  "${CONTACT_NORMAL_DOT_THRESHOLD}" "${CONTACT_TEMPORAL_GAP}" \
    "${CONTACT_MIN_RUN}" "${CONTACT_MIN_FINGERS}" "${JOINT_CONTACT_ENABLED}" \
    "${CONTACT_DEVICE}" "${HAND_MODE}" "${OBJECT_SOURCE}" \
    "${CONTACT_OBJECT_CLOUD_MAX_POINTS}" \
    "${CONTACT_OBJECT_CLOUD_RADIUS_M}" "${CONTACT_OBJECT_CLOUD_VOXEL_M}" \
    "${CONTACT_NEIGHBORHOOD_Z_BAND_M}" <<'PY'
import json, sys
from pathlib import Path
path = sys.argv[1]
expected = {
    "distance_m": float(sys.argv[2]),
    "depth_gap_m": float(sys.argv[3]),
    "normal_abs_dot": float(sys.argv[4]),
    "temporal_gap_frames": int(sys.argv[5]),
    "min_run_frames": int(sys.argv[6]),
    "min_fingers": int(sys.argv[7]),
}
joint_enabled = sys.argv[8] == "1"
requested_device = sys.argv[9]
hand_mode = sys.argv[10]
object_source = sys.argv[11]
object_cloud_max_points = int(sys.argv[12]) if str(sys.argv[12]).strip() else None
object_cloud_radius_m = float(sys.argv[13]) if str(sys.argv[13]).strip() else None
object_cloud_voxel_m = float(sys.argv[14]) if str(sys.argv[14]).strip() else None
neighborhood_z_band_m = float(sys.argv[15]) if str(sys.argv[15]).strip() else None
try:
    payload = json.load(open(path, encoding="utf-8"))
    thresholds = payload.get("contact_thresholds", {})
    return_code = int(
        payload.get("contact_detector") != "source_rgbd_pointcloud"
        or any(thresholds.get(k) != value for k, value in expected.items())
    )
    joint = payload.get("joint_contact_optimization")
    if not isinstance(joint, dict):
        return_code = 1
    elif bool(joint.get("enabled", False)) != joint_enabled:
        return_code = 1
    elif payload.get("contact_compute", {}).get("requested_device") != requested_device:
        return_code = 1
    elif payload.get("hand_mode", "dex3") != hand_mode:
        return_code = 1
    elif payload.get("object_source") not in (None, object_source):
        return_code = 1
    elif object_cloud_max_points is not None and int(
        thresholds.get("object_cloud_max_points") or 0
    ) != int(object_cloud_max_points):
        return_code = 1
    elif object_cloud_radius_m is not None and abs(
        float(thresholds.get("object_cloud_radius_m") or -1.0)
        - float(object_cloud_radius_m)
    ) > 1e-9:
        return_code = 1
    elif object_cloud_voxel_m is not None and abs(
        float(thresholds.get("object_cloud_voxel_m") or -1.0)
        - float(object_cloud_voxel_m)
    ) > 1e-9:
        return_code = 1
    elif neighborhood_z_band_m is not None and abs(
        float(thresholds.get("neighborhood_z_band_m") or -1.0)
        - float(neighborhood_z_band_m)
    ) > 1e-9:
        return_code = 1
    elif hand_mode == "g1_brainco":
        if joint_enabled and payload.get("status") == "contact_pointcloud_2finger":
            for key in ("wrist_candidate", "hand_override"):
                candidate = joint.get(key)
                if not candidate or not Path(candidate).is_file():
                    return_code = 1
            left = joint.get("wrist_candidate_left")
            right = joint.get("wrist_candidate_right")
            if (not left or not Path(left).is_file()) and (
                not right or not Path(right).is_file()
            ):
                return_code = 1
        base_hand = joint.get("base_hand_50hz")
        if joint_enabled and (not base_hand or not Path(base_hand).is_file()):
            return_code = 1
    elif joint_enabled and payload.get("status") == "contact_pointcloud_2finger":
        for key in ("wrist_candidate", "hand_override"):
            candidate = joint.get(key)
            if not candidate or not Path(candidate).is_file():
                return_code = 1
except Exception:
    return_code = 1
raise SystemExit(return_code)
PY
}

if [[ "${HAND_MODE}" == "g1_brainco" ]]; then
  reuse_candidate="${CONTACT_CANDIDATE_RIGHT}"
  if [[ ! -s "${reuse_candidate}" ]]; then
    reuse_candidate="${CONTACT_CANDIDATE_LEFT}"
  fi
  if [[ "${FORCE}" != "1" && -s "${reuse_candidate}" && \
        -s "${MANIFEST}" && -s "${BASE_HAND_50HZ}" ]] && \
        manifest_matches_pointcloud_contract; then
    echo "[contact-precompute] reuse ${reuse_candidate}"
    printf '%s\n' "${reuse_candidate}"
    exit 0
  fi
elif [[ "${FORCE}" != "1" && -s "${CONTACT_CANDIDATE}" && \
      -s "${MANIFEST}" ]] && manifest_matches_pointcloud_contract; then
  echo "[contact-precompute] reuse ${CONTACT_CANDIDATE}"
  printf '%s\n' "${CONTACT_CANDIDATE}"
  exit 0
fi

# Stage 1: derive the template-palm/position contact initializer from the raw
# annotation timeline. Restricting methods keeps this adapter deterministic and
# avoids generating unused ablation archives.
if [[ "${HAND_MODE}" != "g1_brainco" ]]; then
if [[ "${FORCE}" == "1" || ! -s "${WRIST_CANDIDATE}" || ! -s "${FINGER_CANDIDATE}" ]]; then
  env ROOT="${PROJECT_ROOT}/.." \
    SOURCE_ROOT="${INPUT_ROOT}" \
    KIN_ROOT="${KINEMATICS_ROOT}" \
    KINEMATICS_NPZ="${KINEMATICS_NPZ}" \
    TARGET_TIMESTAMPS_NPZ="${TARGET_TIMESTAMPS_NPZ}" \
    TEMPLATE_PALM_FROM_KINEMATICS="${TEMPLATE_PALM_FROM_KINEMATICS:-0}" \
    OUT_ROOT="${EXPERIMENT_ROOT}" \
    SESSION_ID="${SESSION_ID}" \
    WRIST_METHODS="template_palm_kabsch" \
    FINGER_METHODS="position" \
    MAX_NATIVE_FRAMES="${MAX_NATIVE_FRAMES}" \
    "${SCRIPT_DIR}/run_hand_retarget_experiment.sh"
fi
require_file "${WRIST_CANDIDATE}"
require_file "${FINGER_CANDIDATE}"

# The tracking pipeline publishes a synchronized Dex3 trajectory even when an
# episode has no defensible contact interval.  Build the position-retargeted
# 50 Hz hand before contact classification; a successful joint solve replaces
# only its right-hand values later in this script.
if [[ "${JOINT_CONTACT_ENABLED}" == "1" && \
      ( "${FORCE}" == "1" || ! -s "${BASE_HAND_50HZ}" ) ]]; then
  mkdir -p "$(dirname "${BASE_HAND_50HZ}")"
  echo "[contact-precompute] stage base Dex3 position retarget on tracking timeline"
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/resources/xr_teleoperate/teleop/robot_control/dex-retargeting/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${HAND_PY}" -m koda.retargeting.dex3 \
    --session-dir "${SESSION_DIR}" \
    --target-kinematics-npz "${TARGET_TIMESTAMPS_NPZ}" \
    --output "${BASE_HAND_50HZ}" \
    --retarget-type position
fi
if [[ "${JOINT_CONTACT_ENABLED}" == "1" ]]; then
  require_file "${BASE_HAND_50HZ}"
fi
else
BRAINCO_BASE_REFERENCE="${BRAINCO_BASE_REFERENCE:-$(dirname "${KINEMATICS_NPZ}")/template_palm_kabsch_reference.npz}"
BRAINCO_BASE_TARGETS="${BRAINCO_BASE_TARGETS:-${TARGET_TIMESTAMPS_NPZ}}"
BRAINCO_BASE_META="${BRAINCO_BASE_META:-$(dirname "${KINEMATICS_NPZ}")/template_palm_kabsch_meta.json}"
BRAINCO_NATIVE_REF="${EXPERIMENT_DIR}/brainco_precompute_reference.npz"
BRAINCO_NATIVE_TARGETS="${EXPERIMENT_DIR}/brainco_precompute_targets.npz"
BRAINCO_NATIVE_META="${EXPERIMENT_DIR}/brainco_precompute_meta.json"
require_file "${BRAINCO_BASE_REFERENCE}"
require_file "${BRAINCO_BASE_TARGETS}"
require_file "${BRAINCO_BASE_META}"
mkdir -p "${JOINT_CONTACT_DIR}" "${EXPERIMENT_DIR}"
if [[ "${FORCE}" == "1" || ! -s "${BASE_HAND_50HZ}" || \
      ! -s "${WRIST_CANDIDATE}" || ! -s "${FINGER_CANDIDATE}" ]]; then
  echo "[contact-precompute] stage BrainCo palm refine seed + native candidates"
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${PROJECT_ROOT}/resources/xr_teleoperate/teleop/robot_control/dex-retargeting/src" \
    "${HAND_PY}" "${SCRIPT_DIR}/brainco_hand_retarget.py" \
    --annotation-hdf5 "${SESSION_DIR}/annotation.hdf5" \
    --base-reference "${BRAINCO_BASE_REFERENCE}" \
    --base-targets "${BRAINCO_BASE_TARGETS}" \
    --base-meta "${BRAINCO_BASE_META}" \
    --output-reference "${BRAINCO_NATIVE_REF}" \
    --output-targets "${BRAINCO_NATIVE_TARGETS}" \
    --output-hand "${BASE_HAND_50HZ}" \
    --meta-json "${BRAINCO_NATIVE_META}" \
    --output-native-experiment-dir "${EXPERIMENT_DIR}"
fi
require_file "${BASE_HAND_50HZ}"
require_file "${WRIST_CANDIDATE}"
require_file "${FINGER_CANDIDATE}"
fi

# Stage 2: EfficientSAM3 object proposals.  All prompts share one loaded
# checkpoint: a persistent GPU worker when EFFICIENTSAM3_WORKER_SOCK is set,
# otherwise a single in-process load in efficientsam3_client.py.  Existing
# prompt outputs are reused unless FORCE=1.
# Decode source RGB once for SAM, adapter, warp, and blue-cup HSV.
"${SAM_PY}" "${SCRIPT_DIR}/packed_source_rgb.py" \
  --video "${SESSION_DIR}/rgb.mp4" \
  --out-dir "${SOURCE_RGB_DIR}" \
  --max-frames "${MAX_NATIVE_FRAMES:-0}"

selection_rc=1
if [[ "${OBJECT_SOURCE}" == "mask" ]]; then
  mkdir -p "${MASK_ROOT}"
  object_prompt_args=(
    "${SAM_PY}" "${SCRIPT_DIR}/efficientsam3_client.py"
    --video "${SESSION_DIR}/rgb.mp4"
    --packed-rgb "${SOURCE_RGB_DIR}"
    --mask-root "${MASK_ROOT}"
    --prompts-pipe "${CUP_PROMPTS}"
    --start-frame 0
    --max-frames "${MAX_NATIVE_FRAMES:-0}"
    --confidence "${SAM_CONFIDENCE}"
    --close-px 0 --dilation-px 0 --temporal-radius 0
    --batch-size 16 --resolution 1008 --device cuda
  )
  if [[ -n "${EFFICIENTSAM3_WORKER_SOCK:-}" ]]; then
    object_prompt_args+=(--socket "${EFFICIENTSAM3_WORKER_SOCK}")
  fi
  if [[ "${FORCE}" == "1" ]]; then
    object_prompt_args+=(--force)
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" "${object_prompt_args[@]}"
fi

# Stage 2b: EfficientSAM3 full-body masks for the observed RGB-D point cloud.
# Hands, arms, torso, and legs are removed.  A held object segmented apart
# from the hand stays in the source stream.  Explicit SAM3 mode fails loudly
# instead of silently changing the contact evidence to the keypoint fallback.
if [[ "${FORCE}" == "1" || ! -s "${HUMAN_MASK_ROOT}/human_mask_metadata.json" ]] || \
   { [[ "${MASK_SCOPE:-body}" == "hands" ]] && ! grep -q '"mask_kind": "human_hands_and_arms_only"' "${HUMAN_MASK_ROOT}/human_mask_metadata.json"; } || \
   { [[ "${MASK_SCOPE:-body}" != "hands" ]] && ! grep -q '"mask_kind": "human_body"' "${HUMAN_MASK_ROOT}/human_mask_metadata.json"; }; then
  human_args=(
    --rgb-mp4 "${SESSION_DIR}/rgb.mp4"
    --source-rgb-dir "${SOURCE_RGB_DIR}"
    --annotation-h5 "${SESSION_DIR}/annotation.hdf5"
    --out-dir "${HUMAN_MASK_ROOT}"
    --backend "${HUMAN_MASK_BACKEND}"
    --device cuda
    --prompt "human arms and hands"
  )
  if [[ "${MASK_SCOPE:-body}" == "hands" ]]; then
    human_args+=(--body-prompt "" --leg-prompt "")
  fi
  if [[ -n "${MAX_NATIVE_FRAMES}" ]]; then
    human_args+=(--max-frames "${MAX_NATIVE_FRAMES}")
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" "${SAM_PY}" \
    "${SCRIPT_DIR}/make_human_arm_masks.py" "${human_args[@]}"
fi
require_file "${HUMAN_MASK_ROOT}/human_mask_metadata.json"

contact_status="no_contact_detected"
contact_interval="none"
if [[ "${OBJECT_SOURCE}" == "mask" ]]; then
# Select the prompt/window with the largest stable RGB-D-supported run.  The
# helper uses the registered depth contract and writes a small JSON audit.
set +e
selection_args=(
  --session-dir "${SESSION_DIR}"
  --mask-root "${MASK_ROOT}"
  --output "${MASK_ROOT}/selection.json"
  --min-points "${MIN_MASK_POINTS}"
  --max-gap "${MAX_WINDOW_GAP}"
  --min-frames "${MIN_WINDOW_FRAMES}"
  --device "${CONTACT_DEVICE}"
)
if [[ -n "${SOURCE_RGB_DIR}" && -d "${SOURCE_RGB_DIR}" ]]; then
  selection_args+=(--source-rgb-dir "${SOURCE_RGB_DIR}")
fi
if [[ -n "${MAX_NATIVE_FRAMES}" ]]; then
  selection_args+=(--max-frames "${MAX_NATIVE_FRAMES}")
fi
CUDA_VISIBLE_DEVICES="${GPU}" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "${TRACKING_PY}" "${SCRIPT_DIR}/select_contact_mask_window.py" \
  "${selection_args[@]}"
selection_rc=$?
set -e

if [[ "${selection_rc}" -eq 0 ]]; then
  source <("${TRACKING_PY}" - <<PY
import json
p=json.load(open("${MASK_ROOT}/selection.json"))
print("SELECTED_MASK_DIR=" + p["mask_dir"])
print("CONTACT_START=" + str(p["window_start"]))
print("CONTACT_END=" + str(p["window_end"]))
print("CONTACT_KEYFRAME=" + str(p["keyframe"]))
PY
  )
  MASK_SELECTED="${SELECTED_MASK_DIR}"
  echo "[contact-precompute] selected mask=${MASK_SELECTED} window=${CONTACT_START}..${CONTACT_END} keyframe=${CONTACT_KEYFRAME}"
else
  frame_count="$("${TRACKING_PY}" - "${SESSION_DIR}/annotation.hdf5" <<'PY'
import h5py, sys
with h5py.File(sys.argv[1], "r") as archive:
    print(int(archive["cam-pose/timestamps"].shape[0]))
PY
)"
  if [[ "${CONTACT_NO_EVIDENCE_POLICY}" == "fail" && \
        "${frame_count}" -ge "${MIN_WINDOW_FRAMES}" ]]; then
    echo "[contact-precompute] no stable contact mask for ${frame_count}-frame session" >&2
    exit 2
  fi
  echo "[contact-precompute] no stable contact mask; writing explicit identity candidate (frames=${frame_count})"
fi
fi

# Stage 2c: observed RGB-D point-cloud contact evidence.
pointcloud_rc=1
run_pointcloud=0
if [[ "${OBJECT_SOURCE}" == "fingertip_neighborhood" ]]; then
  run_pointcloud=1
elif [[ "${selection_rc}" -eq 0 ]]; then
  run_pointcloud=1
fi
if [[ "${run_pointcloud}" -eq 1 ]]; then
  rm -rf "${POINTCLOUD_CONTACT_DIR}"
  mkdir -p "${POINTCLOUD_CONTACT_DIR}"
  pointcloud_args=(
    --session-dir "${SESSION_DIR}"
    --human-mask-dir "${HUMAN_MASK_ROOT}"
    --output-dir "${POINTCLOUD_CONTACT_DIR}"
    --object-source "${OBJECT_SOURCE}"
    --neighborhood-radius-m "${CONTACT_NEIGHBORHOOD_M}"
    --neighborhood-mask-margin-px "${CONTACT_NEIGHBORHOOD_MASK_MARGIN_PX}"
    --neighborhood-hand-clearance-m "${CONTACT_NEIGHBORHOOD_HAND_CLEARANCE_M}"
    --neighborhood-dilation-px "${CONTACT_NEIGHBORHOOD_DILATION_PX}"
    --neighborhood-z-band-m "${CONTACT_NEIGHBORHOOD_Z_BAND_M}"
    --object-cloud-max-points "${CONTACT_OBJECT_CLOUD_MAX_POINTS}"
    --object-cloud-radius-m "${CONTACT_OBJECT_CLOUD_RADIUS_M}"
    --object-cloud-voxel-m "${CONTACT_OBJECT_CLOUD_VOXEL_M}"
    --distance-threshold-m "${CONTACT_DISTANCE_THRESHOLD_M}"
    --depth-gap-threshold-m "${CONTACT_DEPTH_GAP_THRESHOLD_M}"
    --normal-dot-threshold "${CONTACT_NORMAL_DOT_THRESHOLD}"
    --temporal-gap "${CONTACT_TEMPORAL_GAP}"
    --min-run "${CONTACT_MIN_RUN}"
    --min-fingers "${CONTACT_MIN_FINGERS}"
    --max-object-gap "${CONTACT_OBJECT_MASK_MAX_GAP}"
    --device "${CONTACT_DEVICE}"
  )
  if [[ "${HAND_MODE}" == "g1_brainco" ]]; then
    pointcloud_args+=(--sides left,right)
  else
    pointcloud_args+=(--side "${SIDE}")
  fi
  if [[ "${OBJECT_SOURCE}" == "mask" ]]; then
    pointcloud_args+=(--object-mask-root "${MASK_ROOT}" --object-mask-dir "${MASK_SELECTED}")
  fi
  if [[ "${CONTACT_DEPTH_FOOTPRINT}" == "0" ]]; then
    pointcloud_args+=(--no-depth-footprint)
  else
    pointcloud_args+=(--depth-footprint)
  fi
  if [[ -n "${MAX_NATIVE_FRAMES}" ]]; then
    pointcloud_args+=(--max-frames "${MAX_NATIVE_FRAMES}")
  fi
  set +e
  run_contact_py "${SCRIPT_DIR}/contact_pointcloud_analysis.py" \
    "${pointcloud_args[@]}"
  pointcloud_rc=$?
  set -e
  if [[ "${pointcloud_rc}" -eq 0 && -s "${POINTCLOUD_INTERVAL_JSON}" ]]; then
    read -r pointcloud_start pointcloud_end < <(
      "${TRACKING_PY}" - "${POINTCLOUD_INTERVAL_JSON}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
frames = list(p.get("all_frames") or [])
for side_payload in (p.get("per_side") or {}).values():
    frames.extend(side_payload.get("all_frames") or [])
if p.get("first_contact") is not None and p.get("last_contact") is not None and not frames:
    frames = [p["first_contact"], p["last_contact"]]
if not frames:
    raise SystemExit(1)
print(int(min(frames)), int(max(frames)))
PY
    ) && {
      CONTACT_START="${pointcloud_start}"
      CONTACT_END="${pointcloud_end}"
      contact_interval="${CONTACT_START}:${CONTACT_END}"
      contact_status="contact_pointcloud_2finger"
      echo "[contact-precompute] pointcloud contact interval=${contact_interval}"
    } || echo "[contact-precompute] pointcloud contact detector found no valid interval"
  else
    echo "[contact-precompute] pointcloud contact detector found no valid interval"
  fi
fi

JOINT_CONTACT_STATUS="disabled"
if [[ "${HAND_MODE}" == "g1_brainco" ]]; then
  if [[ "${pointcloud_rc}" -eq 0 && -s "${POINTCLOUD_INTERVAL_JSON}" ]]; then
    cp --reflink=auto "${POINTCLOUD_INTERVAL_JSON}" "${INTERVAL_JSON}"
  fi
  mkdir -p "${OUTPUT_DIR}" "${JOINT_CONTACT_DIR}"
  if [[ "${contact_status}" != "contact_pointcloud_2finger" && \
        "${CONTACT_NO_EVIDENCE_POLICY}" == "fail" ]]; then
    echo "[contact-precompute] no point-cloud contact interval detected" >&2
    exit 2
  fi

  enabled_sides=()
  ANCHORS_NPZ="${POINTCLOUD_CONTACT_DIR}/contact_anchors.npz"
  if [[ "${contact_status}" == "contact_pointcloud_2finger" && \
        "${JOINT_CONTACT_ENABLED}" == "1" && -s "${ANCHORS_NPZ}" ]]; then
    require_file "${BRAINCO_MJCF}"
    require_file "${BRAINCO_NATIVE_REF}"
    require_file "${BRAINCO_NATIVE_TARGETS}"
    require_file "${BASE_HAND_50HZ}"
    echo "[contact-precompute] BrainCo neighborhood Laplacian wrist+6"
    run_contact_py "${SCRIPT_DIR}/optimize_brainco_contact_anchors.py" \
      --session-dir "${SESSION_DIR}" \
      --human-mask-dir "${HUMAN_MASK_ROOT}" \
      --contact-anchors "${ANCHORS_NPZ}" \
      --input-reference "${BRAINCO_NATIVE_REF}" \
      --input-targets "${BRAINCO_NATIVE_TARGETS}" \
      --input-hand "${BASE_HAND_50HZ}" \
      --mjcf "${BRAINCO_MJCF}" \
      --output-reference "${OUTPUT_DIR}/brainco_laplacian_reference.npz" \
      --output-targets "${OUTPUT_DIR}/brainco_laplacian_targets.npz" \
      --output-hand "${JOINT_CONTACT_DIR}/hand_brainco_50hz_joint_contact.npz" \
      --meta-json "${OUTPUT_DIR}/brainco_laplacian_meta.json" \
      --output-wrist-candidate-left "${CONTACT_CANDIDATE_LEFT}" \
      --output-wrist-candidate-right "${CONTACT_CANDIDATE_RIGHT}" \
      --mesh-samples-per-geom "${MESH_SAMPLES_PER_GEOM}" \
      --laplacian-object-cloud-max-points "${CONTACT_LAPLACIAN_OBJECT_CLOUD_MAX_POINTS}" \
      --seed "${CONTACT_SEED}"
    require_file "${JOINT_CONTACT_DIR}/hand_brainco_50hz_joint_contact.npz"
    cp --reflink=auto \
      "${JOINT_CONTACT_DIR}/hand_brainco_50hz_joint_contact.npz" \
      "${JOINT_HAND_OVERRIDE}"
  fi
  for side in left right; do
    side_out="${OUTPUT_DIR}/contact_aware_wrist_candidates_native_${side}.npz"
    side_start=""
    side_end=""
    if [[ "${contact_status}" == "contact_pointcloud_2finger" && \
          -s "${POINTCLOUD_INTERVAL_JSON}" ]]; then
      read -r side_start side_end < <(
        "${TRACKING_PY}" - "${POINTCLOUD_INTERVAL_JSON}" "${side}" <<'PY' || true
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
side = sys.argv[2]
side_payload = (payload.get("per_side") or {}).get(side) or {}
frames = list(side_payload.get("all_frames") or [])
if side_payload.get("first_contact") is not None and side_payload.get("last_contact") is not None:
    print(int(side_payload["first_contact"]), int(side_payload["last_contact"]))
elif frames:
    print(int(min(frames)), int(max(frames)))
else:
    raise SystemExit(1)
PY
      ) || true
    fi
    if [[ -n "${side_start}" && -s "${side_out}" && \
          "${JOINT_CONTACT_ENABLED}" == "1" ]]; then
      echo "[contact-precompute] BrainCo ${side} Laplacian schedule ${side_start}:${side_end}"
      "${TRACKING_PY}" "${SCRIPT_DIR}/expand_contact_wrist_schedule.py" \
        --candidates "${side_out}" \
        --output "${OUTPUT_DIR}/wrist_6d_offset_schedule_${side}.npz" \
        --contact-json "${OUTPUT_DIR}/contact_frames_${side}.json" \
        --side "${side}"
      enabled_sides+=("${side}")
    else
      echo "[contact-precompute] BrainCo ${side} identity / skipped_no_contact"
      "${TRACKING_PY}" "${SCRIPT_DIR}/make_identity_contact_candidate.py" \
        --experiment-dir "${EXPERIMENT_DIR}" --output "${side_out}" \
        --reason "${contact_status}" --side "${side}" \
        --wrist-method g1_brainco_palm_refine
    fi
    require_file "${side_out}"
  done
  if [[ "${#enabled_sides[@]}" -gt 0 ]]; then
    JOINT_CONTACT_STATUS="enabled"
  else
    if [[ -e "${JOINT_HAND_OVERRIDE}" ]]; then
      rm -f -- "${JOINT_HAND_OVERRIDE}"
    fi
    if [[ "${JOINT_CONTACT_ENABLED}" == "1" ]]; then
      JOINT_CONTACT_STATUS="skipped_no_contact"
    fi
  fi
  "${TRACKING_PY}" - "${OUTPUT_DIR}" "${contact_status}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
status = sys.argv[2]
left_frames = []
right_frames = []
for side, dest in (("left", "left_frames"), ("right", "right_frames")):
    path = root / f"contact_frames_{side}.json"
    if not path.is_file():
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = list(payload.get("contact_frames") or [])
    if side == "left":
        left_frames = frames
    else:
        right_frames = frames
union = sorted(set(left_frames) | set(right_frames))
out = {
    "schema_version": 1,
    "contact_frames": union,
    "left_contact_frames": left_frames,
    "right_contact_frames": right_frames,
    "first_contact": union[0] if union else None,
    "last_contact": union[-1] if union else None,
    "optimization_space": "per_hand_wrist_yaw_link_SE3",
    "not_joint_yaw_offset": True,
    "policy": {
        "before": "first_contact_offset",
        "contact": "per_frame_offset",
        "after": "last_contact_offset",
    },
    "status": "contact_pointcloud_2finger" if union else status,
}
(root / "contact_frames.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(json.dumps(out, indent=2))
PY
  if [[ -s "${OUTPUT_DIR}/contact_frames.json" ]]; then
    contact_status="$("${TRACKING_PY}" - "${OUTPUT_DIR}/contact_frames.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", "no_contact_detected"))
PY
)"
  fi
else
# Stage 3: fit the tapered cup and score hand-object interaction.  Empty or
# low-support windows are a normal no-contact outcome, not a baseline fallback.
if [[ "${selection_rc}" -eq 0 ]]; then
  rm -rf "${INTERACTION_DIR}"
  mkdir -p "${INTERACTION_DIR}"
  set +e
  run_contact_py "${SCRIPT_DIR}/evaluate_hand_object_interaction.py" \
    --session-dir "${SESSION_DIR}" \
    --experiment-dir "${EXPERIMENT_DIR}" \
    --mask-dir "${MASK_SELECTED}" \
    --out-dir "${INTERACTION_DIR}" \
    --start-frame "${CONTACT_START}" --end-frame "${CONTACT_END}" \
    --keyframe "${CONTACT_KEYFRAME}" --side "${SIDE}" \
    --wrist-methods template_palm_kabsch --finger-methods position \
    --dbscan-eps-m "${DBSCAN_EPS_M}" --dbscan-min-samples "${DBSCAN_MIN_SAMPLES}" \
    --allow-object-fit-fallback --max-object-fallback-run 1000 \
    --ransac-iterations "${RANSAC_ITERATIONS}" \
    --mesh-samples-per-geom "${MESH_SAMPLES_PER_GEOM}" --seed "${CONTACT_SEED}" \
    --device "${CONTACT_DEVICE}" \
    --skip-plots
  interaction_rc=$?
  set -e
  if [[ "${pointcloud_rc}" -eq 0 && -s "${POINTCLOUD_INTERVAL_JSON}" ]]; then
    # The point-cloud detector, rather than the old CupModel ranking, owns the
    # frame interval.  Preserve a compatibility copy at the historical path.
    cp --reflink=auto "${POINTCLOUD_INTERVAL_JSON}" "${INTERVAL_JSON}"
  fi
  if [[ "${contact_status}" != "contact_pointcloud_2finger" && \
        "${CONTACT_NO_EVIDENCE_POLICY}" == "fail" ]]; then
    echo "[contact-precompute] no point-cloud contact interval detected" >&2
    exit 2
  fi
fi

if [[ "${contact_status}" == "contact_pointcloud_2finger" && \
      -s "${INTERACTION_DIR}/object_models.jsonl" ]]; then
  # OUTPUT_DIR is normally AUTO_ROOT, which also owns object_interaction/.
  # Removing it here deleted the freshly fitted object_models.jsonl before
  # optimize_contact_aware_wrist.py could consume it.  The optimizer replaces
  # its own artifacts, so preserve the evidence directory.
  mkdir -p "${OUTPUT_DIR}"
  run_contact_py "${SCRIPT_DIR}/optimize_contact_aware_wrist.py" \
    --experiment-dir "${EXPERIMENT_DIR}" \
    --interaction-dir "${INTERACTION_DIR}" \
    --out-dir "${OUTPUT_DIR}" --side "${SIDE}" \
    --wrist-method template_palm_kabsch --finger-method position \
    --start-frame "${CONTACT_START}" --end-frame "${CONTACT_END}" \
    --keyframe "${CONTACT_KEYFRAME}" --seed "${CONTACT_SEED}"
else
  # Explicit identity candidate keeps the same contact-aware injection and
  # provenance contract when this episode has no defensible contact evidence.
  mkdir -p "${OUTPUT_DIR}"
  "${TRACKING_PY}" "${SCRIPT_DIR}/make_identity_contact_candidate.py" \
    --experiment-dir "${EXPERIMENT_DIR}" --output "${CONTACT_CANDIDATE}" \
    --reason "${contact_status}" --side "${SIDE}"
fi
require_file "${CONTACT_CANDIDATE}"

# The wrist-only solve above remains the deterministic initializer.  The
# formal contact-aware contract then refines that candidate jointly with the
# right Dex3 hand, so the wrist schedule and the hand override describe the
# same physical grasp.  Episodes without point-cloud contact evidence keep the
# wrist-only/identity path and do not publish a stale hand override.
JOINT_CONTACT_STATUS="disabled"
if [[ "${contact_status}" == "contact_pointcloud_2finger" && \
      -s "${INTERACTION_DIR}/object_models.jsonl" && \
      "${JOINT_CONTACT_ENABLED}" == "1" ]]; then
  mkdir -p "${JOINT_CONTACT_DIR}" "$(dirname "${WRIST_ONLY_CANDIDATE}")"
  cp --reflink=auto "${CONTACT_CANDIDATE}" "${WRIST_ONLY_CANDIDATE}"

  require_file "${BASE_HAND_50HZ}"

  echo "[contact-precompute] stage joint wrist SE(3) + right Dex3 7 DoF contact optimization"
  run_contact_py "${SCRIPT_DIR}/optimize_contact_aware_wrist_and_fingers.py" \
    --experiment-dir "${EXPERIMENT_DIR}" \
    --interaction-dir "${INTERACTION_DIR}" \
    --wrist-candidate "${CONTACT_CANDIDATE}" \
    --base-hand-50hz "${BASE_HAND_50HZ}" \
    --out-dir "${JOINT_CONTACT_DIR}" \
    --start-frame "${CONTACT_START}" --end-frame "${CONTACT_END}" \
    --keyframe "${CONTACT_KEYFRAME}" --side "${SIDE}" \
    --transition-frames "${JOINT_CONTACT_TRANSITION_FRAMES}" \
    --penetration-tolerance-m "${JOINT_CONTACT_PENETRATION_TOLERANCE_M}" \
    --penetration-ratio-tolerance "${JOINT_CONTACT_PENETRATION_RATIO_TOLERANCE}" \
    --robust-penetration-m "${JOINT_CONTACT_ROBUST_PENETRATION_M}" \
    --robust-penetration-ratio "${JOINT_CONTACT_ROBUST_PENETRATION_RATIO}" \
    --penetration-target-m "${JOINT_CONTACT_PENETRATION_TARGET_M}" \
    --penetration-ratio-target "${JOINT_CONTACT_PENETRATION_RATIO_TARGET}" \
    --penetration-depth-weight "${JOINT_CONTACT_PENETRATION_DEPTH_WEIGHT}" \
    --penetration-ratio-weight "${JOINT_CONTACT_PENETRATION_RATIO_WEIGHT}" \
    --fingertip-penetration-max-m "${JOINT_CONTACT_FINGERTIP_PENETRATION_MAX_M}" \
    --fingertip-penetration-p95-m "${JOINT_CONTACT_FINGERTIP_PENETRATION_P95_M}" \
    --fingertip-penetration-ratio "${JOINT_CONTACT_FINGERTIP_PENETRATION_RATIO}" \
    --fingertip-penetration-weight "${JOINT_CONTACT_FINGERTIP_PENETRATION_WEIGHT}" \
    --seed "${CONTACT_SEED}"
  require_file "${JOINT_CONTACT_DIR}/contact_aware_wrist_candidates_joint_contact_native.npz"
  require_file "${JOINT_CONTACT_DIR}/hand_dex3_50hz_joint_contact.npz"
  cp --reflink=auto \
    "${JOINT_CONTACT_DIR}/contact_aware_wrist_candidates_joint_contact_native.npz" \
    "${CONTACT_CANDIDATE}"
  cp --reflink=auto \
    "${JOINT_CONTACT_DIR}/hand_dex3_50hz_joint_contact.npz" \
    "${JOINT_HAND_OVERRIDE}"
  JOINT_CONTACT_STATUS="enabled"
else
  # Do not let a previous joint experiment leak into a no-contact run.
  if [[ -e "${JOINT_HAND_OVERRIDE}" ]]; then
    rm -f -- "${JOINT_HAND_OVERRIDE}"
  fi
  if [[ "${JOINT_CONTACT_ENABLED}" == "1" ]]; then
    JOINT_CONTACT_STATUS="skipped_no_contact"
  fi
fi

"${TRACKING_PY}" "${SCRIPT_DIR}/expand_contact_wrist_schedule.py" \
  --candidates "${CONTACT_CANDIDATE}" \
  --output "${OUTPUT_DIR}/wrist_6d_offset_schedule.npz" \
  --contact-json "${OUTPUT_DIR}/contact_frames.json"
fi

"${TRACKING_PY}" - <<PY
import json
from pathlib import Path
selection = {}
pointcloud = {}
try:
  selection = json.loads(Path("${MASK_ROOT}/selection.json").read_text())
except Exception:
  pass
try:
  pointcloud = json.loads(Path("${POINTCLOUD_CONTACT_DIR}/summary.json").read_text())
except Exception:
  pass
is_brainco = "${HAND_MODE}" == "g1_brainco"
candidate = Path("${CONTACT_CANDIDATE}")
candidate_left = Path("${CONTACT_CANDIDATE_LEFT}")
candidate_right = Path("${CONTACT_CANDIDATE_RIGHT}")
schedule = Path("${OUTPUT_DIR}/wrist_6d_offset_schedule.npz")
override = Path("${JOINT_HAND_OVERRIDE}")
base_hand = Path("${BASE_HAND_50HZ}")
joint_enabled = "${JOINT_CONTACT_STATUS}" == "enabled"
brainco_candidate = candidate_right if candidate_right.is_file() else candidate_left
p={
  "schema_version": 2,
  "session_id": "${SESSION_ID}",
  "hand_mode": "${HAND_MODE}",
  "hand_stage": (
      "g1_brainco_palm_refine+neighborhood_laplacian" if is_brainco else "dex3_joint_contact_optimization"
  ),
  "object_source": "${OBJECT_SOURCE}",
  "contact_sides": "left+right" if is_brainco else "${SIDE}",
  "status": "${contact_status}",
  "contact_interval": "${contact_interval}",
  "contact_detector": "source_rgbd_pointcloud",
  "contact_compute": {
    "requested_device": "${CONTACT_DEVICE}",
    "gpu": "${GPU}",
    "mask_window_device": selection.get("device", "unknown"),
    "mask_window_backend": selection.get("backend", "unknown"),
    "pointcloud_device": pointcloud.get("device", "unknown"),
    "pointcloud_projection_backend": pointcloud.get("projection_backend", "unknown"),
    "pointcloud_nearest_backend": pointcloud.get("nearest_backend", "unknown"),
  },
  "pointcloud_contact_dir": str(Path("${POINTCLOUD_CONTACT_DIR}").resolve()),
  "pointcloud_contact_interval": str(Path("${POINTCLOUD_INTERVAL_JSON}").resolve()),
  "contact_thresholds": {
    "distance_m": float("${CONTACT_DISTANCE_THRESHOLD_M}"),
    "depth_gap_m": float("${CONTACT_DEPTH_GAP_THRESHOLD_M}"),
    "normal_abs_dot": float("${CONTACT_NORMAL_DOT_THRESHOLD}"),
    "temporal_gap_frames": int("${CONTACT_TEMPORAL_GAP}"),
    "min_run_frames": int("${CONTACT_MIN_RUN}"),
    "primary_fingers": (
        ["thumb", "index", "middle", "ring", "pinky"] if is_brainco else ["thumb", "index", "middle"]
    ),
    "min_fingers": int("${CONTACT_MIN_FINGERS}"),
    "neighborhood_radius_m": float("${CONTACT_NEIGHBORHOOD_M}"),
    "neighborhood_mask_margin_px": float("${CONTACT_NEIGHBORHOOD_MASK_MARGIN_PX}"),
    "neighborhood_hand_clearance_m": float("${CONTACT_NEIGHBORHOOD_HAND_CLEARANCE_M}"),
    "neighborhood_dilation_px": float("${CONTACT_NEIGHBORHOOD_DILATION_PX}"),
    "neighborhood_z_band_m": float("${CONTACT_NEIGHBORHOOD_Z_BAND_M}"),
    "object_cloud_radius_m": float("${CONTACT_OBJECT_CLOUD_RADIUS_M}" or 0),
    "object_cloud_voxel_m": float("${CONTACT_OBJECT_CLOUD_VOXEL_M}" or 0),
    "depth_footprint": int("${CONTACT_DEPTH_FOOTPRINT}") == 1,
    "object_cloud_max_points": int("${CONTACT_OBJECT_CLOUD_MAX_POINTS}"),
    "laplacian_object_cloud_max_points": int("${CONTACT_LAPLACIAN_OBJECT_CLOUD_MAX_POINTS}" or 0),
  },
  "human_mask_root": str(Path("${HUMAN_MASK_ROOT}").resolve()),
  "mask_selection": str(Path("${MASK_ROOT}/selection.json").resolve()),
  "interaction_dir": str(Path("${INTERACTION_DIR}").resolve()),
  "candidate": str((brainco_candidate if is_brainco else candidate).resolve()) if (brainco_candidate if is_brainco else candidate).is_file() else None,
  "candidate_left": str(candidate_left.resolve()) if is_brainco and candidate_left.is_file() else None,
  "candidate_right": str(candidate_right.resolve()) if is_brainco and candidate_right.is_file() else None,
  "schedule": None if not schedule.is_file() else str(schedule.resolve()),
  "optimization_space": (
      "per_hand_wrist_yaw_link_SE3" if is_brainco else "right_wrist_yaw_link_SE3"
  ),
  "not_joint_yaw_offset": True,
  "contact_policy": "first_contact / per_contact / last_contact",
  "identity_on_no_contact": True,
  "reference_bootstrap": {
    "kind": "raw_annotation_template_palm",
    "archive": None,
    "archive_sha256": "${REFERENCE_BOOTSTRAP_SHA256}" if "${REFERENCE_BOOTSTRAP_SHA256}" else None,
    "tracking_bootstrap_used": False,
  },
  "joint_contact_optimization": {
    "enabled": "${JOINT_CONTACT_ENABLED}" == "1",
    "status": "${JOINT_CONTACT_STATUS}",
    "contract": (
        "dual_wrist_yaw_link_SE3_plus_g1_brainco_6DoF_per_hand"
        if is_brainco else "right_wrist_yaw_link_SE3_plus_right_Dex3_7DoF"
    ),
    "wrist_candidate": (
        str((brainco_candidate if is_brainco else candidate).resolve())
        if joint_enabled and (brainco_candidate if is_brainco else candidate).is_file()
        else None
    ),
    "wrist_candidate_left": str(candidate_left.resolve()) if is_brainco and candidate_left.is_file() else None,
    "wrist_candidate_right": str(candidate_right.resolve()) if is_brainco and candidate_right.is_file() else None,
    "hand_override": (
        str(override.resolve()) if joint_enabled and override.is_file() else None
    ),
    "base_hand_50hz": str(base_hand.resolve()) if base_hand.is_file() else None,
    "transition_frames": int("${JOINT_CONTACT_TRANSITION_FRAMES}"),
    "object_source": "${OBJECT_SOURCE}",
    "contact_sides": "left+right" if is_brainco else "${SIDE}",
  },
}
Path("${MANIFEST}").write_text(json.dumps(p, indent=2)+"\\n")
print(json.dumps(p, indent=2))
PY
if [[ "${HAND_MODE}" == "g1_brainco" ]]; then
  if [[ -s "${CONTACT_CANDIDATE_RIGHT}" ]]; then
    printf '%s\n' "${CONTACT_CANDIDATE_RIGHT}"
  else
    printf '%s\n' "${CONTACT_CANDIDATE_LEFT}"
  fi
else
  printf '%s\n' "${CONTACT_CANDIDATE}"
fi
