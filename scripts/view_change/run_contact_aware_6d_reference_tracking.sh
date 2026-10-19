#!/usr/bin/env bash
set -Eeuo pipefail

# Unified offline contract-aware view-change entry point.
# The wrist correction is SE(3) on right_wrist_yaw_link, never a joint yaw.
# Optional q0 initialization example:
#   Q0_SHOULDER_CLEARANCE=1 Q0_SHOULDER_CLEARANCE_COST=5 bash "$0"

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SESSION_ID="${SESSION_ID:-session_20260804_162223_iphone14_3}"
INPUT_ROOT="${INPUT_ROOT:-${ROOT}/data/egocentric_data/pick_up_the_cup_smooth}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${ROOT}/data/experiments/hand_retarget/${SESSION_ID}}"
CONTACT_OUT_DIR="${CONTACT_OUT_DIR:-${EXPERIMENT_DIR}/contact_aware_auto}"
CONTACT_CANDIDATES="${CONTACT_CANDIDATES:-${CONTACT_OUT_DIR}/contact_aware_wrist_candidates_native.npz}"
CONTACT_BUNDLE_ROOT="${CONTACT_BUNDLE_ROOT:-$(dirname "${CONTACT_CANDIDATES}")}"
INTERACTION_DIR="${INTERACTION_DIR:-${CONTACT_OUT_DIR}/object_interaction}"
JOINT_CONTACT_ENABLED="${JOINT_CONTACT_ENABLED:-1}"
# dex3 keeps the contact-aware Dex3 CupModel path. g1_brainco uses palm refine,
# and when JOINT_CONTACT_ENABLED=1 a neighborhood Laplacian wrist+finger refine.
HAND_MODE="${HAND_MODE:-dex3}"
# Independent of whether a contact stage runs, the formal pipeline always
# publishes the paired wrist/hand tracking contract, so the two conditions need
# separate flags.
REQUIRE_PAIRED_HAND_CONTRACT=1
case "${HAND_MODE}" in
  dex3) HAND_DOF=14 ;;
  g1_brainco)
    HAND_DOF=12
    [[ "${JOINT_CONTACT_ENABLED}" == "0" || "${JOINT_CONTACT_ENABLED}" == "1" ]] || \
      { echo "[contact-aware-6d] ERROR: JOINT_CONTACT_ENABLED must be 0 or 1" >&2; exit 2; }
    ;;
  *) echo "[contact-aware-6d] ERROR: HAND_MODE must be dex3 or g1_brainco" >&2; exit 2 ;;
esac
TRACKING_ROOT="${TRACKING_ROOT:-${ROOT}/data/kinematics/contact_aware_6d_reference_tracking}"
TRACKING_CACHE_ROOT="${TRACKING_CACHE_ROOT:-}"
VIEW_ROOT="${VIEW_ROOT:-${ROOT}/data/view_change/contact_aware_6d_reference_tracking}"
WORK_ROOT="${WORK_ROOT:-${KODA_CACHE:-${ROOT}/.cache}/contact_aware_6d_reference_tracking}"

GPU="${GPU:-1}"
CONTACT_DEVICE="${CONTACT_DEVICE:-cuda}"
MPC_BACKEND="${MPC_BACKEND:-jax}"
[[ "${MPC_BACKEND}" == "jax" ]] || {
  echo "[contact-aware-6d] ERROR: MPC_BACKEND must be jax" >&2
  exit 2
}
FORCE="${FORCE:-0}"
# View-change can be forced independently so RERUN_VIEW_CHANGE can rebuild
# RGB from a tracking cache hit without clearing that cache.
VIEW_FORCE="${VIEW_FORCE:-${FORCE}}"
SKIP_TRACKING="${SKIP_TRACKING:-0}"
AUTO_OPTIMIZE="${AUTO_OPTIMIZE:-1}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
KEEP_INTERMEDIATE="${KEEP_INTERMEDIATE:-0}"
# Forward stage-level cleanup to the single-frame renderer.  The wrapper still
# publishes only the final contract when KEEP_INTERMEDIATE=0.
PRUNE_INTERMEDIATES="${PRUNE_INTERMEDIATES:-0}"
CONTACT_START="${CONTACT_START:-300}"
CONTACT_END="${CONTACT_END:-360}"
CONTACT_KEYFRAME="${CONTACT_KEYFRAME:-331}"
CONTACT_SEED="${CONTACT_SEED:-20260807}"
TRACKING_PY="${TRACKING_PY:-${KODA_PYTHON:-${PYTHON:-python3}}}"
CONTACT_PY="${CONTACT_PY:-${KODA_PYTHON:-${PYTHON:-python3}}}"
VIEW_SCRIPT="${VIEW_SCRIPT:-${ROOT}/scripts/view_change/run_singleframe_view_change_30to50.sh}"
PRECOMPUTE_SCRIPT="${PRECOMPUTE_SCRIPT:-${ROOT}/scripts/view_change/run_contact_precompute.sh}"
PREPARE_REFERENCE_SCRIPT="${PREPARE_REFERENCE_SCRIPT:-${ROOT}/scripts/view_change/prepare_contact_template_palm_reference.sh}"
TRACKING_SCRIPT="${TRACKING_SCRIPT:-${ROOT}/scripts/run_three_point_batch.py}"
SCHEDULE_SCRIPT="${SCHEDULE_SCRIPT:-${ROOT}/scripts/view_change/expand_contact_wrist_schedule.py}"
CONTACT_EVAL_SCRIPT="${CONTACT_EVAL_SCRIPT:-${ROOT}/scripts/view_change/evaluate_post_ik_hand_object_interaction.py}"
CAMERA_KINEMATICS_SCRIPT="${CAMERA_KINEMATICS_SCRIPT:-${ROOT}/scripts/view_change/extract_camera_kinematics_30hz.py}"
CONTACT_MASK_DIR="${CONTACT_MASK_DIR:-${EXPERIMENT_DIR}/cup_segmentation_auto/selected/masks}"

SESSION_DIR="${INPUT_ROOT}/${SESSION_ID}"
TRACKING_SESSION_DIR="${TRACKING_ROOT}/${SESSION_ID}"
VIEW_SESSION_DIR="${VIEW_ROOT}/${SESSION_ID}"
RUN_DIR="${VIEW_SESSION_DIR}"
mkdir -p "${VIEW_SESSION_DIR}" "${WORK_ROOT}" "${TRACKING_ROOT}" "${VIEW_ROOT}"
LOG="${VIEW_SESSION_DIR}/run.log"
exec > >(tee -a "${LOG}") 2>&1

die() { echo "[contact-aware-6d] ERROR: $*" >&2; exit 2; }
require_file() { [[ -f "$1" ]] || die "missing file: $1"; }
require_dir() { [[ -d "$1" ]] || die "missing directory: $1"; }

prune_work_path() {
  [[ "${PRUNE_INTERMEDIATES}" == "1" ]] || return 0
  local path="$1" work_abs path_abs
  work_abs="$(realpath -m -- "${WORK_ROOT}")"
  path_abs="$(realpath -m -- "${path}")"
  [[ "${path_abs}" == "${work_abs}"/* ]] || die "refusing to prune outside WORK_ROOT: ${path_abs}"
  if [[ -e "${path_abs}" || -L "${path_abs}" ]]; then
    rm -rf -- "${path_abs}"
    echo "[contact-aware-6d] pruned work cache ${path_abs}"
  fi
}

require_dir "${SESSION_DIR}"
require_file "${SESSION_DIR}/rgb.mp4"
require_file "${SESSION_DIR}/annotation.hdf5"
require_file "${SESSION_DIR}/calibrations/rgb_K.npy"
require_file "${TRACKING_SCRIPT}"
require_file "${PREPARE_REFERENCE_SCRIPT}"
require_file "${VIEW_SCRIPT}"
require_file "${PRECOMPUTE_SCRIPT}"
require_file "${SCHEDULE_SCRIPT}"
require_file "${CAMERA_KINEMATICS_SCRIPT}"
require_file "${TRACKING_PY}"

if [[ "${PRUNE_INTERMEDIATES}" != "0" && "${PRUNE_INTERMEDIATES}" != "1" ]]; then
  die "PRUNE_INTERMEDIATES must be 0 or 1; got ${PRUNE_INTERMEDIATES}"
fi
if [[ "${HAND_MODE}" != "g1_brainco" ]]; then
  [[ "${JOINT_CONTACT_ENABLED}" == "1" ]] || \
    die "the formal pipeline requires JOINT_CONTACT_ENABLED=1"
fi

contact_manifest_matches_device() {
  "${TRACKING_PY}" - "${CONTACT_BUNDLE_ROOT}/contact_precompute_manifest.json" \
    "${CONTACT_DEVICE}" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    manifest = json.load(stream)
raise SystemExit(
    int(manifest.get("contact_compute", {}).get("requested_device") != sys.argv[2])
)
PY
}

NEED_CONTACT_PRECOMPUTE=0
if [[ "${HAND_MODE}" == "g1_brainco" ]]; then
  if [[ "${JOINT_CONTACT_ENABLED}" == "1" ]]; then
    CONTACT_BUNDLE_ROOT="${CONTACT_OUT_DIR}"
    if [[ ! -s "${CONTACT_OUT_DIR}/contact_precompute_manifest.json" ||
          ! -s "${CONTACT_OUT_DIR}/contact_frames.json" ||
          ! -s "${CONTACT_OUT_DIR}/joint_contact/base_hand_brainco_50hz.npz" ||
          ( ! -s "${CONTACT_OUT_DIR}/contact_aware_wrist_candidates_native_left.npz" &&
            ! -s "${CONTACT_OUT_DIR}/contact_aware_wrist_candidates_native_right.npz" ) ]] ||
       ! contact_manifest_matches_device; then
      NEED_CONTACT_PRECOMPUTE=1
    fi
  fi
elif [[ ! -f "${CONTACT_CANDIDATES}" ]]; then
  NEED_CONTACT_PRECOMPUTE=1
elif [[ "${JOINT_CONTACT_ENABLED}" == "1" &&
        ( ! -s "${CONTACT_BUNDLE_ROOT}/joint_contact/base_hand_dex3_50hz.npz" ||
          ! -s "${CONTACT_BUNDLE_ROOT}/contact_precompute_manifest.json" ||
          ! -s "${CONTACT_BUNDLE_ROOT}/contact_frames.json" ) ]] ||
     ! contact_manifest_matches_device; then
  # The latest tracking contract always publishes a synchronized 14-DoF hand,
  # including the base position-retargeted hand for no-contact episodes.
  NEED_CONTACT_PRECOMPUTE=1
fi
if [[ "${NEED_CONTACT_PRECOMPUTE}" == "1" ]]; then
  [[ "${AUTO_OPTIMIZE}" == "1" ]] || die "contact candidate missing: ${CONTACT_CANDIDATES}; set CONTACT_CANDIDATES or AUTO_OPTIMIZE=1"
  require_dir "${EXPERIMENT_DIR}"
  CONTACT_REFERENCE_ROOT="${WORK_ROOT}/contact_reference"
  CONTACT_REFERENCE_KINEMATICS="${CONTACT_REFERENCE_ROOT}/template_palm_reference_kinematics.npz"
  CONTACT_REFERENCE_TARGETS="${CONTACT_REFERENCE_ROOT}/template_palm_kabsch_targets.npz"
  CONTACT_REFERENCE_SHA256=""
  echo "[contact-aware-6d] preparing raw template-palm reference/timeline (no tracking bootstrap)"
  env SESSION_DIR="${SESSION_DIR}" \
    REFERENCE_OUTPUT_DIR="${CONTACT_REFERENCE_ROOT}" \
    bash "${PREPARE_REFERENCE_SCRIPT}"
  CONTACT_REFERENCE_SHA256="$(sha256sum "${CONTACT_REFERENCE_KINEMATICS}" | awk '{print $1}')"
  echo "[contact-aware-6d] contact candidate missing; running point-cloud contact precompute"
  env ROOT="${ROOT}" SESSION_ID="${SESSION_ID}" \
    INPUT_ROOT="${INPUT_ROOT}" KINEMATICS_ROOT="${CONTACT_REFERENCE_ROOT}" \
    KINEMATICS_NPZ="${CONTACT_REFERENCE_KINEMATICS}" \
    TARGET_TIMESTAMPS_NPZ="${CONTACT_REFERENCE_TARGETS}" \
    REFERENCE_BOOTSTRAP_NPZ="${CONTACT_REFERENCE_KINEMATICS}" \
    REFERENCE_BOOTSTRAP_SHA256="${CONTACT_REFERENCE_SHA256}" \
    TEMPLATE_PALM_FROM_KINEMATICS="1" \
    EXPERIMENT_ROOT="$(dirname "${EXPERIMENT_DIR}")" EXPERIMENT_DIR="${EXPERIMENT_DIR}" \
    OUTPUT_DIR="${CONTACT_OUT_DIR}" GPU="${GPU}" FORCE="1" \
    CONTACT_NO_EVIDENCE_POLICY="${CONTACT_NO_EVIDENCE_POLICY:-identity}" \
    CONTACT_SEED="${CONTACT_SEED}" \
    JOINT_CONTACT_ENABLED="${JOINT_CONTACT_ENABLED}" \
    HAND_MODE="${HAND_MODE}" \
    CONTACT_DEVICE="${CONTACT_DEVICE}" \
    MAX_NATIVE_FRAMES="${MAX_SOURCE_FRAMES:-}" \
    EFFICIENTSAM3_WORKER_SOCK="${EFFICIENTSAM3_WORKER_SOCK:-}" \
    bash "${PRECOMPUTE_SCRIPT}"
  CONTACT_BUNDLE_ROOT="${CONTACT_OUT_DIR}"
  if [[ "${HAND_MODE}" == "g1_brainco" ]]; then
    if [[ -s "${CONTACT_OUT_DIR}/contact_aware_wrist_candidates_native_right.npz" ]]; then
      CONTACT_CANDIDATES="${CONTACT_OUT_DIR}/contact_aware_wrist_candidates_native_right.npz"
    else
      CONTACT_CANDIDATES="${CONTACT_OUT_DIR}/contact_aware_wrist_candidates_native_left.npz"
    fi
  else
    CONTACT_CANDIDATES="${CONTACT_OUT_DIR}/contact_aware_wrist_candidates_native.npz"
  fi
fi
CONTACT_METADATA="${VIEW_SESSION_DIR}/contact_frames.json"
CONTACT_SCHEDULE_ROOT="${VIEW_ROOT}/contact_schedules"
CANDIDATE_SCHEDULE="${VIEW_SESSION_DIR}/candidate_wrist_6d_offset_schedule.npz"
SCHEDULE="${CONTACT_SCHEDULE_ROOT}/${SESSION_ID}/wrist_6d_offset_schedule.npz"
if [[ "${HAND_MODE}" == "g1_brainco" ]]; then
  MANIFEST_OPTIMIZATION_SPACE="dual_wrist_yaw_link_SE3"
  MANIFEST_BUNDLE="none"
  MANIFEST_CANDIDATES="none"
  MANIFEST_CONTACT_FRAMES="none"
  MANIFEST_SCHEDULE="none"
  MANIFEST_CONTACT_POLICY="none"
  MANIFEST_HAND_STAGE="g1_brainco_palm_refine"
  MANIFEST_OBJECT_SOURCE="none"
  if [[ "${JOINT_CONTACT_ENABLED}" == "1" ]]; then
    require_file "${CONTACT_OUT_DIR}/contact_precompute_manifest.json"
    require_file "${CONTACT_OUT_DIR}/contact_frames.json"
    require_file "${CONTACT_OUT_DIR}/joint_contact/base_hand_brainco_50hz.npz"
    CONTACT_CANDIDATE_LEFT="${CONTACT_OUT_DIR}/contact_aware_wrist_candidates_native_left.npz"
    CONTACT_CANDIDATE_RIGHT="${CONTACT_OUT_DIR}/contact_aware_wrist_candidates_native_right.npz"
    [[ -s "${CONTACT_CANDIDATE_LEFT}" || -s "${CONTACT_CANDIDATE_RIGHT}" ]] || \
      die "BrainCo joint-contact bundle is missing left/right wrist candidates"
    cp --reflink=auto "${CONTACT_OUT_DIR}/contact_frames.json" "${CONTACT_METADATA}"
    if [[ -s "${CONTACT_CANDIDATE_LEFT}" ]]; then
      "${TRACKING_PY}" "${SCHEDULE_SCRIPT}" \
        --candidates "${CONTACT_CANDIDATE_LEFT}" --side left \
        --output "${VIEW_SESSION_DIR}/candidate_wrist_6d_offset_schedule_left.npz" \
        --contact-json "${CONTACT_METADATA}"
    fi
    if [[ -s "${CONTACT_CANDIDATE_RIGHT}" ]]; then
      "${TRACKING_PY}" "${SCHEDULE_SCRIPT}" \
        --candidates "${CONTACT_CANDIDATE_RIGHT}" --side right \
        --output "${VIEW_SESSION_DIR}/candidate_wrist_6d_offset_schedule_right.npz" \
        --contact-json "${CONTACT_METADATA}"
      CONTACT_CANDIDATES="${CONTACT_CANDIDATE_RIGHT}"
    else
      CONTACT_CANDIDATES="${CONTACT_CANDIDATE_LEFT}"
    fi
    MANIFEST_OPTIMIZATION_SPACE="per_hand_wrist_yaw_link_SE3"
    MANIFEST_BUNDLE="${CONTACT_OUT_DIR}"
    MANIFEST_CANDIDATES="$(realpath -m "${CONTACT_CANDIDATES}")"
    MANIFEST_CONTACT_FRAMES="${CONTACT_METADATA}"
    MANIFEST_SCHEDULE="${CONTACT_SCHEDULE_ROOT}"
    MANIFEST_CONTACT_POLICY="first_contact / per_contact / last_contact"
    MANIFEST_HAND_STAGE="g1_brainco_palm_refine+neighborhood_laplacian"
    MANIFEST_OBJECT_SOURCE="fingertip_neighborhood"
  fi
else
  require_file "${CONTACT_CANDIDATES}"
  require_file "${CONTACT_BUNDLE_ROOT}/joint_contact/base_hand_dex3_50hz.npz"
  require_file "${CONTACT_BUNDLE_ROOT}/contact_precompute_manifest.json"
  require_file "${CONTACT_BUNDLE_ROOT}/contact_frames.json"
  "${TRACKING_PY}" "${SCHEDULE_SCRIPT}" \
    --candidates "${CONTACT_CANDIDATES}" --output "${CANDIDATE_SCHEDULE}" \
    --contact-json "${CONTACT_METADATA}"
  MANIFEST_OPTIMIZATION_SPACE="right_wrist_yaw_link_SE3"
  MANIFEST_BUNDLE="${CONTACT_BUNDLE_ROOT}"
  MANIFEST_CANDIDATES="$(realpath -m "${CONTACT_CANDIDATES}")"
  MANIFEST_CONTACT_FRAMES="${CONTACT_METADATA}"
  MANIFEST_SCHEDULE="${SCHEDULE}"
  MANIFEST_CONTACT_POLICY="first_contact / per_contact / last_contact"
  MANIFEST_HAND_STAGE="dex3_joint_contact_optimization"
  MANIFEST_OBJECT_SOURCE="mask"
fi

SESSION_LIST="${VIEW_SESSION_DIR}/session_list.txt"
printf '%s\n' "${SESSION_ID}" > "${SESSION_LIST}"

cat > "${VIEW_SESSION_DIR}/reference_manifest.json" <<EOF
{
  "schema_version": 1,
  "session_id": "${SESSION_ID}",
  "hand_mode": "${HAND_MODE}",
  "hand_dof": ${HAND_DOF},
  "optimization_space": "${MANIFEST_OPTIMIZATION_SPACE}",
  "joint_contact_optimization": ${JOINT_CONTACT_ENABLED},
  "hand_stage": "${MANIFEST_HAND_STAGE}",
  "object_source": "${MANIFEST_OBJECT_SOURCE:-none}",
  "contact_sides": "$([ "${HAND_MODE}" = "g1_brainco" ] && [ "${JOINT_CONTACT_ENABLED}" = "1" ] && echo "left+right" || echo "right")",
  "hand_stage": "${MANIFEST_HAND_STAGE}",
  "object_source": "${MANIFEST_OBJECT_SOURCE:-none}",
  "contact_sides": "$([ "${HAND_MODE}" = "g1_brainco" ] && [ "${JOINT_CONTACT_ENABLED}" = "1" ] && echo "left+right" || echo "right")",
  "hand_stage": "${MANIFEST_HAND_STAGE}",
  "joint_contact_bundle": "${MANIFEST_BUNDLE}",
  "not_joint_yaw_offset": true,
  "contact_policy": "${MANIFEST_CONTACT_POLICY}",
  "contact_candidates": "${MANIFEST_CANDIDATES}",
  "contact_frames": "${MANIFEST_CONTACT_FRAMES}",
  "wrist_6d_offset_schedule": "${MANIFEST_SCHEDULE}",
  "tracking_error": "${TRACKING_SESSION_DIR}/tracking_error.json",
  "contact_metrics": "${RUN_DIR}/contact_metrics.jsonl",
  "tracking_script": "$(realpath "${TRACKING_SCRIPT}")",
  "view_script": "$(realpath "${VIEW_SCRIPT}")"
}
EOF

echo "[contact-aware-6d] session=${SESSION_ID} gpu=${GPU} force=${FORCE} view_force=${VIEW_FORCE} smoke=${SMOKE} dry_run=${DRY_RUN} prune_intermediates=${PRUNE_INTERMEDIATES}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[contact-aware-6d] dry-run complete"
  exit 0
fi

TRACK_ARGS=(
  "--input-root" "${INPUT_ROOT}" "--output-root" "${TRACKING_ROOT}"
  "--work-root" "${WORK_ROOT}/tracking" "--session-list" "${SESSION_LIST}"
  "--gpus" "${GPU}" "--workers" "1"
  "--mpc-backend" "${MPC_BACKEND}"
  "--reference-savgol-window" "${REFERENCE_SAVGOL_WINDOW:-5}"
  "--reference-savgol-polyorder" "${REFERENCE_SAVGOL_POLYORDER:-2}"
  "--reference-savgol-gap-factor" "${REFERENCE_SAVGOL_GAP_FACTOR:-3.0}"
  "--reference-wrist-outlier-max-duration-s" "${REFERENCE_WRIST_OUTLIER_MAX_DURATION_S:-1.1}"
  "--reference-wrist-outlier-position-floor-m" "${REFERENCE_WRIST_OUTLIER_POSITION_FLOOR_M:-0.05}"
  "--reference-wrist-outlier-orientation-floor-deg" "${REFERENCE_WRIST_OUTLIER_ORIENTATION_FLOOR_DEG:-45.0}"
)
if [[ "${HAND_MODE}" == "g1_brainco" ]]; then
  TRACK_ARGS+=("--hand-mode" "g1_brainco")
  if [[ "${JOINT_CONTACT_ENABLED}" == "1" ]]; then
    TRACK_ARGS+=("--contact-schedule-root" "${CONTACT_SCHEDULE_ROOT}")
    TRACK_ARGS+=("--contact-edge-hold")
    TRACK_ARGS+=("--joint-contact-root" "${CONTACT_BUNDLE_ROOT}")
  fi
else
  TRACK_ARGS+=(
    "--contact-schedule-root" "${CONTACT_SCHEDULE_ROOT}"
    "--contact-edge-hold"
    "--joint-contact-root" "${CONTACT_BUNDLE_ROOT}"
  )
fi
if [[ "${FORCE}" == "1" ]]; then TRACK_ARGS+=("--force"); fi
if [[ "${SMOKE}" == "1" ]]; then TRACK_ARGS+=("--keep-work"); fi

# Optional frame-0 Mink initialization/clearance branch.  The contact-aware
# reference itself is unchanged; these flags only affect the first arm solve.
if [[ "${Q0_ZERO_ARM_NO_SMOOTHNESS:-1}" == "1" ]]; then
  TRACK_ARGS+=("--q0-zero-arm-no-smoothness")
fi
if [[ "${Q0_SHOULDER_CLEARANCE:-1}" == "1" ]]; then
  TRACK_ARGS+=("--q0-shoulder-clearance")
fi
if [[ -n "${Q0_MAX_ARM_DELTA:-}" ]]; then
  TRACK_ARGS+=("--q0-max-arm-delta" "${Q0_MAX_ARM_DELTA}")
fi
if [[ -n "${Q0_ARM_VELOCITY_LIMIT:-}" ]]; then
  TRACK_ARGS+=("--q0-arm-velocity-limit" "${Q0_ARM_VELOCITY_LIMIT}")
fi
if [[ -n "${Q0_POSTURE_COST:-}" ]]; then
  TRACK_ARGS+=("--q0-posture-cost" "${Q0_POSTURE_COST}")
fi
if [[ -n "${Q0_ORIENTATION_COST:-}" ]]; then
  TRACK_ARGS+=("--q0-orientation-cost" "${Q0_ORIENTATION_COST}")
fi
if [[ -n "${Q0_COLLISION_MARGIN:-}" ]]; then
  TRACK_ARGS+=("--q0-collision-margin" "${Q0_COLLISION_MARGIN}")
fi
if [[ -z "${Q0_SHOULDER_CLEARANCE_COST:-}" ]]; then
  Q0_SHOULDER_CLEARANCE_COST="5"
fi
if [[ -n "${Q0_SHOULDER_CLEARANCE_COST:-}" ]]; then
  TRACK_ARGS+=("--q0-shoulder-clearance-cost" "${Q0_SHOULDER_CLEARANCE_COST}")
fi
if [[ -z "${Q0_SHOULDER_CLEARANCE_THRESHOLD:-}" ]]; then
  Q0_SHOULDER_CLEARANCE_THRESHOLD="0.06"
fi
if [[ -n "${Q0_SHOULDER_CLEARANCE_THRESHOLD:-}" ]]; then
  TRACK_ARGS+=("--q0-shoulder-clearance-threshold" "${Q0_SHOULDER_CLEARANCE_THRESHOLD}")
fi
if [[ -n "${Q0_SHOULDER_CLEARANCE_SAMPLE_START:-}" ]]; then
  TRACK_ARGS+=("--q0-shoulder-clearance-sample-start" "${Q0_SHOULDER_CLEARANCE_SAMPLE_START}")
fi
if [[ -n "${Q0_SHOULDER_CLEARANCE_SAMPLE_COUNT:-}" ]]; then
  TRACK_ARGS+=("--q0-shoulder-clearance-sample-count" "${Q0_SHOULDER_CLEARANCE_SAMPLE_COUNT}")
fi
if [[ "${Q0_OPEN_BRANCH_FALLBACK:-0}" == "1" ]]; then
  TRACK_ARGS+=("--q0-open-branch-fallback")
fi
if [[ -z "${Q0_MIN_SHOULDER_CLEARANCE:-}" ]]; then
  Q0_MIN_SHOULDER_CLEARANCE="0.01"
fi
if [[ -n "${Q0_MIN_SHOULDER_CLEARANCE:-}" ]]; then
  TRACK_ARGS+=("--q0-min-shoulder-clearance" "${Q0_MIN_SHOULDER_CLEARANCE}")
fi
if [[ -n "${MAX_GAP_FRAMES:-}" ]]; then
  TRACK_ARGS+=("--max-gap-frames" "${MAX_GAP_FRAMES}")
fi

verify_tracking_contract() {
  local kinematics="$1"
  require_file "${kinematics}"
  "${TRACKING_PY}" - "${kinematics}" "${SESSION_ID}" \
    "${REQUIRE_PAIRED_HAND_CONTRACT}" "${HAND_DOF}" <<'PY'
import json
import sys
import numpy as np

path, session_id, require_paired_hand, hand_dof = sys.argv[1:5]
require_paired_hand = require_paired_hand == "1"
hand_dof = int(hand_dof)
with np.load(path, allow_pickle=False) as archive:
    required = {
        "joint_pos", "joint_vel", "joint_names", "timestamps",
        "reference_smoothing_config_json", "reference_smoothing_config_sha256",
    }
    if require_paired_hand:
        required |= {
            "hand_qpos", "hand_qvel", "hand_joint_names",
            "contact_30hz", "contact_50hz", "joint_contact_bundle_sha256",
            "joint_contact_contract",
        }
    missing = sorted(required - set(archive.files))
    if missing:
        raise SystemExit(f"{session_id}: tracking contract missing {missing}")
    body = np.asarray(archive["joint_pos"])
    body_vel = np.asarray(archive["joint_vel"])
    timestamps = np.asarray(archive["timestamps"])
    if body.ndim != 2 or body.shape[1] != 29 or body_vel.shape != body.shape:
        raise SystemExit(f"{session_id}: invalid 29-DoF body {body.shape}/{body_vel.shape}")
    if timestamps.shape != (body.shape[0],):
        raise SystemExit(f"{session_id}: invalid tracking timestamps {timestamps.shape}")
    smoothing = json.loads(str(np.asarray(archive["reference_smoothing_config_json"]).item()))
    if not bool(smoothing.get("enabled")) or smoothing.get("stage") != "post_contact_pre_tracking":
        raise SystemExit(f"{session_id}: post-contact reference smoothing is not enabled")
    if require_paired_hand:
        hand = np.asarray(archive["hand_qpos"])
        hand_vel = np.asarray(archive["hand_qvel"])
        if hand.shape != (body.shape[0], hand_dof) or hand_vel.shape != hand.shape:
            raise SystemExit(
                f"{session_id}: invalid {hand_dof}-DoF hand "
                f"{hand.shape}/{hand_vel.shape}"
            )
        if np.asarray(archive["contact_50hz"]).shape != (body.shape[0],):
            raise SystemExit(f"{session_id}: invalid 50 Hz contact timeline")
print(f"{session_id}: verified smoothed 29+{hand_dof} DoF tracking frames={body.shape[0]}")
PY
}

tracking_cache_ready() {
  local cache="${TRACKING_CACHE_ROOT}"
  [[ -n "${cache}" ]] || return 1
  [[ -s "${cache}/kinematics.npz" && -s "${cache}/meta.json" && -s "${cache}/tracking_error.json" ]] || return 1
  verify_tracking_contract "${cache}/kinematics.npz" || return 1
}

publish_tracking_cache() {
  local source="$1" dest="$2"
  [[ -n "${dest}" ]] || return 0
  mkdir -p "${dest}"
  cp --reflink=auto "${source}/kinematics.npz" "${dest}/kinematics.npz"
  cp --reflink=auto "${source}/meta.json" "${dest}/meta.json"
  cp --reflink=auto "${source}/tracking_error.json" "${dest}/tracking_error.json"
}

if [[ "${FORCE}" == "1" && -n "${TRACKING_CACHE_ROOT}" ]]; then
  rm -rf -- "${TRACKING_CACHE_ROOT}"
  echo "[contact-aware-6d] FORCE=1 cleared tracking cache ${TRACKING_CACHE_ROOT}"
fi

if [[ "${FORCE}" != "1" ]] && tracking_cache_ready; then
  echo "[contact-aware-6d] skip tracking cache_hit=${TRACKING_CACHE_ROOT}"
  mkdir -p "${TRACKING_SESSION_DIR}"
  publish_tracking_cache "${TRACKING_CACHE_ROOT}" "${TRACKING_SESSION_DIR}"
elif [[ "${SKIP_TRACKING}" == "1" ]]; then
  die "SKIP_TRACKING=1 requires a reusable tracking cache at ${TRACKING_CACHE_ROOT:-<unset>}"
else
  echo "[contact-aware-6d] stage tracking: Kimodo -> ScaleTrack/${MPC_BACKEND} MPC -> Mink"
  "${TRACKING_PY}" "${TRACKING_SCRIPT}" "${TRACK_ARGS[@]}"
  require_file "${TRACKING_SESSION_DIR}/kinematics.npz"
  require_file "${TRACKING_SESSION_DIR}/meta.json"
  require_file "${TRACKING_SESSION_DIR}/tracking_error.json"
  if [[ -n "${TRACKING_CACHE_ROOT}" ]]; then
    publish_tracking_cache "${TRACKING_SESSION_DIR}" "${TRACKING_CACHE_ROOT}"
    echo "[contact-aware-6d] wrote tracking cache ${TRACKING_CACHE_ROOT}"
  fi
fi

TRACKING_KINEMATICS="${TRACKING_SESSION_DIR}/kinematics.npz"
require_file "${TRACKING_KINEMATICS}"
require_file "${TRACKING_SESSION_DIR}/meta.json"
require_file "${TRACKING_SESSION_DIR}/tracking_error.json"
verify_tracking_contract "${TRACKING_KINEMATICS}"
# Tracking work files are not inputs to rendering or contact evaluation once
# the formal kinematics/error contract exists.  Remove them before the much
# larger view-change stages start, but keep the published tracking root.
prune_work_path "${WORK_ROOT}/tracking"
ACTUAL_SCHEDULE="${CONTACT_SCHEDULE_ROOT}/${SESSION_ID}/wrist_6d_offset_schedule.npz"
if [[ -f "${ACTUAL_SCHEDULE}" ]]; then
  SCHEDULE="${ACTUAL_SCHEDULE}"
else
  SCHEDULE="${CANDIDATE_SCHEDULE}"
fi

SOURCE_COUNT_ENV="${MAX_SOURCE_FRAMES:-}"
if [[ "${SMOKE}" == "1" ]]; then SOURCE_COUNT_ENV="${MAX_SOURCE_FRAMES:-30}"; fi
echo "[contact-aware-6d] stage novel view: warp -> MuJoCo ID/depth composite -> DiffuEraser"
env GPU="${GPU}" FORCE="${VIEW_FORCE}" SESSION="${SESSION_DIR}" \
  TARGET_KINEMATICS="${TRACKING_KINEMATICS}" TARGET_REFERENCE="${TRACKING_KINEMATICS}" \
  SOURCE_RGB_K="${SESSION_DIR}/calibrations/rgb_K.npy" OUTPUT_ROOT="${RUN_DIR}" \
  PRUNE_INTERMEDIATES="${PRUNE_INTERMEDIATES}" \
  STOP_AFTER="${STOP_AFTER:-}" \
  REUSE_BASE_ROOT="${REUSE_BASE_ROOT:-}" HAND_MODE="${HAND_MODE}" \
  REUSE_HUMAN_MASKS_DIR="${EXPERIMENT_DIR}/human_masks" \
  SOURCE_RGB_DIR="${EXPERIMENT_DIR}/source_rgb" \
  SOURCE_ADAPTER_DIR="${EXPERIMENT_DIR}/source_adapter" \
  LOG_ROOT="${LOG_ROOT:-}" \
  DIFFUERASER_WORKER_SOCK="${DIFFUERASER_WORKER_SOCK:-}" \
  EFFICIENTSAM3_WORKER_SOCK="${EFFICIENTSAM3_WORKER_SOCK:-}" \
  RETARGET_TYPE="position" LEFT_WRIST_YAW_OFFSET_DEG="0" RIGHT_WRIST_YAW_OFFSET_DEG="0" \
  MAX_SOURCE_FRAMES="${SOURCE_COUNT_ENV}" MAX_TARGET_FRAMES="${MAX_TARGET_FRAMES:-}" \
  bash "${VIEW_SCRIPT}"

if [[ "${STOP_AFTER:-}" == "composite" ]]; then
  require_file "${RUN_DIR}/processed_30hz/eraser/composite_trim.mp4"
  echo "[contact-aware-6d] STOP_AFTER=composite; wrote ${RUN_DIR}/processed_30hz/eraser/composite_trim.mp4"
  exit 0
fi

FINAL_VIDEO="${RUN_DIR}/final_50hz/rgb_final_50hz.mp4"
require_file "${FINAL_VIDEO}"

# target_robot/kinematics.npz is the 43-joint MuJoCo render archive.  Publish
# the original tracking archive instead so its 29-DoF body + separate 14-DoF
# hand, contact flags, smoothing configuration, and provenance remain intact.
# The render archive is used only to derive the exact 30 Hz camera sidecar.
FULL_KINEMATICS="${RUN_DIR}/target_robot/kinematics.npz"
require_file "${FULL_KINEMATICS}"
FINAL_KINEMATICS="${RUN_DIR}/final_50hz/kinematics.npz"
cp --reflink=auto "${TRACKING_KINEMATICS}" "${FINAL_KINEMATICS}"
TIMELINE_MANIFEST="${RUN_DIR}/timeline_manifest.json"
require_file "${TIMELINE_MANIFEST}"
FINAL_CAMERA_KINEMATICS="${RUN_DIR}/final_50hz/camera_kinematics_30hz.npz"
"${TRACKING_PY}" "${CAMERA_KINEMATICS_SCRIPT}" \
  --target-kinematics "${FULL_KINEMATICS}" \
  --timeline-manifest "${TIMELINE_MANIFEST}" \
  --output "${FINAL_CAMERA_KINEMATICS}"

# Contact evaluation is a sidecar: it never blocks RGB rendering.  When the
# fitted object model and cup masks are available, report realized post-Mink
# metrics; otherwise leave an explicit unavailable record instead of silently
# treating pre-IK metrics as post-IK truth.
CONTACT_METRICS="${RUN_DIR}/contact_metrics.jsonl"
HAND_KINEMATICS="${RUN_DIR}/dex3_kinematics/hand_from_tracking_50hz.npz"
OBJECT_MODELS="${INTERACTION_DIR}/object_models.jsonl"
if [[ "${HAND_MODE}" == "g1_brainco" && "${JOINT_CONTACT_ENABLED}" == "0" ]]; then
  printf '%s\n' "{\"status\":\"unavailable\",\"reason\":\"joint_contact_disabled\"}" > "${CONTACT_METRICS}"
elif [[ "${HAND_MODE}" == "g1_brainco" ]]; then
  if [[ -f "${CONTACT_EVAL_SCRIPT}" && -f "${HAND_KINEMATICS}" && \
        -f "${OBJECT_MODELS}" && -d "${CONTACT_MASK_DIR}" ]]; then
    CONTACT_EVAL_DIR="${RUN_DIR}/contact_evaluation"
    rm -f -- "${CONTACT_METRICS}"
    contact_eval_ran=0
    contact_eval_rc=0
    for side in left right; do
      side_candidate="${CONTACT_OUT_DIR}/contact_aware_wrist_candidates_native_${side}.npz"
      [[ -s "${side_candidate}" ]] || continue
      side_dir="${CONTACT_EVAL_DIR}/${side}"
      mkdir -p "${side_dir}"
      set +e
      "${CONTACT_PY}" "${CONTACT_EVAL_SCRIPT}" \
        --ik-kinematics "${TRACKING_KINEMATICS}" \
        --hand-kinematics "${HAND_KINEMATICS}" \
        --contact-candidates "${side_candidate}" \
        --object-models "${OBJECT_MODELS}" \
        --session-dir "${SESSION_DIR}" --mask-dir "${CONTACT_MASK_DIR}" \
        --out-dir "${side_dir}" --side "${side}" \
        --hand-mode g1_brainco \
        --keyframe "${CONTACT_KEYFRAME}" --seed "${CONTACT_SEED}" \
        --skip-plots
      side_rc=$?
      set -e
      if [[ "${side_rc}" -ne 0 ]]; then
        contact_eval_rc="${side_rc}"
        continue
      fi
      if [[ -f "${side_dir}/post_ik_interaction_metrics.jsonl" ]]; then
        cat "${side_dir}/post_ik_interaction_metrics.jsonl" >> "${CONTACT_METRICS}"
        contact_eval_ran=1
      fi
    done
    if [[ "${contact_eval_ran}" -eq 0 ]]; then
      printf '%s\n' "{\"status\":\"failed\",\"exit_code\":${contact_eval_rc},\"stage\":\"post_ik_contact_evaluation\"}" > "${CONTACT_METRICS}"
    fi
  else
    printf '%s\n' "{\"status\":\"unavailable\",\"reason\":\"object_models_or_cup_masks_missing\"}" > "${CONTACT_METRICS}"
  fi
elif [[ "${HAND_MODE}" == "g1_brainco" && "${JOINT_CONTACT_ENABLED}" == "0" ]]; then
  printf '%s\n' "{\"status\":\"unavailable\",\"reason\":\"joint_contact_disabled\"}" > "${CONTACT_METRICS}"
elif [[ "${HAND_MODE}" == "g1_brainco" ]]; then
  printf '%s\n' "{\"status\":\"unavailable\",\"reason\":\"brainco_contact_anchors_has_no_cupmodel_eval\"}" > "${CONTACT_METRICS}"
elif [[ -f "${CONTACT_EVAL_SCRIPT}" && -f "${HAND_KINEMATICS}" && \
      -f "${OBJECT_MODELS}" && -d "${CONTACT_MASK_DIR}" ]]; then
  CONTACT_EVAL_DIR="${RUN_DIR}/contact_evaluation"
  set +e
  "${CONTACT_PY}" "${CONTACT_EVAL_SCRIPT}" \
    --ik-kinematics "${TRACKING_KINEMATICS}" \
    --hand-kinematics "${HAND_KINEMATICS}" \
    --contact-candidates "${CONTACT_CANDIDATES}" \
    --object-models "${OBJECT_MODELS}" \
    --session-dir "${SESSION_DIR}" --mask-dir "${CONTACT_MASK_DIR}" \
    --out-dir "${CONTACT_EVAL_DIR}" --side right \
    --keyframe "${CONTACT_KEYFRAME}" --seed "${CONTACT_SEED}"
  contact_eval_rc=$?
  set -e
  if [[ "${contact_eval_rc}" -eq 0 && -f "${CONTACT_EVAL_DIR}/post_ik_interaction_metrics.jsonl" ]]; then
    cp "${CONTACT_EVAL_DIR}/post_ik_interaction_metrics.jsonl" "${CONTACT_METRICS}"
  else
    printf '%s\n' "{\"status\":\"failed\",\"exit_code\":${contact_eval_rc},\"stage\":\"post_ik_contact_evaluation\"}" > "${CONTACT_METRICS}"
  fi
else
  printf '%s\n' "{\"status\":\"unavailable\",\"reason\":\"object_models_or_cup_masks_missing\"}" > "${CONTACT_METRICS}"
fi
cp "${TRACKING_SESSION_DIR}/tracking_error.json" "${RUN_DIR}/episode_summary.json"

# Keep the published output minimal.  All frame-wise products, calibration
# diagnostics, renderer caches and DiffuEraser intermediates remain available
# only when KEEP_INTERMEDIATE=1 is explicitly requested for debugging.
if [[ "${KEEP_INTERMEDIATE}" != "1" ]]; then
  FINAL_DIR="${RUN_DIR}/final_50hz"
  find "${FINAL_DIR}" -mindepth 1 -maxdepth 1 \
    ! -name "rgb_final_50hz.mp4" ! -name "kinematics.npz" \
    ! -name "camera_kinematics_30hz.npz" \
    -exec rm -rf -- {} +
  find "${RUN_DIR}" -mindepth 1 -maxdepth 1 ! -name "final_50hz" \
    -exec rm -rf -- {} +
fi

echo "[contact-aware-6d] completed"
echo "[contact-aware-6d] output=${RUN_DIR}"
echo "[contact-aware-6d] video=${FINAL_VIDEO}"
echo "[contact-aware-6d] kinematics=${FINAL_KINEMATICS}"
echo "[contact-aware-6d] camera_kinematics=${FINAL_CAMERA_KINEMATICS}"
