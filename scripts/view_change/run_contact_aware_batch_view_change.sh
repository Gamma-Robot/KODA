#!/usr/bin/env bash
set -Eeuo pipefail

# Multi-GPU batch entry point for the validated offline 30 Hz -> 50 Hz
# view-change pipeline. Every session uses the latest paired wrist/hand bundle contract:
# detected contact publishes the jointly optimized right wrist/hand, while a
# no-contact session publishes the identity wrist plus base position-retargeted
# 14-DoF hand. Both paths run the same smoothed tracking and view-change stages.
# HAND_MODE=g1_brainco uses the BrainCo MANO palm refinement and, when
# JOINT_CONTACT_ENABLED=1, neighborhood Laplacian wrist+finger contact refine,
# then publishes 12 hand DoF.
# CONTACT_DETECTOR=pointcloud keeps the SAM/point-cloud contact flags.
# CONTACT_DETECTOR=egohumanoid writes EgoHumanoid 0-1-0 close/open from native
# MANO-21 3D hands into contact_frames.json and published kinematics. With
# JOINT_CONTACT_ENABLED=0 this is the only contact signal (tracking writes zeros).
# RERUN_VIEW_CHANGE=1 re-renders from published tracking kinematics and skips
# sessions that do not already have a reusable tracking archive.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCALER_PY="${KODA_PYTHON:-${PYTHON:-python3}}"
CONTACT_SCRIPT="${PROJECT_ROOT}/scripts/view_change/run_contact_aware_6d_reference_tracking.sh"
PRECOMPUTE_SCRIPT="${PROJECT_ROOT}/scripts/view_change/run_contact_precompute.sh"
PREPARE_REFERENCE_SCRIPT="${PROJECT_ROOT}/scripts/view_change/prepare_contact_template_palm_reference.sh"
EGOHUMANOID_CONTACT_SCRIPT="${PROJECT_ROOT}/scripts/view_change/write_egohumanoid_contact_frames.py"

SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/data/egocentric_data/pick_up_the_cup_smooth}"
CONTACT_ROOT="${CONTACT_ROOT:-${PROJECT_ROOT}/data/experiments/hand_retarget}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/data/whole_body_data/pick_up_the_cup_smooth}"
OUTPUT_LAYOUT="${OUTPUT_LAYOUT:-nested}"
# /tmp is a small root partition on the experiment host.  Keep transient
# adapters, masks, and renderer chunks on the large DATA4 volume instead.
WORK_ROOT="${WORK_ROOT:-${PROJECT_ROOT}/.cache/contact_aware_batch_view_change}"
# GPUS is a comma-separated worker pool. GPU remains a backward-compatible
# single-device alias when GPUS is not set.
GPUS="${GPUS:-${GPU:-1}}"
CONTACT_DEVICE="${CONTACT_DEVICE:-cuda}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
WORKERS="${WORKERS:-${#GPU_LIST[@]}}"
FORCE="${FORCE:-0}"
# 1 = reuse published tracking kinematics and re-run only view-change
# (MuJoCo render + composite + DiffuEraser). Sessions without reusable
# tracking/{kinematics.npz,meta.json,tracking_error.json} are skipped.
# Incompatible with FORCE=1, which clears tracking and re-runs everything.
RERUN_VIEW_CHANGE="${RERUN_VIEW_CHANGE:-0}"
MAX_EPISODES="${MAX_EPISODES:-0}"
SESSION_ID="${SESSION_ID:-}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
CONTACT_MISSING_POLICY="${CONTACT_MISSING_POLICY:-precompute}"
JOINT_CONTACT_ENABLED="${JOINT_CONTACT_ENABLED:-1}"
# pointcloud: SAM + RGB-D contact (default). egohumanoid: native 3D finger
# curvature square-wave, used as the published left/right contact flags.
CONTACT_DETECTOR="${CONTACT_DETECTOR:-pointcloud}"
# dex3 keeps the contact-aware Dex3 path unchanged. g1_brainco uses the BrainCo
# palm refinement and, when JOINT_CONTACT_ENABLED=1, neighborhood Laplacian contact.
HAND_MODE="${HAND_MODE:-dex3}"
# jax keeps Hydrax fragment-knot CEM. sil-tts runs Warp+ONNX ScaleBFM SIL
# (configs/experiments/scalebfm_torso_umi3_ego2humanoid.yaml) and skips JAX workers.
MPC_BACKEND="${MPC_BACKEND:-jax}"
MAX_SOURCE_FRAMES="${MAX_SOURCE_FRAMES:-}"
MAX_TARGET_FRAMES="${MAX_TARGET_FRAMES:-}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
# 1 = finish contact SAM/precompute for the whole queue (resident EfficientSAM3),
# then switch every GPU to tracking + view-change (resident Kimodo + JAX;
# DiffuEraser + DA3 start on that GPU's first view-change and stay hot).
# 0 restores the old per-session full pipeline.
STAGE_BARRIER="${STAGE_BARRIER:-1}"

# Keep only the final contract in OUTPUT_ROOT.  All high-volume frame assets
# stay in one per-session temporary directory and are removed immediately
# after the three final files are verified.
KEEP_INTERMEDIATE="${KEEP_INTERMEDIATE:-0}"
# Batch runs opt into stage-level pruning by default.  Set to 0 when retaining
# every frame-wise adapter/renderer artifact for debugging is required.
if [[ "${KEEP_INTERMEDIATE}" == "1" && -z "${PRUNE_INTERMEDIATES:-}" ]]; then
  PRUNE_INTERMEDIATES="0"
else
  PRUNE_INTERMEDIATES="${PRUNE_INTERMEDIATES:-1}"
fi
# The contact precompute stage writes large per-frame masks under CONTACT_ROOT,
# outside the per-session WORK_ROOT cleanup.  Remove those masks after a valid
# contact bundle has been consumed by final tracking, or after an incomplete
# precompute has failed and will be recomputed from scratch.
PRUNE_CONTACT_EVIDENCE="${PRUNE_CONTACT_EVIDENCE:-${PRUNE_INTERMEDIATES}}"

# Persistent JAX CEM-MPC worker (one per GPU). Contact-aware sessions each
# start a new run_three_point_batch process; keeping the compiled rollouts
# alive on this socket is what avoids paying JIT on every clip.
HYDRAX_PY="${HYDRAX_PY:-${KODA_PYTHON:-${PYTHON:-python3}}}"
JAX_MPC_WORKER="${PROJECT_ROOT}/resources/hydrax/scripts/pi_mjx_sim2sim/diagnostics/jax_mpc_worker.py"
THREE_POINT_ARTIFACT_MANIFEST="${THREE_POINT_ARTIFACT_MANIFEST:-${PROJECT_ROOT}/resources/hydrax/scripts/pi_mjx_sim2sim/artifacts/artifact_manifest.json}"
SAM_PY="${SAM_PY:-${KODA_PYTHON:-${PYTHON:-python3}}}"
EFFICIENTSAM3_WORKER="${PROJECT_ROOT}/scripts/view_change/efficientsam3_worker.py"
KIMODO_WORKER="${PROJECT_ROOT}/scripts/view_change/kimodo_init_worker.py"
DA3_WORKER="${PROJECT_ROOT}/scripts/view_change/da3_depth_worker.py"
DIFFUERASER_PY="${DIFFUERASER_PY:-${KODA_PYTHON:-${PYTHON:-python3}}}"
DIFFUERASER_WORKER="${DIFFUERASER_WORKER:-${PROJECT_ROOT}/resources/DiffuEraser/diffueraser_worker.py}"
PROPAINTER_WEIGHTS="${PROPAINTER_WEIGHTS:-${KODA_ASSETS:-${PROJECT_ROOT}/assets}/models/diffueraser/propainter}"
RESIDENT_WORKERS="${PROJECT_ROOT}/scripts/view_change/resident_gpu_workers.sh"

for required in "${SCALER_PY}" "${CONTACT_SCRIPT}" "${PRECOMPUTE_SCRIPT}" \
  "${PREPARE_REFERENCE_SCRIPT}" "${EGOHUMANOID_CONTACT_SCRIPT}" \
  "${EFFICIENTSAM3_WORKER}" "${DIFFUERASER_WORKER}" \
  "${KIMODO_WORKER}" "${DA3_WORKER}" "${RESIDENT_WORKERS}" \
  "${SAM_PY}" "${DIFFUERASER_PY}"; do
  [[ -f "${required}" || -x "${required}" ]] || { echo "missing required file: ${required}" >&2; exit 2; }
done
command -v flock >/dev/null || { echo "missing required command: flock" >&2; exit 2; }
[[ -d "${SOURCE_ROOT}" ]] || { echo "missing source root: ${SOURCE_ROOT}" >&2; exit 2; }
[[ "${CONTACT_MISSING_POLICY}" == "precompute" || "${CONTACT_MISSING_POLICY}" == "fail" ]] || {
  echo "CONTACT_MISSING_POLICY must be precompute or fail" >&2; exit 2;
}
[[ "${CONTACT_DETECTOR}" == "pointcloud" || "${CONTACT_DETECTOR}" == "egohumanoid" ]] || {
  echo "CONTACT_DETECTOR must be pointcloud or egohumanoid" >&2; exit 2;
}
[[ "${OUTPUT_LAYOUT}" == "nested" || "${OUTPUT_LAYOUT}" == "flat" ]] || {
  echo "OUTPUT_LAYOUT must be nested or flat" >&2; exit 2;
}
[[ "${HAND_MODE}" == "dex3" || "${HAND_MODE}" == "g1_brainco" ]] || {
  echo "HAND_MODE must be dex3 or g1_brainco" >&2; exit 2;
}
[[ "${MPC_BACKEND}" == "jax" || "${MPC_BACKEND}" == "sil-tts" ]] || {
  echo "MPC_BACKEND must be jax or sil-tts" >&2; exit 2;
}
if [[ "${HAND_MODE}" == "g1_brainco" ]]; then
  [[ "${JOINT_CONTACT_ENABLED}" == "0" || "${JOINT_CONTACT_ENABLED}" == "1" ]] || {
    echo "JOINT_CONTACT_ENABLED must be 0 or 1" >&2; exit 2;
  }
  HAND_DOF=12
else
  [[ "${JOINT_CONTACT_ENABLED}" == "1" ]] || {
    echo "the formal pipeline requires JOINT_CONTACT_ENABLED=1" >&2; exit 2;
  }
  HAND_DOF=14
fi
[[ "${FORCE}" == "0" || "${FORCE}" == "1" ]] || {
  echo "FORCE must be 0 or 1" >&2; exit 2;
}
[[ "${RERUN_VIEW_CHANGE}" == "0" || "${RERUN_VIEW_CHANGE}" == "1" ]] || {
  echo "RERUN_VIEW_CHANGE must be 0 or 1" >&2; exit 2;
}
if [[ "${FORCE}" == "1" && "${RERUN_VIEW_CHANGE}" == "1" ]]; then
  echo "FORCE=1 and RERUN_VIEW_CHANGE=1 cannot be set together" >&2
  exit 2
fi
[[ "${STAGE_BARRIER}" == "0" || "${STAGE_BARRIER}" == "1" ]] || {
  echo "STAGE_BARRIER must be 0 or 1" >&2; exit 2;
}
[[ "${PRUNE_CONTACT_EVIDENCE}" == "0" || "${PRUNE_CONTACT_EVIDENCE}" == "1" ]] || {
  echo "PRUNE_CONTACT_EVIDENCE must be 0 or 1" >&2; exit 2;
}
[[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "WORKERS must be a positive integer" >&2; exit 2;
}
(( WORKERS <= ${#GPU_LIST[@]} )) || {
  echo "WORKERS=${WORKERS} exceeds GPUS count ${#GPU_LIST[@]}" >&2; exit 2;
}
declare -A SEEN_GPUS=()
for gpu_index in "${GPU_LIST[@]}"; do
  [[ "${gpu_index}" =~ ^[0-9]+$ ]] || {
    echo "invalid GPU index in GPUS=${GPUS}: ${gpu_index}" >&2; exit 2;
  }
  [[ -z "${SEEN_GPUS[${gpu_index}]:-}" ]] || {
    echo "duplicate GPU index in GPUS=${GPUS}: ${gpu_index}" >&2; exit 2;
  }
  SEEN_GPUS["${gpu_index}"]=1
done

if [[ -n "${SESSION_ID}" ]]; then
  sessions=("${SOURCE_ROOT}/${SESSION_ID}")
else
  mapfile -d '' sessions < <(
    find "${SOURCE_ROOT}" -mindepth 1 -maxdepth 1 -xtype d -name 'session_*' -print0 | sort -z
  )
fi
[[ "${#sessions[@]}" -gt 0 ]] || { echo "no sessions under ${SOURCE_ROOT}" >&2; exit 2; }

selected=()
for session in "${sessions[@]}"; do
  [[ -d "${session}" ]] || { echo "missing session: ${session}" >&2; exit 2; }
  selected+=("$(basename "${session}")")
  if [[ "${MAX_EPISODES}" =~ ^[1-9][0-9]*$ && "${#selected[@]}" -ge "${MAX_EPISODES}" ]]; then break; fi
done

mkdir -p "${OUTPUT_ROOT}" "${WORK_ROOT}"
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_ROOT="${WORK_ROOT}/logs_${RUN_TAG}"
export LOG_ROOT
mkdir -p "${LOG_ROOT}"
STATUS_FILE="${LOG_ROOT}/status.tsv"
STATUS_LOCK="${LOG_ROOT}/status.lock"
STOP_FILE="${LOG_ROOT}/stop_after_failure"
QUEUE_INDEX_FILE="${LOG_ROOT}/queue.index"
QUEUE_LOCK="${LOG_ROOT}/queue.lock"
printf 'session_id\tstatus\texit_code\tgpu\tcontact_candidate\toutput\n' > "${STATUS_FILE}"
printf '0\n' > "${QUEUE_INDEX_FILE}"
# start_jax_mpc_worker / start_kimodo_worker / start_diffueraser_worker /
# start_da3_worker / start_efficientsam3_worker
# shellcheck source=resident_gpu_workers.sh
. "${RESIDENT_WORKERS}"

record_status() {
  local sid="$1" status="$2" rc="$3" gpu="$4" candidate="$5" out="$6"
  (
    flock -x 9
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${sid}" "${status}" "${rc}" "${gpu}" "${candidate}" "${out}" \
      >> "${STATUS_FILE}"
  ) 9>"${STATUS_LOCK}"
}

find_joint_candidate() {
  local sid="$1"
  local root="${CONTACT_ROOT}/${sid}/contact_aware_auto"
  if [[ "${HAND_MODE}" == "g1_brainco" ]]; then
    [[ "${JOINT_CONTACT_ENABLED}" == "1" ]] || return 0
    local left="${root}/contact_aware_wrist_candidates_native_left.npz"
    local right="${root}/contact_aware_wrist_candidates_native_right.npz"
    local base_hand="${root}/joint_contact/base_hand_brainco_50hz.npz"
    [[ ( -s "${left}" || -s "${right}" ) &&
       -s "${base_hand}" && -s "${root}/contact_precompute_manifest.json" &&
       -s "${root}/contact_frames.json" ]] || return 0
    "${SCALER_PY}" - "${root}/contact_precompute_manifest.json" "${CONTACT_DEVICE}" <<'PY' || return 0
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    manifest = json.load(stream)
bootstrap = manifest.get("reference_bootstrap", {})
if bootstrap.get("kind") != "raw_annotation_template_palm":
    raise SystemExit(1)
if bootstrap.get("tracking_bootstrap_used") is not False:
    raise SystemExit(1)
if manifest.get("contact_compute", {}).get("requested_device") != sys.argv[2]:
    raise SystemExit(1)
if manifest.get("hand_mode") not in (None, "g1_brainco"):
    raise SystemExit(1)
PY
    if [[ -s "${right}" ]]; then
      printf '%s\n' "${right}"
    else
      printf '%s\n' "${left}"
    fi
    return 0
  fi
  local candidate="${root}/contact_aware_wrist_candidates_native.npz"
  local base_hand="${root}/joint_contact/base_hand_dex3_50hz.npz"
  [[ -s "${candidate}" &&
     -s "${base_hand}" && -s "${root}/contact_precompute_manifest.json" &&
     -s "${root}/contact_frames.json" ]] || return 0
  "${SCALER_PY}" - "${root}/contact_precompute_manifest.json" "${CONTACT_DEVICE}" <<'PY' || return 0
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    manifest = json.load(stream)
bootstrap = manifest.get("reference_bootstrap", {})
if bootstrap.get("kind") != "raw_annotation_template_palm":
    raise SystemExit(1)
if bootstrap.get("tracking_bootstrap_used") is not False:
    raise SystemExit(1)
if manifest.get("contact_compute", {}).get("requested_device") != sys.argv[2]:
    raise SystemExit(1)
PY
  printf '%s\n' "${candidate}"
}

prune_contact_evidence() {
  local sid="$1" reason="$2"
  [[ "${PRUNE_CONTACT_EVIDENCE}" == "1" ]] || return 0
  case "${sid}" in
    session_*) ;;
    *) echo "[contact-batch] refusing to prune invalid session id: ${sid}" >&2; return 1 ;;
  esac

  local session_root="${CONTACT_ROOT}/${sid}"
  local complete=0
  local path
  [[ -n "$(find_joint_candidate "${sid}" || true)" ]] && complete=1
  if [[ "${reason}" == "completed" && "${complete}" -ne 1 ]]; then
    echo "[contact-batch] retain contact evidence for ${sid}: bundle is incomplete" >&2
  elif [[ "${reason}" == "failed_precompute" && "${complete}" -eq 1 ]]; then
    echo "[contact-batch] retain complete contact evidence for ${sid}" >&2
  else
    for path in "${session_root}/cup_segmentation_auto"; do
      if [[ -d "${path}" ]]; then
        find "${path}" -depth -delete
        echo "[contact-batch] pruned ${path} reason=${reason}"
      fi
    done
  fi

  # View-change frame caches are ~14GB/session at 1080p. Delete them as soon as
  # this session's final video is published; do not wait for the batch to finish,
  # and do not require a joint-contact bundle (BrainCo JOINT_CONTACT_ENABLED=0).
  local out="${OUTPUT_ROOT}/${sid}"
  if is_complete "${out}" "${sid}"; then
    for path in "${session_root}/human_masks" "${session_root}/source_rgb" "${session_root}/source_adapter"; do
      if [[ -d "${path}" ]]; then
        find "${path}" -depth -delete
        echo "[contact-batch] pruned ${path} reason=${reason} final_50hz=complete"
      fi
    done
  else
    echo "[contact-batch] retain human_masks/source_rgb/source_adapter for ${sid}: final_50hz contract missing" >&2
  fi
}

verify_final_contract() {
  local out="$1" sid="$2"
  [[ -s "${out}/rgb_final_50hz.mp4" && -s "${out}/kinematics.npz" &&
     -s "${out}/camera_kinematics_30hz.npz" ]] || return 1
  "${SCALER_PY}" - "${out}/kinematics.npz" \
    "${out}/camera_kinematics_30hz.npz" "${sid}" "${HAND_DOF}" <<'PY'
import json
import sys
import numpy as np

kin_path, cam_path, sid, hand_dof = sys.argv[1:]
hand_dof = int(hand_dof)
with np.load(kin_path, allow_pickle=False) as archive:
    required = {
        "joint_pos", "joint_vel", "joint_names", "timestamps",
        "reference_smoothing_config_json", "reference_smoothing_config_sha256",
    }
    required |= {
        "hand_qpos", "hand_qvel", "hand_joint_names",
        "contact_30hz", "contact_50hz", "joint_contact_bundle_sha256",
        "joint_contact_contract",
    }
    missing = sorted(required - set(archive.files))
    if missing:
        raise SystemExit(f"{sid}: final kinematics missing {missing}")
    body = np.asarray(archive["joint_pos"])
    body_vel = np.asarray(archive["joint_vel"])
    timestamps = np.asarray(archive["timestamps"])
    if body.ndim != 2 or body.shape[1] != 29 or body_vel.shape != body.shape:
        raise SystemExit(f"{sid}: invalid 29-DoF body {body.shape}/{body_vel.shape}")
    if timestamps.shape != (body.shape[0],) or not np.all(np.isfinite(timestamps)):
        raise SystemExit(f"{sid}: invalid tracking timestamps")
    smoothing = json.loads(str(np.asarray(archive["reference_smoothing_config_json"]).item()))
    if not bool(smoothing.get("enabled")) or smoothing.get("stage") != "post_contact_pre_tracking":
        raise SystemExit(f"{sid}: post-contact reference smoothing is not enabled")
    hand = np.asarray(archive["hand_qpos"])
    hand_vel = np.asarray(archive["hand_qvel"])
    if hand.shape != (body.shape[0], hand_dof) or hand_vel.shape != hand.shape:
        raise SystemExit(
            f"{sid}: invalid {hand_dof}-DoF hand {hand.shape}/{hand_vel.shape}"
        )
    if np.asarray(archive["contact_50hz"]).shape != (body.shape[0],):
        raise SystemExit(f"{sid}: invalid 50 Hz contact timeline")
    if not np.all(np.isin(np.asarray(archive["contact_30hz"]), (0, 1))):
        raise SystemExit(f"{sid}: invalid 30 Hz contact flags")
with np.load(cam_path, allow_pickle=False) as camera:
    if "timestamps" not in camera or np.asarray(camera["timestamps"]).ndim != 1:
        raise SystemExit(f"{sid}: invalid camera kinematics contract")
print(f"{sid}: verified final smoothed 29+{hand_dof} DoF frames={body.shape[0]}")
PY
}

published_video() {
  local out="$1"
  if [[ "${OUTPUT_LAYOUT}" == "flat" ]]; then
    printf '%s\n' "${out}/rgb_final_50hz.mp4"
  else
    printf '%s\n' "${out}/final_50hz/rgb_final_50hz.mp4"
  fi
}

published_kinematics() {
  local out="$1"
  if [[ -s "${out}/tracking/kinematics.npz" ]]; then
    printf '%s\n' "${out}/tracking/kinematics.npz"
    return 0
  fi
  if [[ -s "${out}/final_50hz/kinematics.npz" ]]; then
    printf '%s\n' "${out}/final_50hz/kinematics.npz"
    return 0
  fi
  if [[ -s "${out}/kinematics.npz" ]]; then
    printf '%s\n' "${out}/kinematics.npz"
    return 0
  fi
  return 1
}

# Skip re-tracking and view-change when the 50 Hz RGB and a kinematics
# archive already exist. ``tracking/kinematics.npz`` counts; the camera
# sidecar is not required. Newly produced runs still pass
# ``verify_final_contract`` (video + kinematics + camera).
# RERUN_VIEW_CHANGE=1 bypasses this skip so the video can be rebuilt from
# the same published tracking archive.
is_complete() {
  local out="$1"
  local video
  video="$(published_video "${out}")"
  [[ -s "${video}" ]] || return 1
  published_kinematics "${out}" >/dev/null
}

reusable_tracking_root() {
  local out="$1"
  local cache
  for cache in "${out}/tracking" "${out}/tracking/${MPC_BACKEND}"; do
    if [[ -s "${cache}/kinematics.npz" && -s "${cache}/meta.json" &&
          -s "${cache}/tracking_error.json" ]]; then
      printf '%s\n' "${cache}"
      return 0
    fi
  done
  return 1
}

skip_missing_kinematics_for_view_rerun() {
  local sid="$1" gpu="$2" out="$3" candidate="${4:-none}"
  [[ "${RERUN_VIEW_CHANGE}" == "1" ]] || return 1
  if reusable_tracking_root "${out}" >/dev/null; then
    return 1
  fi
  record_status "${sid}" "skipped_missing_kinematics" 0 "${gpu}" "${candidate}" "${out}"
  echo "[contact-batch gpu=${gpu}] skip missing kinematics ${sid}"
  return 0
}

sync_published_kinematics() {
  local out="$1"
  local video dest kin
  video="$(published_video "${out}")"
  dest="$(dirname "${video}")/kinematics.npz"
  [[ -s "${dest}" ]] && return 0
  kin="$(published_kinematics "${out}")" || return 0
  mkdir -p "$(dirname "${dest}")"
  cp --reflink=auto "${kin}" "${dest}"
}

# EgoHumanoid close/open as published contact. JOINT_CONTACT_ENABLED=0 tracking
# writes zeros, so this is the only contact signal. JOINT_CONTACT_ENABLED=1 keeps
# the pointcloud JSON for wrist/hand injection and writes a sidecar plus patched
# kinematics flags.
apply_egohumanoid_contact() {
  local sid="$1" out="$2"
  [[ "${CONTACT_DETECTOR}" == "egohumanoid" ]] || return 0
  local json_out="${CONTACT_ROOT}/${sid}/contact_aware_auto/contact_frames.json"
  if [[ "${JOINT_CONTACT_ENABLED}" == "1" ]]; then
    json_out="${CONTACT_ROOT}/${sid}/contact_aware_auto/contact_frames_egohumanoid.json"
  fi
  local cmd=(
    "${SCALER_PY}" "${EGOHUMANOID_CONTACT_SCRIPT}"
    --session-dir "${SOURCE_ROOT}/${sid}"
    --output-json "${json_out}"
  )
  local path resolved
  local -A seen=()
  for path in \
    "$(dirname "$(published_video "${out}")")/kinematics.npz" \
    "${out}/final_50hz/kinematics.npz" \
    "${out}/kinematics.npz" \
    "${out}/tracking/kinematics.npz" \
    "${out}/tracking/${MPC_BACKEND}/kinematics.npz"
  do
    [[ -s "${path}" ]] || continue
    resolved="$(realpath -e "${path}")" || continue
    [[ -z "${seen[${resolved}]:-}" ]] || continue
    seen["${resolved}"]=1
    cmd+=(--kinematics "${resolved}")
  done
  echo "[contact-batch] EgoHumanoid contact ${sid} json=${json_out}"
  "${cmd[@]}"
}

copy_and_verify() {
  local src="$1" dst="$2" sid="$3" mode="$4"
  local publish="${dst}"
  [[ "${OUTPUT_LAYOUT}" == "flat" ]] || publish="${dst}/final_50hz"
  mkdir -p "${publish}"
  for name in rgb_final_50hz.mp4 kinematics.npz camera_kinematics_30hz.npz; do
    [[ -s "${src}/final_50hz/${name}" ]] || {
      echo "${sid}: missing final artifact ${src}/final_50hz/${name}" >&2
      return 1
    }
    cp --reflink=auto "${src}/final_50hz/${name}" "${publish}/${name}"
  done
  verify_final_contract "${publish}" "${sid}"
}

run_contact_session() {
  local sid="$1" candidate="$2" out="$3" tmp="$4" gpu="$5"
  local experiment_dir="${CONTACT_ROOT}/${sid}"
  local bundle_root="${experiment_dir}/contact_aware_auto"
  local tracking_cache="${out}/tracking/${MPC_BACKEND}"
  local reused_tracking track_force view_force
  if reused_tracking="$(reusable_tracking_root "${out}")"; then
    tracking_cache="${reused_tracking}"
  fi
  if [[ "${FORCE}" == "1" ]]; then
    rm -rf -- "${tracking_cache}"
    echo "[contact-batch gpu=${gpu}] FORCE=1 cleared tracking cache ${tracking_cache}"
  fi
  track_force="${FORCE}"
  view_force="${FORCE}"
  if [[ "${RERUN_VIEW_CHANGE}" == "1" ]]; then
    track_force="0"
    view_force="1"
    echo "[contact-batch gpu=${gpu}] rerun view-change from ${tracking_cache}"
  fi
  env GPU="${gpu}" FORCE="${track_force}" VIEW_FORCE="${view_force}" SMOKE="0" KEEP_INTERMEDIATE="${KEEP_INTERMEDIATE}" \
    PRUNE_INTERMEDIATES="${PRUNE_INTERMEDIATES}" \
    ROOT="${PROJECT_ROOT}" SESSION_ID="${sid}" \
    EXPERIMENT_DIR="${experiment_dir}" CONTACT_OUT_DIR="${bundle_root}" \
    INPUT_ROOT="${SOURCE_ROOT}" CONTACT_CANDIDATES="${candidate}" \
    CONTACT_BUNDLE_ROOT="${bundle_root}" \
    JOINT_CONTACT_ENABLED="${JOINT_CONTACT_ENABLED}" \
    HAND_MODE="${HAND_MODE}" \
    MPC_BACKEND="${MPC_BACKEND}" \
    CONTACT_DEVICE="${CONTACT_DEVICE}" \
    SKIP_TRACKING="$([ "${RERUN_VIEW_CHANGE}" == "1" ] && echo 1 || echo 0)" \
    TRACKING_ROOT="${tmp}/tracking" VIEW_ROOT="${tmp}/view" WORK_ROOT="${tmp}/work" \
    TRACKING_CACHE_ROOT="${tracking_cache}" \
    TRACKING_PY="${SCALER_PY}" MAX_SOURCE_FRAMES="${MAX_SOURCE_FRAMES}" \
    MAX_TARGET_FRAMES="${MAX_TARGET_FRAMES}" \
    LOG_ROOT="${LOG_ROOT}" \
    JAX_MPC_WORKER_SOCK="${JAX_MPC_WORKER_SOCK:-}" \
    KIMODO_WORKER_SOCK="${KIMODO_WORKER_SOCK:-}" \
    DIFFUERASER_WORKER_SOCK="${DIFFUERASER_WORKER_SOCK:-}" \
    DA3_WORKER_SOCK="${DA3_WORKER_SOCK:-}" \
    EFFICIENTSAM3_WORKER_SOCK="${EFFICIENTSAM3_WORKER_SOCK:-}" \
    JAX_COMPILATION_CACHE_DIR="${WORK_ROOT}/jax_cache" \
    Q0_ZERO_ARM_NO_SMOOTHNESS="${Q0_ZERO_ARM_NO_SMOOTHNESS:-1}" \
    Q0_SHOULDER_CLEARANCE="${Q0_SHOULDER_CLEARANCE:-1}" \
    Q0_SHOULDER_CLEARANCE_COST="${Q0_SHOULDER_CLEARANCE_COST:-5}" \
    Q0_SHOULDER_CLEARANCE_THRESHOLD="${Q0_SHOULDER_CLEARANCE_THRESHOLD:-0.06}" \
    Q0_MIN_SHOULDER_CLEARANCE="${Q0_MIN_SHOULDER_CLEARANCE:-0.01}" \
    bash "${CONTACT_SCRIPT}" || return $?
  copy_and_verify "${tmp}/view/${sid}" "${out}" "${sid}" "contact_aware_feedback1" || return $?
  apply_egohumanoid_contact "${sid}" "${out}"
  prune_contact_evidence "${sid}" completed || true
}

run_precompute_session() {
  local sid="$1" tmp="$2" gpu="$3" session_out="${CONTACT_ROOT}/${sid}"
  local reference_root="${tmp}/contact_reference"
  local reference_kinematics="${reference_root}/template_palm_reference_kinematics.npz"
  local reference_targets="${reference_root}/template_palm_kabsch_targets.npz"
  local reference_sha256
  local extra_env=()
  mkdir -p "${session_out}"
  env SESSION_DIR="${SOURCE_ROOT}/${sid}" \
    REFERENCE_OUTPUT_DIR="${reference_root}" \
    bash "${PREPARE_REFERENCE_SCRIPT}"
  reference_sha256="$(sha256sum "${reference_kinematics}" | awk '{print $1}')"
  extra_env+=(HAND_MODE="${HAND_MODE}")
  if [[ -n "${CUP_PROMPTS+x}" ]]; then
    extra_env+=(CUP_PROMPTS="${CUP_PROMPTS}")
  fi
  env GPU="${gpu}" SESSION_ID="${sid}" \
    INPUT_ROOT="${SOURCE_ROOT}" EXPERIMENT_ROOT="${CONTACT_ROOT}" \
    KINEMATICS_ROOT="${reference_root}" \
    KINEMATICS_NPZ="${reference_kinematics}" \
    TARGET_TIMESTAMPS_NPZ="${reference_targets}" \
    REFERENCE_BOOTSTRAP_NPZ="${reference_kinematics}" \
    REFERENCE_BOOTSTRAP_SHA256="${reference_sha256}" \
    TEMPLATE_PALM_FROM_KINEMATICS="1" \
    OUTPUT_DIR="${session_out}/contact_aware_auto" FORCE="1" \
    CONTACT_NO_EVIDENCE_POLICY="${CONTACT_NO_EVIDENCE_POLICY:-identity}" \
    MAX_NATIVE_FRAMES="${MAX_SOURCE_FRAMES}" CONTACT_SEED="${SEED}" \
    CONTACT_DISTANCE_THRESHOLD_M="${CONTACT_DISTANCE_THRESHOLD_M:-0.020}" \
    CONTACT_DEPTH_GAP_THRESHOLD_M="${CONTACT_DEPTH_GAP_THRESHOLD_M:-0.020}" \
    CONTACT_NORMAL_DOT_THRESHOLD="${CONTACT_NORMAL_DOT_THRESHOLD:-0.10}" \
    CONTACT_TEMPORAL_GAP="${CONTACT_TEMPORAL_GAP:-3}" \
    CONTACT_MIN_RUN="${CONTACT_MIN_RUN:-3}" \
    CONTACT_MIN_FINGERS="${CONTACT_MIN_FINGERS:-2}" \
    CONTACT_OBJECT_MASK_MAX_GAP="${CONTACT_OBJECT_MASK_MAX_GAP:-4}" \
    JOINT_CONTACT_ENABLED="${JOINT_CONTACT_ENABLED}" \
    "${extra_env[@]}" \
    CONTACT_DEVICE="${CONTACT_DEVICE}" \
    HUMAN_MASK_BACKEND="${HUMAN_MASK_BACKEND:-efficientsam3}" \
    EFFICIENTSAM3_WORKER_SOCK="${EFFICIENTSAM3_WORKER_SOCK:-}" \
    WORK_ROOT="${tmp}/precompute" \
    bash "${PRECOMPUTE_SCRIPT}"
}

echo "[contact-batch] source=${SOURCE_ROOT} reference_bootstrap=raw_template_palm contact=${CONTACT_ROOT}"
echo "[contact-batch] output=${OUTPUT_ROOT} layout=${OUTPUT_LAYOUT} sessions=${#selected[@]} gpus=${GPUS} workers=${WORKERS} scheduling=dynamic_queue stage_barrier=${STAGE_BARRIER} hand_mode=${HAND_MODE} hand_dof=${HAND_DOF} mpc_backend=${MPC_BACKEND} contact_device=${CONTACT_DEVICE} contact_detector=${CONTACT_DETECTOR} policy=${CONTACT_MISSING_POLICY} prune_intermediates=${PRUNE_INTERMEDIATES} prune_contact_evidence=${PRUNE_CONTACT_EVIDENCE} rerun_view_change=${RERUN_VIEW_CHANGE}"
if [[ "${DRY_RUN}" == "1" ]]; then
  for ((index=0; index<${#selected[@]}; index++)); do
    sid="${selected[$index]}"
    candidate="$(find_joint_candidate "${sid}" || true)"
    if [[ "${RERUN_VIEW_CHANGE}" == "1" ]]; then
      if reusable_tracking_root "${OUTPUT_ROOT}/${sid}" >/dev/null; then
        mode="rerun_view_change"
      else
        mode="skipped_missing_kinematics"
      fi
    elif [[ "${HAND_MODE}" == "g1_brainco" && "${JOINT_CONTACT_ENABLED}" == "0" ]]; then
      mode="g1_brainco_palm_refine"
    elif [[ "${HAND_MODE}" == "g1_brainco" && -n "${candidate}" ]]; then
      mode="g1_brainco_palm_refine+neighborhood_laplacian"
    elif [[ "${HAND_MODE}" == "g1_brainco" ]]; then
      mode="g1_brainco_contact_precompute_required"
    elif [[ -n "${candidate}" ]]; then
      mode="latest_tracking_joint_bundle"
    else
      mode="contact_precompute_required"
    fi
    printf '[contact-batch] plan queue_position=%s session=%s gpu=dynamic_pool(%s) mode=%s candidate=%s contact_detector=%s\n' \
      "$((index + 1))" "${sid}" "${GPUS}" "${mode}" "${candidate:-none}" "${CONTACT_DETECTOR}"
  done
  echo "[contact-batch] dry-run complete; no episode was processed (runtime scheduling=dynamic_queue)"
  exit 0
fi

process_session() {
  local sid="$1" gpu="$2"
  local out="${OUTPUT_ROOT}/${sid}"
  local candidate tmp rc status
  candidate="$(find_joint_candidate "${sid}" || true)"
  if skip_missing_kinematics_for_view_rerun "${sid}" "${gpu}" "${out}" "${candidate:-none}"; then
    return 0
  fi
  if [[ "${RERUN_VIEW_CHANGE}" == "1" ]]; then
    tmp="$(mktemp -d "${WORK_ROOT}/${sid}.XXXXXX")"
    rc=0
    echo "[contact-batch gpu=${gpu}] start ${sid} mode=rerun_view_change"
    run_contact_session "${sid}" "${candidate}" "${out}" "${tmp}" "${gpu}" || rc=$?
    rm -rf -- "${tmp}"
    [[ "${rc}" -eq 0 ]] && status="completed" || status="failed"
    record_status "${sid}" "${status}" "${rc}" "${gpu}" "${candidate:-none}" "${out}"
    echo "[contact-batch gpu=${gpu}] ${status} ${sid} rc=${rc}"
    return "${rc}"
  fi

  if [[ "${HAND_MODE}" == "g1_brainco" && "${JOINT_CONTACT_ENABLED}" == "0" ]]; then
    # Palm refine only: no contact detection or anchor optimization.
    if [[ "${FORCE}" != "1" ]] && is_complete "${out}" "${sid}"; then
      sync_published_kinematics "${out}"
      apply_egohumanoid_contact "${sid}" "${out}"
      record_status "${sid}" "skipped_complete" 0 "${gpu}" "none" "${out}"
      echo "[contact-batch gpu=${gpu}] skip complete ${sid} (video + kinematics)"
      return 0
    fi
    tmp="$(mktemp -d "${WORK_ROOT}/${sid}.XXXXXX")"
    rc=0
    echo "[contact-batch gpu=${gpu}] start ${sid} mode=g1_brainco_palm_refine"
    run_contact_session "${sid}" "" "${out}" "${tmp}" "${gpu}" || rc=$?
    rm -rf -- "${tmp}"
    [[ "${rc}" -eq 0 ]] && status="completed" || status="failed"
    record_status "${sid}" "${status}" "${rc}" "${gpu}" "none" "${out}"
    echo "[contact-batch gpu=${gpu}] ${status} ${sid} rc=${rc}"
    return "${rc}"
  fi

  candidate="$(find_joint_candidate "${sid}" || true)"
  if [[ "${FORCE}" != "1" ]] && is_complete "${out}" "${sid}" && \
        [[ -n "${candidate}" ]]; then
    sync_published_kinematics "${out}"
    apply_egohumanoid_contact "${sid}" "${out}"
    record_status "${sid}" "skipped_complete" 0 "${gpu}" "${candidate}" "${out}"
    echo "[contact-batch gpu=${gpu}] skip complete ${sid} (video + kinematics)"
    return 0
  fi
  if [[ "${CONTACT_MISSING_POLICY}" == "fail" && -z "${candidate}" ]]; then
    echo "[contact-batch gpu=${gpu}] missing complete contact bundle: ${sid}" >&2
    record_status "${sid}" "missing_contact_evidence" 2 "${gpu}" "none" "${out}"
    return 2
  fi
  tmp="$(mktemp -d "${WORK_ROOT}/${sid}.XXXXXX")"
  rc=0
  if [[ -z "${candidate}" ]]; then
    echo "[contact-batch gpu=${gpu}] joint-contact artifacts missing; precomputing ${sid}"
    if run_precompute_session "${sid}" "${tmp}" "${gpu}"; then
      rc=0
    else
      rc=$?
    fi
    candidate="$(find_joint_candidate "${sid}" || true)"
    if [[ "${rc}" -ne 0 && -z "${candidate}" ]]; then
      prune_contact_evidence "${sid}" failed_precompute || true
    fi
  fi
  echo "[contact-batch gpu=${gpu}] start ${sid} mode=$([[ -n "${candidate}" ]] && echo contact_aware_joint || echo contact_precompute)"
  if [[ "${rc}" -eq 0 && -n "${candidate}" ]]; then
    if run_contact_session "${sid}" "${candidate}" "${out}" "${tmp}" "${gpu}"; then
      rc=0
      prune_contact_evidence "${sid}" completed || true
    else
      rc=$?
    fi
  elif [[ "${rc}" -eq 0 ]]; then
    echo "[contact-batch gpu=${gpu}] precompute produced no candidate for ${sid}" >&2
    rc=2
  fi
  rm -rf -- "${tmp}"
  if [[ "${rc}" -eq 0 ]]; then
    status="completed"
  else
    status="failed"
  fi
  record_status "${sid}" "${status}" "${rc}" "${gpu}" "${candidate:-none}" "${out}"
  echo "[contact-batch gpu=${gpu}] ${status} ${sid} rc=${rc}"
  return "${rc}"
}

# Worker start/stop helpers are sourced from resident_gpu_workers.sh
# (start_jax_mpc_worker, start_kimodo_worker, start_diffueraser_worker,
#  start_da3_worker, start_efficientsam3_worker).

claim_next_session() {
  (
    flock -x 9
    read -r index < "${QUEUE_INDEX_FILE}"
    if (( index >= ${#selected[@]} )); then
      exit 1
    fi
    next_index=$((index + 1))
    printf '%s\n' "${next_index}" > "${QUEUE_INDEX_FILE}"
    printf '%s\n' "${selected[$index]}"
  ) 9>"${QUEUE_LOCK}"
}

process_precompute_only() {
  local sid="$1" gpu="$2"
  local out="${OUTPUT_ROOT}/${sid}"
  local candidate tmp rc=0
  if [[ "${HAND_MODE}" == "g1_brainco" && "${JOINT_CONTACT_ENABLED}" == "0" ]]; then
    record_status "${sid}" "precompute_skipped_palm_refine" 0 "${gpu}" "none" "${out}"
    echo "[contact-batch gpu=${gpu}] phase=precompute skip palm-refine ${sid}"
    return 0
  fi
  candidate="$(find_joint_candidate "${sid}" || true)"
  if [[ "${FORCE}" != "1" && "${RERUN_VIEW_CHANGE}" != "1" ]] && is_complete "${out}" "${sid}" && [[ -n "${candidate}" ]]; then
    sync_published_kinematics "${out}"
    apply_egohumanoid_contact "${sid}" "${out}"
    record_status "${sid}" "skipped_complete" 0 "${gpu}" "${candidate}" "${out}"
    echo "[contact-batch gpu=${gpu}] phase=precompute skip complete ${sid} (video + kinematics)"
    return 0
  fi
  if [[ "${FORCE}" != "1" && -n "${candidate}" ]]; then
    record_status "${sid}" "precompute_skipped_existing" 0 "${gpu}" "${candidate}" "${out}"
    echo "[contact-batch gpu=${gpu}] phase=precompute reuse bundle ${sid}"
    return 0
  fi
  if [[ "${CONTACT_MISSING_POLICY}" == "fail" ]]; then
    record_status "${sid}" "missing_contact_evidence" 2 "${gpu}" "none" "${out}"
    return 2
  fi
  tmp="$(mktemp -d "${WORK_ROOT}/${sid}.pre.XXXXXX")"
  echo "[contact-batch gpu=${gpu}] phase=precompute start ${sid}"
  if run_precompute_session "${sid}" "${tmp}" "${gpu}"; then
    candidate="$(find_joint_candidate "${sid}" || true)"
    rm -rf -- "${tmp}"
    if [[ -z "${candidate}" ]]; then
      prune_contact_evidence "${sid}" failed_precompute || true
      record_status "${sid}" "precompute_failed" 2 "${gpu}" "none" "${out}"
      return 2
    fi
    record_status "${sid}" "precompute_completed" 0 "${gpu}" "${candidate}" "${out}"
    echo "[contact-batch gpu=${gpu}] phase=precompute completed ${sid}"
    return 0
  fi
  rc=$?
  rm -rf -- "${tmp}"
  prune_contact_evidence "${sid}" failed_precompute || true
  record_status "${sid}" "precompute_failed" "${rc}" "${gpu}" "none" "${out}"
  echo "[contact-batch gpu=${gpu}] phase=precompute failed ${sid} rc=${rc}"
  return "${rc}"
}

process_tracking_only() {
  local sid="$1" gpu="$2"
  local out="${OUTPUT_ROOT}/${sid}"
  local candidate tmp rc=0 status
  candidate="$(find_joint_candidate "${sid}" || true)"
  if skip_missing_kinematics_for_view_rerun "${sid}" "${gpu}" "${out}" "${candidate:-none}"; then
    return 0
  fi
  if [[ "${RERUN_VIEW_CHANGE}" == "1" ]]; then
    tmp="$(mktemp -d "${WORK_ROOT}/${sid}.track.XXXXXX")"
    echo "[contact-batch gpu=${gpu}] phase=tracking start ${sid} mode=rerun_view_change"
    run_contact_session "${sid}" "${candidate}" "${out}" "${tmp}" "${gpu}" || rc=$?
    rm -rf -- "${tmp}"
    [[ "${rc}" -eq 0 ]] && status="completed" || status="failed"
    record_status "${sid}" "${status}" "${rc}" "${gpu}" "${candidate:-none}" "${out}"
    echo "[contact-batch gpu=${gpu}] phase=tracking ${status} ${sid} rc=${rc}"
    return "${rc}"
  fi
  if [[ "${FORCE}" != "1" ]] && is_complete "${out}" "${sid}"; then
    sync_published_kinematics "${out}"
    apply_egohumanoid_contact "${sid}" "${out}"
    record_status "${sid}" "skipped_complete" 0 "${gpu}" "${candidate:-none}" "${out}"
    echo "[contact-batch gpu=${gpu}] phase=tracking skip complete ${sid} (video + kinematics)"
    return 0
  fi
  tmp="$(mktemp -d "${WORK_ROOT}/${sid}.track.XXXXXX")"
  if [[ "${HAND_MODE}" == "g1_brainco" && "${JOINT_CONTACT_ENABLED}" == "0" ]]; then
    echo "[contact-batch gpu=${gpu}] phase=tracking start ${sid} mode=g1_brainco_palm_refine"
    run_contact_session "${sid}" "" "${out}" "${tmp}" "${gpu}" || rc=$?
    rm -rf -- "${tmp}"
    [[ "${rc}" -eq 0 ]] && status="completed" || status="failed"
    record_status "${sid}" "${status}" "${rc}" "${gpu}" "none" "${out}"
    echo "[contact-batch gpu=${gpu}] phase=tracking ${status} ${sid} rc=${rc}"
    return "${rc}"
  fi
  candidate="$(find_joint_candidate "${sid}" || true)"
  if [[ -z "${candidate}" ]]; then
    echo "[contact-batch gpu=${gpu}] phase=tracking missing contact bundle ${sid}" >&2
    rm -rf -- "${tmp}"
    record_status "${sid}" "missing_contact_evidence" 2 "${gpu}" "none" "${out}"
    return 2
  fi
  echo "[contact-batch gpu=${gpu}] phase=tracking start ${sid}"
  if run_contact_session "${sid}" "${candidate}" "${out}" "${tmp}" "${gpu}"; then
    rc=0
    prune_contact_evidence "${sid}" completed || true
  else
    rc=$?
  fi
  rm -rf -- "${tmp}"
  [[ "${rc}" -eq 0 ]] && status="completed" || status="failed"
  record_status "${sid}" "${status}" "${rc}" "${gpu}" "${candidate}" "${out}"
  echo "[contact-batch gpu=${gpu}] phase=tracking ${status} ${sid} rc=${rc}"
  return "${rc}"
}

run_worker() {
  local worker_index="$1" phase="$2"
  local gpu="${GPU_LIST[$worker_index]}"
  local sid rc worker_failed=0
  GPU="${gpu}"
  export GPU
  JAX_MPC_WORKER_PID=""
  JAX_MPC_WORKER_SOCK=""
  KIMODO_WORKER_PID=""
  KIMODO_WORKER_SOCK=""
  EFFICIENTSAM3_WORKER_PID=""
  EFFICIENTSAM3_WORKER_SOCK=""
  DIFFUERASER_WORKER_PID=""
  DIFFUERASER_WORKER_SOCK=""
  DA3_WORKER_PID=""
  DA3_WORKER_SOCK=""
  if [[ "${phase}" == "precompute" ]]; then
    start_efficientsam3_worker "${gpu}"
    trap 'stop_efficientsam3_worker "${GPU}"' EXIT
  elif [[ "${RERUN_VIEW_CHANGE}" == "1" ]]; then
    echo "[contact-batch gpu=${gpu}] rerun view-change; skipping Kimodo/JAX workers"
    trap 'stop_all_gpu_workers "${GPU}"' EXIT
  elif [[ "${phase}" == "tracking" ]]; then
    start_kimodo_worker "${gpu}" || echo "[contact-batch gpu=${gpu}] Kimodo worker unavailable; falling back to per-session subprocess" >&2
    if [[ "${MPC_BACKEND}" == "jax" ]]; then
      start_jax_mpc_worker "${gpu}"
    else
      echo "[contact-batch gpu=${gpu}] mpc_backend=${MPC_BACKEND}; skipping JAX CEM worker"
    fi
    trap 'stop_all_gpu_workers "${GPU}"' EXIT
  else
    start_kimodo_worker "${gpu}" || echo "[contact-batch gpu=${gpu}] Kimodo worker unavailable; falling back to per-session subprocess" >&2
    if [[ "${MPC_BACKEND}" == "jax" ]]; then
      start_jax_mpc_worker "${gpu}"
    else
      echo "[contact-batch gpu=${gpu}] mpc_backend=${MPC_BACKEND}; skipping JAX CEM worker"
    fi
    trap 'stop_all_gpu_workers "${GPU}"' EXIT
  fi
  while true; do
    if [[ "${CONTINUE_ON_ERROR}" != "1" && -e "${STOP_FILE}" ]]; then
      echo "[contact-batch gpu=${gpu}] stop requested; no new session will start"
      break
    fi
    if ! sid="$(claim_next_session)"; then
      break
    fi
    echo "[contact-batch gpu=${gpu}] phase=${phase} claimed ${sid}"
    rc=0
    if [[ "${phase}" == "precompute" ]]; then
      process_precompute_only "${sid}" "${gpu}" || rc=$?
    elif [[ "${phase}" == "tracking" ]]; then
      process_tracking_only "${sid}" "${gpu}" || rc=$?
    else
      process_session "${sid}" "${gpu}" || rc=$?
    fi
    if [[ "${rc}" -eq 0 ]]; then
      continue
    fi
    worker_failed=1
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      : > "${STOP_FILE}"
      echo "[contact-batch gpu=${gpu}] stopping after ${sid} rc=${rc}"
      break
    fi
  done
  [[ "${worker_failed}" -eq 0 || "${CONTINUE_ON_ERROR}" == "1" ]]
}

run_phase_pool() {
  local phase="$1"
  local worker_index gpu pid
  local pids=()
  printf '0\n' > "${QUEUE_INDEX_FILE}"
  echo "[contact-batch] phase=${phase} starting ${WORKERS} GPU workers"
  for ((worker_index=0; worker_index<WORKERS; worker_index++)); do
    gpu="${GPU_LIST[$worker_index]}"
    : >> "${LOG_ROOT}/worker_gpu_${gpu}.log"
    (
      run_worker "${worker_index}" "${phase}"
    ) > >(tee -a "${LOG_ROOT}/worker_gpu_${gpu}.log") 2>&1 &
    pids+=("$!")
  done
  local worker_wait_failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      worker_wait_failed=1
    fi
  done
  [[ "${worker_wait_failed}" -eq 0 ]]
}

if [[ "${STAGE_BARRIER}" == "1" ]]; then
  if [[ "${RERUN_VIEW_CHANGE}" == "1" ]]; then
    echo "[contact-batch] skip contact/SAM; rerun view-change from published kinematics"
  else
    run_phase_pool precompute
    echo "[contact-batch] contact/SAM phase complete; switching to tracking + view-change"
  fi
  run_phase_pool tracking
else
  run_phase_pool session
fi

read -r completed failed skipped < <(
  awk -F '\t' 'NR > 1 {
    if ($2 == "completed") completed++
    if ($2 == "skipped_complete" || $2 == "skipped_missing_kinematics") skipped++
    if ($3 != 0) failed++
  } END {print completed + 0, failed + 0, skipped + 0}' "${STATUS_FILE}"
)

echo "[contact-batch] completed=${completed} skipped=${skipped} failed=${failed} status=${STATUS_FILE}"
[[ "${failed}" -eq 0 ]]
