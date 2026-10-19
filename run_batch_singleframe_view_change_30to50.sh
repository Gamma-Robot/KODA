#!/usr/bin/env bash
set -Eeuo pipefail

# Batch wrapper for run_singleframe_view_change_30to50.sh. Each source session
# is paired with the same-named 50 Hz body-kinematics directory and processed
# independently. The underlying per-session geometry/render/inpaint stages are
# intentionally unchanged. Both the 29-DoF body and contact-aware 14-DoF Dex3
# hand are read from the paired tracking kinematics; view change never retargets
# the hand and never applies an additional wrist offset.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/data/egocentric_data/pick_up_the_cup_smooth}"
KINEMATICS_ROOT="${KINEMATICS_ROOT:-${PROJECT_ROOT}/data/kinematics/pick_up_the_cup_smooth}"
BATCH_OUTPUT_ROOT="${BATCH_OUTPUT_ROOT:-${PROJECT_ROOT}/data/view_change/pick_up_the_cup_smooth}"
PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-${PROJECT_ROOT}/scripts/view_change/run_singleframe_view_change_30to50.sh}"
VALIDATOR="${PROJECT_ROOT}/scripts/validate_pipeline_layout.py"
TARGET_CAMERA_CALIBRATION="${TARGET_CAMERA_CALIBRATION:-${PROJECT_ROOT}/scripts/view_change/configs/target_camera_672x376.json}"

"${KODA_PYTHON:-${PYTHON:-python3}}" "${VALIDATOR}" >/dev/null

GPU="${GPU:-1}"
FORCE="${FORCE:-0}"
SEED="${SEED:-42}"
SESSION_ID="${SESSION_ID:-}"
MAX_EPISODES="${MAX_EPISODES:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
DRY_RUN="${DRY_RUN:-0}"

for directory in "${SOURCE_ROOT}" "${KINEMATICS_ROOT}"; do
  [[ -d "${directory}" ]] || {
    echo "missing batch input directory: ${directory}" >&2
    exit 2
  }
done
[[ -f "${TARGET_CAMERA_CALIBRATION}" ]] || {
  echo "missing target camera calibration: ${TARGET_CAMERA_CALIBRATION}" >&2
  exit 2
}
[[ -f "${PIPELINE_SCRIPT}" ]] || {
  echo "missing single-session pipeline: ${PIPELINE_SCRIPT}" >&2
  exit 2
}
for integer_setting in MAX_EPISODES FORCE CONTINUE_ON_ERROR DRY_RUN; do
  value="${!integer_setting}"
  [[ "${value}" =~ ^[0-9]+$ ]] || {
    echo "${integer_setting} must be a non-negative integer, got ${value}" >&2
    exit 2
  }
done
if [[ -n "${SESSION_ID}" ]]; then
  [[ "${SESSION_ID}" =~ ^session_[A-Za-z0-9_]+$ ]] || {
    echo "invalid SESSION_ID: ${SESSION_ID}" >&2
    exit 2
  }
  sessions=("${SOURCE_ROOT}/${SESSION_ID}")
else
  mapfile -d '' sessions < <(
    find "${SOURCE_ROOT}" -mindepth 1 -maxdepth 1 -type d \
      -name 'session_*' -print0 | sort -z
  )
fi
[[ "${#sessions[@]}" -gt 0 ]] || {
  echo "no source sessions selected under ${SOURCE_ROOT}" >&2
  exit 2
}

selected=()
input_errors=0
for session in "${sessions[@]}"; do
  session_id="$(basename "${session}")"
  kinematics_dir="${KINEMATICS_ROOT}/${session_id}"
  for required in \
    "${session}/rgb.mp4" \
    "${session}/annotation.hdf5" \
    "${session}/calibrations/rgb_K.npy" \
    "${kinematics_dir}/kinematics.npz" \
    "${kinematics_dir}/meta.json"; do
    if [[ ! -f "${required}" ]]; then
      echo "missing paired input for ${session_id}: ${required}" >&2
      input_errors=$((input_errors + 1))
    fi
  done
  if [[ -f "${kinematics_dir}/kinematics.npz" ]] && ! \
    "${KODA_PYTHON:-${PYTHON:-python3}}" -c '
import sys
import numpy as np
path = sys.argv[1]
with np.load(path, allow_pickle=False) as z:
    required = {"joint_pos", "joint_vel", "joint_names", "timestamps", "hand_qpos", "hand_qvel", "hand_joint_names"}
    missing = sorted(required - set(z.files))
    if missing:
        raise SystemExit(f"missing fields {missing}")
    body = np.asarray(z["joint_pos"])
    body_vel = np.asarray(z["joint_vel"])
    hand = np.asarray(z["hand_qpos"])
    hand_vel = np.asarray(z["hand_qvel"])
    body_names = np.asarray(z["joint_names"]).reshape(-1)
    hand_names = np.asarray(z["hand_joint_names"]).reshape(-1)
    timestamps = np.asarray(z["timestamps"])
    if body.ndim != 2 or body.shape[1] != 29 or body_vel.shape != body.shape:
        raise SystemExit(f"invalid 29-DoF body contract pos={body.shape} vel={body_vel.shape}")
    if hand.shape != (body.shape[0], 14) or hand_vel.shape != hand.shape:
        raise SystemExit(f"invalid 14-DoF hand contract pos={hand.shape} vel={hand_vel.shape}")
    if body_names.shape != (29,) or hand_names.shape != (14,) or timestamps.shape != (body.shape[0],):
        raise SystemExit("invalid body/hand names or timestamp contract")
    if not all(np.all(np.isfinite(v)) for v in (body, body_vel, hand, hand_vel, timestamps)):
        raise SystemExit("tracking kinematics contains non-finite values")
' "${kinematics_dir}/kinematics.npz"; then
    echo "invalid tracking body+hand contract for ${session_id}: ${kinematics_dir}/kinematics.npz" >&2
    input_errors=$((input_errors + 1))
  fi
  selected+=("${session_id}")
  if [[ "${MAX_EPISODES}" -gt 0 && "${#selected[@]}" -ge "${MAX_EPISODES}" ]]; then
    break
  fi
done
[[ "${input_errors}" -eq 0 ]] || {
  echo "batch preflight failed with ${input_errors} missing inputs" >&2
  exit 2
}

echo "[batch] source_root=${SOURCE_ROOT}"
echo "[batch] kinematics_root=${KINEMATICS_ROOT}"
echo "[batch] output_root=${BATCH_OUTPUT_ROOT}"
echo "[batch] episodes=${#selected[@]} GPU=${GPU} force=${FORCE} dry_run=${DRY_RUN}"
echo "[batch] robot_state_source=tracking_kinematics_29dof_body_plus_14dof_hand"
echo "[batch] target_camera=${TARGET_CAMERA_CALIBRATION}"

if [[ "${DRY_RUN}" == "1" ]]; then
  for session_id in "${selected[@]}"; do
    echo "[batch] pair ${session_id}: ${SOURCE_ROOT}/${session_id} -> ${KINEMATICS_ROOT}/${session_id}/kinematics.npz"
  done
  echo "[batch] dry-run complete; no episode was processed"
  exit 0
fi

mkdir -p "${BATCH_OUTPUT_ROOT}/_batch_logs"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
status_file="${BATCH_OUTPUT_ROOT}/_batch_logs/status_${run_stamp}.tsv"
printf 'session_id\tstatus\texit_code\toutput_root\n' > "${status_file}"

completed=0
failed=0
for session_id in "${selected[@]}"; do
  session="${SOURCE_ROOT}/${session_id}"
  kinematics="${KINEMATICS_ROOT}/${session_id}/kinematics.npz"
  episode_output="${BATCH_OUTPUT_ROOT}/${session_id}"
  echo "[batch] start ${session_id}"

  set +e
  env \
    GPU="${GPU}" \
    FORCE="${FORCE}" \
    SEED="${SEED}" \
    SESSION="${session}" \
    TARGET_KINEMATICS="${kinematics}" \
    TARGET_REFERENCE="${kinematics}" \
    SOURCE_RGB_K="${session}/calibrations/rgb_K.npy" \
    TARGET_CAMERA_CALIBRATION="${TARGET_CAMERA_CALIBRATION}" \
    OUTPUT_ROOT="${episode_output}" \
    REUSE_BASE_ROOT= \
    MAX_SOURCE_FRAMES="${MAX_SOURCE_FRAMES:-}" \
    MAX_TARGET_FRAMES="${MAX_TARGET_FRAMES:-}" \
    DIFFUERASER_MAX_IMG_SIZE="${DIFFUERASER_MAX_IMG_SIZE:-768}" \
    DIFFUERASER_CKPT="${DIFFUERASER_CKPT:-Normal CFG 8-Step}" \
    bash "${PIPELINE_SCRIPT}"
  exit_code=$?
  set -e

  if [[ "${exit_code}" -eq 0 ]]; then
    status="completed"
    completed=$((completed + 1))
  else
    status="failed"
    failed=$((failed + 1))
  fi
  printf '%s\t%s\t%s\t%s\n' \
    "${session_id}" "${status}" "${exit_code}" "${episode_output}" \
    >> "${status_file}"
  echo "[batch] ${status} ${session_id} rc=${exit_code}"

  if [[ "${exit_code}" -ne 0 && "${CONTINUE_ON_ERROR}" != "1" ]]; then
    break
  fi
done

echo "[batch] finished completed=${completed} failed=${failed} status=${status_file}"
[[ "${failed}" -eq 0 ]]
