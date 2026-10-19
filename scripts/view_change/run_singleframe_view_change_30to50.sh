#!/usr/bin/env bash
set -Eeuo pipefail

# Offline single-frame RGB-D view change.  The source clock is 30 Hz and the
# final clock is the measured 50 Hz robot timeline; no frame is warped twice and
# no historical background is fused.  Source RGB-D registration keeps each
# session's source calibration, while every target-view stage shares one fixed
# target calibration.  The tracking archive is the sole source of both the
# 29-DoF body and the contact-aware 14-DoF Dex3 hand.

GPU="${GPU:-1}"
FORCE="${FORCE:-0}"
SEED="${SEED:-42}"
# Pose-convention preflight is off. Re-enable with GEOMETRY_GATE_MODE=strict|fallback.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESOURCE_ROOT="${PROJECT_ROOT}/resources"
ASSET_ROOT="${KODA_ASSETS:-${PROJECT_ROOT}/assets}"
# When set, reuse an accepted run's masks, warp and timeline, and rebuild only
# kinematics/render/composite/DiffuEraser/final outputs in OUTPUT_ROOT.
REUSE_BASE_ROOT="${REUSE_BASE_ROOT:-}"
# Stage-level cleanup for long offline batch runs.  Standalone/debug runs keep
# the historical behavior unless explicitly enabled by the caller.
PRUNE_INTERMEDIATES="${PRUNE_INTERMEDIATES:-0}"
# composite: encode processed_30hz/eraser/composite_trim.mp4 and exit before DiffuEraser.
STOP_AFTER="${STOP_AFTER:-}"
ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/data/view_change/single_session}"
SESSION="${SESSION:-${PROJECT_ROOT}/data/egocentric_data/pick_up_the_cup_smooth/session_20260804_155314_iphone14_3}"
TARGET_KINEMATICS="${TARGET_KINEMATICS:-${PROJECT_ROOT}/data/kinematics/pick_up_the_cup_smooth/$(basename "${SESSION}")/kinematics.npz}"
TARGET_REFERENCE="${TARGET_REFERENCE:-${TARGET_KINEMATICS}}"
SOURCE_RGB_K="${SOURCE_RGB_K:-${SESSION}/calibrations/rgb_K.npy}"
SOURCE_RGB_DIR="${SOURCE_RGB_DIR:-}"
TARGET_CAMERA_CALIBRATION="${TARGET_CAMERA_CALIBRATION:-${PROJECT_ROOT}/scripts/view_change/configs/target_camera_672x376.json}"
SCRIPTS="${PROJECT_ROOT}/scripts/view_change"
KODA_SRC="${PROJECT_ROOT}/src"
# dex3 renders the visual Dex3 hands. g1_brainco renders the BrainCo hands from
# the same 29-DoF body, so only the hand model and its mesh root change here.
HAND_MODE="${HAND_MODE:-dex3}"
# The 29-DoF body XML always resolves its meshes here, independently of which
# hand model the render stage compiles.
BODY_MESH="${RESOURCE_ROOT}/ScaleBFM/scalebridge/data/robot/g1_29dof/meshes"
case "${HAND_MODE}" in
  dex3)
    DEX3_XML="${RESOURCE_ROOT}/robots/g1_dex3/g1_with_visual_dex3_hands.xml"
    DEX3_MESH="${BODY_MESH}"
    TRACKING_HAND_DOF=14
    RENDER_HAND_DOF=14
    RENDER_DOF=43
    ;;
  g1_brainco)
    DEX3_XML="${RESOURCE_ROOT}/brainco/robots/g1_brainco/g1_mocap_29dof_with_brainco_hands.xml"
    # The BrainCo MJCF references its meshes as ``meshes/<name>.STL``, so the
    # compiler meshdir is the directory that contains ``meshes``.
    DEX3_MESH="${RESOURCE_ROOT}/brainco/robots/g1_brainco"
    TRACKING_HAND_DOF=12
    # Five distal joints per hand are driven by equality constraints.
    RENDER_HAND_DOF=22
    RENDER_DOF=51
    ;;
  *)
    echo "HAND_MODE must be dex3 or g1_brainco; got ${HAND_MODE}" >&2
    exit 2
    ;;
esac
VIEW_PY="${VIEW_PY:-${KODA_PYTHON:-${PYTHON:-python3}}}"
DIFFUERASER_PY="${DIFFUERASER_PY:-${KODA_PYTHON:-${PYTHON:-python3}}}"
ROBOT_PY="${ROBOT_PY:-${KODA_PYTHON:-${PYTHON:-python3}}}"
RESIDENT_WORKERS="${SCRIPTS}/resident_gpu_workers.sh"
LOG="${ROOT}/run.log"

ensure_resident_view_workers() {
  [[ -n "${LOG_ROOT:-}" ]] || return 0
  [[ -f "${RESIDENT_WORKERS}" ]] || return 0
  # shellcheck source=resident_gpu_workers.sh
  . "${RESIDENT_WORKERS}"
  ensure_view_change_workers "${GPU}"
}

for required in \
  "${SESSION}/rgb.mp4" \
  "${SESSION}/annotation.hdf5" \
  "${SOURCE_RGB_K}" \
  "${TARGET_CAMERA_CALIBRATION}" \
  "${TARGET_KINEMATICS}" \
  "${TARGET_REFERENCE}"; do
  [[ -f "${required}" ]] || {
    echo "missing required pipeline input: ${required}" >&2
    exit 2
  }
done

read -r CONFIG_TARGET_WIDTH CONFIG_TARGET_HEIGHT < <("${VIEW_PY}" -c '
import json, sys
meta = json.load(open(sys.argv[1], encoding="utf-8"))
print(int(meta["width"]), int(meta["height"]))
' "${TARGET_CAMERA_CALIBRATION}")
TARGET_WIDTH="${TARGET_WIDTH:-${CONFIG_TARGET_WIDTH}}"
TARGET_HEIGHT="${TARGET_HEIGHT:-${CONFIG_TARGET_HEIGHT}}"
if [[ "${TARGET_WIDTH}" != "${CONFIG_TARGET_WIDTH}" || \
      "${TARGET_HEIGHT}" != "${CONFIG_TARGET_HEIGHT}" ]]; then
  echo "target dimensions must match ${TARGET_CAMERA_CALIBRATION}: expected ${CONFIG_TARGET_WIDTH}x${CONFIG_TARGET_HEIGHT}, got ${TARGET_WIDTH}x${TARGET_HEIGHT}" >&2
  exit 2
fi

AVAILABLE_SOURCE_COUNT="$("${VIEW_PY}" -c '
import h5py, sys
with h5py.File(sys.argv[1], "r") as archive:
    print(int(archive["cam-pose/timestamps"].shape[0]))
' "${SESSION}/annotation.hdf5")"
AVAILABLE_TARGET_COUNT="$("${VIEW_PY}" -c '
import numpy as np, sys
with np.load(sys.argv[1], allow_pickle=False) as archive:
    print(int(archive["joint_pos"].shape[0]))
' "${TARGET_KINEMATICS}")"
if [[ ! "${AVAILABLE_SOURCE_COUNT}" =~ ^[0-9]+$ || "${AVAILABLE_SOURCE_COUNT}" -le 0 ]]; then
  echo "invalid source frame count: ${AVAILABLE_SOURCE_COUNT}" >&2
  exit 2
fi
if [[ ! "${AVAILABLE_TARGET_COUNT}" =~ ^[0-9]+$ || "${AVAILABLE_TARGET_COUNT}" -le 0 ]]; then
  echo "invalid target frame count: ${AVAILABLE_TARGET_COUNT}" >&2
  exit 2
fi

SOURCE_COUNT="${MAX_SOURCE_FRAMES:-${AVAILABLE_SOURCE_COUNT}}"
if [[ ! "${SOURCE_COUNT}" =~ ^[0-9]+$ ]] || \
   (( SOURCE_COUNT <= 0 || SOURCE_COUNT > AVAILABLE_SOURCE_COUNT )); then
  echo "SOURCE_COUNT must be in 1..${AVAILABLE_SOURCE_COUNT}" >&2
  exit 2
fi
if [[ -z "${MAX_TARGET_FRAMES:-}" ]]; then
  if [[ "${SOURCE_COUNT}" -eq "${AVAILABLE_SOURCE_COUNT}" ]]; then
    TARGET_COUNT="${AVAILABLE_TARGET_COUNT}"
  else
    # For a smoke prefix, retain exactly the measured 50 Hz samples covered by
    # the selected source interval. The rendering/mapping policy stays the same.
    TARGET_COUNT="$("${VIEW_PY}" -c '
import h5py, numpy as np, sys
source_path, target_path, source_count = sys.argv[1], sys.argv[2], int(sys.argv[3])
with h5py.File(source_path, "r") as archive:
    source = np.asarray(archive["cam-pose/timestamps"][:source_count], dtype=np.float64)
with np.load(target_path, allow_pickle=False) as archive:
    target = np.asarray(archive["timestamps"], dtype=np.float64)
source = source - source[0]
target = target - target[0]
print(int(np.searchsorted(target, source[-1] + 1e-9, side="right")))
' "${SESSION}/annotation.hdf5" "${TARGET_KINEMATICS}" "${SOURCE_COUNT}")"
    [[ "${TARGET_COUNT}" -lt 1 ]] && TARGET_COUNT=1
  fi
else
  TARGET_COUNT="${MAX_TARGET_FRAMES}"
fi
if [[ ! "${TARGET_COUNT}" =~ ^[0-9]+$ ]] || \
   (( TARGET_COUNT <= 0 || TARGET_COUNT > AVAILABLE_TARGET_COUNT )); then
  echo "TARGET_COUNT must be in 1..${AVAILABLE_TARGET_COUNT}" >&2
  exit 2
fi
mkdir -p "${ROOT}"
exec > >(tee -a "${LOG}") 2>&1
echo "[run] root=${ROOT} GPU=${GPU} source=${SOURCE_COUNT} target=${TARGET_COUNT} force=${FORCE} prune_intermediates=${PRUNE_INTERMEDIATES} stop_after=${STOP_AFTER:-none}"
echo "[run] session=${SESSION}"
echo "[run] tracking_kinematics=${TARGET_KINEMATICS}"
echo "[run] hand_source=tracking_kinematics.hand_qpos/hand_qvel"
echo "[run] hand_mode=${HAND_MODE} tracking_hand_dof=${TRACKING_HAND_DOF} render_dof=${RENDER_DOF}"
echo "[run] source_rgb_K=${SOURCE_RGB_K}"
[[ -n "${SOURCE_RGB_DIR}" ]] && echo "[run] source_rgb_dir=${SOURCE_RGB_DIR}"
echo "[run] target_camera=${TARGET_CAMERA_CALIBRATION} ${TARGET_WIDTH}x${TARGET_HEIGHT}"
[[ -z "${REUSE_BASE_ROOT}" ]] || echo "[run] reuse_base_root=${REUSE_BASE_ROOT}"

ADAPTER="${ROOT}/source_adapter"
MASKS="${ROOT}/human_masks"
VIEW_SOURCE_RGB="${ROOT}/source_rgb"
ROBOT="${ROOT}/target_robot"
DEX3_ROOT="${ROOT}/dex3_kinematics"
HAND_FROM_TRACKING="${DEX3_ROOT}/hand_from_tracking_50hz.npz"
DEX3_KINEMATICS="${DEX3_ROOT}/kinematics.npz"
WARP="${ROOT}/warp_30hz"
TIMELINE="${ROOT}/timeline_manifest.json"
PROCESSED="${ROOT}/processed_30hz"
FINAL="${ROOT}/final_50hz"

stage_marker() { printf '%s/.stage_%s_%s.done' "${ROOT}" "$1" "${SOURCE_COUNT}"; }

encode_composite_trim() {
  local rgb_dir="${PROCESSED}/rgb_geometry"
  local out_mp4="${PROCESSED}/eraser/composite_trim.mp4"
  mkdir -p "${PROCESSED}/eraser"
  echo "[run] encoding composite_trim ${out_mp4}"
  PYTHONPATH="${SCRIPTS}${PYTHONPATH:+:${PYTHONPATH}}" "${VIEW_PY}" - "${rgb_dir}" "${out_mp4}" <<'PY'
import sys
from pathlib import Path

import cv2

from frame_volume import as_rgb, open_volume

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
vol = open_volume(src, color_space="rgb")
n = len(vol)
if n <= 0:
    raise SystemExit(f"empty rgb_geometry volume: {src}")
frame0 = as_rgb(vol[0])
height, width = frame0.shape[:2]
dst.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(
    str(dst),
    cv2.VideoWriter_fourcc(*"mp4v"),
    30.0,
    (width, height),
)
if not writer.isOpened():
    raise SystemExit(f"failed to open VideoWriter for {dst}")
try:
    for index in range(n):
        bgr = cv2.cvtColor(as_rgb(vol[index]), cv2.COLOR_RGB2BGR)
        writer.write(bgr)
finally:
    writer.release()
if not dst.is_file() or dst.stat().st_size <= 0:
    raise SystemExit(f"composite_trim encode produced no file: {dst}")
print(f"wrote {dst} frames={n} size={dst.stat().st_size}")
PY
}

stop_after_if_requested() {
  local stage="$1"
  [[ "${STOP_AFTER}" == "${stage}" ]] || return 0
  encode_composite_trim
  echo "[run] STOP_AFTER=${STOP_AFTER}; skipping later stages"
  echo "[run] composite_trim: ${PROCESSED}/eraser/composite_trim.mp4"
  exit 0
}

if [[ "${PRUNE_INTERMEDIATES}" != "0" && "${PRUNE_INTERMEDIATES}" != "1" ]]; then
  echo "PRUNE_INTERMEDIATES must be 0 or 1; got ${PRUNE_INTERMEDIATES}" >&2
  exit 2
fi

# Remove only explicitly named children of this run.  The realpath guard is
# intentional: a malformed ROOT must never turn a cleanup into a workspace
# delete.  Stage markers for deleted inputs are removed so a later retry
# recomputes those inputs instead of skipping into a missing directory.
prune_path() {
  [[ "${PRUNE_INTERMEDIATES}" == "1" ]] || return 0
  local path="$1" root_abs path_abs
  root_abs="$(realpath -m -- "${ROOT}")"
  path_abs="$(realpath -m -- "${path}")"
  [[ "${path_abs}" == "${root_abs}"/* ]] || {
    echo "refusing to prune outside ROOT: ${path_abs}" >&2
    return 2
  }
  if [[ -e "${path_abs}" || -L "${path_abs}" ]]; then
    rm -rf -- "${path_abs}"
    echo "[run] pruned ${path_abs}"
  fi
}

prune_dir_except() {
  [[ "${PRUNE_INTERMEDIATES}" == "1" ]] || return 0
  local dir="$1" keep="$2" root_abs dir_abs
  root_abs="$(realpath -m -- "${ROOT}")"
  dir_abs="$(realpath -m -- "${dir}")"
  [[ "${dir_abs}" == "${root_abs}"/* ]] || {
    echo "refusing to prune outside ROOT: ${dir_abs}" >&2
    return 2
  }
  [[ -d "${dir_abs}" ]] || return 0
  find "${dir_abs}" -mindepth 1 -maxdepth 1 ! -name "${keep}" -exec rm -rf -- {} +
  echo "[run] pruned ${dir_abs} except ${keep}"
}

prune_after_warp() {
  [[ "${PRUNE_INTERMEDIATES}" == "1" ]] || return 0
  prune_path "${ADAPTER}"
  prune_path "${MASKS}"
  rm -f -- "$(stage_marker adapter)" "$(stage_marker human_masks)"
}

prune_after_composite() {
  [[ "${PRUNE_INTERMEDIATES}" == "1" ]] || return 0
  # Composite has consumed the per-frame MuJoCo streams.  Keep the complete
  # kinematics archive for the contact-aware wrapper's published sidecar.
  for name in rgb depth robot_id robot_instance.json robot_rgb.mp4; do
    prune_path "${ROBOT}/${name}"
  done
  rm -f -- "$(stage_marker target_robot)"
}

prune_after_assemble() {
  [[ "${PRUNE_INTERMEDIATES}" == "1" ]] || return 0
  # The assembled processed bundle is the only input needed by DiffuEraser;
  # retain warp quality/frame status for quality and structural verification.
  for name in rgb_geometry depth valid hole confidence source_count provenance; do
    prune_path "${WARP}/${name}"
  done
  prune_path "${WARP}/timeline_render_manifest.json"
  prune_path "${PROCESSED}/composite_debug"
  prune_path "${PROCESSED}/rgb_warp"
}

prune_after_diffueraser() {
  [[ "${PRUNE_INTERMEDIATES}" == "1" ]] || return 0
  # Keep the small status contract; native DiffuEraser videos/prior consume
  # many GB and are not read by later stages.
  [[ -s "${PROCESSED}/eraser/diffueraser_status.json" ]] || {
    echo "refusing to prune DiffuEraser work without a status contract" >&2
    return 2
  }
  prune_dir_except "${PROCESSED}/eraser" "diffueraser_status.json"
}

prune_after_resample() {
  [[ "${PRUNE_INTERMEDIATES}" == "1" ]] || return 0
  prune_path "${ROOT}/source_rgb"
}

run_stage() {
  local name="$1"; shift
  local marker start_s elapsed
  marker="$(stage_marker "${name}")"
  if [[ "${FORCE}" != "1" && -f "${marker}" ]]; then
    echo "[run] skip ${name} (marker exists)"
    return
  fi
  start_s="$(date +%s)"
  echo "[run] stage ${name}"
  "$@"
  elapsed="$(( $(date +%s) - start_s ))"
  echo "[run] stage ${name} elapsed_s=${elapsed}"
  date -u +%FT%TZ > "${marker}"
}

compose_tracking_render_kinematics() {
  mkdir -p "${DEX3_ROOT}"
  "${VIEW_PY}" "${SCRIPTS}/extract_hand_archive.py" \
    --full-kinematics "${TARGET_KINEMATICS}" \
    --output "${HAND_FROM_TRACKING}" \
    --hand-dof "${TRACKING_HAND_DOF}" \
    --require-hand-qvel
  PYTHONPATH="${KODA_SRC}${PYTHONPATH:+:${PYTHONPATH}}" "${ROBOT_PY}" \
    -m koda.kinematics.compose \
    --body-kinematics "${TARGET_KINEMATICS}" \
    --hand-kinematics "${HAND_FROM_TRACKING}" \
    --output "${DEX3_KINEMATICS}" \
    --body-model "${RESOURCE_ROOT}/ScaleBFM/scalebridge/data/robot/g1_29dof/g1_29dof_mjx.xml" \
    --dex3-model "${DEX3_XML}" \
    --meshdir "${BODY_MESH}" \
    --dex3-meshdir "${DEX3_MESH}" \
    --hand-mode "${HAND_MODE}" \
    --expected-hand-joints "${TRACKING_HAND_DOF}"
  "${VIEW_PY}" "${SCRIPTS}/verify_tracking_render_kinematics.py" \
    --tracking "${TARGET_KINEMATICS}" \
    --render "${DEX3_KINEMATICS}" \
    --hand-dof "${TRACKING_HAND_DOF}" \
    --render-dof "${RENDER_DOF}"
}

packed_volume_ready() {
  local root="$1"
  [[ -e "${root}/volume_meta.json" || -e "${root}/frames.npy" ]]
}

human_masks_ready() {
  local root="$1"
  [[ -s "${root}/human_mask_metadata.json" ]] || return 1
  packed_volume_ready "${root}/human_mask" || return 1
  # body: require torso/leg masks. hands: keep the arm-and-hand mask only.
  if [[ "${MASK_SCOPE:-body}" == "hands" ]]; then
    grep -q '"mask_kind": "human_hands_and_arms_only"' "${root}/human_mask_metadata.json"
  else
    grep -q '"mask_kind": "human_body"' "${root}/human_mask_metadata.json"
  fi
}

adapter_cache_ready() {
  local root="$1"
  [[ -s "${root}/camera.h5" && -s "${root}/alignment_manifest.json" ]] || return 1
  packed_volume_ready "${root}/depth_m_rgb"
}

publish_tree() {
  local src="$1" dst="$2" src_abs dst_abs
  src_abs="$(realpath -m -- "${src}")"
  dst_abs="$(realpath -m -- "${dst}")"
  [[ "${src_abs}" != "${dst_abs}" ]] || return 0
  rm -rf -- "${dst_abs}"
  mkdir -p "${dst_abs}"
  cp -al -- "${src_abs}/." "${dst_abs}/" 2>/dev/null || cp -a -- "${src_abs}/." "${dst_abs}/"
}

ensure_view_source_rgb() {
  local durable="${SOURCE_RGB_DIR:-}"
  local pack=(
    "${VIEW_PY}" "${SCRIPTS}/packed_source_rgb.py"
    --video "${SESSION}/rgb.mp4"
    --max-frames "${SOURCE_COUNT}"
  )
  [[ "${FORCE}" == "1" ]] && pack+=(--force)
  if [[ -n "${durable}" ]]; then
    mkdir -p "${durable}"
    "${pack[@]}" --out-dir "${durable}"
    "${VIEW_PY}" "${SCRIPTS}/packed_source_rgb.py" \
      --video "${SESSION}/rgb.mp4" \
      --out-dir "${VIEW_SOURCE_RGB}" \
      --reuse-from "${durable}" \
      --max-frames "${SOURCE_COUNT}"
  else
    "${pack[@]}" --out-dir "${VIEW_SOURCE_RGB}"
  fi
}

run_adapter() {
  local durable="${SOURCE_ADAPTER_DIR:-}"
  local args=(
    env CUDA_VISIBLE_DEVICES="${GPU}" "${VIEW_PY}" "${SCRIPTS}/stera_episode_to_pipeline_inputs.py"
    --episode "${SESSION}"
    --rgb-k "${SOURCE_RGB_K}"
    --source-rgb-dir "${VIEW_SOURCE_RGB}"
    --depth-aligned-to-rgb
    --rgb-frame-locked-to-camera-pose
    --strict-calibration
    --max-frames "${SOURCE_COUNT}"
  )
  if [[ -n "${durable}" ]]; then
    if [[ "${FORCE}" == "1" ]] || ! adapter_cache_ready "${durable}"; then
      mkdir -p "${durable}"
      "${args[@]}" --out-dir "${durable}"
    else
      echo "[adapter] cache hit ${durable}"
    fi
    publish_tree "${durable}" "${ADAPTER}"
  else
    "${args[@]}" --out-dir "${ADAPTER}"
  fi
}

run_human_masks() {
  local durable="${REUSE_HUMAN_MASKS_DIR:-}"
  local args=(
    env CUDA_VISIBLE_DEVICES="${GPU}" "${VIEW_PY}" "${SCRIPTS}/make_human_arm_masks.py"
    --rgb-mp4 "${SESSION}/rgb.mp4"
    --source-rgb-dir "${VIEW_SOURCE_RGB}"
    --annotation-h5 "${SESSION}/annotation.hdf5"
    --max-frames "${SOURCE_COUNT}"
    --backend "${MASK_BACKEND:-efficientsam3}"
    --device cuda
  )
  if [[ "${MASK_SCOPE:-body}" == "hands" ]]; then
    args+=(--body-prompt "" --leg-prompt "")
  fi
  if [[ -n "${durable}" ]]; then
    if [[ "${FORCE}" == "1" ]] || ! human_masks_ready "${durable}"; then
      mkdir -p "${durable}"
      "${args[@]}" --out-dir "${durable}"
    else
      echo "[human_masks] cache hit ${durable}"
    fi
    "${args[@]}" --reuse-from "${durable}" --out-dir "${MASKS}"
  else
    "${args[@]}" --out-dir "${MASKS}"
  fi
}

if [[ -n "${REUSE_BASE_ROOT}" ]]; then
  BASE_ROOT="$(realpath "${REUSE_BASE_ROOT}")"
  WARP="${BASE_ROOT}/warp_30hz"

for required in \
    "${BASE_ROOT}/timeline_manifest.json" \
    "${BASE_ROOT}/processed_30hz/frame_status.jsonl" \
    "${WARP}/quality/geometry_gate.json" \
    "${WARP}/frame_status.jsonl"; do
    [[ -f "${required}" ]] || {
      echo "missing accepted baseline artifact: ${required}" >&2
      exit 2
  }
done

  reuse_marker() { printf '%s/.stage_%s.done' "${ROOT}" "$1"; }
  reuse_stage() {
    local name="$1"; shift
    local marker start_s elapsed
    marker="$(reuse_marker "${name}")"
    if [[ "${FORCE}" != "1" && -f "${marker}" ]]; then
      echo "[run] skip ${name} (reuse marker exists)"
      return
    fi
    start_s="$(date +%s)"
    echo "[run] stage ${name}"
    "$@"
    elapsed="$(( $(date +%s) - start_s ))"
    echo "[run] stage ${name} elapsed_s=${elapsed}"
    date -u +%FT%TZ > "${marker}"
  }

  # Geometry can be reused, but the robot archive is always rebuilt from the
  # current tracking file so body, wrists and contact-aware fingers stay paired.
  reuse_stage tracking_render_kinematics compose_tracking_render_kinematics

  reuse_stage target_robot env MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" \
    PYTHONPATH="${KODA_SRC}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${ROBOT_PY}" "${SCRIPTS}/render_torso_robot_rgb_depth.py" \
    --kinematics-npz "${DEX3_KINEMATICS}" \
    --reference-npz "${TARGET_REFERENCE}" \
    --camera-calibration-json "${TARGET_CAMERA_CALIBRATION}" \
    --output-dir "${ROBOT}" \
    --robot-xml "${DEX3_XML}" --meshdir "${DEX3_MESH}" \
    --hand-dof "${RENDER_HAND_DOF}" \
    --width "${TARGET_WIDTH}" --height "${TARGET_HEIGHT}" \
    --max-frames "${TARGET_COUNT}" \
    $([[ "${FORCE}" == "1" ]] && printf '%s' '--force' || true)

  reuse_stage target_robot_camera_contract "${VIEW_PY}" \
    "${SCRIPTS}/verify_target_camera_contract.py" \
    --expected "${TARGET_CAMERA_CALIBRATION}" \
    --actual "${ROBOT}/camera_K.json" \
    --rgb-dir "${ROBOT}/rgb" --depth-dir "${ROBOT}/depth" \
    --robot-id-dir "${ROBOT}/robot_id" \
    --warp-meta "${WARP}/warp_meta.json" \
    --warp-rgb-dir "${WARP}/rgb_geometry" \
    --warp-depth-dir "${WARP}/depth"

  link_file() {
    local source="$1" destination="$2"
    mkdir -p "$(dirname "${destination}")"
    [[ -e "${destination}" ]] || ln "${source}" "${destination}"
  }
  link_dir() {
    local source="$1" destination="$2"
    if [[ ! -d "${destination}" ]]; then
      mkdir -p "$(dirname "${destination}")"
      cp -al "${source}" "${destination}"
    fi
  }
  link_reused_bundle() {
    link_file "${BASE_ROOT}/timeline_manifest.json" "${TIMELINE}"
    for stream in depth valid hole confidence source_count rgb_warp; do
      link_dir "${BASE_ROOT}/processed_30hz/${stream}" "${PROCESSED}/${stream}"
    done
    link_dir "${PROCESSED}/composite_debug/robot_mask" "${PROCESSED}/robot_mask"
    link_dir "${PROCESSED}/composite_debug/paste_mask" "${PROCESSED}/paste_mask"
    link_dir "${PROCESSED}/composite_debug/provenance" "${PROCESSED}/provenance_geometry"
    link_file "${BASE_ROOT}/processed_30hz/frame_status.jsonl" "${PROCESSED}/frame_status.jsonl"
  }

  reuse_stage timeline link_file "${BASE_ROOT}/timeline_manifest.json" "${TIMELINE}"

  ensure_resident_view_workers

  reuse_stage composite env CUDA_VISIBLE_DEVICES="${GPU}" \
    DIFFUERASER_WORKER_SOCK="${DIFFUERASER_WORKER_SOCK:-}" \
    "${VIEW_PY}" \
    "${SCRIPTS}/composite_st_depth_overlay.py" \
    --final-rgb-dir "${WARP}/rgb_geometry" \
    --st-rgb-dir "${ROBOT}/rgb" \
    --st-depth-dir "${ROBOT}/depth" \
    --hole-masks "${WARP}/hole" \
    --bg-depth-dir "${WARP}/depth" \
    --robot-mask-dir "${ROBOT}/robot_id" \
    --robot-instance-json "${ROBOT}/robot_instance.json" \
    --timeline-manifest "${TIMELINE}" \
    --require-robot-id --paste-mode B --neighbor-radius 3 \
    --geometry-provenance-dir "${WARP}/provenance" \
    --rgb-out-dir "${PROCESSED}/rgb_geometry" \
    --dump-dir "${PROCESSED}/composite_debug" \
    --fps 30 --device cuda

  reuse_stage link_bundle link_reused_bundle
  stop_after_if_requested composite

  reuse_stage diffueraser env CUDA_VISIBLE_DEVICES="${GPU}" \
    DIFFUERASER_WORKER_SOCK="${DIFFUERASER_WORKER_SOCK:-}" \
    "${VIEW_PY}" "${SCRIPTS}/run_remaining_hole_eraser.py" \
    --composite-rgb-dir "${PROCESSED}/rgb_geometry" \
    --hole-dir "${PROCESSED}/hole" \
    --choice-dir "${PROCESSED}/paste_mask" \
    --geometry-provenance-dir "${PROCESSED}/provenance_geometry" \
    --work-dir "${PROCESSED}/eraser" \
    --synthetic-dir "${PROCESSED}/synthetic" \
    --provenance-dir "${PROCESSED}/provenance" \
    --rgb-final-dir "${PROCESSED}/rgb_final" \
    --fps 30 --inpaint-mode visual_only --require-diffueraser \
    --seed "${SEED}" \
    --propainter-model-dir "${ASSET_ROOT}/models/diffueraser/propainter" \
    --max-img-size "${DIFFUERASER_MAX_IMG_SIZE:-768}" \
    --diffueraser-ckpt "${DIFFUERASER_CKPT:-Normal CFG 8-Step}" \
    --diffueraser-python "${DIFFUERASER_PY}" \
    --diffueraser-script "${RESOURCE_ROOT}/DiffuEraser/run_diffueraser.py"

  reuse_stage quality "${VIEW_PY}" "${SCRIPTS}/quality_report.py" \
    --frame-metrics "${WARP}/quality/frame_metrics.jsonl" \
    --geometry-gate "${WARP}/quality/geometry_gate.json" \
    --valid-dir "${PROCESSED}/valid" \
    --provenance-dir "${PROCESSED}/provenance" \
    --out "${ROOT}/quality/episode_quality.json" \
    --calibration-status complete

  reuse_stage resample "${VIEW_PY}" "${SCRIPTS}/resample_processed_30to50.py" \
    --processed-dir "${PROCESSED}" \
    --timeline-manifest "${TIMELINE}" \
    --output-dir "${FINAL}" \
    --streams rgb_geometry depth valid hole confidence source_count robot_mask paste_mask rgb_final synthetic provenance \
    --expected-source-frames "${SOURCE_COUNT}" \
    --expected-output-frames "${TARGET_COUNT}" \
    --fps 50 --width "${TARGET_WIDTH}" --height "${TARGET_HEIGHT}" --codec mp4v \
    --skip-50hz-volumes \
    $([[ "${FORCE}" == "1" ]] && printf '%s' '--force' || true)

  reuse_stage verify "${VIEW_PY}" "${SCRIPTS}/verify_singleframe_outputs.py" \
    --processed-30hz "${PROCESSED}" \
    --final-50hz "${FINAL}" \
    --timeline "${TIMELINE}" \
    --warp-status "${WARP}/frame_status.jsonl" \
    --diffueraser-status "${PROCESSED}/eraser/diffueraser_status.json" \
    --expected-source "${SOURCE_COUNT}" --expected-target "${TARGET_COUNT}" \
    --target-calibration "${TARGET_CAMERA_CALIBRATION}" \
    --actual-target-calibration "${ROBOT}/camera_K.json" \
    --warp-meta "${WARP}/warp_meta.json" \
    --out "${ROOT}/quality/structural_acceptance.json"

  echo "[run] completed reused single-frame 30->50 view change"
  echo "[run] final video: ${FINAL}/rgb_final_50hz.mp4"
  exit 0
fi

ensure_view_source_rgb

run_stage adapter run_adapter
run_stage human_masks run_human_masks

run_stage tracking_render_kinematics compose_tracking_render_kinematics

# Re-render RGB, metric depth, object IDs and target camera poses from the new
# three-point torso tracking result.  One camera instance and one K are shared
# by robot rendering and novel-view warping.
run_stage target_robot env MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" \
  PYTHONPATH="${KODA_SRC}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${ROBOT_PY}" \
  "${SCRIPTS}/render_torso_robot_rgb_depth.py" \
  --kinematics-npz "${DEX3_KINEMATICS}" \
  --reference-npz "${TARGET_REFERENCE}" \
  --camera-calibration-json "${TARGET_CAMERA_CALIBRATION}" \
  --output-dir "${ROBOT}" \
  --robot-xml "${DEX3_XML}" --meshdir "${DEX3_MESH}" \
  --hand-dof "${RENDER_HAND_DOF}" \
  --width "${TARGET_WIDTH}" --height "${TARGET_HEIGHT}" \
  --max-frames "${TARGET_COUNT}" \
  $([[ "${FORCE}" == "1" ]] && printf '%s' '--force' || true)

run_stage target_robot_camera_contract "${VIEW_PY}" \
  "${SCRIPTS}/verify_target_camera_contract.py" \
  --expected "${TARGET_CAMERA_CALIBRATION}" \
  --actual "${ROBOT}/camera_K.json" \
  --rgb-dir "${ROBOT}/rgb" --depth-dir "${ROBOT}/depth" \
  --robot-id-dir "${ROBOT}/robot_id"

run_stage timeline "${VIEW_PY}" "${SCRIPTS}/timeline_alignment.py" \
  --annotation-hdf5 "${SESSION}/annotation.hdf5" \
  --target-kinematics-npz "${ROBOT}/kinematics.npz" \
  --out "${TIMELINE}" \
  --max-source-frames "${SOURCE_COUNT}" \
  --max-target-frames "${TARGET_COUNT}" \
  --expected-source-frames "${SOURCE_COUNT}" \
  --expected-target-frames "${TARGET_COUNT}"

warp_cmd=(
  "${VIEW_PY}" "${SCRIPTS}/viewport_gt_depth_warp.py"
  --video_file "${SESSION}/rgb.mp4"
  --source-rgb-dir "${VIEW_SOURCE_RGB}"
  --h5_file "${ADAPTER}/camera.h5"
  --gt_depth_dir "${ADAPTER}/depth_m_rgb"
  --target_poses_npz "${ROBOT}/kinematics.npz"
  --target_intrinsics_json "${ROBOT}/camera_K.json"
  --target-width "${TARGET_WIDTH}" --target-height "${TARGET_HEIGHT}"
  --target-time-match nearest --target-time-domain relative
  --alignment-manifest "${ADAPTER}/alignment_manifest.json"
  --timeline-manifest "${WARP}/timeline_render_manifest.json"
  --geometry-gate-report "${WARP}/quality/geometry_gate.json"
  --gate-mode "${GEOMETRY_GATE_MODE:-off}"
  --human-mask-dir "${MASKS}/human_mask"
  --depth-confidence-dir "${ADAPTER}/depth_confidence"
  --depth-edge-invalid-dir "${ADAPTER}/depth_edge_invalid"
  --fusion-backend single --device cuda --fps 30
  --max_frames "${SOURCE_COUNT}" --output_dir "${WARP}"
)
if [[ "${FORCE}" == "1" ]]; then
  # The renderer overwrites frame maps; its output directory is intentionally
  # retained so a failed run remains inspectable.
  true
fi
if [[ "${FORCE}" == "1" || ! -f "$(stage_marker warp)" ]]; then
  echo "[run] stage warp"
  start_s="$(date +%s)"
  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" "${warp_cmd[@]}"
  warp_rc=$?
  set -e
  elapsed="$(( $(date +%s) - start_s ))"
  echo "[run] stage warp elapsed_s=${elapsed}"
  if [[ "${warp_rc}" -ne 0 ]]; then
    "${VIEW_PY}" "${SCRIPTS}/write_diagnostic_passthrough.py" \
      --video "${SESSION}/rgb.mp4" --out-dir "${ROOT}" --count "${SOURCE_COUNT}" \
      --reason "geometry_gate_or_warp_failed_rc_${warp_rc}"
    echo "[run] formal view-change stopped; diagnostic passthrough written" >&2
    exit "${warp_rc}"
  fi
  date -u +%FT%TZ > "$(stage_marker warp)"
else
  echo "[run] skip warp (marker exists)"
fi
run_stage warp_camera_contract "${VIEW_PY}" \
  "${SCRIPTS}/verify_target_camera_contract.py" \
  --expected "${TARGET_CAMERA_CALIBRATION}" \
  --actual "${ROBOT}/camera_K.json" \
  --rgb-dir "${ROBOT}/rgb" --depth-dir "${ROBOT}/depth" \
  --robot-id-dir "${ROBOT}/robot_id" \
  --warp-meta "${WARP}/warp_meta.json" \
  --warp-rgb-dir "${WARP}/rgb_geometry" \
  --warp-depth-dir "${WARP}/depth"
prune_after_warp

ensure_resident_view_workers

run_stage composite env CUDA_VISIBLE_DEVICES="${GPU}" \
  DIFFUERASER_WORKER_SOCK="${DIFFUERASER_WORKER_SOCK:-}" \
  "${VIEW_PY}" "${SCRIPTS}/composite_st_depth_overlay.py" \
  --final-rgb-dir "${WARP}/rgb_geometry" \
  --st-rgb-dir "${ROBOT}/rgb" \
  --st-depth-dir "${ROBOT}/depth" \
  --hole-masks "${WARP}/hole" \
  --bg-depth-dir "${WARP}/depth" \
  --robot-mask-dir "${ROBOT}/robot_id" \
  --robot-instance-json "${ROBOT}/robot_instance.json" \
  --timeline-manifest "${TIMELINE}" \
  --require-robot-id \
  --paste-mode B --neighbor-radius 3 \
  --geometry-provenance-dir "${WARP}/provenance" \
  --rgb-out-dir "${PROCESSED}/rgb_geometry" \
  --dump-dir "${PROCESSED}/composite_debug" \
  --fps 30 --device cuda
prune_after_composite

run_stage assemble "${VIEW_PY}" "${SCRIPTS}/assemble_singleframe_bundle.py" \
  --warp-dir "${WARP}" \
  --composite-debug-dir "${PROCESSED}/composite_debug" \
  --out-dir "${PROCESSED}" \
  --count "${SOURCE_COUNT}" \
  $([[ "${FORCE}" == "1" ]] && printf '%s' '--force' || true)
prune_after_assemble
stop_after_if_requested composite

run_stage diffueraser env CUDA_VISIBLE_DEVICES="${GPU}" \
  DIFFUERASER_WORKER_SOCK="${DIFFUERASER_WORKER_SOCK:-}" \
  "${VIEW_PY}" "${SCRIPTS}/run_remaining_hole_eraser.py" \
  --composite-rgb-dir "${PROCESSED}/rgb_geometry" \
  --hole-dir "${PROCESSED}/hole" \
  --choice-dir "${PROCESSED}/paste_mask" \
  --geometry-provenance-dir "${PROCESSED}/provenance_geometry" \
  --work-dir "${PROCESSED}/eraser" \
  --synthetic-dir "${PROCESSED}/synthetic" \
  --provenance-dir "${PROCESSED}/provenance" \
  --rgb-final-dir "${PROCESSED}/rgb_final" \
  --fps 30 --inpaint-mode visual_only --require-diffueraser \
  --seed "${SEED}" --propainter-model-dir "${ASSET_ROOT}/models/diffueraser/propainter" \
  --max-img-size "${DIFFUERASER_MAX_IMG_SIZE:-768}" \
  --diffueraser-ckpt "${DIFFUERASER_CKPT:-Normal CFG 8-Step}" \
  --diffueraser-python "${DIFFUERASER_PY}" \
  --diffueraser-script "${RESOURCE_ROOT}/DiffuEraser/run_diffueraser.py"
prune_after_diffueraser

run_stage quality "${VIEW_PY}" "${SCRIPTS}/quality_report.py" \
  --frame-metrics "${WARP}/quality/frame_metrics.jsonl" \
  --geometry-gate "${WARP}/quality/geometry_gate.json" \
  --valid-dir "${PROCESSED}/valid" \
  --provenance-dir "${PROCESSED}/provenance" \
  --out "${ROOT}/quality/episode_quality.json" \
  --calibration-status complete

run_stage resample "${VIEW_PY}" "${SCRIPTS}/resample_processed_30to50.py" \
  --processed-dir "${PROCESSED}" \
  --timeline-manifest "${TIMELINE}" \
  --output-dir "${FINAL}" \
  --streams rgb_geometry depth valid hole confidence source_count robot_mask paste_mask rgb_final synthetic provenance \
  --expected-source-frames "${SOURCE_COUNT}" \
  --expected-output-frames "${TARGET_COUNT}" \
  --fps 50 --width "${TARGET_WIDTH}" --height "${TARGET_HEIGHT}" --codec mp4v \
  --skip-50hz-volumes \
  $([[ "${FORCE}" == "1" ]] && printf '%s' '--force' || true)
prune_after_resample

run_stage verify "${VIEW_PY}" "${SCRIPTS}/verify_singleframe_outputs.py" \
  --processed-30hz "${PROCESSED}" \
  --final-50hz "${FINAL}" \
  --timeline "${TIMELINE}" \
  --warp-status "${WARP}/frame_status.jsonl" \
  --diffueraser-status "${PROCESSED}/eraser/diffueraser_status.json" \
  --expected-source "${SOURCE_COUNT}" \
  --expected-target "${TARGET_COUNT}" \
  --target-calibration "${TARGET_CAMERA_CALIBRATION}" \
  --actual-target-calibration "${ROBOT}/camera_K.json" \
  --warp-meta "${WARP}/warp_meta.json" \
  --out "${ROOT}/quality/structural_acceptance.json"

# Publish the synchronized body-plus-hand tracking archive next to the final
# RGB video, preserving tracking timestamps, contacts, and provenance.
cp --reflink=auto "${TARGET_KINEMATICS}" "${FINAL}/kinematics.npz"

echo "[run] completed single-frame 30->50 view change"
echo "[run] final video: ${FINAL}/rgb_final_50hz.mp4"
echo "[run] final kinematics: ${FINAL}/kinematics.npz"
