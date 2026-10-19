#!/usr/bin/env bash
set -euo pipefail

SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPTS}/../.." && pwd)"
ROOT="${ROOT:-${PROJECT_ROOT}}"
SESSION_ID="${SESSION_ID:-session_20260804_162223_iphone14_3}"
SOURCE_ROOT="${SOURCE_ROOT:-${ROOT}/ego2humanoid/egocentric_data/pick_up_the_cup_smooth}"
KIN_ROOT="${KIN_ROOT:-${ROOT}/ego2humanoid/data/kinematics/pick_up_the_cup_smooth}"
KINEMATICS_NPZ="${KINEMATICS_NPZ:-${KIN_ROOT}/${SESSION_ID}/kinematics.npz}"
TARGET_TIMESTAMPS_NPZ="${TARGET_TIMESTAMPS_NPZ:-${KINEMATICS_NPZ}}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/ego2humanoid/data/experiments/hand_retarget}"
PYTHON="${PYTHON:-${KODA_PYTHON:-python3}}"

ARGS=(
  --session-dir "${SOURCE_ROOT}/${SESSION_ID}"
  --kinematics-npz "${KINEMATICS_NPZ}"
  --target-timestamps-npz "${TARGET_TIMESTAMPS_NPZ}"
  --out-dir "${OUT_ROOT}/${SESSION_ID}"
  --flexion-scale "${FLEXION_SCALE:-0.7}"
  --scaling-factor "${SCALING_FACTOR:-1.0}"
  --low-pass-alpha "${LOW_PASS_ALPHA:-0.2}"
  --wrist-methods "${WRIST_METHODS:-historical_tip_wrist,mcp_frame_wrist,mcp_mid_palm_site,mcp_arc_palm_site,palm_polygon_site,template_palm_kabsch,projected_hull_centroid,mano_palm_surface_centroid,calibrated_mcp_center,root_mid_index_ring,root_mid_index_little,kabsch_index_ring,kabsch_index_little}"
  --finger-methods "${FINGER_METHODS:-dexpilot,vector,position,dexpilot_close}"
)

if [[ -n "${MAX_NATIVE_FRAMES:-}" ]]; then
  ARGS+=(--max-native-frames "${MAX_NATIVE_FRAMES}")
fi
if [[ "${TEMPLATE_PALM_FROM_KINEMATICS:-0}" == "1" ]]; then
  ARGS+=(--template-palm-from-current-kinematics)
fi

"${PYTHON}" "${SCRIPTS}/hand_retarget_experiment.py" "${ARGS[@]}"
