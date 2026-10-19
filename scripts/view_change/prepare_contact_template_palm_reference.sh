#!/usr/bin/env bash
set -Eeuo pipefail

# Prepare the contact bootstrap directly from raw annotation data.  This
# intentionally stops before Kimodo, ScaleTrack, and Mink: the resulting
# files are temporary reference/timeline inputs for contact precompute only.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${KODA_PYTHON:-${PYTHON:-python3}}"
PREPARE="${PROJECT_ROOT}/resources/ScaleBFM/ScaleTrack/scripts/eval_session/prepare_output_clean_raw_eval.py"
RETARGET="${PROJECT_ROOT}/scripts/view_change/retarget_template_palm_reference.py"

SESSION_DIR="${SESSION_DIR:?SESSION_DIR is required}"
OUTPUT_DIR="${REFERENCE_OUTPUT_DIR:?REFERENCE_OUTPUT_DIR is required}"
ANNOTATION="${SESSION_DIR}/annotation.hdf5"
BASE_DIR="${OUTPUT_DIR}/base_reference"
RAW_REFERENCE="${BASE_DIR}/raw_dual_wrist_reference.npz"
RAW_TARGETS="${BASE_DIR}/raw_dual_wrist_targets.npz"
PREPARE_META="${BASE_DIR}/prepare_meta.json"
REFERENCE="${OUTPUT_DIR}/template_palm_kabsch_reference.npz"
TARGETS="${OUTPUT_DIR}/template_palm_kabsch_targets.npz"
REFERENCE_META="${OUTPUT_DIR}/template_palm_kabsch_meta.json"
TIMELINE_KINEMATICS="${OUTPUT_DIR}/template_palm_reference_kinematics.npz"

for required in "${PYTHON}" "${PREPARE}" "${RETARGET}" "${ANNOTATION}"; do
  [[ -f "${required}" ]] || {
    echo "[contact-reference] missing required file: ${required}" >&2
    exit 2
  }
done

mkdir -p "${BASE_DIR}" "${OUTPUT_DIR}"

PREPARE_ARGS=(
  --episode "${SESSION_DIR}"
  --out-dir "${BASE_DIR}"
  --out-fps 50
  --use-full-source
  --repair-transient-wrist-outliers
)
if [[ -n "${MAX_GAP_FRAMES:-}" ]]; then
  PREPARE_ARGS+=(--max-gap-frames "${MAX_GAP_FRAMES}")
fi

echo "[contact-reference] prepare raw dual-wrist reference at 50 Hz"
"${PYTHON}" "${PREPARE}" "${PREPARE_ARGS[@]}"

for required in "${RAW_REFERENCE}" "${RAW_TARGETS}" "${PREPARE_META}"; do
  [[ -s "${required}" ]] || {
    echo "[contact-reference] missing preparation output: ${required}" >&2
    exit 2
  }
done

echo "[contact-reference] retarget template-palm Kabsch reference"
"${PYTHON}" "${RETARGET}" \
  --annotation-hdf5 "${ANNOTATION}" \
  --base-reference "${RAW_REFERENCE}" \
  --base-targets "${RAW_TARGETS}" \
  --prepare-meta "${PREPARE_META}" \
  --output-reference "${REFERENCE}" \
  --output-targets "${TARGETS}" \
  --meta-json "${REFERENCE_META}"

echo "[contact-reference] derive temporary body/timeline archive for hand retarget"
"${PYTHON}" - "${REFERENCE}" "${TARGETS}" "${TIMELINE_KINEMATICS}" <<'PY'
from pathlib import Path
import sys

import numpy as np

reference_path, targets_path, output_path = map(Path, sys.argv[1:])
with np.load(reference_path, allow_pickle=False) as reference, np.load(
    targets_path, allow_pickle=False
) as targets:
    required = {"body_names", "body_pos_w", "body_quat_w"}
    missing = sorted(required - set(reference.files))
    if missing:
        raise SystemExit(f"template reference missing {missing}")
    if "timestamps" not in targets.files:
        raise SystemExit("template targets missing timestamps")
    timestamps = np.asarray(targets["timestamps"], dtype=np.float64)
    if timestamps.ndim != 1 or timestamps.size == 0 or not np.all(np.isfinite(timestamps)):
        raise SystemExit("invalid template target timestamps")
    if np.any(np.diff(timestamps) <= 0.0):
        raise SystemExit("template target timestamps are not strictly increasing")
    arrays = {
        key: np.asarray(reference[key])
        for key in ("body_names", "body_pos_w", "body_quat_w")
    }
    arrays["timestamps"] = timestamps
    arrays["fps"] = np.asarray(
        float(np.asarray(targets["fps_session"]).reshape(()))
        if "fps_session" in targets.files
        else 50.0,
        dtype=np.float64,
    )
    if arrays["body_pos_w"].shape[0] != timestamps.size:
        raise SystemExit("template body/timeline frame mismatch")
output_path.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(output_path, **arrays)
print(f"[contact-reference] wrote {output_path} frames={timestamps.size}")
PY

for required in "${REFERENCE}" "${TARGETS}" "${REFERENCE_META}" "${TIMELINE_KINEMATICS}"; do
  [[ -s "${required}" ]] || {
    echo "[contact-reference] missing final bootstrap output: ${required}" >&2
    exit 2
  }
done

echo "[contact-reference] complete reference=${REFERENCE} targets=${TARGETS}"
