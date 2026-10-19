#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-1}"
FRAMES="${FRAMES:-30}"
SESSION_ID="${SESSION_ID:-session_smoke_template_palm_position}"
SOURCE_SESSION="${SOURCE_SESSION:-${PROJECT_ROOT}/data/egocentric_data/pick_up_the_cup/session_20260804_155314_iphone14_3}"
SMOKE_ROOT="${SMOKE_ROOT:-${PROJECT_ROOT}/data/smoke}"
INPUT_ROOT="${SMOKE_ROOT}/egocentric_data"
KINEMATICS_ROOT="${SMOKE_ROOT}/kinematics"
VIEW_ROOT="${SMOKE_ROOT}/view_change"
WORK_ROOT="${SMOKE_ROOT}/work"
LOG_ROOT="${SMOKE_ROOT}/logs"
PYTHON="${KODA_PYTHON:-${PYTHON:-python3}}"

mkdir -p "$INPUT_ROOT" "$KINEMATICS_ROOT" "$VIEW_ROOT" "$WORK_ROOT" "$LOG_ROOT"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/full_pipeline_smoke.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[smoke] project=$PROJECT_ROOT gpu=$GPU frames=$FRAMES session=$SESSION_ID"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits | awk -F, -v gpu="$GPU" '
    $1 + 0 == gpu + 0 {
      free = $3 - $2
      if (free < 18000) {
        printf "GPU %s has only %d MiB free\n", gpu, free > "/dev/stderr"
        exit 2
      }
      found = 1
    }
    END { if (!found) exit 2 }
  '

creator_args=(
  --source-session "$SOURCE_SESSION"
  --output-root "$INPUT_ROOT"
  --session-id "$SESSION_ID"
  --frames "$FRAMES"
)
if [[ "${FORCE_SMOKE_INPUT:-0}" == "1" ]]; then
  creator_args+=(--force)
fi
"$PYTHON" "$PROJECT_ROOT/scripts/create_smoke_session.py" "${creator_args[@]}"

INPUT_ROOT="$INPUT_ROOT" \
OUTPUT_ROOT="$KINEMATICS_ROOT" \
WORK_ROOT="$WORK_ROOT/tracking" \
GPUS="$GPU" \
WORKERS=1 \
LIMIT=1 \
ALLOW_SMOOTHING_REVIEW=1 \
FORCE="${FORCE:-0}" \
"$PROJECT_ROOT/pipeline_smooth_motion.txt"

SOURCE_ROOT="$INPUT_ROOT" \
KINEMATICS_ROOT="$KINEMATICS_ROOT" \
BATCH_OUTPUT_ROOT="$VIEW_ROOT" \
GPU="$GPU" \
SESSION_ID="$SESSION_ID" \
MAX_EPISODES=1 \
MAX_SOURCE_FRAMES="$FRAMES" \
DIFFUERASER_MAX_IMG_SIZE="${DIFFUERASER_MAX_IMG_SIZE:-512}" \
FORCE="${FORCE:-0}" \
"$PROJECT_ROOT/run_batch_singleframe_view_change_30to50.sh"

"$PYTHON" "$PROJECT_ROOT/scripts/verify_full_pipeline_smoke.py" \
  --session-id "$SESSION_ID" \
  --kinematics-root "$KINEMATICS_ROOT" \
  --view-root "$VIEW_ROOT" \
  --output "$SMOKE_ROOT/smoke_acceptance.json"

echo "[smoke] completed report=$SMOKE_ROOT/smoke_acceptance.json"
