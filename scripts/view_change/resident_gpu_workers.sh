#!/usr/bin/env bash
# Shared GPU Unix-socket workers for contact-aware batch / view-change.
# Source this file; do not execute it. Callers must set PROJECT_ROOT and LOG_ROOT.

wait_for_worker_socket() {
  local gpu="$1" pid="$2" sock="$3" log="$4" name="$5"
  local waited=0
  echo "[gpu-workers gpu=${gpu}] starting ${name} pid=${pid} sock=${sock}" >&2
  while (( waited < 900 )); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[gpu-workers gpu=${gpu}] ${name} exited; log=${log}" >&2
      tail -n 80 "${log}" >&2 || true
      return 2
    fi
    if [[ -S "${sock}" ]]; then
      echo "[gpu-workers gpu=${gpu}] ${name} ready pid=${pid}" >&2
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
    if (( waited % 15 == 0 )); then
      echo "[gpu-workers gpu=${gpu}] waiting for ${name} (${waited}s)" >&2
    fi
  done
  echo "[gpu-workers gpu=${gpu}] timed out waiting for ${name}; log=${log}" >&2
  kill "${pid}" 2>/dev/null || true
  return 2
}

stop_pid_sock() {
  local pid="$1" sock="$2"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
  [[ -n "${sock}" && -e "${sock}" ]] && rm -f -- "${sock}"
}

_worker_paths() {
  local name="$1" gpu="$2"
  _WORKER_SOCK="${LOG_ROOT}/${name}_gpu_${gpu}.sock"
  _WORKER_LOG="${LOG_ROOT}/${name}_gpu_${gpu}.log"
  _WORKER_PIDFILE="${LOG_ROOT}/${name}_gpu_${gpu}.pid"
}

_read_pidfile() {
  local pidfile="$1"
  if [[ -f "${pidfile}" ]]; then
    tr -d '[:space:]' < "${pidfile}" || true
  fi
}

_worker_alive() {
  local pid="$1" sock="$2"
  [[ -n "${pid}" && -S "${sock}" ]] && kill -0 "${pid}" 2>/dev/null
}

stop_named_gpu_worker() {
  local name="$1" gpu="$2"
  [[ -n "${LOG_ROOT:-}" && -n "${gpu}" ]] || return 0
  _worker_paths "${name}" "${gpu}"
  local pid
  pid="$(_read_pidfile "${_WORKER_PIDFILE}")"
  stop_pid_sock "${pid}" "${_WORKER_SOCK}"
  rm -f -- "${_WORKER_PIDFILE}"
}

_launch_worker() {
  local gpu="$1" name="$2"
  shift 2
  _worker_paths "${name}" "${gpu}"
  local pid existing
  existing="$(_read_pidfile "${_WORKER_PIDFILE}")"
  if _worker_alive "${existing}" "${_WORKER_SOCK}"; then
    echo "[gpu-workers gpu=${gpu}] reuse ${name} pid=${existing}" >&2
    printf '%s\n' "${existing}"
    return 0
  fi
  rm -f -- "${_WORKER_SOCK}" "${_WORKER_PIDFILE}"
  mkdir -p "${LOG_ROOT}"
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" CUDA_DEVICE_ORDER=PCI_BUS_ID "$@" \
    > "${_WORKER_LOG}" 2>&1 &
  pid=$!
  disown "${pid}" 2>/dev/null || true
  if ! wait_for_worker_socket "${gpu}" "${pid}" "${_WORKER_SOCK}" "${_WORKER_LOG}" "${name}"; then
    return 2
  fi
  printf '%s\n' "${pid}" > "${_WORKER_PIDFILE}"
  printf '%s\n' "${pid}"
}

resident_worker_defaults() {
  HYDRAX_PY="${HYDRAX_PY:-${KODA_PYTHON:-${PYTHON:-python3}}}"
  JAX_MPC_WORKER="${JAX_MPC_WORKER:-${PROJECT_ROOT}/resources/hydrax/scripts/pi_mjx_sim2sim/diagnostics/jax_mpc_worker.py}"
  THREE_POINT_ARTIFACT_MANIFEST="${THREE_POINT_ARTIFACT_MANIFEST:-${PROJECT_ROOT}/resources/hydrax/scripts/pi_mjx_sim2sim/artifacts/artifact_manifest.json}"
  SAM_PY="${SAM_PY:-${KODA_PYTHON:-${PYTHON:-python3}}}"
  EFFICIENTSAM3_WORKER="${EFFICIENTSAM3_WORKER:-${PROJECT_ROOT}/scripts/view_change/efficientsam3_worker.py}"
  SCALER_PY="${SCALER_PY:-${KODA_PYTHON:-${PYTHON:-python3}}}"
  # Kimodo generation needs peft/LLM2Vec. That lives in envs/kimodo, not scalebfm.
  # Keep this aligned with DEFAULT_KIMODO_PYTHON in run_clean_kimodo_init_sweep.py.
  if [[ -z "${KIMODO_PY:-}" ]]; then
    KIMODO_PY="${KODA_PYTHON:-${PYTHON:-python3}}"
  fi
  KIMODO_WORKER="${KIMODO_WORKER:-${PROJECT_ROOT}/scripts/view_change/kimodo_init_worker.py}"
  KIMODO_CODE="${KIMODO_CODE:-${PROJECT_ROOT}/resources/kimodo}"
  KIMODO_SWEEP_DIR="${KIMODO_SWEEP_DIR:-${PROJECT_ROOT}/resources/ScaleBFM/ScaleTrack/scripts/eval_session}"
  DIFFUERASER_PY="${DIFFUERASER_PY:-${KODA_PYTHON:-${PYTHON:-python3}}}"
  DIFFUERASER_WORKER="${DIFFUERASER_WORKER:-${PROJECT_ROOT}/resources/DiffuEraser/diffueraser_worker.py}"
  PROPAINTER_WEIGHTS="${PROPAINTER_WEIGHTS:-${KODA_ASSETS:-${PROJECT_ROOT}/assets}/models/diffueraser/propainter}"
  WORK_ROOT="${WORK_ROOT:-${LOG_ROOT%/logs_*}}"
}

start_jax_mpc_worker() {
  local gpu="$1"
  resident_worker_defaults
  [[ -x "${HYDRAX_PY}" ]] || { echo "missing hydrax python: ${HYDRAX_PY}" >&2; return 2; }
  [[ -f "${JAX_MPC_WORKER}" ]] || { echo "missing jax mpc worker: ${JAX_MPC_WORKER}" >&2; return 2; }
  [[ -f "${THREE_POINT_ARTIFACT_MANIFEST}" ]] || {
    echo "missing artifact manifest: ${THREE_POINT_ARTIFACT_MANIFEST}" >&2
    return 2
  }
  mkdir -p "${WORK_ROOT}/jax_cache"
  local pid
  pid="$(_launch_worker "${gpu}" jax_mpc \
    env XLA_PYTHON_CLIENT_PREALLOCATE=false \
      JAX_COMPILATION_CACHE_DIR="${WORK_ROOT}/jax_cache" \
      "${HYDRAX_PY}" "${JAX_MPC_WORKER}" \
        --socket "${LOG_ROOT}/jax_mpc_gpu_${gpu}.sock" \
        --artifact-manifest "${THREE_POINT_ARTIFACT_MANIFEST}")" || return $?
  JAX_MPC_WORKER_PID="${pid}"
  JAX_MPC_WORKER_SOCK="${LOG_ROOT}/jax_mpc_gpu_${gpu}.sock"
  export JAX_MPC_WORKER_SOCK
}

stop_jax_mpc_worker() {
  local gpu="${1:-${GPU:-}}"
  stop_named_gpu_worker jax_mpc "${gpu}"
  JAX_MPC_WORKER_PID=""
  JAX_MPC_WORKER_SOCK=""
  unset JAX_MPC_WORKER_SOCK
}

start_efficientsam3_worker() {
  local gpu="$1"
  resident_worker_defaults
  local pid
  pid="$(_launch_worker "${gpu}" efficientsam3 \
    "${SAM_PY}" "${EFFICIENTSAM3_WORKER}" \
      --socket "${LOG_ROOT}/efficientsam3_gpu_${gpu}.sock" \
      --device cuda --confidence "${SAM_CONFIDENCE:-0.20}")" || return $?
  EFFICIENTSAM3_WORKER_PID="${pid}"
  EFFICIENTSAM3_WORKER_SOCK="${LOG_ROOT}/efficientsam3_gpu_${gpu}.sock"
  export EFFICIENTSAM3_WORKER_SOCK
}

stop_efficientsam3_worker() {
  local gpu="${1:-${GPU:-}}"
  stop_named_gpu_worker efficientsam3 "${gpu}"
  EFFICIENTSAM3_WORKER_PID=""
  EFFICIENTSAM3_WORKER_SOCK=""
  unset EFFICIENTSAM3_WORKER_SOCK
}

start_kimodo_worker() {
  local gpu="$1"
  resident_worker_defaults
  [[ -x "${KIMODO_PY}" ]] || { echo "missing kimodo python: ${KIMODO_PY}" >&2; return 2; }
  [[ -f "${KIMODO_WORKER}" ]] || { echo "missing kimodo worker: ${KIMODO_WORKER}" >&2; return 2; }
  local pid
  pid="$(_launch_worker "${gpu}" kimodo \
    env CHECKPOINT_DIR="${KODA_ASSETS:-${PROJECT_ROOT}/assets}/models/kimodo" \
      LOCAL_CACHE=True \
      TEXT_ENCODER_MODE=local \
      TEXT_ENCODERS_DIR="${KODA_ASSETS:-${PROJECT_ROOT}/assets}/models/kimodo/text_encoders" \
      TEXT_ENCODER_DEVICE=cpu \
      PYTHONPATH="${KIMODO_CODE}:${KIMODO_SWEEP_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${KIMODO_PY}" "${KIMODO_WORKER}" \
        --socket "${LOG_ROOT}/kimodo_gpu_${gpu}.sock" \
        --model-name Kimodo-G1-RP-v1)" || return $?
  KIMODO_WORKER_PID="${pid}"
  KIMODO_WORKER_SOCK="${LOG_ROOT}/kimodo_gpu_${gpu}.sock"
  export KIMODO_WORKER_SOCK
}

stop_kimodo_worker() {
  local gpu="${1:-${GPU:-}}"
  stop_named_gpu_worker kimodo "${gpu}"
  KIMODO_WORKER_PID=""
  KIMODO_WORKER_SOCK=""
  unset KIMODO_WORKER_SOCK
}

start_diffueraser_worker() {
  local gpu="$1"
  resident_worker_defaults
  local pid de_root="${KODA_ASSETS:-${PROJECT_ROOT}/assets}/models/diffueraser"
  pid="$(_launch_worker "${gpu}" diffueraser \
    "${DIFFUERASER_PY}" "${DIFFUERASER_WORKER}" \
      --socket "${LOG_ROOT}/diffueraser_gpu_${gpu}.sock" \
      --ckpt "Normal CFG 8-Step" \
      --base-model-path "${de_root}/weights/stable-diffusion-v1-5" \
      --vae-path "${de_root}/weights/sd-vae-ft-mse" \
      --diffueraser-path "${de_root}/weights/diffuEraser" \
      --propainter-model-dir "${PROPAINTER_WEIGHTS}")" || return $?
  DIFFUERASER_WORKER_PID="${pid}"
  DIFFUERASER_WORKER_SOCK="${LOG_ROOT}/diffueraser_gpu_${gpu}.sock"
  export DIFFUERASER_WORKER_SOCK
}

stop_diffueraser_worker() {
  local gpu="${1:-${GPU:-}}"
  stop_named_gpu_worker diffueraser "${gpu}"
  DIFFUERASER_WORKER_PID=""
  DIFFUERASER_WORKER_SOCK=""
  unset DIFFUERASER_WORKER_SOCK
}

ensure_view_change_workers() {
  local gpu="${1:-${GPU:-}}"
  [[ -n "${LOG_ROOT:-}" && -n "${gpu}" ]] || return 0
  resident_worker_defaults
  start_diffueraser_worker "${gpu}" || \
    echo "[gpu-workers gpu=${gpu}] DiffuEraser worker unavailable; falling back to per-session load" >&2
}

stop_view_change_workers() {
  local gpu="${1:-${GPU:-}}"
  stop_diffueraser_worker "${gpu}"
}

stop_all_gpu_workers() {
  local gpu="${1:-${GPU:-}}"
  stop_view_change_workers "${gpu}"
  stop_kimodo_worker "${gpu}"
  stop_jax_mpc_worker "${gpu}"
  stop_efficientsam3_worker "${gpu}"
}
