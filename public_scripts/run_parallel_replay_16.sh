#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${SCRIPT_DIR}/replay_runs"
DEFAULT_PYTHON_BIN="/data/hongzefu/maniskillenv1114/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
TARGET_SCRIPT="${SCRIPT_DIR}/evaluate_jointangle_dataset_replay.py"

ENV_IDS=(
  PickXtimes
  StopCube
  SwingXtimes
  BinFill
  VideoUnmaskSwap
  VideoUnmask
  ButtonUnmaskSwap
  ButtonUnmask
  VideoRepick
  VideoPlaceButton
  VideoPlaceOrder
  PickHighlight
  InsertPeg
  MoveCube
  PatternLock
  RouteStick
)

usage() {
  cat <<'EOF'
Usage:
  PYTHON_BIN=/path/to/python run_parallel_replay_16.sh start [run_name] [extra_python_args...]
  run_parallel_replay_16.sh start [run_name] [extra_python_args...]
  run_parallel_replay_16.sh monitor [run_name]
  run_parallel_replay_16.sh status [run_name]
  run_parallel_replay_16.sh stop [run_name]
  run_parallel_replay_16.sh list

Examples:
  ./run_parallel_replay_16.sh start
  PYTHON_BIN=/data/hongzefu/maniskillenv1114/bin/python ./run_parallel_replay_16.sh start
  ./run_parallel_replay_16.sh start my_run
  ./run_parallel_replay_16.sh monitor my_run
  ./run_parallel_replay_16.sh status my_run
  ./run_parallel_replay_16.sh stop my_run
EOF
}

latest_run() {
  if [[ ! -d "${RUN_ROOT}" ]]; then
    return 1
  fi
  ls -1dt "${RUN_ROOT}"/* 2>/dev/null | head -n 1
}

resolve_run_dir() {
  local name="${1:-}"
  if [[ -n "${name}" ]]; then
    echo "${RUN_ROOT}/${name}"
    return 0
  fi
  latest_run
}

cmd_start() {
  local run_name="${1:-}"
  if [[ -n "${run_name}" && "${run_name}" == --* ]]; then
    run_name=""
  else
    shift || true
  fi

  if [[ ! -f "${TARGET_SCRIPT}" ]]; then
    echo "Script not found: ${TARGET_SCRIPT}" >&2
    exit 1
  fi
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python interpreter not executable: ${PYTHON_BIN}" >&2
    exit 1
  fi

  if [[ -z "${run_name}" ]]; then
    run_name="jointangle_replay_$(date +%Y%m%d_%H%M%S)"
  fi

  local run_dir="${RUN_ROOT}/${run_name}"
  local log_dir="${run_dir}/logs"
  local pid_dir="${run_dir}/pids"

  mkdir -p "${log_dir}" "${pid_dir}"

  echo "mode=jointangle_replay" > "${run_dir}/meta.txt"
  echo "script=${TARGET_SCRIPT}" >> "${run_dir}/meta.txt"
  echo "python_bin=${PYTHON_BIN}" >> "${run_dir}/meta.txt"
  echo "started_at=$(date -Iseconds)" >> "${run_dir}/meta.txt"
  echo "run_name=${run_name}" >> "${run_dir}/meta.txt"

  echo "Starting 16 parallel workers..."
  echo "Python: ${PYTHON_BIN}"
  for env_id in "${ENV_IDS[@]}"; do
    local log_file="${log_dir}/${env_id}.log"
    local pid_file="${pid_dir}/${env_id}.pid"

    # -u + PYTHONUNBUFFERED=1 保证日志尽可能实时刷新。
    nohup env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u "${TARGET_SCRIPT}" --env-id "${env_id}" "$@" \
      > "${log_file}" 2>&1 < /dev/null &
    echo "$!" > "${pid_file}"
    echo "  started ${env_id} pid=$(cat "${pid_file}") log=${log_file}"
  done

  echo
  echo "Run directory: ${run_dir}"
  echo "Monitor: ./run_parallel_replay_16.sh monitor ${run_name}"
  echo "Status : ./run_parallel_replay_16.sh status ${run_name}"
  echo "Stop   : ./run_parallel_replay_16.sh stop ${run_name}"
}

cmd_monitor() {
  local run_name="${1:-}"
  local run_dir
  run_dir="$(resolve_run_dir "${run_name}")"
  if [[ -z "${run_dir}" || ! -d "${run_dir}/logs" ]]; then
    echo "Run not found. Please specify run_name or start a run first." >&2
    exit 1
  fi
  echo "Monitoring logs under: ${run_dir}/logs"
  tail -n 60 -F "${run_dir}/logs/"*.log
}

cmd_status() {
  local run_name="${1:-}"
  local run_dir
  run_dir="$(resolve_run_dir "${run_name}")"
  if [[ -z "${run_dir}" || ! -d "${run_dir}/pids" ]]; then
    echo "Run not found. Please specify run_name or start a run first." >&2
    exit 1
  fi

  local total=0
  local alive=0
  for pid_file in "${run_dir}/pids/"*.pid; do
    [[ -e "${pid_file}" ]] || continue
    total=$((total + 1))
    local env_id
    env_id="$(basename "${pid_file}" .pid)"
    local pid
    pid="$(cat "${pid_file}")"
    if kill -0 "${pid}" 2>/dev/null; then
      alive=$((alive + 1))
      echo "[RUNNING] ${env_id} pid=${pid}"
    else
      echo "[EXITED ] ${env_id} pid=${pid}"
    fi
  done
  echo "Summary: ${alive}/${total} running"
}

cmd_stop() {
  local run_name="${1:-}"
  local run_dir
  run_dir="$(resolve_run_dir "${run_name}")"
  if [[ -z "${run_dir}" || ! -d "${run_dir}/pids" ]]; then
    echo "Run not found. Please specify run_name or start a run first." >&2
    exit 1
  fi

  for pid_file in "${run_dir}/pids/"*.pid; do
    [[ -e "${pid_file}" ]] || continue
    local env_id
    env_id="$(basename "${pid_file}" .pid)"
    local pid
    pid="$(cat "${pid_file}")"
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" || true
      echo "Stopped ${env_id} pid=${pid}"
    else
      echo "Already exited ${env_id} pid=${pid}"
    fi
  done
}

cmd_list() {
  if [[ ! -d "${RUN_ROOT}" ]]; then
    echo "No runs yet."
    exit 0
  fi
  ls -1dt "${RUN_ROOT}"/* 2>/dev/null || true
}

main() {
  local action="${1:-help}"
  shift || true

  case "${action}" in
    start) cmd_start "$@" ;;
    monitor) cmd_monitor "$@" ;;
    status) cmd_status "$@" ;;
    stop) cmd_stop "$@" ;;
    list) cmd_list ;;
    help|-h|--help) usage ;;
    *)
      echo "Unknown action: ${action}" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
