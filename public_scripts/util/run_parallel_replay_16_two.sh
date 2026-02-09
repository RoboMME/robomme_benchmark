#!/usr/bin/env bash
# 用法:
# 1) 启动顺序回放(先 jointangle, 再 endeffector_fk):
#    ./run_parallel_replay_16_two.sh start [run_name] [extra_python_args...]
# 2) 查看状态:
#    ./run_parallel_replay_16_two.sh status [run_name]
# 3) 实时看日志:
#    ./run_parallel_replay_16_two.sh monitor [run_name]
# 4) 停止当前 run:
#    ./run_parallel_replay_16_two.sh stop [run_name]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${SCRIPT_DIR}/replay_runs"
DEFAULT_PYTHON_BIN="/data/hongzefu/maniskillenv1114/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"

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
  PYTHON_BIN=/path/to/python run_parallel_replay_16_two.sh start [run_name] [extra_python_args...]
  run_parallel_replay_16_two.sh start [run_name] [extra_python_args...]
  run_parallel_replay_16_two.sh monitor [run_name]
  run_parallel_replay_16_two.sh status [run_name]
  run_parallel_replay_16_two.sh stop [run_name]
  run_parallel_replay_16_two.sh list

start behavior:
  fixed order: jointangle -> endeffector_fk
  each mode runs 16 envs in parallel
  next mode starts immediately after previous mode fully finishes

Examples:
  ./run_parallel_replay_16_two.sh start
  PYTHON_BIN=/data/hongzefu/maniskillenv1114/bin/python ./run_parallel_replay_16_two.sh start
  ./run_parallel_replay_16_two.sh start my_run
EOF
}

resolve_script() {
    local mode="$1"
  case "$mode" in
    jointangle) echo "${SCRIPT_DIR}/evaluate_jointangle_dataset_replay.py" ;;
    endeffector_fk) echo "${SCRIPT_DIR}/evaluate_endeffector_FK.py" ;;
    *)
      echo "Unknown mode: ${mode}" >&2
      echo "Valid: jointangle | endeffector_fk" >&2
      exit 1
      ;;
  esac
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

  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python interpreter not executable: ${PYTHON_BIN}" >&2
    exit 1
  fi

  if [[ -z "${run_name}" ]]; then
    run_name="seq_jointangle_endeffector_fk_$(date +%Y%m%d_%H%M%S)"
  fi

  local run_dir="${RUN_ROOT}/${run_name}"
  mkdir -p "${run_dir}"

  echo "modes=jointangle,endeffector_fk" > "${run_dir}/meta.txt"
  echo "python_bin=${PYTHON_BIN}" >> "${run_dir}/meta.txt"
  echo "started_at=$(date -Iseconds)" >> "${run_dir}/meta.txt"
  echo "run_name=${run_name}" >> "${run_dir}/meta.txt"

  local modes=(jointangle endeffector_fk)
  local mode
  for mode in "${modes[@]}"; do
    local py_script
    py_script="$(resolve_script "${mode}")"
    if [[ ! -f "${py_script}" ]]; then
      echo "Script not found: ${py_script}" >&2
      exit 1
    fi

    local mode_dir="${run_dir}/${mode}"
    local log_dir="${mode_dir}/logs"
    local pid_dir="${mode_dir}/pids"
    mkdir -p "${log_dir}" "${pid_dir}"

    echo "mode=${mode}" > "${mode_dir}/meta.txt"
    echo "script=${py_script}" >> "${mode_dir}/meta.txt"
    echo "started_at=$(date -Iseconds)" >> "${mode_dir}/meta.txt"

    echo
    echo "Starting mode=${mode} with 16 parallel workers..."
    echo "Python: ${PYTHON_BIN}"
    for env_id in "${ENV_IDS[@]}"; do
      local log_file="${log_dir}/${env_id}.log"
      local pid_file="${pid_dir}/${env_id}.pid"

      # -u + PYTHONUNBUFFERED=1 保证日志尽可能实时刷新。
      env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u "${py_script}" --env-id "${env_id}" "$@" \
        > "${log_file}" 2>&1 &
      echo "$!" > "${pid_file}"
      echo "  started ${mode}/${env_id} pid=$(cat "${pid_file}") log=${log_file}"
    done

    local failed=0
    for pid_file in "${pid_dir}/"*.pid; do
      [[ -e "${pid_file}" ]] || continue
      local pid
      pid="$(cat "${pid_file}")"
      if ! wait "${pid}"; then
        failed=1
      fi
    done

    if [[ "${failed}" -ne 0 ]]; then
      echo "Mode ${mode} finished with failures. Stop running next mode." >&2
      echo "finished_at=$(date -Iseconds)" >> "${mode_dir}/meta.txt"
      echo "status=failed" >> "${mode_dir}/meta.txt"
      exit 1
    fi

    echo "Mode ${mode} finished successfully."
    echo "finished_at=$(date -Iseconds)" >> "${mode_dir}/meta.txt"
    echo "status=success" >> "${mode_dir}/meta.txt"
  done

  echo "finished_at=$(date -Iseconds)" >> "${run_dir}/meta.txt"
  echo "status=success" >> "${run_dir}/meta.txt"
  echo
  echo "All modes finished successfully in order: jointangle -> endeffector_fk"
  echo "Run directory: ${run_dir}"
}

cmd_monitor() {
  local run_name="${1:-}"
  local run_dir
  run_dir="$(resolve_run_dir "${run_name}")"
  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "Run not found. Please specify run_name or start a run first." >&2
    exit 1
  fi

  local log_files=()
  local mode
  for mode in jointangle endeffector_fk; do
    local log_dir="${run_dir}/${mode}/logs"
    [[ -d "${log_dir}" ]] || continue
    local f
    for f in "${log_dir}/"*.log; do
      [[ -e "${f}" ]] || continue
      log_files+=("${f}")
    done
  done

  if [[ "${#log_files[@]}" -eq 0 ]]; then
    echo "No logs found under: ${run_dir}" >&2
    exit 1
  fi

  echo "Monitoring logs under: ${run_dir}/{jointangle,endeffector_fk}/logs"
  tail -n 60 -F "${log_files[@]}"
}

cmd_status() {
  local run_name="${1:-}"
  local run_dir
  run_dir="$(resolve_run_dir "${run_name}")"
  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "Run not found. Please specify run_name or start a run first." >&2
    exit 1
  fi

  local modes=(jointangle endeffector_fk)
  local mode
  for mode in "${modes[@]}"; do
    local pid_dir="${run_dir}/${mode}/pids"
    [[ -d "${pid_dir}" ]] || continue

    local total=0
    local alive=0
    echo "[${mode}]"
    for pid_file in "${pid_dir}/"*.pid; do
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
    echo "Summary(${mode}): ${alive}/${total} running"
  done
}

cmd_stop() {
  local run_name="${1:-}"
  local run_dir
  run_dir="$(resolve_run_dir "${run_name}")"
  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "Run not found. Please specify run_name or start a run first." >&2
    exit 1
  fi

  local modes=(jointangle endeffector_fk)
  local mode
  for mode in "${modes[@]}"; do
    local pid_dir="${run_dir}/${mode}/pids"
    [[ -d "${pid_dir}" ]] || continue

    for pid_file in "${pid_dir}/"*.pid; do
      [[ -e "${pid_file}" ]] || continue
      local env_id
      env_id="$(basename "${pid_file}" .pid)"
      local pid
      pid="$(cat "${pid_file}")"
      if kill -0 "${pid}" 2>/dev/null; then
        kill "${pid}" || true
        echo "Stopped ${mode}/${env_id} pid=${pid}"
      else
        echo "Already exited ${mode}/${env_id} pid=${pid}"
      fi
    done
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
