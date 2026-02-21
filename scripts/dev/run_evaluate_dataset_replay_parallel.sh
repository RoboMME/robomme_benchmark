#!/usr/bin/env bash
# run_evaluate_dataset_replay_parallel.sh
# Micromamba environment: /data/hongzefu/maniskillenv1114
#
# Usage examples:
# 1) Start parallel replay（default 16 envs, one process per env_id; auto-enter aggregated log monitor after start）
#    bash run_evaluate_dataset_replay_parallel.sh start
# 2) Start parallel replay（manually specify env_ids）
#    bash run_evaluate_dataset_replay_parallel.sh start --env_ids PickXtimes,StopCube
# 3) show only current active run status
#    bash run_evaluate_dataset_replay_parallel.sh status
# 4) reconnect log monitor (Ctrl+C exits monitor only, does not stop jobs)
#    bash run_evaluate_dataset_replay_parallel.sh monitor
# 5) stop all processes in the current active run
#    bash run_evaluate_dataset_replay_parallel.sh stop
# 6) restart (stop first, then start with new env_ids; defaults to 16 if omitted)
#    bash run_evaluate_dataset_replay_parallel.sh restart --env_ids PickXtimes,StopCube

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/evaluate_dataset_replay-parallel.py"
MICROMAMBA_ENV="/data/hongzefu/maniskillenv1114"
PYTHON_BIN="${MICROMAMBA_ENV}/bin/python"
DEFAULT_ENV_IDS_CSV="PickXtimes,StopCube,SwingXtimes,BinFill,VideoUnmaskSwap,VideoUnmask,ButtonUnmaskSwap,ButtonUnmask,VideoRepick,VideoPlaceButton,VideoPlaceOrder,PickHighlight,InsertPeg,MoveCube,PatternLock,RouteStick"

LOG_ROOT="${SCRIPT_DIR}/logs/evaluate_dataset_replay_parallel"
ACTIVE_RUN_FILE="${LOG_ROOT}/active_run"

mkdir -p "${LOG_ROOT}"

show_usage() {
    echo "Usage: $0 {start|monitor|status|stop|restart} [--env_ids A,B,C]"
    echo ""
    echo "Commands:"
    echo "  start   [--env_ids A,B,C] Start one process per envid and attach monitor."
    echo "                            If omitted, defaults to all 16 env ids."
    echo "  monitor                    Monitor logs of the current active run."
    echo "  status                     Show status of the current active run."
    echo "  stop                       Stop all processes in the current active run."
    echo "  restart [--env_ids A,B,C]  Stop current active run, then start a new run."
    echo "                            If omitted, defaults to all 16 env ids."
}

trim_whitespace() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf "%s" "${value}"
}

is_pid_alive() {
    local pid="$1"
    ps -p "${pid}" > /dev/null 2>&1
}

is_process_tree_alive() {
    local pid="$1"
    [ -z "${pid}" ] && return 1
    if is_pid_alive "${pid}"; then
        return 0
    fi
    ps -o pid= --ppid "${pid}" 2>/dev/null | grep -q .
}

kill_process_tree() {
    local pid="$1"
    local signal="${2:-15}"
    [ -z "${pid}" ] && return

    local children
    children=$(ps -o pid= --ppid "${pid}" 2>/dev/null)
    if [ -n "${children}" ]; then
        for child in ${children}; do
            kill_process_tree "${child}" "${signal}"
        done
    fi

    kill "-${signal}" "${pid}" 2>/dev/null || true
}

get_active_run_dir() {
    if [ ! -f "${ACTIVE_RUN_FILE}" ]; then
        return 1
    fi
    local run_dir
    run_dir="$(cat "${ACTIVE_RUN_FILE}")"
    if [ -z "${run_dir}" ] || [ ! -d "${run_dir}" ]; then
        return 1
    fi
    printf "%s\n" "${run_dir}"
}

parse_env_ids_csv() {
    local csv="$1"
    IFS=',' read -r -a raw_env_ids <<< "${csv}"
    ENV_IDS=()
    local env_id
    for env_id in "${raw_env_ids[@]}"; do
        env_id="$(trim_whitespace "${env_id}")"
        if [ -n "${env_id}" ]; then
            ENV_IDS+=("${env_id}")
        fi
    done
    if [ "${#ENV_IDS[@]}" -eq 0 ]; then
        return 1
    fi
}

extract_env_ids_arg() {
    local env_ids_csv=""
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --env_ids)
                if [ -z "${2:-}" ]; then
                    echo "Error: --env_ids requires a value."
                    return 1
                fi
                env_ids_csv="$2"
                shift 2
                ;;
            *)
                echo "Error: unknown argument '$1'."
                return 1
                ;;
        esac
    done

    if [ -z "${env_ids_csv}" ]; then
        env_ids_csv="${DEFAULT_ENV_IDS_CSV}"
        echo "Info: --env_ids not provided, using default 16 env ids." >&2
    fi

    printf "%s\n" "${env_ids_csv}"
}

validate_runtime() {
    if [ ! -f "${PYTHON_SCRIPT}" ]; then
        echo "Error: script not found: ${PYTHON_SCRIPT}"
        return 1
    fi
    if [ ! -d "${MICROMAMBA_ENV}" ]; then
        echo "Error: micromamba env not found: ${MICROMAMBA_ENV}"
        return 1
    fi
    if [ ! -x "${PYTHON_BIN}" ]; then
        echo "Error: python binary not executable: ${PYTHON_BIN}"
        return 1
    fi
}

run_has_alive_process() {
    local run_dir="$1"
    local pids_file="${run_dir}/pids.tsv"
    [ -f "${pids_file}" ] || return 1

    local env_id pid log_file
    while IFS=$'\t' read -r env_id pid log_file; do
        [ -z "${pid}" ] && continue
        if is_process_tree_alive "${pid}"; then
            return 0
        fi
    done < "${pids_file}"
    return 1
}

monitor_run() {
    local run_dir="${1:-}"
    if [ -z "${run_dir}" ]; then
        if ! run_dir="$(get_active_run_dir)"; then
            echo "No active run found."
            return 1
        fi
    fi

    local pids_file="${run_dir}/pids.tsv"
    if [ ! -s "${pids_file}" ]; then
        echo "No pids.tsv found for run: ${run_dir}"
        return 1
    fi

    local log_files=()
    local env_id pid log_file
    while IFS=$'\t' read -r env_id pid log_file; do
        [ -z "${log_file}" ] && continue
        log_files+=("${log_file}")
    done < "${pids_file}"

    if [ "${#log_files[@]}" -eq 0 ]; then
        echo "No log files registered in ${pids_file}"
        return 1
    fi

    echo "Monitoring run: ${run_dir}"
    echo "Press Ctrl+C to exit monitor. Processes keep running."
    tail -n 0 -F "${log_files[@]}"
}

status_run() {
    local run_dir
    if ! run_dir="$(get_active_run_dir)"; then
        echo "Status: no active run."
        return 0
    fi

    local pids_file="${run_dir}/pids.tsv"
    if [ ! -f "${pids_file}" ]; then
        echo "Status: active_run points to ${run_dir}, but pids.tsv is missing."
        return 1
    fi

    local total=0
    local alive=0
    local env_id pid log_file state
    echo "Active run: ${run_dir}"
    while IFS=$'\t' read -r env_id pid log_file; do
        [ -z "${pid}" ] && continue
        total=$((total + 1))
        if is_process_tree_alive "${pid}"; then
            state="RUNNING"
            alive=$((alive + 1))
        else
            state="EXITED"
        fi
        printf "  [%s] pid=%s state=%s log=%s\n" "${env_id}" "${pid}" "${state}" "${log_file}"
    done < "${pids_file}"

    echo "Summary: alive=${alive}/${total}"
}

stop_run() {
    local run_dir
    if ! run_dir="$(get_active_run_dir)"; then
        echo "No active run to stop."
        return 0
    fi

    local pids_file="${run_dir}/pids.tsv"
    if [ ! -f "${pids_file}" ]; then
        echo "pids.tsv missing for run ${run_dir}. Clearing active run pointer."
        rm -f "${ACTIVE_RUN_FILE}"
        return 0
    fi

    local pids=()
    local env_id pid log_file
    while IFS=$'\t' read -r env_id pid log_file; do
        [ -z "${pid}" ] && continue
        pids+=("${pid}")
    done < "${pids_file}"

    if [ "${#pids[@]}" -eq 0 ]; then
        echo "No PIDs recorded for run ${run_dir}."
        rm -f "${ACTIVE_RUN_FILE}"
        return 0
    fi

    echo "Stopping run: ${run_dir}"
    local p
    for p in "${pids[@]}"; do
        kill_process_tree "${p}" 15
    done

    local i has_alive
    for i in {1..15}; do
        has_alive=0
        for p in "${pids[@]}"; do
            if is_process_tree_alive "${p}"; then
                has_alive=1
                break
            fi
        done
        [ "${has_alive}" -eq 0 ] && break
        sleep 1
    done

    for p in "${pids[@]}"; do
        if is_process_tree_alive "${p}"; then
            kill_process_tree "${p}" 9
        fi
    done
    sleep 1

    local remaining=0
    for p in "${pids[@]}"; do
        if is_process_tree_alive "${p}"; then
            remaining=$((remaining + 1))
        fi
    done

    rm -f "${ACTIVE_RUN_FILE}"
    if [ "${remaining}" -eq 0 ]; then
        echo "Stop complete: all processes from active run have exited."
    else
        echo "Stop complete with warnings: ${remaining} process trees still alive."
        return 1
    fi
}

start_run() {
    local env_ids_csv="$1"

    if ! validate_runtime; then
        return 1
    fi
    if ! parse_env_ids_csv "${env_ids_csv}"; then
        echo "Error: --env_ids is empty after parsing."
        return 1
    fi

    local current_run
    if current_run="$(get_active_run_dir 2>/dev/null)"; then
        if run_has_alive_process "${current_run}"; then
            echo "Error: active run is still alive: ${current_run}"
            echo "Use: $0 stop"
            return 1
        fi
    fi

    local run_id
    run_id="$(date +%Y%m%d_%H%M%S)"
    local run_dir="${LOG_ROOT}/${run_id}"
    mkdir -p "${run_dir}"
    local pids_file="${run_dir}/pids.tsv"
    : > "${pids_file}"

    echo "Starting run: ${run_dir}"
    local env_id safe_env log_file pid
    for env_id in "${ENV_IDS[@]}"; do
        safe_env="$(printf "%s" "${env_id}" | tr '/ ' '__')"
        log_file="${run_dir}/${safe_env}.log"

        if command -v stdbuf >/dev/null 2>&1; then
            nohup env PATH="${MICROMAMBA_ENV}/bin:${PATH}" \
                PYTHONUNBUFFERED=1 \
                PYTHONIOENCODING=utf-8 \
                stdbuf -oL -eL "${PYTHON_BIN}" -u "${PYTHON_SCRIPT}" --envid "${env_id}" > "${log_file}" 2>&1 &
        else
            nohup env PATH="${MICROMAMBA_ENV}/bin:${PATH}" \
                PYTHONUNBUFFERED=1 \
                PYTHONIOENCODING=utf-8 \
                "${PYTHON_BIN}" -u "${PYTHON_SCRIPT}" --envid "${env_id}" > "${log_file}" 2>&1 &
        fi

        pid=$!
        printf "%s\t%s\t%s\n" "${env_id}" "${pid}" "${log_file}" >> "${pids_file}"
        echo "  started envid=${env_id} pid=${pid} log=${log_file}"
    done

    printf "%s\n" "${run_dir}" > "${ACTIVE_RUN_FILE}"
    echo "Run is detached with nohup. active_run=${run_dir}"
    monitor_run "${run_dir}"
}

restart_run() {
    local env_ids_csv="$1"
    stop_run || true
    start_run "${env_ids_csv}"
}

COMMAND="${1:-}"
case "${COMMAND}" in
    start)
        shift
        ENV_IDS_CSV="$(extract_env_ids_arg "$@")" || { show_usage; exit 1; }
        start_run "${ENV_IDS_CSV}"
        ;;
    monitor)
        shift
        if [ "$#" -ne 0 ]; then
            echo "Error: monitor takes no extra arguments."
            show_usage
            exit 1
        fi
        monitor_run
        ;;
    status)
        shift
        if [ "$#" -ne 0 ]; then
            echo "Error: status takes no extra arguments."
            show_usage
            exit 1
        fi
        status_run
        ;;
    stop)
        shift
        if [ "$#" -ne 0 ]; then
            echo "Error: stop takes no extra arguments."
            show_usage
            exit 1
        fi
        stop_run
        ;;
    restart)
        shift
        ENV_IDS_CSV="$(extract_env_ids_arg "$@")" || { show_usage; exit 1; }
        restart_run "${ENV_IDS_CSV}"
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
