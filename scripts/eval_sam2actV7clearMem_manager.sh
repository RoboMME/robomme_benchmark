#!/bin/bash

# eval_sam2actV7clearMem_manager.sh - 评估脚本管理器
# 用于管理 eval_sam2actV7clearMem.py 的运行、监控、停止
# 功能：
# 1. 使用nohup确保SSH断开后仍能运行
# 2. 使用Python -u参数实现无缓冲输出
# 3. 实时查看日志输出
# 4. 进程管理（启动、停止、监控、状态查看）

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="eval_sam2actV7clearMem.py"
SCRIPT_PATH="${SCRIPT_DIR}/${SCRIPT_NAME}"

# 日志文件路径
LOG_DIR="${SCRIPT_DIR}/logs"
PID_FILE="${LOG_DIR}/eval_sam2actV7clearMem.pid"

# 创建日志目录
mkdir -p "${LOG_DIR}"

# Micromamba/Conda环境路径
CONDA_ENV="/data/hongzefu/maniskillenv1114"

# 显示使用说明
show_usage() {
    echo "用法: $0 {start|monitor|stop|status|restart} [脚本参数...]"
    echo ""
    echo "命令:"
    echo "  start    - 启动评估脚本并实时监控日志"
    echo "  monitor  - 监控正在运行的评估脚本日志"
    echo "  stop     - 停止正在运行的评估脚本"
    echo "  status   - 查看评估脚本运行状态"
    echo "  restart  - 重启评估脚本（先停止再启动）"
    echo ""
    echo "示例:"
    echo "  $0 start"
    echo "  $0 start --api_url http://141.212.48.176:8002 --max_steps 40"
    echo "  $0 monitor"
    echo "  $0 stop"
    echo "  $0 status"
}

# 检查进程状态
check_status() {
    if [ ! -f "${PID_FILE}" ]; then
        echo "状态: 未运行（无PID文件）"
        return 1
    fi
    
    PID=$(cat "${PID_FILE}")
    if ps -p "${PID}" > /dev/null 2>&1; then
        echo "状态: 运行中"
        echo "PID: ${PID}"
        
        # 查找最新的日志文件
        LATEST_LOG=$(ls -t "${LOG_DIR}"/eval_sam2actV7clearMem_*.log 2>/dev/null | head -1)
        if [ -n "${LATEST_LOG}" ]; then
            echo "日志文件: ${LATEST_LOG}"
            echo "日志大小: $(du -h "${LATEST_LOG}" | cut -f1)"
        fi
        
        # 显示进程信息
        echo "进程信息:"
        ps -p "${PID}" -o pid,ppid,cmd,etime,pcpu,pmem 2>/dev/null || echo "  无法获取进程信息"
        return 0
    else
        echo "状态: 未运行（进程不存在）"
        echo "清理PID文件"
        rm -f "${PID_FILE}"
        return 1
    fi
}

# 停止进程
stop_process() {
    if [ ! -f "${PID_FILE}" ]; then
        echo "未找到PID文件，可能没有进程在运行"
        return 0
    fi
    
    PID=$(cat "${PID_FILE}")
    
    if ! ps -p "${PID}" > /dev/null 2>&1; then
        echo "进程 ${PID} 已不存在"
        rm -f "${PID_FILE}"
        return 0
    fi
    
    echo "正在停止进程 ${PID}..."
    
    # 发送SIGTERM信号（优雅退出）
    kill "${PID}" 2>/dev/null
    
    # 等待进程结束
    for i in {1..10}; do
        if ! ps -p "${PID}" > /dev/null 2>&1; then
            echo "进程已停止"
            rm -f "${PID_FILE}"
            return 0
        fi
        sleep 1
    done
    
    # 如果进程仍在运行，强制杀死
    if ps -p "${PID}" > /dev/null 2>&1; then
        echo "进程未响应，强制终止..."
        kill -9 "${PID}" 2>/dev/null
        sleep 1
    fi
    
    # 清理PID文件
    rm -f "${PID_FILE}"
    echo "进程已停止"
}

# 启动进程
start_process() {
    # 检查脚本是否存在
    if [ ! -f "${SCRIPT_PATH}" ]; then
        echo "错误: eval_sam2actV7clearMem.py 文件不存在: ${SCRIPT_PATH}"
        exit 1
    fi
    
    # 检查是否已有进程在运行
    if [ -f "${PID_FILE}" ]; then
        OLD_PID=$(cat "${PID_FILE}")
        if ps -p "${OLD_PID}" > /dev/null 2>&1; then
            echo "警告: 检测到已有进程在运行 (PID: ${OLD_PID})"
            echo "如果要重新运行，请先停止旧进程: $0 stop"
            exit 1
        else
            echo "清理旧的PID文件"
            rm -f "${PID_FILE}"
        fi
    fi
    
    # 生成日志文件名
    LOG_FILE="${LOG_DIR}/eval_sam2actV7clearMem_$(date +%Y%m%d_%H%M%S).log"
    
    # 激活conda环境并运行脚本
    echo "=========================================="
    echo "开始运行 eval_sam2actV7clearMem.py"
    echo "脚本路径: ${SCRIPT_PATH}"
    echo "日志文件: ${LOG_FILE}"
    echo "Micromamba环境: ${CONDA_ENV}"
    echo "=========================================="
    
    # 使用nohup运行，确保SSH断开后仍能继续
    # -u: Python无缓冲输出
    # 2>&1: 将stderr重定向到stdout
    # 获取所有传入的参数（除了第一个命令参数）
    SCRIPT_ARGS="${@:2}"
    
    # 检查Python可执行文件
    PYTHON_BIN="${CONDA_ENV}/bin/python"
    if [ ! -f "${PYTHON_BIN}" ]; then
        echo "错误: Python可执行文件不存在: ${PYTHON_BIN}"
        exit 1
    fi
    
    # 设置环境变量
    export PATH="${CONDA_ENV}/bin:${PATH}"
    export CONDA_PREFIX="${CONDA_ENV}"
    export CONDA_DEFAULT_ENV="${CONDA_ENV}"
    
    # 设置LD_LIBRARY_PATH
    if [ -d "${CONDA_ENV}/lib" ]; then
        export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
    fi
    
    # 查找site-packages目录并添加到PYTHONPATH
    if [ -d "${CONDA_ENV}/lib" ]; then
        SITE_PACKAGES=$(find "${CONDA_ENV}/lib" -type d -name "site-packages" 2>/dev/null | head -1)
        if [ -n "${SITE_PACKAGES}" ]; then
            export PYTHONPATH="${SITE_PACKAGES}:${PYTHONPATH:-}"
        fi
    fi
    
    # 使用nohup直接启动Python进程，确保SSH断开后仍能运行
    # 使用exec替换shell进程，确保PID正确
    nohup env PATH="${CONDA_ENV}/bin:${PATH}" \
         CONDA_PREFIX="${CONDA_ENV}" \
         CONDA_DEFAULT_ENV="${CONDA_ENV}" \
         LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}" \
         PYTHONPATH="${SITE_PACKAGES}:${PYTHONPATH:-}" \
         "${PYTHON_BIN}" -u "${SCRIPT_PATH}" ${SCRIPT_ARGS} > "${LOG_FILE}" 2>&1 &
    
    # 获取进程ID
    PID=$!
    
    # 等待一下让进程启动
    sleep 2
    
    # 验证进程是否真的在运行
    if ! ps -p "${PID}" > /dev/null 2>&1; then
        # 如果PID无效，尝试通过进程名查找
        PID=$(pgrep -f "python.*${SCRIPT_NAME}" | head -1)
        if [ -z "${PID}" ]; then
            PID=""
        fi
    fi
    
    if [ -z "${PID}" ] || ! ps -p "${PID}" > /dev/null 2>&1; then
        echo "错误: 无法启动Python进程或进程立即退出"
        echo "请检查:"
        echo "  1. Micromamba环境路径: ${CONDA_ENV}"
        echo "  2. Python脚本路径: ${SCRIPT_PATH}"
        echo "  3. 日志文件（如果已生成）: ${LOG_FILE}"
        if [ -f "${LOG_FILE}" ]; then
            echo ""
            echo "日志文件内容:"
            cat "${LOG_FILE}" 2>/dev/null || echo "无法读取日志文件"
        else
            echo ""
            echo "提示: 日志文件尚未生成，可能启动脚本执行失败"
            echo "尝试手动测试: ${PYTHON_BIN} -u ${SCRIPT_PATH}"
        fi
        exit 1
    fi
    
    echo "${PID}" > "${PID_FILE}"
    
    # 验证进程是否真的在运行
    if ! ps -p "${PID}" > /dev/null 2>&1; then
        echo "错误: 进程启动失败，PID ${PID} 不存在"
        echo "请检查日志文件: ${LOG_FILE}"
        rm -f "${PID_FILE}"
        exit 1
    fi
    
    echo "进程已启动，PID: ${PID}"
    echo "日志文件: ${LOG_FILE}"
    echo ""
    echo "实时查看日志 (按Ctrl+C退出监控，进程会继续运行):"
    echo "----------------------------------------"
    
    # 等待日志文件创建
    sleep 1
    
    # 实时查看日志（无缓冲）
    if [ -f "${LOG_FILE}" ]; then
        tail -f "${LOG_FILE}"
    else
        echo "等待日志文件生成..."
        sleep 2
        if [ -f "${LOG_FILE}" ]; then
            tail -f "${LOG_FILE}"
        else
            echo "警告: 日志文件尚未生成，但进程已启动"
            echo "PID: ${PID}"
            echo "稍后可以使用 '$0 monitor' 查看日志"
            echo ""
            echo "提示: 如果进程立即退出，请检查:"
            echo "  1. Conda环境是否正确: ${CONDA_ENV}"
            echo "  2. Python脚本是否有语法错误"
            echo "  3. 查看系统日志或使用 '$0 status' 检查进程状态"
        fi
    fi
}

# 监控日志
monitor_log() {
    if [ ! -f "${PID_FILE}" ]; then
        echo "错误: 未找到PID文件 ${PID_FILE}"
        echo "可能没有进程在运行，请先使用 '$0 start' 启动"
        exit 1
    fi
    
    PID=$(cat "${PID_FILE}")
    
    # 检查进程是否还在运行
    if ! ps -p "${PID}" > /dev/null 2>&1; then
        echo "警告: 进程 ${PID} 已不存在"
        echo "清理PID文件"
        rm -f "${PID_FILE}"
        exit 1
    fi
    
    # 查找最新的日志文件
    LATEST_LOG=$(ls -t "${LOG_DIR}"/eval_sam2actV7clearMem_*.log 2>/dev/null | head -1)
    
    if [ -z "${LATEST_LOG}" ]; then
        echo "错误: 未找到 eval_sam2actV7clearMem 的日志文件"
        exit 1
    fi
    
    echo "=========================================="
    echo "监控 eval_sam2actV7clearMem.py 日志"
    echo "进程PID: ${PID}"
    echo "日志文件: ${LATEST_LOG}"
    echo "按Ctrl+C退出监控（进程会继续运行）"
    echo "=========================================="
    echo ""
    
    # 实时查看日志（无缓冲）
    tail -f "${LATEST_LOG}"
}

# 主逻辑
case "${1}" in
    start)
        start_process "$@"
        ;;
    monitor)
        monitor_log
        ;;
    stop)
        stop_process
        ;;
    status)
        check_status
        ;;
    restart)
        echo "重启 eval_sam2actV7clearMem.py..."
        stop_process
        sleep 2
        start_process "$@"
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
