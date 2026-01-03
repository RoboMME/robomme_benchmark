#!/bin/bash

# 运行 oraclev4.py 的脚本
# 使用指定的 micromamba 环境路径，支持后台运行和实时查看日志

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR" && pwd)"
PYTHON_SCRIPT="$PROJECT_DIR/history_bench_sim/oraclev4.py"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/oraclev4_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$LOG_DIR/oraclev4.pid"
MICROMAMBA_ENV_PATH="/home/hongzefu/micromamba/envs/maniskillenv1228"

# 环境变量配置
export GOOGLE_API_KEY="AIzaSyBozLZAq6up8sCd2VSjXWW4CPwvtNHUu8s"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 函数：启动脚本
start() {
    # 检查是否已经在运行
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "脚本已经在运行中 (PID: $PID)"
            echo "使用 'bash $0 view' 查看日志"
            exit 1
        else
            echo "发现旧的 PID 文件，正在清理..."
            rm -f "$PID_FILE"
        fi
    fi

    echo "正在启动脚本..."
    echo "日志文件: $LOG_FILE"
    echo "项目目录: $PROJECT_DIR"
    
    # 检查指定的 micromamba 环境路径
    if [ ! -d "$MICROMAMBA_ENV_PATH" ]; then
        echo "错误: 环境路径不存在: $MICROMAMBA_ENV_PATH"
        exit 1
    fi
    
    PYTHON_BIN="$MICROMAMBA_ENV_PATH/bin/python"
    if [ ! -f "$PYTHON_BIN" ]; then
        echo "错误: Python 解释器不存在: $PYTHON_BIN"
        exit 1
    fi
    
    echo "使用环境路径: $MICROMAMBA_ENV_PATH"
    echo "使用 Python: $PYTHON_BIN"
    
    # 使用 nohup 在后台运行，直接使用指定环境的 Python
    cd "$PROJECT_DIR"
    nohup "$PYTHON_BIN" "$PYTHON_SCRIPT" > "$LOG_FILE" 2>&1 &
    
    PID=$!
    echo $PID > "$PID_FILE"
    
    echo "脚本已启动 (PID: $PID)"
    echo "使用以下命令查看日志:"
    echo "  bash $0 view        # 实时查看日志"
    echo "  bash $0 tail        # 查看最后几行日志"
    echo "  bash $0 status      # 查看运行状态"
    echo "  bash $0 stop        # 停止脚本"
}

# 函数：查看实时日志
view() {
    if [ ! -f "$PID_FILE" ]; then
        echo "错误: 未找到 PID 文件，脚本可能未运行"
        exit 1
    fi
    
    PID=$(cat "$PID_FILE")
    if ! ps -p "$PID" > /dev/null 2>&1; then
        echo "错误: 进程不存在 (PID: $PID)，脚本可能已停止"
        rm -f "$PID_FILE"
        exit 1
    fi
    
    # 找到最新的日志文件
    LATEST_LOG=$(ls -t "$LOG_DIR"/oraclev4_*.log 2>/dev/null | head -1)
    if [ -z "$LATEST_LOG" ]; then
        echo "错误: 未找到日志文件"
        exit 1
    fi
    
    echo "正在实时查看日志: $LATEST_LOG"
    echo "按 Ctrl+C 退出查看（不会停止脚本）"
    echo "----------------------------------------"
    tail -f "$LATEST_LOG"
}

# 函数：查看最后几行日志
tail_log() {
    LATEST_LOG=$(ls -t "$LOG_DIR"/oraclev4_*.log 2>/dev/null | head -1)
    if [ -z "$LATEST_LOG" ]; then
        echo "错误: 未找到日志文件"
        exit 1
    fi
    
    echo "最后 50 行日志: $LATEST_LOG"
    echo "----------------------------------------"
    tail -n 50 "$LATEST_LOG"
}

# 函数：查看运行状态
status() {
    if [ ! -f "$PID_FILE" ]; then
        echo "状态: 未运行"
        exit 0
    fi
    
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "状态: 运行中"
        echo "PID: $PID"
        LATEST_LOG=$(ls -t "$LOG_DIR"/oraclev4_*.log 2>/dev/null | head -1)
        if [ -n "$LATEST_LOG" ]; then
            echo "日志文件: $LATEST_LOG"
            echo "日志大小: $(du -h "$LATEST_LOG" | cut -f1)"
            echo ""
            echo "最后 5 行日志:"
            echo "----------------------------------------"
            tail -n 5 "$LATEST_LOG"
        fi
    else
        echo "状态: 已停止 (PID 文件存在但进程不存在)"
        rm -f "$PID_FILE"
    fi
}

# 函数：停止脚本
stop() {
    if [ ! -f "$PID_FILE" ]; then
        echo "错误: 未找到 PID 文件，脚本可能未运行"
        exit 1
    fi
    
    PID=$(cat "$PID_FILE")
    if ! ps -p "$PID" > /dev/null 2>&1; then
        echo "进程不存在 (PID: $PID)"
        rm -f "$PID_FILE"
        exit 0
    fi
    
    echo "正在停止进程 (PID: $PID)..."
    kill "$PID"
    
    # 等待进程结束
    for i in {1..10}; do
        if ! ps -p "$PID" > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    
    # 如果还在运行，强制杀死
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "强制停止进程..."
        kill -9 "$PID"
    fi
    
    rm -f "$PID_FILE"
    echo "脚本已停止"
}

# 主逻辑
case "$1" in
    start)
        start
        ;;
    view)
        view
        ;;
    tail)
        tail_log
        ;;
    status)
        status
        ;;
    stop)
        stop
        ;;
    *)
        echo "用法: $0 {start|view|tail|status|stop}"
        echo ""
        echo "命令说明:"
        echo "  start   - 启动脚本（后台运行）"
        echo "  view    - 实时查看日志（tail -f）"
        echo "  tail    - 查看最后 50 行日志"
        echo "  status  - 查看运行状态"
        echo "  stop    - 停止脚本"
        exit 1
        ;;
esac

