#!/bin/bash

# 视频标注工具后台运行脚本
# 支持断开SSH后继续运行

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/15-filterVideo.py"
PYTHON_INTERPRETER="/data/hongzefu/maniskillenv1114/bin/python"
LOG_DIR="${SCRIPT_DIR}/logs"
PID_FILE="${SCRIPT_DIR}/filterVideo.pid"
LOG_FILE="${LOG_DIR}/filterVideo_$(date +%Y%m%d_%H%M%S).log"

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 检查Python解释器是否存在
if [ ! -f "${PYTHON_INTERPRETER}" ]; then
    echo "错误: 找不到Python解释器: ${PYTHON_INTERPRETER}"
    exit 1
fi

# 检查Python脚本是否存在
if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "错误: 找不到Python脚本: ${PYTHON_SCRIPT}"
    exit 1
fi

# 函数：获取局域网IP地址
get_local_ip() {
    # 尝试多种方法获取局域网IP
    local ip=""
    
    # 方法1: 使用 ip 命令（优先）
    if command -v ip > /dev/null 2>&1; then
        ip=$(ip route get 8.8.8.8 2>/dev/null | grep -oP 'src \K[\d.]+' | head -1)
    fi
    
    # 方法2: 使用 hostname -I
    if [ -z "$ip" ] && command -v hostname > /dev/null 2>&1; then
        ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    fi
    
    # 方法3: 使用 ifconfig
    if [ -z "$ip" ] && command -v ifconfig > /dev/null 2>&1; then
        ip=$(ifconfig 2>/dev/null | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1' | head -1)
    fi
    
    # 方法4: 使用 /proc/net/route
    if [ -z "$ip" ]; then
        ip=$(awk '/^[0-9]/{print $1}' /proc/net/route 2>/dev/null | head -1 | xargs -I {} ip addr show {} 2>/dev/null | grep -oP 'inet \K[\d.]+' | head -1)
    fi
    
    echo "$ip"
}

# 函数：从日志中提取端口信息
extract_ports_from_log() {
    local log_file="$1"
    local ports=()
    
    if [ -f "$log_file" ]; then
        # 从日志中提取端口号（查找 "端口: 7860" 或 "访问地址: http://0.0.0.0:7860" 这样的模式）
        while IFS= read -r line; do
            # 匹配 "端口: 7860" 格式
            if [[ "$line" =~ 端口:[[:space:]]*([0-9]+) ]]; then
                ports+=("${BASH_REMATCH[1]}")
            # 匹配 "访问地址: http://0.0.0.0:7860" 格式
            elif [[ "$line" =~ http://[^:]+:([0-9]+) ]]; then
                ports+=("${BASH_REMATCH[1]}")
            fi
        done < "$log_file"
    fi
    
    # 去重并返回
    printf '%s\n' "${ports[@]}" | sort -u
}

# 函数：启动服务
start_service() {
    # 检查是否已经在运行
    if [ -f "${PID_FILE}" ]; then
        PID=$(cat "${PID_FILE}")
        if ps -p "${PID}" > /dev/null 2>&1; then
            echo "服务已经在运行中 (PID: ${PID})"
            echo "使用 'bash $0 stop' 停止服务"
            exit 1
        else
            # PID文件存在但进程不存在，删除旧的PID文件
            rm -f "${PID_FILE}"
        fi
    fi
    
    echo "正在启动视频标注服务..."
    echo "Python解释器: ${PYTHON_INTERPRETER}"
    echo "日志文件: ${LOG_FILE}"
    
    # 使用nohup在后台运行，并将输出重定向到日志文件
    nohup "${PYTHON_INTERPRETER}" "${PYTHON_SCRIPT}" > "${LOG_FILE}" 2>&1 &
    
    # 保存进程ID
    echo $! > "${PID_FILE}"
    
    # 等待一下，检查进程是否成功启动
    sleep 2
    PID=$(cat "${PID_FILE}")
    if ps -p "${PID}" > /dev/null 2>&1; then
        echo "✅ 服务已启动 (PID: ${PID})"
        echo "📝 日志文件: ${LOG_FILE}"
        echo "🔍 查看日志: tail -f ${LOG_FILE}"
        echo "🛑 停止服务: bash $0 stop"
        echo ""
        
        # 等待服务完全启动并获取端口信息
        echo "等待服务启动完成..."
        sleep 5
        
        # 获取局域网IP地址
        local_ip=$(get_local_ip)
        
        # 从日志中提取端口信息
        ports=$(extract_ports_from_log "${LOG_FILE}")
        
        if [ -n "$ports" ]; then
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "🌐 访问地址："
            if [ -n "$local_ip" ]; then
                while IFS= read -r port; do
                    if [ -n "$port" ]; then
                        echo "   本地访问: http://localhost:${port}"
                        echo "   局域网访问: http://${local_ip}:${port}"
                    fi
                done <<< "$ports"
            else
                while IFS= read -r port; do
                    if [ -n "$port" ]; then
                        echo "   访问地址: http://0.0.0.0:${port}"
                        echo "   (无法获取局域网IP，请手动查看日志获取完整地址)"
                    fi
                done <<< "$ports"
            fi
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        else
            echo "⚠️  无法从日志中提取端口信息，请查看日志文件获取访问地址"
            if [ -n "$local_ip" ]; then
                echo "   局域网IP: ${local_ip}"
            fi
        fi
        
        echo ""
        echo "服务将在后台运行，即使断开SSH连接也会继续运行。"
    else
        echo "❌ 服务启动失败，请查看日志: ${LOG_FILE}"
        rm -f "${PID_FILE}"
        exit 1
    fi
}

# 函数：停止服务
stop_service() {
    if [ ! -f "${PID_FILE}" ]; then
        echo "服务未运行（找不到PID文件）"
        exit 1
    fi
    
    PID=$(cat "${PID_FILE}")
    if ps -p "${PID}" > /dev/null 2>&1; then
        echo "正在停止服务 (PID: ${PID})..."
        kill "${PID}"
        
        # 等待进程结束
        for i in {1..10}; do
            if ! ps -p "${PID}" > /dev/null 2>&1; then
                break
            fi
            sleep 1
        done
        
        # 如果还在运行，强制杀死
        if ps -p "${PID}" > /dev/null 2>&1; then
            echo "强制停止服务..."
            kill -9 "${PID}"
        fi
        
        rm -f "${PID_FILE}"
        echo "✅ 服务已停止"
    else
        echo "服务未运行（进程不存在）"
        rm -f "${PID_FILE}"
    fi
}

# 函数：查看状态
status_service() {
    if [ ! -f "${PID_FILE}" ]; then
        echo "服务未运行"
        exit 0
    fi
    
    PID=$(cat "${PID_FILE}")
    if ps -p "${PID}" > /dev/null 2>&1; then
        echo "✅ 服务正在运行 (PID: ${PID})"
        echo "进程信息:"
        ps -p "${PID}" -o pid,ppid,cmd,etime,pcpu,pmem
        echo ""
        
        # 获取局域网IP地址
        local_ip=$(get_local_ip)
        
        # 查找日志文件
        log_to_check="${LOG_FILE}"
        if [ ! -f "${log_to_check}" ]; then
            log_to_check=$(ls -t ${LOG_DIR}/filterVideo_*.log 2>/dev/null | head -n 1)
        fi
        
        # 从日志中提取端口信息
        if [ -n "${log_to_check}" ]; then
            ports=$(extract_ports_from_log "${log_to_check}")
            
            if [ -n "$ports" ]; then
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                echo "🌐 访问地址："
                if [ -n "$local_ip" ]; then
                    while IFS= read -r port; do
                        if [ -n "$port" ]; then
                            echo "   本地访问: http://localhost:${port}"
                            echo "   局域网访问: http://${local_ip}:${port}"
                        fi
                    done <<< "$ports"
                else
                    while IFS= read -r port; do
                        if [ -n "$port" ]; then
                            echo "   访问地址: http://0.0.0.0:${port}"
                        fi
                    done <<< "$ports"
                fi
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                echo ""
            fi
        fi
        
        echo "最新日志 (最后20行):"
        echo "----------------------------------------"
        if [ -n "${log_to_check}" ] && [ -f "${log_to_check}" ]; then
            tail -n 20 "${log_to_check}"
        else
            echo "未找到日志文件"
        fi
    else
        echo "❌ 服务未运行（PID文件存在但进程不存在）"
        rm -f "${PID_FILE}"
    fi
}

# 函数：查看日志
view_log() {
    if [ -f "${LOG_FILE}" ]; then
        tail -f "${LOG_FILE}"
    else
        # 查找最新的日志文件
        LATEST_LOG=$(ls -t ${LOG_DIR}/filterVideo_*.log 2>/dev/null | head -n 1)
        if [ -n "${LATEST_LOG}" ]; then
            tail -f "${LATEST_LOG}"
        else
            echo "未找到日志文件"
            exit 1
        fi
    fi
}

# 函数：重启服务
restart_service() {
    echo "重启服务..."
    stop_service
    sleep 2
    start_service
}

# 主逻辑
case "${1}" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    status)
        status_service
        ;;
    log|logs)
        view_log
        ;;
    *)
        echo "用法: $0 {start|stop|restart|status|log}"
        echo ""
        echo "命令说明:"
        echo "  start   - 启动服务（后台运行）"
        echo "  stop    - 停止服务"
        echo "  restart - 重启服务"
        echo "  status  - 查看服务状态"
        echo "  log     - 实时查看日志"
        exit 1
        ;;
esac

