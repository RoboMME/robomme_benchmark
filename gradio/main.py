#不在播放且已经被按下两个都满足 则​execute才可以被按下



"""
主入口模块
组装应用并启动服务器

本模块是整个应用的入口点，负责：
1. 创建 FastAPI 应用实例
2. 注册视频流路由（MJPEG流式传输）
3. 创建 Gradio UI 界面
4. 将 Gradio 应用挂载到 FastAPI
5. 启动 uvicorn 服务器

模块依赖关系及功能说明：

main.py (本模块 - 主进程)
  ├── streaming_service.py
  │   └── 功能：注册 FastAPI 路由 /video_feed/{uid}，提供 MJPEG 流式视频端点
  │       负责处理实时视频流的生成和传输
  │       从 ProcessSessionProxy 的本地缓存读取视频帧
  │
  └── ui_layout.py
      └── 功能：创建 Gradio Blocks 界面，定义所有 UI 组件和事件绑定
           返回配置好的 demo 对象供挂载使用

ui_layout.py (视图层 - 主进程)
  ├── gradio_callbacks.py
  │   └── 功能：提供所有 UI 事件回调函数
  │       - login_and_load_task: 用户登录并加载任务
  │       - execute_step: 执行动作步骤
  │       - on_map_click: 处理图片点击事件
  │       - on_option_select: 处理选项选择事件
  │       - init_app: 初始化应用（自动登录）
  │
  ├── user_manager.py
  │   └── 功能：用户管理和任务分配
  │       - 管理用户登录状态和租约（lease）
  │       - 跟踪用户任务进度
  │       - 提供可用用户名列表
  │
  ├── config.py
  │   └── 功能：应用配置常量
  │       - RESTRICT_VIDEO_PLAYBACK: 是否限制视频播放控制
  │       - USE_SEGMENTED_VIEW: 是否使用分割视图
  │       - ENV_IDS: 环境ID列表
  │
  └── state_manager.py
      └── 功能：管理 Gradio State 组件（uid_state, username_state）
           用于在 UI 组件间传递状态

gradio_callbacks.py (控制层 - 主进程)
  ├── state_manager.py
  │   └── 功能：全局状态管理
  │       - GLOBAL_SESSIONS: 存储所有 ProcessSessionProxy 实例（每个用户一个代理）
  │       - TASK_INDEX_MAP: 存储任务索引和进度信息
  │       - COORDINATE_CLICKS: 跟踪坐标点击事件
  │       - OPTION_SELECTS: 跟踪选项选择事件
  │       - FRAME_QUEUES: 管理视频帧队列（用于流式传输）
  │       提供线程安全的访问方法
  │
  ├── process_session.py
  │   └── 功能：多进程会话管理（核心架构）
  │       - ProcessSessionProxy: 主进程中的代理类，提供与 OracleSession 相同的接口
  │       - session_worker_loop: 工作进程中的循环，运行实际的 OracleSession
  │       - 通过 multiprocessing.Queue 进行进程间通信
  │       - 后台线程实时同步视频帧到主进程缓存
  │
  ├── streaming_service.py
  │   └── 功能：流媒体服务管理
  │       - FrameQueueManager: 管理帧队列的初始化和清理
  │       - 启动/停止后台监控线程
  │       - 从 ProcessSessionProxy 的本地缓存读取帧并加入队列
  │       - 处理实时帧的入队和出队
  │
  ├── image_utils.py
  │   └── 功能：图像处理工具函数（纯函数，无状态）
  │       - save_video: 将帧序列保存为视频文件
  │       - concatenate_frames_horizontally: 水平拼接两个帧序列
  │       - draw_marker: 在图片上绘制标记（红圈和十字）
  │
  ├── oracle_logic.py
  │   └── 功能：核心算法逻辑（在工作进程中运行）
  │       - OracleSession: 管理环境会话，执行重计算任务
  │       - 执行动作、加载任务、获取图像等核心功能
  │       - 每个用户一个独立进程，互不干扰
  │
  ├── user_manager.py
  │   └── 功能：用户鉴权和租约管理
  │       - 验证用户登录状态
  │       - 检查租约有效性（防止多设备登录）
  │       - 管理用户任务进度
  │
  └── logger.py
      └── 功能：日志记录
          - log_user_action: 记录用户操作
          - create_new_attempt: 创建新的尝试记录
          - has_existing_actions: 检查是否存在已有操作

streaming_service.py (流媒体服务层 - 主进程)
  ├── state_manager.py
  │   └── 功能：获取 Session 和队列信息
  │       - 通过 get_session() 获取 ProcessSessionProxy（代理对象）
  │       - 通过 FRAME_QUEUES 访问帧队列
  │       - 监控 ProcessSessionProxy.base_frames 的变化
  │       - 这些帧数据由后台同步线程从工作进程实时更新到主进程缓存
  │
  └── image_utils.py
      └── 功能：使用 concatenate_frames_horizontally 函数
            处理 base_frames 并添加标注和坐标系

image_utils.py (工具层)
  └── 功能：纯工具函数库，无业务逻辑依赖
      - 所有函数都是无状态的
      - 只进行图像处理和格式转换
      - 易于测试和复用

config.py (配置层)
  └── 功能：集中管理应用配置常量
      - 避免硬编码
      - 便于统一修改配置
      - 无依赖，可被任何模块导入

多进程架构说明：  
1. 主进程（Main Process）：
   - 运行 FastAPI/Gradio Web 服务器
   - 处理所有 HTTP/WebSocket 请求
   - 管理 UI 交互和用户界面
   - 每个用户会话对应一个 ProcessSessionProxy 实例
   - 通过代理对象与工作进程通信

2. 工作进程（Worker Process，每个用户一个）：
   - 运行实际的 OracleSession 实例
   - 执行重计算任务（环境加载、路径规划、动作执行等）
   - 通过 multiprocessing.Queue 与主进程通信
   - 将新产生的视频帧推送到流队列，由主进程同步线程接收

3. 进程间通信：
   - cmd_queue: 主进程发送命令到工作进程
   - result_queue: 工作进程返回命令执行结果
   - stream_queue: 工作进程推送新的视频帧到主进程

4. 数据同步：
   - ProcessSessionProxy 维护本地状态缓存（base_frames 等）
   - 后台同步线程持续从 stream_queue 接收新帧并更新缓存
   - streaming_service 从代理的本地缓存读取帧数据，无需直接访问工作进程

架构设计原则：
1. 关注点分离：每个模块只负责一个明确的功能领域
2. 单向依赖：依赖关系清晰，避免循环依赖
3. 接口稳定：回调函数签名保持不变，确保 Gradio 事件绑定正常工作
4. 进程隔离：每个用户的计算任务在独立进程中运行，互不干扰
5. 线程安全：状态访问通过 state_manager 进行，保证并发安全
6. 易于测试：模块化设计便于单元测试和集成测试

"""

import socket
import uvicorn
import gradio as gr
from fastapi import FastAPI
from streaming_service import create_video_feed_route
from ui_layout import create_ui_blocks, CSS, SYNC_JS
from state_manager import create_session, start_timeout_monitor
from user_manager import user_manager

import os
import multiprocessing
import signal
import sys

def find_free_port(start_port=7860):
    """查找单个可用端口"""
    for port in range(start_port, start_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    return 7860

def find_free_ports(start_port=7860, count=1):
    """查找多个连续可用端口"""
    ports = []
    current_port = start_port
    max_attempts = 1000  # 最多尝试1000个端口
    
    while len(ports) < count and current_port < start_port + max_attempts:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                # 尝试绑定端口以检查是否可用
                s.bind(('localhost', current_port))
                ports.append(current_port)
            except OSError:
                # 端口已被占用，跳过
                pass
            current_port += 1
    
    if len(ports) < count:
        raise RuntimeError(f"无法找到 {count} 个连续可用端口（从 {start_port} 开始）")
    
    return ports

def get_available_gpu_count():
    """检测可用的GPU数量"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        else:
            return 1
    except ImportError:
        # 如果torch不可用，尝试使用nvidia-smi
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                gpu_count = len([line for line in result.stdout.split('\n') if 'GPU' in line])
                return gpu_count if gpu_count > 0 else 1
        except:
            pass
        return 1

def get_gpu_for_user(username, available_users, num_gpus):
    """为指定用户分配GPU（轮询方式）"""
    user_index = sorted(available_users).index(username)
    gpu_id = user_index % num_gpus
    return gpu_id

def start_user_server(username, port, gpu_id):
    """为指定用户启动独立的服务器进程"""
    # 必须在导入torch等库之前设置环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 确保每个子进程都有 session 和 timeout monitor
    # 注意：在多进程环境下，每个进程都有独立的内存空间
    create_session()
    start_timeout_monitor()
    
    # 创建独立的 FastAPI 应用实例
    fastapi_app = FastAPI(title=f"HistoryBench Oracle Planner - {username}")
    
    # 注册视频流路由
    create_video_feed_route(fastapi_app)
    
    # 创建 Gradio UI
    demo = create_ui_blocks()
    
    # 使用 Gradio 的 mount_gradio_app 函数正确挂载 Gradio 应用到 FastAPI
    fastapi_app = gr.mount_gradio_app(
        fastapi_app,
        demo,
        path="/",
        css=CSS,
        js=SYNC_JS
    )
    
    # 使用 uvicorn 运行 FastAPI 应用
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True,
        use_colors=False
    )


def get_all_network_ips():
    """获取所有网络接口的 IP 地址"""
    ips = []
    
    # 方法1: 使用 socket 连接外部地址获取默认路由 IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # 连接到一个远程地址（不需要实际连接）
            s.connect(('8.8.8.8', 80))
            local_ip = s.getsockname()[0]
            if local_ip and local_ip != "127.0.0.1":
                ips.append(("default", local_ip))
        except Exception:
            pass
        finally:
            s.close()
    except Exception:
        pass
    
    # 方法2: 尝试使用 netifaces 获取所有接口（如果可用）
    try:
        import netifaces
        interfaces = netifaces.interfaces()
        for interface in interfaces:
            addrs = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in addrs:
                for addr_info in addrs[netifaces.AF_INET]:
                    ip = addr_info.get('addr')
                    if ip and ip != "127.0.0.1" and ip not in [ip_addr for _, ip_addr in ips]:
                        ips.append((interface, ip))
    except ImportError:
        # netifaces 不可用，跳过
        pass
    except Exception:
        pass
    
    # 方法3: 使用 psutil（如果可用）
    try:
        import psutil
        for interface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == socket.AF_INET:
                    ip = addr.address
                    if ip and ip != "127.0.0.1" and ip not in [ip_addr for _, ip_addr in ips]:
                        ips.append((interface, ip))
    except ImportError:
        # psutil 不可用，跳过
        pass
    except Exception:
        pass
    
    # 方法4: 使用 ip 命令（Linux）
    try:
        import subprocess
        import re
        result = subprocess.run(['ip', 'addr', 'show'], 
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            # 解析 ip addr show 的输出
            pattern = r'inet\s+(\d+\.\d+\.\d+\.\d+)/\d+'
            matches = re.findall(pattern, result.stdout)
            for ip in matches:
                if ip and ip != "127.0.0.1" and ip not in [ip_addr for _, ip_addr in ips]:
                    ips.append(("interface", ip))
    except (ImportError, FileNotFoundError):
        pass
    except Exception:
        # 包括 subprocess.TimeoutExpired 等其他异常
        pass
    
    return ips


if __name__ == "__main__":
    # Ensure session created for imports
    create_session()
    
    # 启动session超时监控线程
    start_timeout_monitor()
    
    # 获取所有用户列表
    available_users = list(user_manager.user_tasks.keys())
    
    if not available_users:
        print("\n⚠️  警告: 未找到任何用户配置")
        sys.exit(1)
    
    # 获取所有网络接口 IP
    network_ips = get_all_network_ips()
    base_ip = network_ips[0][1] if network_ips else "localhost"
    
    # 为每个用户分配端口
    num_users = len(available_users)
    try:
        ports = find_free_ports(start_port=7860, count=num_users)
    except RuntimeError as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)
    
    # 创建用户名到端口的映射
    user_port_map = {username: ports[i] for i, username in enumerate(sorted(available_users))}
    
    # 存储所有子进程
    processes = []
    
    # 信号处理函数，用于优雅关闭
    def signal_handler(sig, frame):
        print("\n\n收到退出信号，正在关闭所有服务器...")
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
        sys.exit(0)
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 检测可用GPU数量并分配GPU
    num_gpus = get_available_gpu_count()
    print(f"\n检测到 {num_gpus} 个可用GPU，使用轮询方式分配")
    
    # 为每个用户启动独立的服务器进程
    print("\n" + "="*60)
    print("正在启动多用户服务器...")
    print("="*60)
    
    for username in sorted(available_users):
        port = user_port_map[username]
        gpu_id = get_gpu_for_user(username, available_users, num_gpus)
        process = multiprocessing.Process(
            target=start_user_server,
            args=(username, port, gpu_id),
            name=f"Server-{username}"
        )
        process.start()
        processes.append(process)
        print(f"✓ 用户 {username:20s} 的服务器已启动在端口 {port} (GPU {gpu_id})")
    
    # 等待一下确保所有服务器都已启动
    import time
    time.sleep(2)
    
    # 打印服务器信息
    print("\n" + "="*60)
    print("所有服务器已启动:")
    print("="*60)
    
    # 打印端口分配映射表
    print("\n端口分配映射:")
    print("-" * 60)
    for username in sorted(available_users):
        port = user_port_map[username]
        print(f"  用户: {username:20s} -> 端口: {port}")
    print("-" * 60)
    print(f"共 {len(available_users)} 个用户服务器")
    
    # 打印GPU分配映射表
    print("\nGPU分配映射:")
    print("-" * 60)
    for username in sorted(available_users):
        port = user_port_map[username]
        gpu_id = get_gpu_for_user(username, available_users, num_gpus)
        print(f"  用户: {username:20s} -> 端口: {port:5d} -> GPU: {gpu_id}")
    print("-" * 60)
    
    # 打印每个用户的访问链接
    if network_ips:
        print("\n所有可用的访问地址:")
        print("-" * 60)
        for interface, ip in network_ips:
            print(f"  网络接口: {interface}")
        print("-" * 60)
    
    print("\n用户访问链接:")
    print("-" * 60)
    for username in sorted(available_users):
        port = user_port_map[username]
        login_link = f"http://{base_ip}:{port}/?user={username}&__theme=light"
        print(f"  {username:20s} -> {login_link}")
    print("-" * 60)
    
    print("\nMJPEG 流媒体端点:")
    print("-" * 60)
    for username in sorted(available_users):
        port = user_port_map[username]
        stream_endpoint = f"http://{base_ip}:{port}/video_feed/{{uid}}"
        print(f"  {username:20s} -> {stream_endpoint}")
    print("-" * 60)
    
    print("="*60)
    print("所有服务器正在运行中...")
    print("按 Ctrl+C 停止所有服务器")
    print("="*60 + "\n")
    
    # 主进程等待所有子进程
    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        signal_handler(None, None)
