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

main.py (本模块)
  ├── streaming_service.py
  │   └── 功能：注册 FastAPI 路由 /video_feed/{uid}，提供 MJPEG 流式视频端点
  │       负责处理实时视频流的生成和传输
  │
  └── ui_layout.py
      └── 功能：创建 Gradio Blocks 界面，定义所有 UI 组件和事件绑定
           返回配置好的 demo 对象供挂载使用

ui_layout.py (视图层)
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

gradio_callbacks.py (控制层)
  ├── state_manager.py
  │   └── 功能：全局状态管理
  │       - GLOBAL_SESSIONS: 存储所有 OracleSession 实例
  │       - TASK_INDEX_MAP: 存储任务索引和进度信息
  │       - COORDINATE_CLICKS: 跟踪坐标点击事件
  │       - OPTION_SELECTS: 跟踪选项选择事件
  │       - FRAME_QUEUES: 管理视频帧队列（用于流式传输）
  │       提供线程安全的访问方法
  │
  ├── streaming_service.py
  │   └── 功能：流媒体服务管理
  │       - FrameQueueManager: 管理帧队列的初始化和清理
  │       - 启动/停止后台监控线程
  │       - 处理实时帧的入队和出队
  │
  ├── image_utils.py
  │   └── 功能：图像处理工具函数（纯函数，无状态）
  │       - save_video: 将帧序列保存为视频文件
  │       - concatenate_frames_horizontally: 水平拼接两个帧序列
  │       - draw_marker: 在图片上绘制标记（红圈和十字）
  │
  ├── oracle_logic.py
  │   └── 功能：核心算法逻辑
  │       - OracleSession: 管理环境会话
  │       - 执行动作、加载任务、获取图像等核心功能
  │
  ├── user_manager.py
  │   └── 功能：用户鉴权和租约管理
  │       - 验证用户登录状态
  │       - 检查租约有效性（防止多设备登录）
  │       - 管理用户任务进度
  │
  └── logger.py
      └── 功能：日志记录
          - log_session: 记录会话信息
          - log_user_action: 记录用户操作
          - create_new_attempt: 创建新的尝试记录
          - has_existing_actions: 检查是否存在已有操作

streaming_service.py (流媒体服务层)
  ├── state_manager.py
  │   └── 功能：获取 Session 和队列信息
  │       - 通过 get_session() 获取 OracleSession
  │       - 通过 FRAME_QUEUES 访问帧队列
  │       - 监控 session.base_frames 和 session.wrist_frames 的变化
  │
  └── image_utils.py
      └── 功能：使用 concatenate_frames_horizontally 函数
            将 base_frames 和 wrist_frames 水平拼接成组合视图

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

架构设计原则：
1. 关注点分离：每个模块只负责一个明确的功能领域
2. 单向依赖：依赖关系清晰，避免循环依赖
3. 接口稳定：回调函数签名保持不变，确保 Gradio 事件绑定正常工作
4. 线程安全：状态访问通过 state_manager 进行，保证并发安全
5. 易于测试：模块化设计便于单元测试和集成测试

"""

import socket
import uvicorn
import gradio as gr
from fastapi import FastAPI
from streaming_service import create_video_feed_route
from ui_layout import create_ui_blocks, CSS, SYNC_JS
from state_manager import create_session
from user_manager import user_manager


def find_free_port(start_port=7860):
    """查找可用端口"""
    for port in range(start_port, start_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    return 7860


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
    
    # 创建 FastAPI 应用
    fastapi_app = FastAPI(title="HistoryBench Oracle Planner")
    
    # 注册视频流路由
    create_video_feed_route(fastapi_app)
    
    # 创建 Gradio UI
    demo = create_ui_blocks()
    
    # 查找可用端口
    port = find_free_port()
    print(f"Starting server on port {port}")
    
    # 使用 Gradio 的 mount_gradio_app 函数正确挂载 Gradio 应用到 FastAPI
    # 这会正确初始化所有必要的配置，包括 config 对象
    fastapi_app = gr.mount_gradio_app(
        fastapi_app,
        demo,
        path="/",
        css=CSS,
        js=SYNC_JS
    )
    
    # 获取所有网络接口 IP
    network_ips = get_all_network_ips()
    
    print("\n" + "="*60)
    print("SERVER STARTING:")
    print("="*60)
    print(f"FastAPI + Gradio server running on http://0.0.0.0:{port}")
    print(f"MJPEG stream endpoint: http://0.0.0.0:{port}/video_feed/{{uid}}")
    print("="*60)
    
    # 打印所有可用的公共 IP 地址
    if network_ips:
        print("\n所有可用的 Gradio 公共 IP 地址:")
        print("-" * 60)
        for interface, ip in network_ips:
            print(f"  {interface:15s} -> http://{ip}:{port}")
        print("-" * 60)
        print(f"共 {len(network_ips)} 个网络接口")
    else:
        print("\n⚠️  警告: 无法获取网络接口 IP 地址")
        print("   请使用 http://0.0.0.0:{port} 或 http://localhost:{port} 访问")
    
    # 打印每个用户的登录链接（使用第一个可用 IP）
    available_users = list(user_manager.user_tasks.keys())
    if available_users:
        print("\n用户登录链接:")
        print("-" * 60)
        # 使用第一个可用 IP，如果没有则使用 localhost
        base_ip = network_ips[0][1] if network_ips else "localhost"
        for username in sorted(available_users):
            login_link = f"http://{base_ip}:{port}/?user={username}"
            print(f"  {username:20s} -> {login_link}")
        print("-" * 60)
        print(f"共 {len(available_users)} 个用户")
    else:
        print("\n⚠️  警告: 未找到任何用户配置")
    
    print("="*60 + "\n")
    
    # 使用 uvicorn 运行 FastAPI 应用
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
