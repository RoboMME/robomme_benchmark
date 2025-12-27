"""
多进程会话管理模块

本模块实现了多进程架构，将每个用户的 OracleSession 运行在独立的工作进程中。
这样可以确保重计算任务不会阻塞主进程，多个用户可以并发使用系统。

架构说明：
1. ProcessSessionProxy: 主进程中的代理类，提供与 OracleSession 相同的接口
2. session_worker_loop: 工作进程中的循环函数，运行实际的 OracleSession
3. 进程间通信：通过 multiprocessing.Queue 进行命令和结果的传递
4. 视频帧同步：工作进程产生的新帧通过 stream_queue 推送到主进程，由后台线程同步到代理的本地缓存
"""
import multiprocessing
import queue
import threading
import time
import traceback
import numpy as np
import sys
import os

# 添加父目录到路径（逻辑复制自 oracle_logic.py）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from oracle_logic import OracleSession, DEFAULT_DATASET_ROOT

# 定义命令常量
CMD_LOAD_EPISODE = "load_episode"
CMD_UPDATE_OBSERVATION = "update_observation"
CMD_GET_PIL_IMAGE = "get_pil_image"
CMD_EXECUTE_ACTION = "execute_action"
CMD_CLOSE = "close"

def _sanitize_options(options):
    """
    清理选项数据，移除不可序列化的项（如 'solve' 函数）
    
    在跨进程通信时，需要确保所有数据都可以被 pickle 序列化。
    raw_solve_options 中包含的 'solve' 函数无法序列化，需要移除。
    'available' 字段可能是复杂对象，需要转换为简单的布尔值。
    
    Args:
        options: 原始选项列表
        
    Returns:
        list: 清理后的选项列表
    """
    clean_opts = []
    if not options:
        return clean_opts
    for opt in options:
        clean_opt = opt.copy()
        if "solve" in clean_opt:
            del clean_opt["solve"]
        if "available" in clean_opt:
            # Only keep truthiness for UI logic
            clean_opt["available"] = bool(clean_opt["available"])
        clean_opts.append(clean_opt)
    return clean_opts

def session_worker_loop(cmd_queue, result_queue, stream_queue, dataset_root, gui_render):
    """
    工作进程主循环
    
    此函数在工作进程中运行，负责：
    1. 初始化 OracleSession 实例
    2. 监听来自主进程的命令（通过 cmd_queue）
    3. 执行命令并返回结果（通过 result_queue）
    4. 监控视频帧变化，将新帧推送到流队列（通过 stream_queue）
    5. 处理异常和清理资源
    
    Args:
        cmd_queue: 命令队列，主进程发送命令到此队列
        result_queue: 结果队列，工作进程返回命令执行结果到此队列
        stream_queue: 流队列，工作进程推送新视频帧到此队列
        dataset_root: 数据集根目录路径
        gui_render: 是否使用GUI渲染模式
    """
    session = None
    try:
        session = OracleSession(dataset_root=dataset_root, gui_render=gui_render)
        
        while True:
            try:
                # Check for commands
                cmd_data = cmd_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            cmd = cmd_data["cmd"]
            args = cmd_data.get("args", [])
            kwargs = cmd_data.get("kwargs", {})
            
            if cmd == CMD_CLOSE:
                if session:
                    session.close()
                break
            
            elif cmd == CMD_LOAD_EPISODE:
                # 加载环境episode
                res = session.load_episode(*args, **kwargs)
                
                # 更新帧索引跟踪（用于增量同步）
                session.last_base_frame_idx = len(session.base_frames)
                session.last_wrist_frame_idx = len(session.wrist_frames)
                
                # 获取演示状态（从 DemonstrationWrapper 获取）
                is_demonstration = False
                if session.env:
                    is_demonstration = getattr(session.env, 'current_task_demonstration', False)
                
                # 构建状态更新（完整同步，因为这是加载操作）
                state_update = {
                    "env_id": session.env_id,
                    "episode_idx": session.episode_idx,
                    "language_goal": session.language_goal,
                    "difficulty": session.difficulty,
                    "demonstration_frames": session.demonstration_frames,
                    "base_frames": session.base_frames,  # 加载时完整同步
                    "wrist_frames": session.wrist_frames,  # 加载时完整同步
                    "available_options": session.available_options,
                    "raw_solve_options": _sanitize_options(session.raw_solve_options),
                    "seg_vis": session.seg_vis,
                    "is_demonstration": is_demonstration
                }
                result_queue.put({"status": "success", "result": res, "state": state_update})
                
            elif cmd == CMD_EXECUTE_ACTION:
                # 执行动作（重计算任务）
                res = session.execute_action(*args, **kwargs)
                
                # 增量帧同步：只发送新增的帧
                new_base = session.base_frames[session.last_base_frame_idx:]
                new_wrist = session.wrist_frames[session.last_wrist_frame_idx:]
                
                # 更新帧索引
                session.last_base_frame_idx = len(session.base_frames)
                session.last_wrist_frame_idx = len(session.wrist_frames)
                
                # 如果有新帧，推送到流队列
                if new_base or new_wrist:
                    stream_queue.put({"base": new_base, "wrist": new_wrist})

                # 获取演示状态（从 DemonstrationWrapper 获取）
                is_demonstration = False
                if session.env:
                    is_demonstration = getattr(session.env, 'current_task_demonstration', False)

                # 构建状态更新（只更新选项和分割视图，帧通过流队列同步）
                state_update = {
                    "available_options": session.available_options,
                    "raw_solve_options": _sanitize_options(session.raw_solve_options),
                    "seg_vis": session.seg_vis,
                    "is_demonstration": is_demonstration
                }
                result_queue.put({"status": "success", "result": res, "state": state_update})

            elif cmd == CMD_GET_PIL_IMAGE:
                res = session.get_pil_image(*args, **kwargs)
                result_queue.put({"status": "success", "result": res})
                
            elif cmd == CMD_UPDATE_OBSERVATION:
                # 更新观察（获取当前环境状态）
                res = session.update_observation(*args, **kwargs)
                
                # 增量帧同步
                new_base = session.base_frames[session.last_base_frame_idx:]
                new_wrist = session.wrist_frames[session.last_wrist_frame_idx:]
                
                # 更新帧索引
                session.last_base_frame_idx = len(session.base_frames)
                session.last_wrist_frame_idx = len(session.wrist_frames)

                # 如果有新帧，推送到流队列
                if new_base or new_wrist:
                    stream_queue.put({"base": new_base, "wrist": new_wrist})
                
                # 获取演示状态（从 DemonstrationWrapper 获取）
                is_demonstration = False
                if session.env:
                    is_demonstration = getattr(session.env, 'current_task_demonstration', False)
                    
                # 构建状态更新
                state_update = {
                    "available_options": session.available_options,
                    "raw_solve_options": _sanitize_options(session.raw_solve_options),
                    "seg_vis": session.seg_vis,
                    "is_demonstration": is_demonstration
                }
                result_queue.put({"status": "success", "result": res, "state": state_update})
                
            else:
                result_queue.put({"status": "error", "message": f"Unknown command: {cmd}"})
                
    except Exception as e:
        traceback.print_exc()
        result_queue.put({"status": "fatal", "message": str(e)})


class ProcessSessionProxy:
    """
    进程会话代理类
    
    此类在主进程中运行，提供与 OracleSession 相同的接口。
    所有方法调用都会被转发到工作进程中的实际 OracleSession 实例。
    
    主要功能：
    1. 启动和管理工作进程
    2. 通过队列与工作进程通信
    3. 维护本地状态缓存（从工作进程同步）
    4. 后台线程实时同步视频帧
    """
    
    def __init__(self, dataset_root=DEFAULT_DATASET_ROOT, gui_render=False):
        """
        初始化代理对象
        
        Args:
            dataset_root: 数据集根目录路径
            gui_render: 是否使用GUI渲染模式
        """
        # 使用 spawn 上下文以获得更清晰的进程隔离
        ctx = multiprocessing.get_context("spawn")
        
        # 创建进程间通信队列
        self.cmd_queue = ctx.Queue()      # 命令队列：主进程 -> 工作进程
        self.result_queue = ctx.Queue()   # 结果队列：工作进程 -> 主进程
        self.stream_queue = ctx.Queue()    # 流队列：工作进程 -> 主进程（视频帧）
        
        # 启动工作进程
        self.process = ctx.Process(
            target=session_worker_loop,
            args=(self.cmd_queue, self.result_queue, self.stream_queue, dataset_root, gui_render),
            daemon=True
        )
        self.process.start()
        
        # 本地状态缓存（从工作进程同步）
        self.env_id = None
        self.episode_idx = None
        self.language_goal = ""
        self.difficulty = None
        self.demonstration_frames = []
        self.base_frames = []  # 由后台同步线程持续更新
        self.wrist_frames = []  # 由后台同步线程持续更新
        self.available_options = []
        self.raw_solve_options = []
        self.seg_vis = None
        self.is_demonstration = False  # 演示模式标志
        
        # 帧同步线程：从流队列接收新帧并更新本地缓存
        self.stop_sync = False
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()

    def _sync_loop(self):
        """
        后台线程循环：从流队列消费视频帧并更新本地缓存
        
        此线程持续运行，实时接收工作进程推送的新视频帧，
        并将其追加到本地的 base_frames 和 wrist_frames 列表中。
        这样 streaming_service 就可以直接从代理的本地缓存读取帧数据。
        """
        while not self.stop_sync:
            try:
                # Use a short timeout to check stop_sync frequently
                frames = self.stream_queue.get(timeout=0.1)
                new_base = frames.get("base", [])
                new_wrist = frames.get("wrist", [])
                
                # Append to local lists
                if new_base:
                    self.base_frames.extend(new_base)
                if new_wrist:
                    self.wrist_frames.extend(new_wrist)
            except queue.Empty:
                continue
            except Exception:
                break
    
    def _send_cmd(self, cmd, *args, **kwargs):
        """
        发送命令到工作进程并等待结果
        
        Args:
            cmd: 命令名称
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            命令执行结果
            
        Raises:
            RuntimeError: 工作进程返回错误或致命错误
            TimeoutError: 工作进程超时（600秒）
        """
        # 发送命令到工作进程
        self.cmd_queue.put({"cmd": cmd, "args": args, "kwargs": kwargs})
        try:
            # 等待结果（重任务如加载/执行可能需要较长时间，设置600秒超时）
            res = self.result_queue.get(timeout=600) 
            
            # 检查错误状态
            if res.get("status") == "fatal":
                raise RuntimeError(f"工作进程致命错误: {res.get('message')}")
            if res.get("status") == "error":
                raise RuntimeError(f"命令执行错误: {res.get('message')}")
            
            # 更新本地状态缓存（如果工作进程返回了状态更新）
            if "state" in res:
                state = res["state"]
                for k, v in state.items():
                    if k in ["base_frames", "wrist_frames"]:
                        # 对于帧数据：只有在显式发送时才替换（如加载时）
                        # 否则由同步循环处理增量更新
                        if v is not None:
                             setattr(self, k, v)
                    else:
                        # 其他状态直接更新
                        setattr(self, k, v)
                        
            return res.get("result")
        except queue.Empty:
            raise TimeoutError("工作进程超时")

    def load_episode(self, env_id, episode_idx):
        """
        加载环境episode（在工作进程中执行）
        
        Args:
            env_id: 环境ID
            episode_idx: episode索引
            
        Returns:
            tuple: (PIL.Image, str) 图像和状态消息
        """
        return self._send_cmd(CMD_LOAD_EPISODE, env_id, episode_idx)

    def execute_action(self, action_idx, click_coords):
        """
        执行动作（在工作进程中执行，重计算任务）
        
        Args:
            action_idx: 动作索引
            click_coords: 点击坐标 (x, y) 或 None
            
        Returns:
            tuple: (PIL.Image, str, bool) 图像、状态消息、是否完成
        """
        return self._send_cmd(CMD_EXECUTE_ACTION, action_idx, click_coords)
        
    def get_pil_image(self, use_segmented=True):
        """
        获取PIL图像（在工作进程中执行）
        
        Args:
            use_segmented: 是否使用分割视图
            
        Returns:
            PIL.Image: 图像对象
        """
        return self._send_cmd(CMD_GET_PIL_IMAGE, use_segmented=use_segmented)
        
    def update_observation(self, use_segmentation=True):
        """
        更新观察（在工作进程中执行）
        
        Args:
            use_segmentation: 是否使用分割视图
            
        Returns:
            tuple: (PIL.Image, str) 图像和状态消息
        """
        return self._send_cmd(CMD_UPDATE_OBSERVATION, use_segmentation=use_segmentation)
        
    def close(self):
        """
        关闭代理并清理资源
        
        此方法会：
        1. 停止帧同步线程
        2. 发送关闭命令到工作进程
        3. 等待工作进程优雅退出（最多1秒）
        4. 如果进程仍在运行，强制终止
        """
        self.stop_sync = True
        try:
            self.cmd_queue.put({"cmd": CMD_CLOSE})
        except:
            pass
        # 等待工作进程优雅退出
        self.process.join(timeout=1)
        if self.process.is_alive():
            self.process.terminate()

