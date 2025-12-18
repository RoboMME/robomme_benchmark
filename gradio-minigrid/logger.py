import json
import threading
import os
from datetime import datetime
from pathlib import Path

# 线程锁，防止多用户同时写入时文件损坏
lock = threading.Lock()
# 使用基于 logger.py 文件位置的绝对路径，确保日志文件始终保存在 gradio-minigrid/data/ 目录下
BASE_DIR = Path(__file__).parent.absolute()
LOG_FILE = str(BASE_DIR / "data" / "experiment_logs.jsonl")
USER_ACTION_LOG_DIR = str(BASE_DIR / "data" / "user_action_logs")

def log_session(session_data):
    """
    将单个会话的数据追加写入到 JSONL 文件中。
    session_data 应该是一个字典。
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    # 添加写入时间戳
    session_data["logged_at"] = datetime.now().isoformat()
    
    with lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(session_data, ensure_ascii=False) + "\n")

def log_user_action(username, env_id, episode_idx, action_data):
    """
    记录用户的详细操作到JSON文件中。
    
    Args:
        username: 用户名
        env_id: 环境ID
        episode_idx: Episode索引
        action_data: 操作数据字典，包含：
            - action_type: "option_select" | "coordinate_click" | "action_execute"
            - 其他相关字段（option_idx, option_label, coordinates, coords_str, status, done等）
    
    文件路径: data/user_action_logs/{username}/{env_id}_{episode_idx}.json
    文件格式: JSON数组，每个元素是一个操作记录
    """
    if not username or not env_id or episode_idx is None:
        print(f"Warning: Missing required parameters for log_user_action: username={username}, env_id={env_id}, episode_idx={episode_idx}")
        return
    
    # 构建文件路径
    user_dir = os.path.join(USER_ACTION_LOG_DIR, username)
    log_file = os.path.join(user_dir, f"{env_id}_{episode_idx}.json")
    
    # 确保目录存在
    os.makedirs(user_dir, exist_ok=True)
    
    # 添加时间戳
    action_record = {
        **action_data,
        "timestamp": datetime.now().isoformat()
    }
    
    # 使用线程锁确保并发安全
    with lock:
        # 读取现有文件（如果存在）
        actions = []
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    actions = json.load(f)
                    if not isinstance(actions, list):
                        actions = []
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Error reading existing log file {log_file}: {e}. Starting fresh.")
                actions = []
        
        # 追加新操作
        actions.append(action_record)
        
        # 写回文件
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(actions, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"Error writing to log file {log_file}: {e}")
