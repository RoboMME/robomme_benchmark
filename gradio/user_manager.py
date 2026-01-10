import json
import os
import datetime
import threading
from state_manager import cleanup_session, get_task_start_time, clear_task_start_time


class LeaseLost(Exception):
    """Exception raised when a session loses its lease (logged out elsewhere)."""
    pass

class UserManager:
    def __init__(self, tasks_file="user_tasks.json", progress_dir="user_progress"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.tasks_file = os.path.join(self.base_dir, tasks_file)
        self.progress_dir = os.path.join(self.base_dir, progress_dir)
        self.lock = threading.Lock()
        
        # 创建进度目录（如果不存在）
        os.makedirs(self.progress_dir, exist_ok=True)
        
        # In-memory cache for tasks and progress
        self.user_tasks = {}
        self.user_progress = {}
        
        # 会话管理：跟踪每个用户名的活跃 uid
        # {username: active_uid} - 将用户名映射到当前拥有租约的 uid
        # 当同一用户重复登录时，旧会话会被自动清理
        self.active_uid = {}  # {username: uid}
        
        self.load_tasks()
        self.load_progress()
        
    def load_tasks(self):
        """Load user tasks from JSON file."""
        if not os.path.exists(self.tasks_file):
            print(f"Warning: Tasks file {self.tasks_file} not found.")
            return

        try:
            with open(self.tasks_file, 'r', encoding='utf-8') as f:
                self.user_tasks = json.load(f)
            print(f"Loaded tasks for {len(self.user_tasks)} users.")
        except Exception as e:
            print(f"Error loading tasks file: {e}")

    def _get_user_progress_file(self, username):
        """获取用户特定的进度文件路径"""
        safe_username = username.replace("/", "_").replace("\\", "_")
        return os.path.join(self.progress_dir, f"{safe_username}.jsonl")
    
    def has_episode98_success(self, username, env_id):
        """
        检查指定用户和环境的episode98是否有成功记录。
        
        Args:
            username: 用户名
            env_id: 环境ID
        
        Returns:
            bool: 如果存在episode98且status为"success"的记录则返回True，否则返回False
        """
        if not username or not env_id:
            return False
        
        user_progress_file = self._get_user_progress_file(username)
        if not os.path.exists(user_progress_file):
            return False
        
        try:
            with self.lock:
                with open(user_progress_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                            # 检查是否是episode98且匹配env_id且status为success
                            record_env_id = record.get("env_id")
                            record_episode_idx = record.get("episode_idx")
                            record_status = record.get("status", "").lower()
                            
                            if (record_env_id == env_id and 
                                record_episode_idx == 98 and 
                                record_status == "success"):
                                return True
                        except (json.JSONDecodeError, KeyError, ValueError):
                            # 跳过格式错误的记录
                            continue
        except Exception as e:
            print(f"Error checking episode98 success for {username}/{env_id}: {e}")
        
        return False
    
    def check_all_ep98_completed_and_no_other_tasks(self, username):
        """
        检查所有ep98任务是否都已完成且没有任何其他任务完成。
        
        Args:
            username: 用户名
        
        Returns:
            bool: 如果所有ep98任务都已完成且没有其他任务完成，返回True；否则返回False
        """
        if not username:
            return False
        
        # 检查用户是否存在
        if username not in self.user_tasks:
            return False
        
        # 获取用户的所有任务列表
        tasks = self.user_tasks[username]
        
        # 筛选出所有ep98任务，获取它们的env_id集合（去重）
        ep98_tasks = {}
        for task in tasks:
            if task.get("episode_idx") == 98:
                env_id = task.get("env_id")
                if env_id:
                    # 使用env_id作为key，如果同一个env_id有多个ep98任务，只保留一个
                    ep98_tasks[env_id] = True
        
        # 如果没有ep98任务，返回False
        if not ep98_tasks:
            return False
        
        # 检查所有ep98任务是否都有成功记录
        for env_id in ep98_tasks.keys():
            if not self.has_episode98_success(username, env_id):
                return False
        
        # 检查用户进度文件中是否有任何非ep98任务的成功记录
        user_progress_file = self._get_user_progress_file(username)
        if not os.path.exists(user_progress_file):
            # 如果没有进度文件，但所有ep98都检查完成（上面已经检查过），返回True
            return True
        
        try:
            with self.lock:
                with open(user_progress_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                            record_episode_idx = record.get("episode_idx")
                            record_status = record.get("status", "").lower()
                            
                            # 检查是否有非ep98任务的成功记录
                            if (record_episode_idx is not None and 
                                record_episode_idx != 98 and 
                                record_status == "success"):
                                return False
                        except (json.JSONDecodeError, KeyError, ValueError):
                            # 跳过格式错误的记录
                            continue
        except Exception as e:
            print(f"Error checking non-ep98 tasks for {username}: {e}")
            return False
        
        # 所有ep98完成且没有其他任务完成
        return True
    
    def load_progress(self):
        """Load user progress from individual JSONL files. 
        Reconstructs the latest state by reading all user files."""
        if not os.path.exists(self.progress_dir):
            return

        try:
            # 遍历进度目录中的所有用户文件
            for filename in os.listdir(self.progress_dir):
                if not filename.endswith('.jsonl'):
                    continue
                
                user_file = os.path.join(self.progress_dir, filename)
                try:
                    with open(user_file, 'r', encoding='utf-8') as f:
                        # 读取该用户文件的所有记录，保留最新的状态
                        for line in f:
                            if not line.strip():
                                continue
                            try:
                                record = json.loads(line)
                                username = record.get("username")
                                if username:
                                    # #region agent log
                                    import json as json_module
                                    try:
                                        with open('/home/hongzefu/historybench-v5.6.11b5-debug/.cursor/debug.log', 'a', encoding='utf-8') as f_log:
                                            f_log.write(json_module.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"user_manager.py:115","message":"load_progress reading record","data":{"username":username,"record":record,"has_current_task_index":"current_task_index" in record},"timestamp":int(__import__('time').time()*1000)})+"\n")
                                    except: pass
                                    # #endregion
                                    # 只更新包含 current_task_index 的记录，避免用默认值0覆盖
                                    if "current_task_index" in record:
                                        # #region agent log
                                        try:
                                            with open('/home/hongzefu/historybench-v5.6.11b5-debug/.cursor/debug.log', 'a', encoding='utf-8') as f_log:
                                                f_log.write(json_module.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"user_manager.py:123","message":"load_progress updating with current_task_index","data":{"username":username,"current_task_index":record.get("current_task_index"),"completed_tasks":record.get("completed_tasks",[])},"timestamp":int(__import__('time').time()*1000)})+"\n")
                                        except: pass
                                        # #endregion
                                        # Update in-memory state with latest record that has current_task_index
                                        self.user_progress[username] = {
                                            "current_task_index": record.get("current_task_index", 0),
                                            "completed_tasks": set(record.get("completed_tasks", []))
                                        }
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    print(f"Error loading progress file {user_file}: {e}")
        except Exception as e:
            print(f"Error loading progress directory: {e}")

    def save_progress_record(self, username, current_index, completed_tasks, 
                            env_id=None, episode_idx=None, status=None, 
                            difficulty=None, language_goal=None, seed=None,
                            start_time=None, end_time=None, timestamp=None):
        """
        Append a progress record to the user-specific JSONL file.
        
        Args:
            username: 用户名
            current_index: 当前任务索引
            completed_tasks: 已完成任务集合
            env_id: 环境ID（可选）
            episode_idx: Episode索引（可选）
            status: 任务状态（可选）
            difficulty: 难度（可选）
            language_goal: 语言目标（可选）
            seed: 随机种子（可选）
            start_time: 任务开始时间，ISO格式字符串（可选）
            end_time: 任务结束时间，ISO格式字符串（可选）
            timestamp: 向后兼容参数，如果提供且start_time/end_time为None，则同时设置为start_time和end_time
        """
        
        record = {
            "username": username
        }
        
        # 处理时间戳：优先使用 start_time 和 end_time，向后兼容 timestamp
        if start_time is not None:
            record["start_time"] = start_time
        if end_time is not None:
            record["end_time"] = end_time
        
        # 向后兼容：如果只提供了 timestamp，则同时设置为 start_time 和 end_time
        if timestamp is not None:
            if start_time is None:
                record["start_time"] = timestamp
            if end_time is None:
                record["end_time"] = timestamp
        elif start_time is None and end_time is None:
            # 如果都没有提供，使用当前时间作为结束时间
            current_time = datetime.datetime.now().isoformat()
            record["end_time"] = current_time
        
        # 添加 episode 相关信息（如果提供）
        if env_id is not None:
            record["env_id"] = env_id
        if episode_idx is not None:
            record["episode_idx"] = episode_idx
        if status is not None:
            record["status"] = status
        if difficulty is not None:
            record["difficulty"] = difficulty
        if language_goal is not None:
            record["language_goal"] = language_goal
        if seed is not None:
            record["seed"] = seed
        
        # 【修复】保存 current_task_index 和 completed_tasks 到 JSONL 文件
        # 这样 load_progress 才能正确恢复任务索引
        record["current_task_index"] = current_index
        record["completed_tasks"] = list(completed_tasks) if isinstance(completed_tasks, set) else completed_tasks
        
        # #region agent log
        import json as json_module
        try:
            with open('/home/hongzefu/historybench-v5.6.11b5-debug/.cursor/debug.log', 'a', encoding='utf-8') as f:
                f.write(json_module.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"user_manager.py:187","message":"save_progress_record before write","data":{"username":username,"current_index":current_index,"completed_tasks":list(completed_tasks) if isinstance(completed_tasks, set) else completed_tasks,"record":record},"timestamp":int(__import__('time').time()*1000)})+"\n")
        except: pass
        # #endregion
        
        with self.lock:
            try:
                user_progress_file = self._get_user_progress_file(username)
                with open(user_progress_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(record) + "\n")
                
                # Update cache
                self.user_progress[username] = {
                    "current_task_index": current_index,
                    "completed_tasks": set(completed_tasks)
                }
            except Exception as e:
                print(f"Error saving progress for {username}: {e}")

    def login(self, username, uid=None):
        """
        验证用户并返回会话信息。
        如果用户名已被另一个 uid 使用，强制接管并清理旧会话的所有资源。
        
        当检测到同一用户重复登录时：
        1. 自动清理旧会话的工作进程（释放 RAM/VRAM）
        2. 清理旧会话的所有状态数据（任务索引、坐标点击、选项选择、帧队列等）
        3. 终止旧的 MJPEG 流
        
        Args:
            username: 要登录的用户名
            uid: 请求登录的会话 uid（可选，但建议提供）
        
        Returns: (success, message, progress_info)
        """
        if not username:
            return False, "Username cannot be empty", None
        
        if username not in self.user_tasks:
            return False, f"User '{username}' not found in task configuration.", None
            
        # Ensure progress entry exists
        if username not in self.user_progress:
            self.user_progress[username] = {
                "current_task_index": 0,
                "completed_tasks": set()
            }
        
        # 强制接管：如果用户名已被另一个 uid 使用，覆盖它并清理旧会话资源
        # 清理旧会话的工作进程（释放 RAM/VRAM）和所有状态数据
        if uid:
            with self.lock:
                old_uid = self.active_uid.get(username)
                if old_uid and old_uid != uid:
                    print(f"强制接管: 用户 {username} 的旧会话 {old_uid} 被新会话 {uid} 接管")
                    # 清理旧会话的所有资源（进程、RAM、VRAM、状态数据等）
                    print(f"正在清理用户 {username} 的旧会话 {old_uid}...")
                    cleanup_session(old_uid)
                self.active_uid[username] = uid
            
        return True, f"Welcome, {username}!", self.get_user_status(username)
    
    def assert_lease(self, username, uid):
        """
        断言给定的 uid 拥有该用户名的租约。
        如果该 uid 不拥有租约（例如用户在其他地方登录），则抛出 LeaseLost 异常。
        
        Args:
            username: 要检查的用户名
            uid: 声称拥有租约的 uid
        
        Raises:
            LeaseLost: 如果该 uid 不拥有该用户名的租约
        """
        if not username or not uid:
            raise LeaseLost(f"Invalid username or uid")
        
        with self.lock:
            active_uid = self.active_uid.get(username)
            if active_uid != uid:
                raise LeaseLost(f"Lease lost: {username} is now owned by another session. You have been logged out elsewhere.")

    def get_user_status(self, username):
        """Get current status for a user."""
        if username not in self.user_tasks:
            return None
            
        tasks = self.user_tasks[username]
        progress = self.user_progress.get(username, {"current_task_index": 0, "completed_tasks": set()})
        
        current_idx = progress["current_task_index"]
        completed = progress["completed_tasks"]
        
        # #region agent log
        import json as json_module
        try:
            with open('/home/hongzefu/historybench-v5.6.11b5-debug/.cursor/debug.log', 'a', encoding='utf-8') as f:
                f.write(json_module.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"user_manager.py:272","message":"get_user_status","data":{"username":username,"current_idx":current_idx,"total_tasks":len(tasks),"completed_count":len(completed)},"timestamp":int(__import__('time').time()*1000)})+"\n")
        except: pass
        # #endregion
        
        # Ensure index is within bounds
        if current_idx >= len(tasks):
            current_task = None
            is_done_all = True
        else:
            current_task = tasks[current_idx]
            is_done_all = False
            
        return {
            "username": username,
            "total_tasks": len(tasks),
            "current_index": current_idx,
            "completed_count": len(completed),
            "current_task": current_task,
            "is_done_all": is_done_all,
            "tasks": tasks
        }

    def complete_current_task(self, username, env_id=None, episode_idx=None, 
                             status=None, difficulty=None, language_goal=None, seed=None):
        """Mark current task as complete and move to next."""
        # #region agent log
        import json as json_module
        try:
            with open('/home/hongzefu/historybench-v5.6.11b5-debug/.cursor/debug.log', 'a', encoding='utf-8') as f:
                f.write(json_module.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"user_manager.py:293","message":"complete_current_task called","data":{"username":username,"env_id":env_id,"episode_idx":episode_idx,"status":status,"difficulty":difficulty,"language_goal":language_goal,"seed":seed},"timestamp":int(__import__('time').time()*1000)})+"\n")
        except: pass
        # #endregion
        
        user_status = self.get_user_status(username)
        if not user_status or user_status["is_done_all"]:
            return None
            
        current_idx = user_status["current_index"]
        completed = self.user_progress[username]["completed_tasks"]
        
        # Mark as completed
        completed.add(current_idx)
        
        # Move to next task
        next_idx = current_idx + 1
        
        # 获取任务开始时间（如果存在）
        start_time = None
        if env_id is not None and episode_idx is not None:
            start_time = get_task_start_time(username, env_id, episode_idx)
        
        # #region agent log
        try:
            with open('/home/hongzefu/historybench-v5.6.11b5-debug/.cursor/debug.log', 'a', encoding='utf-8') as f:
                f.write(json_module.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"user_manager.py:312","message":"start_time retrieved","data":{"username":username,"env_id":env_id,"episode_idx":episode_idx,"start_time":start_time},"timestamp":int(__import__('time').time()*1000)})+"\n")
        except: pass
        # #endregion
        
        # 获取任务结束时间
        end_time = datetime.datetime.now().isoformat()
        
        # #region agent log
        try:
            with open('/home/hongzefu/historybench-v5.6.11b5-debug/.cursor/debug.log', 'a', encoding='utf-8') as f:
                f.write(json_module.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"user_manager.py:318","message":"before save_progress_record","data":{"username":username,"next_idx":next_idx,"env_id":env_id,"episode_idx":episode_idx,"status":status,"difficulty":difficulty,"language_goal":language_goal,"seed":seed,"start_time":start_time,"end_time":end_time},"timestamp":int(__import__('time').time()*1000)})+"\n")
        except: pass
        # #endregion
        
        # Save persistence with episode information
        self.save_progress_record(
            username, next_idx, completed,
            env_id=env_id,
            episode_idx=episode_idx,
            status=status,
            difficulty=difficulty,
            language_goal=language_goal,
            seed=seed,
            start_time=start_time,
            end_time=end_time
        )
        
        # 清理任务开始时间记录（避免内存泄漏）
        if env_id is not None and episode_idx is not None:
            clear_task_start_time(username, env_id, episode_idx)
        
        return self.get_user_status(username)



    def set_task_index(self, username, index):
        """Manually set task index (if needed)."""
        if username not in self.user_tasks:
            return False
            
        tasks = self.user_tasks[username]
        if 0 <= index <= len(tasks):
             progress = self.user_progress.get(username, {"current_task_index": 0, "completed_tasks": set()})
             self.save_progress_record(username, index, progress["completed_tasks"])
             return True
        return False

# Global instance for simplicity in app.py
user_manager = UserManager()
