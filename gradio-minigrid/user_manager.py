import json
import os
import datetime
from pathlib import Path
import threading

class UserManager:
    def __init__(self, tasks_file="user_tasks.json", progress_file="user_progress.jsonl"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.tasks_file = os.path.join(self.base_dir, tasks_file)
        self.progress_file = os.path.join(self.base_dir, progress_file)
        self.lock = threading.Lock()
        
        # In-memory cache for tasks and progress
        self.user_tasks = {}
        self.user_progress = {}
        
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

    def load_progress(self):
        """Load user progress from JSONL file. 
        Reconstructs the latest state by reading all lines."""
        if not os.path.exists(self.progress_file):
            return

        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        username = record.get("username")
                        if username:
                            # Update in-memory state with latest record
                            self.user_progress[username] = {
                                "current_task_index": record.get("current_task_index", 0),
                                "completed_tasks": set(record.get("completed_tasks", []))
                            }
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error loading progress file: {e}")

    def save_progress_record(self, username, current_index, completed_tasks):
        """Append a progress record to the JSONL file."""
        record = {
            "username": username,
            "current_task_index": current_index,
            "completed_tasks": list(completed_tasks),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        with self.lock:
            try:
                with open(self.progress_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(record) + "\n")
                
                # Update cache
                self.user_progress[username] = {
                    "current_task_index": current_index,
                    "completed_tasks": set(completed_tasks)
                }
            except Exception as e:
                print(f"Error saving progress for {username}: {e}")

    def login(self, username):
        """
        Validate user and return their session info.
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
            
        return True, f"Welcome, {username}!", self.get_user_status(username)

    def get_user_status(self, username):
        """Get current status for a user."""
        if username not in self.user_tasks:
            return None
            
        tasks = self.user_tasks[username]
        progress = self.user_progress.get(username, {"current_task_index": 0, "completed_tasks": set()})
        
        current_idx = progress["current_task_index"]
        completed = progress["completed_tasks"]
        
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

    def complete_current_task(self, username):
        """Mark current task as complete and move to next."""
        status = self.get_user_status(username)
        if not status or status["is_done_all"]:
            return None
            
        current_idx = status["current_index"]
        completed = self.user_progress[username]["completed_tasks"]
        
        # Mark as completed
        completed.add(current_idx)
        
        # Move to next task
        next_idx = current_idx + 1
        
        # Save persistence
        self.save_progress_record(username, next_idx, completed)
        
        return self.get_user_status(username)

    def skip_task(self, username):
        """Skip current task (move to next without marking complete? Or just move index).
        For now, let's treat 'next' button as just moving index, but usually we only move if completed.
        If requirement is 'one by one', maybe we shouldn't allow skip unless completed.
        But user might want to revisit? 
        The prompt says 'user one by one do tasks', 'can exit and re-enter'.
        Let's assume we advance index only on completion.
        """
        pass # Not implementing explicit skip unless requested.

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
