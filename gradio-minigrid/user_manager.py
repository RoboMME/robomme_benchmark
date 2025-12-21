import json
import os
import datetime
from pathlib import Path
import threading


class LeaseLost(Exception):
    """Exception raised when a session loses its lease (logged out elsewhere)."""
    pass

class UserManager:
    def __init__(self, tasks_file="user_tasks.json", progress_file="user_progress.jsonl"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.tasks_file = os.path.join(self.base_dir, tasks_file)
        self.progress_file = os.path.join(self.base_dir, progress_file)
        self.lock = threading.Lock()
        
        # In-memory cache for tasks and progress
        self.user_tasks = {}
        self.user_progress = {}
        
        # Session management: track active uid for each username
        # {username: active_uid} - maps username to the uid that currently owns the lease
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

    def login(self, username, uid=None):
        """
        Validate user and return their session info.
        If username is already in use by another uid, force takeover (kick old session).
        
        Args:
            username: The username to login
            uid: The session uid requesting login (optional, but recommended)
        
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
        
        # Force takeover: if username is already in use by another uid, override it
        # The old uid will fail on next assert_lease() call
        if uid:
            with self.lock:
                old_uid = self.active_uid.get(username)
                if old_uid and old_uid != uid:
                    print(f"Force takeover: {username} was used by {old_uid}, now taken by {uid}")
                self.active_uid[username] = uid
            
        return True, f"Welcome, {username}!", self.get_user_status(username)
    
    def assert_lease(self, username, uid):
        """
        Assert that the given uid owns the lease for the username.
        Raises LeaseLost exception if the lease is not owned by this uid.
        
        Args:
            username: The username to check
            uid: The uid claiming the lease
        
        Raises:
            LeaseLost: If the uid does not own the lease for this username
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
