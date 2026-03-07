import json
import os
import random
import threading
from pathlib import Path

from config import TASK_NAME_LIST
from state_manager import clear_task_start_time, get_task_start_time


METADATA_FILE_GLOB = "record_dataset_*_metadata.json"


class UserManager:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.lock = threading.Lock()

        self.env_to_episodes = self._load_env_episode_pool()
        self.env_choices = self._build_env_choices()

        # Session-local progress only (no disk persistence)
        self.session_progress = {}

    def _resolve_metadata_root(self) -> Path:
        env_root = os.environ.get("ROBOMME_METADATA_ROOT")
        if env_root:
            return Path(env_root)
        return self.base_dir.parent / "src" / "robomme" / "env_metadata" / "train"

    def _load_env_episode_pool(self):
        env_to_episode_set = {}
        metadata_root = self._resolve_metadata_root()
        if not metadata_root.exists():
            print(f"Warning: metadata root not found: {metadata_root}")
            return {}

        for metadata_path in sorted(metadata_root.glob(METADATA_FILE_GLOB)):
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"Warning: failed to read metadata file {metadata_path}: {exc}")
                continue

            fallback_env = str(payload.get("env_id") or "").strip()
            for record in payload.get("records", []):
                env_id = str(record.get("task") or fallback_env or "").strip()
                episode = record.get("episode")
                if not env_id or episode is None:
                    continue
                try:
                    episode_idx = int(episode)
                except (TypeError, ValueError):
                    continue
                env_to_episode_set.setdefault(env_id, set()).add(episode_idx)

        env_to_episodes = {
            env_id: sorted(episodes)
            for env_id, episodes in env_to_episode_set.items()
            if episodes
        }
        print(f"Loaded random env pool: {len(env_to_episodes)} envs from metadata root {metadata_root}")
        return env_to_episodes

    def _build_env_choices(self):
        available_envs = set(self.env_to_episodes.keys())
        ordered_choices = [env_id for env_id in TASK_NAME_LIST if env_id in available_envs]
        remaining_choices = sorted(available_envs - set(ordered_choices))
        return ordered_choices + remaining_choices

    def _ensure_session_entry(self, uid):
        if uid not in self.session_progress:
            self.session_progress[uid] = {
                "completed_count": 0,
                "current_env_id": None,
                "current_episode_idx": None,
            }

    def _set_current_random_task(self, uid, preferred_env=None):
        if not self.env_choices:
            return False
        self._ensure_session_entry(uid)

        env_id = preferred_env if preferred_env in self.env_to_episodes else random.choice(self.env_choices)
        episodes = self.env_to_episodes.get(env_id, [])
        if not episodes:
            return False

        episode_idx = int(random.choice(episodes))
        self.session_progress[uid]["current_env_id"] = env_id
        self.session_progress[uid]["current_episode_idx"] = episode_idx
        return True

    def init_session(self, uid):
        if not uid:
            return False, "Session uid cannot be empty", None
        if not self.env_choices:
            return False, "No available environments found in metadata.", None

        with self.lock:
            self._ensure_session_entry(uid)
            progress = self.session_progress[uid]
            if progress.get("current_env_id") is None or progress.get("current_episode_idx") is None:
                if not self._set_current_random_task(uid):
                    return False, "Failed to assign random task from metadata.", None

        return True, "Session initialized", self.get_session_status(uid)

    def get_session_status(self, uid):
        if not uid:
            return None

        with self.lock:
            self._ensure_session_entry(uid)
            progress = self.session_progress[uid]
            if (
                (progress.get("current_env_id") is None or progress.get("current_episode_idx") is None)
                and self.env_choices
            ):
                self._set_current_random_task(uid)
                progress = self.session_progress[uid]

            current_task = None
            if progress.get("current_env_id") is not None and progress.get("current_episode_idx") is not None:
                current_task = {
                    "env_id": progress["current_env_id"],
                    "episode_idx": int(progress["current_episode_idx"]),
                }

            completed_count = int(progress.get("completed_count", 0))

        return {
            "uid": uid,
            "total_tasks": len(self.env_choices),  # compatibility only
            "current_index": completed_count,  # compatibility only
            "completed_count": completed_count,
            "current_task": current_task,
            "is_done_all": False,
            "tasks": [],  # compatibility only
            "env_choices": list(self.env_choices),
        }

    def complete_current_task(self, uid, env_id=None, episode_idx=None, **_kwargs):
        if not uid:
            return None

        with self.lock:
            self._ensure_session_entry(uid)
            self.session_progress[uid]["completed_count"] = int(self.session_progress[uid]["completed_count"]) + 1

        if env_id is not None and episode_idx is not None:
            _ = get_task_start_time(uid, env_id, episode_idx)
            clear_task_start_time(uid, env_id, episode_idx)

        return self.get_session_status(uid)

    def switch_env_and_random_episode(self, uid, env_id):
        if not uid or env_id not in self.env_to_episodes:
            return None

        with self.lock:
            self._ensure_session_entry(uid)
            if not self._set_current_random_task(uid, preferred_env=env_id):
                return None

        return self.get_session_status(uid)

    def next_episode_same_env(self, uid):
        if not uid:
            return None

        with self.lock:
            self._ensure_session_entry(uid)
            current_env = self.session_progress[uid].get("current_env_id")
            if current_env not in self.env_to_episodes:
                if not self._set_current_random_task(uid):
                    return None
            else:
                if not self._set_current_random_task(uid, preferred_env=current_env):
                    return None

        return self.get_session_status(uid)


user_manager = UserManager()
