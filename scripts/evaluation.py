import os
import torch
import numpy
import random
import numpy as np
import cv2
import imageio

from pathlib import Path
from robomme.env_record_wrapper import BenchmarkEnvBuilder

class VideoRecorder:
    BORDER_COLOR = (255, 0, 0)
    BORDER_THICKNESS = 10

    def __init__(self, fps: int = 30):
        self.fps = fps
        self.frames: list[np.ndarray] = []

    @staticmethod
    def _to_numpy(t) -> np.ndarray:
        return t.cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)

    @classmethod
    def _make_frame(
        cls,
        front: np.ndarray | torch.Tensor,
        wrist: np.ndarray | torch.Tensor,
        is_video_demo: bool = False,
    ) -> np.ndarray:
        frame = np.hstack([cls._to_numpy(front), cls._to_numpy(wrist)]).astype(np.uint8)
        if is_video_demo:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, h), cls.BORDER_COLOR, cls.BORDER_THICKNESS)
        return frame

    def add_initial_obs(self, obs: dict):
        rgb_list = obs["front_rgb_list"]
        for i, (front, wrist) in enumerate(zip(rgb_list, obs["wrist_rgb_list"])):
            self.frames.append(self._make_frame(front, wrist, is_video_demo=i < len(rgb_list) - 1))

    def add_step_obs(self, obs: dict):
        self.frames.append(self._make_frame(
            obs["front_rgb_list"][-1], obs["wrist_rgb_list"][-1],
        ))

    def save(self, file_path: str):
        dir_path = Path(file_path).parent
        dir_path.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(file_path, self.frames, fps=self.fps)
        self.frames = []

class DummyModel:
    def __init__(self, seed: int):
        self.base_action = np.array(
            [0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, np.pi / 4, 1.0],
            dtype=np.float32,
        )
        self.set_model_seed(seed) 
    
    def set_model_seed(self, seed: int):
        # set model seed will not affect the env seed
        # env seed is fixed internally 
        torch.manual_seed(seed)
        numpy.random.seed(seed)
        random.seed(seed)
        self.seed = seed
    
    def predict(self, *args, **kwargs):
        noise = np.random.normal(0, 0.01, self.base_action.shape)
        noise[..., -1:] = 0.0  # Preserve gripper action
        return self.base_action + noise


TASKS = BenchmarkEnvBuilder.get_task_list()
MODEL_SEED = 7 # 7, 42, 0
dummy_model = DummyModel(seed=MODEL_SEED)

total_success = []
for task in TASKS:
    env_builder = BenchmarkEnvBuilder(
        env_id=task,
        dataset="test",
        action_space="joint_angle", # change this to your model's action space
        max_steps=300,  # we set 1300 in MME-VLA experiments.
    )
    episode_count = env_builder.get_episode_num()
    for episode in range(episode_count):
        env = env_builder.make_env_for_episode(episode)
        obs, info = env.reset()
        task_goal = info["task_goal"]
        if isinstance(task_goal, list):
            task_goal = task_goal[0] # you can take alternative task goals if you want
        print(f"\nTask goal: {task_goal}")
        
        recorder = VideoRecorder()
        recorder.add_initial_obs(obs)

        current_front_rgb = obs["front_rgb_list"][-1]
        current_wrist_rgb = obs["wrist_rgb_list"][-1]

        while True:
            dummy_action = dummy_model.predict(current_front_rgb, current_wrist_rgb, task_goal)
            obs, reward, terminated, truncated, info = env.step(dummy_action)
            if info is not None and info.get("status") == "error":
                print(f"Error: {info.get('error_message')}") # often IK error when using ee pose
                total_success.append(False)
                break
            if terminated or truncated:
                outcome = info.get("status", "unknown")
                print(f"Outcome of episode {episode} of task {task}: {outcome}")
                total_success.append(outcome == "success")
                break
            current_front_rgb = obs["front_rgb_list"][-1]
            current_wrist_rgb = obs["wrist_rgb_list"][-1]
            recorder.add_step_obs(obs)
        
        env.close()
        os.makedirs("saved_videos", exist_ok=True)
        recorder.save(file_path=f"saved_videos/{task}_ep_{episode}_{outcome}_{task_goal}.mp4")
        
print(f"Evaluation completed.")
print(f"Success rate: {sum(total_success) / len(total_success)}")