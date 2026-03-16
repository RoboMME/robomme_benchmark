"""
The participant mainly implements this class.
"""

import numpy as np
import cv2
import imageio


BASE_ACTION = np.array(
    [0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, np.pi / 4, 1.0],
    dtype=np.float32,
)

def add_small_noise(
    action: np.ndarray, noise_level: float = 0.1
) -> np.ndarray:
    noise = np.random.normal(0, noise_level, action.shape)
    noise[..., -1:] = 0.0
    return action + noise

   
class Policy:
    def step(self, buffer: dict):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

 
class DummyPolicy(Policy):
    # A random policy that saves video for debugging
    def __init__(self):
        self.imgs = []
        self.chunk_size = 10

    def step(self, buffer: dict):
        import pdb; pdb.set_trace()
        for (img, wrist_img) in zip(buffer["front_rgb_list"], buffer["wrist_rgb_list"]):
            self.imgs.append(np.hstack([img, wrist_img]))
        
        if buffer["is_first_step"]:
            self.exec_id = len(buffer["front_rgb_list"]) - 1 # sample id < self.exec_id is the conditioned video frames
        action_chunk = np.concatenate([BASE_ACTION] * self.chunk_size, axis=0).reshape(-1, 8)
        return {"action": add_small_noise(action_chunk)}

    def reset(self):
        if self.imgs:
            self._save_video(f"test_video.mp4")
        self.imgs = []
        self.exec_id = None
    
    def _save_video(self, file_path: str):
        # just for checking 
        video_frames = []
        for i, img in enumerate(self.imgs):
            if i < self.exec_id:
                # add border to the frame
                img = cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (255, 0, 0), 10)
            video_frames.append(img)
        imageio.mimsave(file_path, video_frames, fps=30)


class YourPolicy(Policy):
    ...