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
    def infer(self, inputs: dict):
        # inputs: dict of observations
        # output is dict of action chunk: {"actions": np.ndarray}
        # If action space is joint_angle, the action shape is (chunk_size, 8)
        # Otherwise, the action shape is (chunk_size, 7)
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

 
class DummyPolicy(Policy):
    # A random policy that saves video for debugging
    def __init__(self):
        self.chunk_size = 10

    def infer(self, inputs: dict):        
        # We need to differentiate the first step from the subsequent steps
        # For video-conditioned tasks, there would be more than one steps in inputs, the last step is the current step ready for execution, all previous steps are the conditioned video frames
        # For normal tasks, there would be only one step in inputs, which is the current step ready for execution
        if inputs["is_first_step"]:
            self.exec_start_idx = len(inputs["front_rgb_list"]) - 1 # sample id < self.exec_id is the conditioned video frames
        
        # Fake action chunk for debugging
        action_chunk = np.concatenate([BASE_ACTION] * self.chunk_size, axis=0).reshape(-1, 8)
        return {"action": add_small_noise(action_chunk)}

    def reset(self):
        self.exec_start_idx = 0
    

class YourPolicy(Policy):
    ...