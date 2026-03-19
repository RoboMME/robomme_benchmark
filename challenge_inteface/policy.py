"""
Participants need to modify this file

This is a sample script about how to adapt a model into a remote evaluation policy for CVPR challenge.

Basically, You need to implement the `step` and `reset` methods.
"""

import numpy as np

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
        """
        The `inputs` is a dict of observations, which includes:
        - task_goal: the goal of the task
        - is_first_step: whether the current step is the first step
        - front_rgb_list: the list of front camera RGB frames
        - wrist_rgb_list: the list of wrist camera RGB frames
        - joint_state_list: the list of joint states
        - eef_state_list: the list of end-effector states
        - gripper_state_list: the list of gripper states
            
        - (optional) front_depth_list: the list of front camera depth frames. return only when you select `use_depth` in EvalAI.
        - (optional) wrist_depth_list: the list of wrist camera depth frames. return only when you select `use_depth` in EvalAI.
        - (optional) front_camera_intrinsic: the intrinsic matrix of the front camera. return only when you select `use_camera_params` in EvalAI.
        - (optional) wrist_camera_intrinsic: the intrinsic matrix of the wrist camera. return only when you select `use_camera_params` in EvalAI.
        - (optional) front_camera_extrinsic_list: the list of extrinsic matrix of the front camera. return only when you select `use_camera_params` in EvalAI.
        - (optional) wrist_camera_extrinsic_list: the list of extrinsic matrix of the wrist camera. return only when you select `use_camera_params` in EvalAI.
        
        
        The output is a dict of action chunk: {"actions": np.ndarray}
        
        if action space is joint_angle, the action shape is (chunk_size, 8)
        otherwise, the action shape is (chunk_size, 7)
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset the policy. If your policy is stateful, you need to reset your model state here.
        The organizers will call this at the beginning of each episode.
        """
        raise NotImplementedError
 
class DummyPolicy(Policy):
    # A random policy that saves video for debugging
    def __init__(self):
        self.chunk_size = 10

    def infer(self, inputs: dict):
        """
        We need to differentiate the first step from the subsequent steps
        For video-conditioned tasks, there would be more than one steps in inputs, the last step is the current step ready for execution, all previous steps are the conditioned video frames.
        For non-video-conditioned tasks, there would be only one step in inputs, which is the current step ready for execution
        """
        if inputs["is_first_step"]:
            self.exec_start_idx = len(inputs["front_rgb_list"]) - 1 # sample id < self.exec_id is the conditioned video frames
        action_chunk = np.concatenate([BASE_ACTION] * self.chunk_size, axis=0).reshape(-1, 8)
        return {"actions": add_small_noise(action_chunk)}

    def reset(self):
        self.exec_start_idx = 0
    

class YourPolicy(Policy):
    ...