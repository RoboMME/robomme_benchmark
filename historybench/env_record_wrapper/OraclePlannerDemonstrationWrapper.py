import gymnasium as gym
import numpy as np
import torch
import cv2
import colorsys
from history_bench_sim.oracle_logic import step_before, step_after

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.motionplanner_stick import (
    PandaStickMotionPlanningSolver,
)

def _generate_color_map(n=10000, s_min=0.70, s_max=0.95, v_min=0.78, v_max=0.95):
    phi = 0.6180339887498948
    color_map = {}
    for i in range(1, n + 1):
        h = (i * phi) % 1.0
        s = s_min + (s_max - s_min) * ((i % 7) / 6)
        v = v_min + (v_max - v_min) * (((i * 3) % 5) / 4)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        color_map[i] = [int(round(r * 255)), int(round(g * 255)), int(round(b * 255))]
    return color_map

def _sync_table_color(env, color_map):
    seg_id_map = getattr(env.unwrapped, "segmentation_id_map", None)
    if not isinstance(seg_id_map, dict):
        return
    for obj_id, obj in seg_id_map.items():
        if getattr(obj, "name", None) == "table-workspace":
            color_map[obj_id] = [0, 0, 0]

def _tensor_to_bool(value):
    """Convert tensor or other types to boolean."""
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().bool().item())
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    return bool(value)

class OraclePlannerDemonstrationWrapper(gym.Wrapper):
    """
    Gym Environment Wrapper for Oracle Planner Logic.
    Wraps the environment to provide oracle planner capabilities.
    """
    def __init__(self, env, env_id, gui_render=True):
        super().__init__(env)
        self.env_id = env_id
        self.gui_render = gui_render
        
        self.planner = None
        self.color_map = None
        self.language_goal = None
        
        # State variables
        self.seg_vis = None
        self.seg_raw = None
        self.base_frames = []
        self.wrist_frames = []
        self.available_options = []
        
        # Metadata
        self.action_space = gym.spaces.Dict({}) 
        self.observation_space = gym.spaces.Dict({})

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        
        # Initialization logic
        try:
            self.language_goal = self.env.unwrapped.demonstration_data.get('language goal')
        except AttributeError:
             # Fallback if demonstration_data is missing, possibly read from metadata if available
             # or simply default to None/Empty
            self.language_goal = None
            print(f"Warning: {self.env_id} object has no attribute 'demonstration_data'. 'language_goal' set to None.")

        # Generate semantic segmentation color map
        self.color_map = _generate_color_map()
        _sync_table_color(self.env, self.color_map)

        # Initialize Planner
        if self.env_id in ("PatternLock", "RouteStick"):
            self.planner = PandaStickMotionPlanningSolver(
                self.env,
                debug=False,
                vis=self.gui_render,
                base_pose=self.env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
                joint_vel_limits=0.3,
            )
        else:
            self.planner = PandaArmMotionPlanningSolver(
                self.env,
                debug=False,
                vis=self.gui_render,
                base_pose=self.env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
            )

        # Initial step_before
        self.seg_vis, self.seg_raw, self.base_frames, self.wrist_frames, self.available_options = \
            step_before(self.env, self.planner, self.env_id, self.color_map)
            
        return self._get_obs(), info

    def step(self, action):
        """
        Args:
            action: command_dict containing "action" and "point"
        """
        command_dict = action
        
        # Execute action (step_after)
        evaluation = step_after(
            self.env, 
            self.planner, 
            self.env_id, 
            self.seg_vis, 
            self.seg_raw, 
            self.base_frames, 
            self.wrist_frames, 
            command_dict
        )
        
        success = False
        fail = False
        
        if evaluation:
            fail = _tensor_to_bool(evaluation.get("fail", False))
            success = _tensor_to_bool(evaluation.get("success", False))
            
        terminated = success or fail
        truncated = False 
        reward = 1.0 if success else 0.0
        
        info = {}
        if evaluation:
            info.update(evaluation)
        
        # If not terminated, prepare next step (step_before)
        if not terminated:
            self.seg_vis, self.seg_raw, self.base_frames, self.wrist_frames, self.available_options = \
                step_before(self.env, self.planner, self.env_id, self.color_map)
        
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return {
            "seg_vis": self.seg_vis,
            "seg_raw": self.seg_raw,
            "base_frames": self.base_frames,
            "wrist_frames": self.wrist_frames,
            "available_options": self.available_options,
            "language_goal": self.language_goal
        }
