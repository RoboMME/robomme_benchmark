from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
import math
import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import SO100, Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

#Robomme
import matplotlib.pyplot as plt

import random
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)

# NOTE: keep wildcard import for legacy helpers that the environment relies on.
from .utils import *
from .utils.subgoal_evaluate_func import *
from .utils.object_generation import *
from .utils import reset_panda
from .utils.difficulty import normalize_robomme_difficulty
from ..logging_utils import logger


PICK_CUBE_DOC_STRING = """**Task Description:**
A simple task where the objective is to grasp a red cube with the {robot_id} robot and move it to a target goal position. This is also the *baseline* task to test whether a robot with manipulation
capabilities can be simulated and trained properly. Hence there is extra code for some robots to set them up properly in this environment as well as the table scene builder.

**Randomizations:**
- the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
- the cube's z-axis rotation is randomized to a random angle
- the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

**Success Conditions:**
- the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
- the robot is static (q velocity < 0.2)
"""


@register_env("PatternLock")
class PatternLock(BaseEnv):

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "xarm6_robotiq",
        "so100",
        "widowxai",
    ]
    agent: Union[Panda]
    goal_thresh = 0.025
    cube_spawn_half_size = 0.05
    cube_spawn_center = (0, 0)

    config_hard = {
        "grid":5,
        "length":[4,8]
    }

    config_easy = {
        "grid":3,
         "length":[2,4]
    }

    config_medium = {
        "grid":4,
        "length":[3,5]
    }

    # Combine into a dictionary
    configs = {
        'hard': config_hard,
        'easy': config_easy,
        'medium': config_medium
    }


    def __init__(self, *args, robot_uids="panda_stick", robot_init_qpos_noise=0,seed=0,Robomme_video_episode=None,Robomme_video_path=None,
                     **kwargs):
        self.achieved_list=[]
        self.match=False
        self.after_demo=False

        self.use_demonstrationwrapper=False
        self.demonstration_record_traj=False
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids in PICK_CUBE_CONFIGS:
            cfg = PICK_CUBE_CONFIGS[robot_uids]
        else:
            cfg = PICK_CUBE_CONFIGS["panda"]
        self.cube_half_size = cfg["cube_half_size"]
        self.goal_thresh = cfg["goal_thresh"]
        self.cube_spawn_half_size = cfg["cube_spawn_half_size"]
        self.cube_spawn_center = cfg["cube_spawn_center"]
        self.max_goal_height = cfg["max_goal_height"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]


        self.robomme_failure_recovery = bool(
            kwargs.pop("robomme_failure_recovery", False)
        )
        self.robomme_failure_recovery_mode = kwargs.pop(
            "robomme_failure_recovery_mode", None
        )
        if isinstance(self.robomme_failure_recovery_mode, str):
            self.robomme_failure_recovery_mode = (
                self.robomme_failure_recovery_mode.lower()
            )
        self.seed = seed
        normalized_robomme_difficulty = normalize_robomme_difficulty(
            kwargs.pop("difficulty", None)
        )
        if normalized_robomme_difficulty is not None:
            self.difficulty = normalized_robomme_difficulty
        else:
            # Determine difficulty based on seed % 3
            seed_mod = seed % 3
            if seed_mod == 0:
                self.difficulty = "easy"
            elif seed_mod == 1:
                self.difficulty = "medium"
            else:  # seed_mod == 2
                self.difficulty = "hard"
        #self.difficulty = "hard"
               # Use seed to determine number of repetitions (1-5) arbitrarily
        generator = torch.Generator()
        generator.manual_seed(seed)


        self.highlight_starts = {}  # Use dictionary to store highlight start time for each button
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        camera_eye=[0.3,0,0.4]
        camera_target =[0,0,-0.2]
        pose = sapien_utils.look_at(
            eye=camera_eye, target=camera_target
        )
        return [CameraConfig("base_camera", pose, 256, 256, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Generate 3x3 grid of buttons
        grid_center = [-0.1, 0]  # Grid center position
        grid_spacing = 0.1  # Spacing between buttons

        self.buttons_grid = []
        self.button_joints_grid = []
        avoid = []
        button_index = 0

        
        num_rows, num_cols = 5, 8
        num_rows, num_cols = self.configs[self.difficulty]["grid"],self.configs[self.difficulty]["grid"]
        row_center = (num_rows - 1) / 2
        col_center = (num_cols - 1) / 2



        for row in range(num_rows):  # 3 rows (x direction)
            for col in range(num_cols):  # 5 columns (y direction)
                x_pos = grid_center[0] + (row - row_center) * grid_spacing
                y_pos = grid_center[1] + (col - col_center) * grid_spacing



                target_name = f"target_{button_index}"

                # Create rotation quaternion for vertical target
                angles = torch.deg2rad(torch.tensor([0.0, 90.0, 0.0], dtype=torch.float32))
                rotate = matrix_to_quaternion(
                    euler_angles_to_matrix(angles, convention="XYZ")
                )

                # Build purple and white target
                target = build_gray_white_target(
                    scene=self.scene,
                    radius=0.02,
                    thickness=0.01,
                    name=target_name,
                    body_type="kinematic",
                    add_collision=False,
                    initial_pose=sapien.Pose(p=[x_pos, y_pos, 0.01], q=rotate),
                )

                self.buttons_grid.append(target)
                # Note: purple_white_target doesn't have joints, so we append None
                self.button_joints_grid.append(None)
                logger.debug(f"Generated target {button_index} at position ({x_pos:.3f}, {y_pos:.3f})")
                button_index += 1

        self.targets_grid = self.buttons_grid

                # Generate task list to move to each button sequentially
        tasks = []

        # start_end_set = [
        #     [0, 1, 8, 9],
        #     [6, 7, 14, 15],
        #     [24, 25, 32, 33],
        #     [30, 31, 38, 39]
        # ]

        # # Randomly select 2 different sets from start_end_set
        # set_indices = torch.randperm(len(start_end_set), generator=generator)[:2].tolist()
        # start_set = start_end_set[set_indices[0]]
        # end_set = start_end_set[set_indices[1]]

        # # Randomly select one node from each set
        # start_idx = torch.randint(0, len(start_set), (1,), generator=generator).item()
        # end_idx = torch.randint(0, len(end_set), (1,), generator=generator).item()

        # num_targets = len(self.targets_grid)
        # node_choices = torch.randperm(num_targets, generator=generator)[:2]
        # start_node, end_node = node_choices.tolist()


        # path_nodes, _, _, _ = find_path_0_to_8(
        #     start=start_node,
        #     target=end_node,
        #     R=num_rows,
        #     C=num_cols,
        #     diagonals=True,
        #     generator=generator,
        # )
        # self.selected_buttons = [self.buttons_grid[i] for i in path_nodes]

        num_targets = len(self.targets_grid)
        max_attempts = 1000  # Safety limit

        for attempt in range(max_attempts):
            node_choices = torch.randperm(num_targets, generator=generator)[:2]
            start_node, end_node = node_choices.tolist()
            
            path_nodes, _, _, _ = find_path_0_to_8(
                start=start_node,
                target=end_node,
                R=num_rows,
                C=num_cols,
                diagonals=True,
                generator=generator,
            )

            length_range = self.configs[self.difficulty]["length"]
            if length_range[0] <= len(path_nodes) <= length_range[1]:
                break
        else:
            # If we couldn't find a path < 5 after max_attempts, use the last one
            logger.debug(f"Warning: Could not find path after {max_attempts} attempts")

        self.selected_buttons = [self.buttons_grid[i] for i in path_nodes]
        current_target=self.selected_buttons[0]
        tasks.append({
            "func":   lambda t=current_target: is_obj_swing_onto(self, obj=self.agent.tcp, target=t),
            "name":  "NO RECORD",
            "subgoal_segment":f"NO RECORD",
            "demonstration": True,
            "failure_func":  lambda expected=current_target: self._wrong_button_touch(expected_button=expected),
            "solve": lambda env, planner, t=current_target: solve_swingonto(env, planner, target=t,record_swing_qpos=True),
        })  
        for i, current_target in enumerate(self.selected_buttons[1:]):
            last_target = self.selected_buttons[i]
            tasks.append({
            "func":   lambda t=current_target: is_obj_swing_onto(self, obj=self.agent.tcp, target=t),
            "name": f"move {direction(current_target, last_target)}",
            "subgoal_segment":f"move {direction(current_target, last_target)}",
            "choice_label": f"move {direction(current_target, last_target)}",
            "demonstration": True,
            "failure_func":  lambda expected=current_target, last=last_target: self._wrong_button_touch(expected_button=expected, last_button=last),
            "solve": lambda env, planner, t=current_target: solve_swingonto(env, planner, target=t),
            #"segment":current_target,
        })  
        
        tasks.append({
                    "func": lambda:reset_check(self,gripper="stick"),
                    "name": "NO RECORD",
                    "subgoal_segment":f"NO RECORD",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: [solve_strong_reset(env,planner,gripper="stick")],
                    },)
        
      
        self.selected_buttons = [self.buttons_grid[i] for i in path_nodes]
        current_target=self.selected_buttons[0]
        tasks.append({
            "func":   lambda:reset_check(self,gripper="stick",target_qpos=self.swing_qpos),
            "name":  "NO RECORD",
            "subgoal_segment":f"NO RECORD",
            "demonstration": True,
            "failure_func":  None,
            "solve": lambda env, planner, t=current_target: [solve_strong_reset(env, planner,gripper="stick",action=self.swing_qpos)],
        })  
        for i, current_target in enumerate(self.selected_buttons[1:]):
            last_target = self.selected_buttons[i]
            tasks.append({
            "func":   lambda t=current_target: is_obj_swing_onto(self, obj=self.agent.tcp, target=t),
            "name": f"move {direction(current_target, last_target)}",
            "subgoal_segment":f"move {direction(current_target, last_target)}",
            "choice_label": f"move {direction(current_target, last_target)}",
            "demonstration": False,
            "failure_func": lambda expected=current_target, last=last_target: self._wrong_button_touch(expected_button=expected, last_button=last),
            "solve": lambda env, planner, t=current_target: solve_swingonto(env, planner, target=t),
            #"segment":current_target,
        })  

        # Store task list for RecordWrapper use
        self.task_list = tasks




    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos=reset_panda.get_reset_panda_param("qpos",gripper="stick")
            self.agent.reset(qpos)



    def _get_obs_extra(self, info: Dict):
        return dict()

    def evaluate(self,solve_complete_eval=False):
        self.successflag=torch.tensor([False])
        self.failureflag = torch.tensor([False])

        for idx, button in enumerate(self.buttons_grid):
           
            if is_obj_swing_onto(self, obj=self.agent.tcp, target=button):# Only execute when gripper is closed
                # Update start time to refresh highlight effect when repeatedly triggered
                self.highlight_starts[idx] =int(self.elapsed_steps[0].item())
                # Only record when not recording
                if self.after_demo==True:
                    if not self.achieved_list or self.achieved_list[-1] is not button:
                        self.achieved_list.append(button) # highlight=reach record, not necessarily target

        def _to_label(item):
            name = getattr(item, "name", None)
            return name if name is not None else str(item)
                # Backtrack from end of completed button sequence to check if it matches current target sequence exactly
        achieved_labels = [_to_label(item) for item in self.achieved_list]
        selected_labels = [_to_label(item) for item in getattr(self, "selected_buttons", [])]
        remaining = [label for label in selected_labels if label not in achieved_labels]
        if selected_labels:
            recent_achieved = achieved_labels[-len(selected_labels):]
            if len(recent_achieved) == len(selected_labels) and recent_achieved == selected_labels:
                logger.debug("match success")
                self.match=True
        # print(f"achieved_list: {achieved_labels}")
        # print(f"selected_buttons: {selected_labels}")
        # print(f"remaining_targets: {len(remaining)}")





        # Use encapsulated sequence task check function
        if(self.use_demonstrationwrapper==False):# change subgoal after planner ends during recording
            if solve_complete_eval==True:
                allow_subgoal_change_this_timestep=True
            else:
                allow_subgoal_change_this_timestep=False
        else:# during demonstration, video needs to call evaluate(solve_complete_eval), video ends and flag changes in demonstrationwrapper
            if solve_complete_eval==True or self.demonstration_record_traj==False:
                allow_subgoal_change_this_timestep=True
            else:
                allow_subgoal_change_this_timestep=False
        all_tasks_completed, current_task_name, task_failed,self.current_task_specialflag = sequential_task_check(self, self.task_list,allow_subgoal_change_this_timestep=allow_subgoal_change_this_timestep)

        if all_tasks_completed and self.match==False:# Manually set to fail if string match fails
            logger.debug("match failure")
            task_failed=True

        # If task failed, mark as failed immediately
        if task_failed:
            self.failureflag = torch.tensor([True])
            logger.debug(f"Task failed: {current_task_name}")

        # If static_check succeeds or all tasks completed, set success flag
        if all_tasks_completed and not task_failed:
            self.successflag = torch.tensor([True])

        return {
            "success": self.successflag,
            "fail": self.failureflag,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # tcp_to_obj_dist = torch.linalg.norm(
        #     self.agent.tcp_pose.p - self.agent.tcp_pose.p, axis=1
        # )
        # reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        # reward = reaching_reward*0
        reward=torch.tensor([0])
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5

    def _wrong_button_touch(self, expected_button, last_button=None):
        # If button touched by is_obj_swing_onto is neither current expected target nor previous button (debounce), check as error
        for button in self.buttons_grid:
            if button is expected_button:
                continue
            if last_button is not None and button is last_button:
                continue
            if is_obj_swing_onto(self, obj=self.agent.tcp, target=button):
                return True
        return False


#Robomme
    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):
        obs, reward, terminated, truncated, info = super().step(action)
        # def _to_label(item):
        #     name = getattr(item, "name", None)
        #     return name if name is not None else str(item)

        # # Backtrack from end of completed button sequence to check if it matches current target sequence exactly
        # achieved_labels = [_to_label(item) for item in self.achieved_list]
        # selected_labels = [_to_label(item) for item in getattr(self, "selected_buttons", [])]
        # remaining = [label for label in selected_labels if label not in achieved_labels]
        # if selected_labels:
        #     recent_achieved = achieved_labels[-len(selected_labels):]
        #     if len(recent_achieved) == len(selected_labels) and recent_achieved == selected_labels:
        #         print("match success")
        #         self.match=True
        # print(f"achieved_list: {achieved_labels}")
        # print(f"selected_buttons: {selected_labels}")
        # print(f"remaining_targets: {len(remaining)}")

        

        # Check if each button is swum onto, and record highlight start time
        cur_step = int(self.elapsed_steps[0].item())
        highlight_position(
            self,
            self.agent.tcp.pose.p,
            start_step=cur_step,
            end_step=cur_step + 40,
            cur_step=cur_step,
            disk_radius=0.005,
        )
        # for idx, button in enumerate(self.buttons_grid):
        #     if is_obj_swing_onto(self, obj=self.agent.tcp, target=button):
        #         # Update start time to refresh highlight effect when repeatedly triggered
        #         self.highlight_starts[idx] = cur_step
        #         # Only record when not recording
        #         if self.after_demo==True:
        #             if not self.achieved_list or self.achieved_list[-1] is not button:
        #                 self.achieved_list.append(button) # highlight=reach record, not necessarily target

        # Apply highlight effect to each triggered button
        for idx, button in enumerate(self.buttons_grid):
            start_step = self.highlight_starts.get(idx)
            if start_step is not None:
                highlight_obj(
                    self,
                    button,
                    start_step=start_step,
                    end_step=start_step + 40,
                    cur_step=cur_step,
                    disk_radius=0.02*1.002,
                    disk_half_length=0.01*2*1.002,
                    highlight_color=[1.0, 0.0, 0.0, 1.0],
                    use_target_style=True,
                )


        return obs, reward, terminated, truncated, info
