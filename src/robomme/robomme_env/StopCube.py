from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

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

from .utils import *
from .utils.subgoal_evaluate_func import static_check
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


@register_env("StopCube")
class StopCube(BaseEnv):

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



    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0,seed=0,Robomme_video_episode=None,Robomme_video_path=None,
                     **kwargs):
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

        self.seed = seed
        self.stop=False

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



        button_obb = build_button(
            self,
            center_xy=(-0.2, 0),
            scale=1.5,
            generator=generator,
            randomize=True,
        )
        #avoid = [button_obb]

        angles = torch.deg2rad(torch.tensor([0.0, 90.0, 0.0], dtype=torch.float32))
        rotate = matrix_to_quaternion(
                    euler_angles_to_matrix(angles, convention="XYZ")
                )
        
        target_x = torch.FloatTensor(1).uniform_(-0.1, 0.1, generator=generator).item()
        target_y = torch.FloatTensor(1).uniform_(-0.1, 0.1, generator=generator).item()
        self.target = build_purple_white_target(
                scene=self.scene,
                radius=self.cube_half_size*1.8,
                thickness=0.01,
                name="target",
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(p=[target_x, target_y, 0.01], q=rotate),
            )
        cube_color_rgb = torch.rand(3, generator=generator).tolist()
        cube_color = (cube_color_rgb[0], cube_color_rgb[1], cube_color_rgb[2], 1.0)
        self.cube= spawn_fixed_cube(
                self,
                position=[-0.3, -0.3,self.cube_half_size/2],
                half_size=self.cube_half_size,
                color=cube_color,
                name_prefix=f"target_cube",
                yaw=0.0,  # No rotation
            )


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):

            

            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos=reset_panda.get_reset_panda_param("qpos")
            self.agent.reset(qpos)
            self.stop = False
            self.stop_timestep = None
            self._task_failed_persistent = False

            # Use generator to generate interval value, floating 5 around 20 (range 15-25)
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            interval = torch.randint(27, 33, (1,), generator=generator).item()
            interval = 30
            self.interval = interval


            move_interval_list = [60,80,120]   
            #move_interval_list=[120]  
            idx = torch.randint(0, len(move_interval_list), (1,), generator=generator).item()
            self.move_interval = move_interval_list[idx]

            stop_time=torch.randint(2, 6, (1,), generator=generator).item()

            self.steps_press=self.move_interval*(stop_time)-self.move_interval/2
            self.stop_time_range = (
                self.move_interval * (stop_time - 1),
                self.move_interval * (stop_time ),
            )
            self.stop_time=stop_time
            # Get target xy coordinates (already randomized in _load_scene)
            target_pose = self.target.pose
            if isinstance(target_pose.p, torch.Tensor):
                target_x = target_pose.p[0, 0].item()
                target_y = target_pose.p[0, 1].item()
            else:
                target_x = target_pose.p[0]
                target_y = target_pose.p[1]
            target_center = np.array([target_x, target_y])

            # Generate random rotation angle (-30 to +30 degrees)
            rotation_angle = torch.FloatTensor(1).uniform_(-30, 30, generator=generator).item()
            rotation_rad = np.deg2rad(rotation_angle)

            # Define original start and end coordinates (around origin (0,0))
            original_start = np.array([0, -0.3])
            original_end = np.array([0, 0.3])

            # Rotation matrix
            cos_theta = np.cos(rotation_rad)
            sin_theta = np.sin(rotation_rad)
            rotation_matrix = np.array([
                [cos_theta, -sin_theta],
                [sin_theta, cos_theta]
            ])

            # Apply rotation (around origin), then add target xy coordinates
            self.start_pos_xy = rotation_matrix @ original_start + target_center
            self.end_pos_xy = rotation_matrix @ original_end + target_center

            # Set cube initial position to rotated start point
            self.cube.set_pose(sapien.Pose(p=[self.start_pos_xy[0], self.start_pos_xy[1], self.cube_half_size/2]))

            # Generate task list to move to each button sequentially
            tasks = []

            tasks.append(             {
                                "func": lambda: button_hover(self,button=self.button),
                                "name": "move to the top of the button to prepare",
                                "subgoal_segment": "move to the top of the button at <> to prepare",
                                "choice_label": "move to the top of the button to prepare",
                                "demonstration": False,
                                "failure_func": None,
                                "specialflag":"swap",
                                "solve": lambda env, planner: [solve_button_ready(env, planner, obj=self.button)],
                                "segment":self.cap_link 
                                },)

            final_abs_timestep = self.steps_press - interval
            static_checkpoints = list(range(100, int(final_abs_timestep), 100))
            if not static_checkpoints or static_checkpoints[-1] != final_abs_timestep:
                static_checkpoints.append(final_abs_timestep)

            for target_timestep in static_checkpoints:
                tasks.append({
                                    "func": lambda target_timestep=target_timestep: before_absTimestep(self, absTimestep=target_timestep),
                                    "name": "remain static",
                                    "subgoal_segment": "remain static",
                                    "choice_label": "remain static",
                                    "demonstration": False,
                                    "failure_func": None,
                                    "specialflag":"swap",
                                    "solve": lambda env, planner, target_timestep=target_timestep: solve_hold_obj_absTimestep(env, planner,absTimestep=target_timestep),
                                    },)
            tasks.append({
                        "func": lambda: is_obj_stopped_onto(self, obj=self.cube, target=self.target, stop=self.stop),
                        "name": "press the button to stop the cube on the target",
                        "subgoal_segment": "press the button to stop the cube on the target at <>",
                        "choice_label": "press button to stop the cube",
                        "demonstration": False,
                        "failure_func": lambda: None,
                        "solve": lambda env, planner: [solve_button(env, planner, obj=self.button,without_hold=True)
                                                       ],

                        "segment":self.target 
                        },
            )


            # Store task list for RecordWrapper use
            self.task_list = tasks

    def _get_obs_extra(self, info: Dict):
        return dict()




    def evaluate(self,solve_complete_eval=False):
        if not hasattr(self, "_task_failed_persistent"):
            self._task_failed_persistent = False
        self.successflag=torch.tensor([False])
        self.failureflag = torch.tensor([True]) if self._task_failed_persistent else torch.tensor([False])




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
        task_failed = task_failed or self._task_failed_persistent# Ensure overshoot is covered

###################################################
        if all_tasks_completed:
            correct=correct_timestep(self,time_range=self.stop_time_range,stop_timestep=self.stop_timestep)# identify which pass pressed, stopped on, count error
            if correct!= True:
                task_failed=True

        current_stop = self.stop or is_button_pressed(self, obj=self.button)# Extra check for timing issue!
        press_before = (not is_obj_stopped_onto(self, obj=self.cube, target=self.target, stop=current_stop)) and is_button_pressed(self, obj=self.button)
        #print(f"press_before",press_before)
        # Manually set to fail if not stopped on target
        if press_before== True:
            #import pdb; pdb.set_trace()
            task_failed=True
##################################################
        # Fail immediately if exceeded without press
        current_step = int(getattr(self, "elapsed_steps", 0))
        if current_step > self.move_interval * self.stop_time:
            if not all_tasks_completed:
                #The issue is that the environment continues running after the task is successfully completed, 
                # eventually triggering a timeout check that incorrectly marks the episode as a failure.
                task_failed = True


#################################################

        # If task failed, mark as failed immediately
        if task_failed:
            self._task_failed_persistent = True
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
        tcp_to_obj_dist = torch.linalg.norm(
            self.agent.tcp_pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward*0
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5


#Robomme
    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):
        

        if is_button_pressed(self, obj=self.button):# Chronological issue, MUST be placed before super!!!
            self.stop=True

            
        obs, reward, terminated, truncated, info = super().step(action)


        # Use the rotated xy coordinates calculated in _initialize_episode
        start_pos = [self.start_pos_xy[0], self.start_pos_xy[1], self.cube_half_size / 2]
        end_pos = [self.end_pos_xy[0], self.end_pos_xy[1], self.cube_half_size / 2]

        # Alternate between the two waypoints so the cube makes five passes
        for segment in range(5):
            move_straight_line(
                self,
                cube=self.cube,
                start_step=self.move_interval * segment,
                end_step=self.move_interval * (segment + 1),
                cur_step=int(self.elapsed_steps),
                start_pos=start_pos if segment % 2 == 0 else end_pos,
                end_pos=end_pos if segment % 2 == 0 else start_pos,
                stop=self.stop,
            )
        return obs, reward, terminated, truncated, info
