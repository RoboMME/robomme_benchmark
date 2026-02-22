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
from .utils.SceneGenerationError import SceneGenerationError
from .utils.subgoal_evaluate_func import static_check
from .utils.object_generation import spawn_fixed_cube, build_board_with_hole
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


@register_env("VideoPlaceButton")
class VideoPlaceButton(BaseEnv):
   
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



    config_easy = {
        'color': 1, 
        "additional_place":False,
        "swap":False,
        "targets":3
    }
    config_medium= {
        'color': 3, 
        "additional_place":False,
        "swap":False,
        "targets":4
    }
    config_hard = {
        'color': 3, 
        "additional_place":False,
        "swap":True,
        "targets":4
    }


    # Combine into a dictionary
    configs = {
        'hard': config_hard,
        'easy': config_easy,
        'medium': config_medium
    }


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
        self._episode_rng = torch.Generator()
        self._episode_rng.manual_seed(seed)

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

        # Use seed to randomly determine number of repetitions (1-5)
        generator = torch.Generator()
        generator.manual_seed(seed)

        self.onto_goalsite=False
        self.start_step=99999
        self.end_step=99999
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

        try:
            self.table_scene = TableSceneBuilder(
                self, robot_init_qpos_noise=self.robot_init_qpos_noise
            )
            self.table_scene.build()
            try:
                self.goal_site = spawn_random_target(
                    self,
                    avoid=None,  # Use current avoidance list, containing all spawned cubes
                    include_existing=False,  # Manually maintain list
                    include_goal=False,  # Manually maintain list
                    region_center=[-0.1, 0],
                    region_half_size=0.1,
                    radius=self.cube_half_size * 3,  # Use radius instead of half_size
                    thickness=0.005,  # target thickness
                    min_gap=self.cube_half_size * 1,  # Gap requirement same as cube
                    name_prefix=f"goal_site",
                    generator=generator,
                )
            except RuntimeError as exc:
                raise SceneGenerationError("goal_site sampling failed") from exc
            avoid = [self.goal_site]

            button_obb = build_button(
                self,
                center_xy=(0.1, 0),
                scale=1.5,
                generator=generator,
                randomize_range=(0.05, 0.3)
            )
            avoid.append(button_obb)

            self.all_cubes = []  # Save all cube objects

            # Initialize storage for each color group
            self.red_cubes = []
            self.red_cube_names = []
            self.blue_cubes = []
            self.blue_cube_names = []
            self.green_cubes = []
            self.green_cube_names = []

            cubes_per_color = 1
            color_groups = [
                {"color": (1, 0, 0, 1), "name": "red", "list": self.red_cubes, "name_list": self.red_cube_names},
                {"color": (0, 0, 1, 1), "name": "blue", "list": self.blue_cubes, "name_list": self.blue_cube_names},
                {"color": (0, 1, 0, 1), "name": "green", "list": self.green_cubes, "name_list": self.green_cube_names},
            ]
            shuffle_indices = torch.randperm(len(color_groups), generator=generator).tolist()
            color_groups = [color_groups[i] for i in shuffle_indices]

            self.target_color_name = color_groups[0]["name"]
            logger.debug(f"Target color selected: {self.target_color_name}")

            # Generate cubes for each color group
            for idx, group in enumerate(color_groups):
                if idx < self.configs[self.difficulty]["color"]:
                    for cube_idx in range(cubes_per_color):
                        try:
                            cube = spawn_random_cube(
                                self,
                                color=group["color"],
                                avoid=avoid,
                                include_existing=False,
                                include_goal=False,
                                region_center=[0 , 0],
                                region_half_size=0.2,
                                half_size=self.cube_half_size,
                                min_gap=self.cube_half_size,
                                random_yaw=True,
                                name_prefix=f"cube_{group['name']}_{cube_idx}",
                                generator=generator,
                            )
                        except RuntimeError as exc:
                            raise SceneGenerationError(
                                f"Failed to generate {group['name']} cube {cube_idx}: {exc}"
                            ) from exc

                        self.all_cubes.append(cube)
                        group["list"].append(cube)
                        cube_name = f"cube_{group['name']}_{cube_idx}"
                        group["name_list"].append(cube_name)
                        setattr(self, cube_name, cube)
                        avoid.append(cube)

                logger.debug(f"Generated {len(group['list'])} {group['name']} cubes")

            logger.debug(
                f"Generated {len(self.all_cubes)} cubes total (red: {len(self.red_cubes)}, blue: {len(self.blue_cubes)}, green: {len(self.green_cubes)})"
            )

            self.targets = []
            for i in range(4):
                if i < self.configs[self.difficulty]["targets"]:
                    try:
                        target = spawn_random_target(
                            self,
                            avoid=avoid,  # Use current avoidance list, containing all spawned cubes
                            include_existing=False,  # Manually maintain list
                            include_goal=False,  # Manually maintain list
                            region_center=[0, 0],
                            region_half_size=0.2,
                            radius=self.cube_half_size * 2,  # Use radius instead of half_size
                            thickness=0.005,  # target thickness
                            min_gap=self.cube_half_size * 1,  # Gap requirement same as cube
                            name_prefix=f"target_{i}",
                            generator=generator,
                        )
                    except RuntimeError as exc:
                        raise SceneGenerationError(f"Target {i + 1} sampling failed: {exc}") from exc

                    self.targets.append(target)
                    setattr(self, f"target_{i}", target)
                    avoid.append(target)

            if len(self.all_cubes) > 0:
                target_cube_idx = torch.randint(0, len(self.all_cubes), (1,), generator=generator).item()
                self.target_cube = self.all_cubes[target_cube_idx]

                if self.target_cube in self.red_cubes:
                    self.target_color_name = "red"
                elif self.target_cube in self.blue_cubes:
                    self.target_color_name = "blue"
                elif self.target_cube in self.green_cubes:
                    self.target_color_name = "green"

                logger.debug(
                    f"Target cube selected: {self.target_color_name} cube (index {target_cube_idx} in all_cubes)"
                )
            else:
                self.target_cube = None
                self.target_color_name = None
                logger.debug("No cubes generated, no target cube selected")

            self.non_target_cubes = [cube for cube in self.all_cubes if cube != self.target_cube]
            logger.debug(f"Non-target cubes: {len(self.non_target_cubes)}")

            self.swap_target_a = None
            self.swap_target_b = None
            self.swap_target_other = []

            if self.configs[self.difficulty]["swap"] == True:
                if len(self.targets) >= 2:
                    perm = torch.randperm(len(self.targets), generator=generator)
                    swap_idx_a = perm[0].item()
                    swap_idx_b = perm[1].item()
                    self.swap_target_a = self.targets[swap_idx_a]
                    self.swap_target_b = self.targets[swap_idx_b]
                    self.swap_target_other = [
                        target
                        for idx, target in enumerate(self.targets)
                        if idx not in (swap_idx_a, swap_idx_b)
                    ]
                    logger.debug(
                        f"Swap targets selected: target_{swap_idx_a} <-> target_{swap_idx_b}"
                    )

            if self.configs[self.difficulty]["additional_place"] == True:
                self.pre_flag = torch.rand(1, generator=generator).item() < 0.5
                self.post_flag = torch.rand(1, generator=generator).item() < 0.5
            else:
                self.pre_flag = 0
                self.post_flag = 0
            self.task_flag = torch.rand(1, generator=generator).item() < 0.5

            if self.task_flag == 1:
                self.target_target = self.target_0
                self.target_target_language = "before"
            else:
                self.target_target = self.target_1
                self.target_target_language = "after"

            self.targets_not_true = [
                t for i, t in enumerate(self.targets) if self.targets[i] != self.target_target
            ]

            tasks = []
            if self.pre_flag == True:
                tasks.append(
                    {
                        "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                        "name": f"pick up the cube",
                        "subgoal_segment": f"pick up the cube at <>",
                        "choice_label": "pick up the cube",
                        "demonstration": True,
                        "failure_func": None,
                        "solve": lambda env, planner: solve_pickup(
                            env, planner, obj=self.target_cube
                        ),
                        "segment": self.target_cube,
                    }
                )
                tasks.append(
                    {
                        "func": (
                            lambda: is_obj_dropped_onto(
                                self, obj=self.target_cube, target=self.target_2
                            )
                        ),
                        "name": "drop the cube onto target",
                        "subgoal_segment": f"drop the cube onto target at <>",
                        "choice_label": "drop onto",
                        "demonstration": True,
                        "failure_func": None,
                        "solve": lambda env, planner: solve_putonto_whenhold(
                            env, planner, target=self.target_2
                        ),
                        "segment": self.target_2,
                    }
                )

            tasks.append(
                {
                    "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                    "name": f"pick up the cube",
                    "subgoal_segment": f"pick up the cube at <>",
                    "choice_label": "pick up the cube",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: solve_pickup(
                        env, planner, obj=self.target_cube
                    ),
                    "segment": self.target_cube,
                }
            )
            tasks.append(
                {
                    "func": (
                        lambda: is_obj_dropped_onto(
                            self, obj=self.target_cube, target=self.target_0
                        )
                    ),
                    "name": "drop the cube onto target",
                    "subgoal_segment": f"drop the cube onto target at <>",
                    "choice_label": "drop onto",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: solve_putonto_whenhold(
                        env, planner, target=self.target_0
                    ),
                    "segment": self.target_0,
                }
            )

            tasks.append(
                {
                    "func": (lambda: is_button_pressed(self, obj=self.button)),
                    "name": "press the button",
                    "subgoal_segment": f"press the button at <>",
                    "choice_label": "press the button",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: solve_button(env, planner, self.button),
                    "segment": self.cap_link,
                }
            )

            if self.post_flag == True:
                tasks.append(
                    {
                        "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                        "name": f"pick up the cube",
                        "subgoal_segment": f"pick up the cube at <>",
                        "choice_label": "pick up the cube",
                        "demonstration": True,
                        "failure_func": None,
                        "solve": lambda env, planner: solve_pickup(
                            env, planner, obj=self.target_cube
                        ),
                        "segment": self.target_cube,
                    }
                )
                tasks.append(
                    {
                        "func": (
                            lambda: is_obj_dropped_onto(
                                self, obj=self.target_cube, target=self.target_3
                            )
                        ),
                        "name": "drop the cube onto target",
                        "subgoal_segment": f"drop the cube onto target at <>",
                        "choice_label": "drop onto",
                        "demonstration": True,
                        "failure_func": None,
                        "solve": lambda env, planner: solve_putonto_whenhold(
                            env, planner, target=self.target_3
                        ),
                        "segment": self.target_3,
                    }
                )

            tasks.append(
                {
                    "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                    "name": f"pick up the cube",
                    "subgoal_segment": f"pick up the cube at <>",
                    "choice_label": "pick up the cube",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: solve_pickup(
                        env, planner, obj=self.target_cube
                    ),
                    "segment": self.target_cube,
                }
            )
            tasks.append(
                {
                    "func": (
                        lambda: is_obj_dropped_onto(
                            self, obj=self.target_cube, target=self.target_1
                        )
                    ),
                    "name": "drop the cube onto target",
                    "subgoal_segment": f"drop the cube onto target at <>",
                    "choice_label": "drop onto",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: solve_putonto_whenhold(
                        env, planner, target=self.target_1
                    ),
                    "segment": self.target_1,
                }
            )

            tasks.append(
                {
                    "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                    "name": f"pick up the cube",
                    "subgoal_segment": f"pick up the cube at <>",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: solve_pickup(
                        env, planner, obj=self.target_cube
                    ),
                    "segment": self.target_cube,
                }
            )
            tasks.append(
                {
                    "func": (
                        lambda: is_obj_dropped_onto(
                            self, obj=self.target_cube, target=self.goal_site
                        )
                    ),
                    "name": "drop the cube onto table",
                    "subgoal_segment": f"drop the cube onto table",
                    "choice_label": "drop onto",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: [
                        solve_putonto_whenhold(env, planner, target=self.goal_site,height=0.01),
                    ],
                }
            )

            tasks.append(
                {
                    "func": lambda: static_check(
                        self, timestep=int(self.elapsed_steps), static_steps=20
                    ),
                    "name": "static",
                    "subgoal_segment": f"static",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: [
                        solve_reset(env, planner),
                        solve_hold_obj(env, planner, static_steps=20),
                    ],
                },
            )

            tasks.append(
                {
                    "func": lambda: static_check(
                        self, timestep=int(self.elapsed_steps), static_steps=60
                    ),
                    "name": "static",
                    "subgoal_segment": f"static",
                    "specialflag": "swap",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: [
                        solve_hold_obj(env, planner, static_steps=60)
                    ],
                },
            )

            tasks.append(
                {
                    "func": lambda: reset_check(self),
                    "name": "NO RECORD",
                    "subgoal_segment": f"NO RECORD",
                    "demonstration": True,
                    "failure_func": None,
                    "solve": lambda env, planner: [solve_strong_reset(env, planner)],
                },
            )

            tasks.append(
                {
                    "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
                    "name": f"pick up the cube",
                    "subgoal_segment": f"pick up the cube at <>",
                    "choice_label": "pick up the cube",
                    "demonstration": False,
                    "failure_func": lambda: is_any_obj_pickup(self, self.non_target_cubes),
                    "solve": lambda env, planner: [
                        solve_pickup(env, planner, obj=self.target_cube)
                    ],
                    "segment": self.target_cube,
                }
            )
            tasks.append(
                {
                    "func": (
                        lambda: is_obj_dropped_onto(
                            self, obj=self.target_cube, target=self.target_target
                        )
                    ),
                    "name": "place the cube onto the correct target",
                    "subgoal_segment": f"place the cube onto the correct target at <>",
                    "choice_label": "drop onto",
                    "demonstration": False,
                    "failure_func": (
                        lambda: is_obj_dropped_onto_any(
                            self, obj=self.target_cube, target=self.targets_not_true
                        )
                    ),
                    "solve": lambda env, planner: [
                        solve_putonto_whenhold(env, planner, target=self.target_target),
                    ],
                    "segment": self.target_target,
                }
            )

            self.task_list = tasks
            self.recovery_pickup_indices, self.recovery_pickup_tasks = task4recovery(
                self.task_list
            )
            if self.robomme_failure_recovery:
                self.fail_grasp_task_index = inject_fail_grasp(
                    self.task_list,
                    generator=generator,
                    mode=self.robomme_failure_recovery_mode,
                )
            else:
                self.fail_grasp_task_index = None

        except SceneGenerationError:
            raise
        except Exception as exc:
            raise SceneGenerationError(
                f"Failed to load VideoPlaceButton scene for seed {self.seed}"
            ) from exc


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos=reset_panda.get_reset_panda_param("qpos")
            self.agent.reset(qpos)
            pose_p=self.goal_site.pose.p.tolist()[0]
            pose_q=self.goal_site.pose.q.tolist()[0]
            pose_p[2]=-0.05
            self.goal_site.set_pose(sapien.Pose(p=pose_p,q=pose_q))  
            #print(self.goal_site.pose.p)  

    def _get_obs_extra(self, info: Dict):
        return dict()


 
    def evaluate(self,solve_complete_eval=False):
        self.successflag=torch.tensor([False])
        self.failureflag = torch.tensor([False])

      
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



        # highlight_obj(self,self.target_cube, start_step=0, end_step=100, cur_step=timestep)
        
        if self.current_task_specialflag=="swap":
            if self.onto_goalsite==False:
                self.onto_goalsite=True
                self.start_step=int(self.elapsed_steps.item())
                self.end_step=int(self.elapsed_steps.item())+50



        if self.swap_target_a is not None and self.swap_target_b is not None:
            other_bins = self.swap_target_other if self.swap_target_other else None
            swap_flat_two_lane(
                self,
                cube_a=self.swap_target_a,
                cube_b=self.swap_target_b,
                start_step=self.start_step,
                end_step=self.end_step,
                cur_step=self.elapsed_steps,
                lane_offset=0.1,
                smooth=True,
                keep_upright=True,
                other_cube=other_bins,
            )
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info
